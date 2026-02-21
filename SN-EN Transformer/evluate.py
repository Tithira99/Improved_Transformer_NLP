# ==========================================
# Eval Only Script: Dense vs RigL (mask-aware FLOPs)
# ==========================================
# すでに別スクリプトで学習し、
#   transformer_dense_best_state_dict.pth
#   transformer_dense_best_meta.pth
#   transformer_rigl_best_state_dict.pth
#   transformer_rigl_best_meta.pth
# が出来ている前提。

import sys
import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
from nltk import bleu_score

from datasets import load_dataset
import sentencepiece as spm

# RigL で使っていた層集合 W をそのまま使う
from utils_rigl import get_W

# FLOPs 計測 (fvcore)
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FlopCountAnalysis = None
    FVCORE_AVAILABLE = False
    print("[WARN] fvcore is not installed. FLOPs will not be computed.")

# Device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
random_state = 42

print("Python exe:", sys.executable)
print("Torch:", torch.__version__)
print("Using device:", device)

# ==========================================
# 1. Dataset & Tokenizer (元コードと同じ前処理)
# ==========================================
print("Loading Dataset...")

dataset = load_dataset("opus100", "en-si")
train = dataset["train"]
test = dataset["test"] if "test" in dataset else dataset["validation"]

train_df = pd.DataFrame(train["translation"]).rename(columns={"si": "src", "en": "tgt"})
test_df  = pd.DataFrame(test["translation"]).rename(columns={"si": "src", "en": "tgt"})

MAX_TRAIN = 100000
MAX_TEST  = 2000

train_df = train_df[train_df["src"].str.strip().astype(bool)]
train_df = train_df[train_df["tgt"].str.strip().astype(bool)]

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
BOS_TOKEN = '<S>'
EOS_TOKEN = '</S>'
PAD = 0
UNK = 1
BOS = 2
EOS = 3

print("Preparing SentencePiece models...")

# すでに spm_src.model / spm_tgt.model があれば再学習せず再利用
if not (os.path.exists("spm_src.model") and os.path.exists("spm_tgt.model")):
    print("SentencePiece models not found. Training new models...")
    with open("train_src.txt", "w", encoding="utf-8") as f:
        for text in train_df["src"]:
            f.write(str(text) + "\n")
    with open("train_tgt.txt", "w", encoding="utf-8") as f:
        for text in train_df["tgt"]:
            f.write(str(text) + "\n")

    spm.SentencePieceTrainer.train(
        input='train_src.txt', model_prefix='spm_src', vocab_size=8000, model_type='bpe',
        pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
        pad_piece=PAD_TOKEN, unk_piece=UNK_TOKEN, bos_piece=BOS_TOKEN, eos_piece=EOS_TOKEN
    )
    spm.SentencePieceTrainer.train(
        input='train_tgt.txt', model_prefix='spm_tgt', vocab_size=8000, model_type='bpe',
        pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
        pad_piece=PAD_TOKEN, unk_piece=UNK_TOKEN, bos_piece=BOS_TOKEN, eos_piece=EOS_TOKEN
    )
else:
    print("Found existing SentencePiece models. Skip training.")

sp_src = spm.SentencePieceProcessor(model_file='spm_src.model')
sp_tgt = spm.SentencePieceProcessor(model_file='spm_tgt.model')

def sp_tokenize(text, sp_model):
    if not isinstance(text, str):
        return ""
    return " ".join(sp_model.encode(text, out_type=str))

print("Tokenizing Train Data...")
train_df["src"] = train_df["src"].apply(lambda x: sp_tokenize(x, sp_src))
train_df["tgt"] = train_df["tgt"].apply(lambda x: sp_tokenize(x, sp_tgt))
train_df = train_df.iloc[:MAX_TRAIN].reset_index(drop=True)

print("Tokenizing Test Data...")
test_df["src"] = test_df["src"].apply(lambda x: sp_tokenize(x, sp_src))
test_df["tgt"] = test_df["tgt"].apply(lambda x: sp_tokenize(x, sp_tgt))
test_df = test_df.iloc[:MAX_TEST].reset_index(drop=True)

# ==========================================
# 2. Vocab & ID mapping
# ==========================================
class Vocab(object):
    def __init__(self, word2id={}):
        self.word2id = dict(word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def build_vocab(self, sentences, min_count=1):
        word_counter = {}
        for sentence in sentences:
            for word in sentence.split():
                word_counter[word] = word_counter.get(word, 0) + 1

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                break
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word

initial_word2id = {
    PAD_TOKEN: PAD,
    UNK_TOKEN: UNK,
    BOS_TOKEN: BOS,
    EOS_TOKEN: EOS
}
vocab_X = Vocab(word2id=initial_word2id)
vocab_Y = Vocab(word2id=initial_word2id)

train_X_list = train_df["src"].tolist()
train_Y_list = train_df["tgt"].tolist()

print("Splitting Data...")
train_X, valid_X, train_Y, valid_Y = train_test_split(
    train_X_list, train_Y_list,
    test_size=0.2, random_state=random_state
)

vocab_X.build_vocab(train_X, min_count=1)
vocab_Y.build_vocab(train_Y, min_count=1)

vocab_size_X = len(vocab_X.id2word)
vocab_size_Y = len(vocab_Y.id2word)
print('vocabX size = ', vocab_size_X)
print('vocabY size = ', vocab_size_Y)

def sentence_to_ids(vocab, sentence, max_seq_len=200):
    ids = [vocab.word2id.get(word, UNK) for word in sentence.split()]
    ids = ids[:max_seq_len - 2]
    ids = [BOS] + ids + [EOS]
    return ids

print("Converting to IDs...")
train_X = [sentence_to_ids(vocab_X, s) for s in train_X]
train_Y = [sentence_to_ids(vocab_Y, s) for s in train_Y]
valid_X = [sentence_to_ids(vocab_X, s) for s in valid_X]
valid_Y = [sentence_to_ids(vocab_Y, s) for s in valid_Y]

# ==========================================
# 3. DataLoader
# ==========================================
class DataLoader(object):
    def __init__(self, src_insts, tgt_insts, batch_size, shuffle=True):
        self.data = list(zip(src_insts, tgt_insts))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_index = 0
        self.reset()

    def reset(self):
        if self.shuffle:
            self.data = shuffle(self.data, random_state=random_state)
        self.start_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        def preprocess_seqs(seqs):
            max_length = max(len(s) for s in seqs)
            data = [s + [PAD] * (max_length - len(s)) for s in seqs]
            positions = [[pos + 1 if w != PAD else 0
                          for pos, w in enumerate(seq)]
                         for seq in data]
            return (
                torch.tensor(data, dtype=torch.long, device=device),
                torch.tensor(positions, dtype=torch.long, device=device)
            )

        if self.start_index >= len(self.data):
            self.reset()
            raise StopIteration()

        seqs = self.data[self.start_index:self.start_index + self.batch_size]
        src_seqs, tgt_seqs = zip(*seqs)

        src_data, src_pos = preprocess_seqs(src_seqs)
        tgt_data, tgt_pos = preprocess_seqs(tgt_seqs)

        self.start_index += self.batch_size
        return (src_data, src_pos), (tgt_data, tgt_pos)

MAX_LENGTH = 250
max_length = MAX_LENGTH + 2
batch_size = 16

train_dataloader = DataLoader(train_X, train_Y, batch_size)
valid_dataloader = DataLoader(valid_X, valid_Y, batch_size, shuffle=False)

# ==========================================
# 4. Transformer 定義（元コードと同じ）
# ==========================================
def position_encoding_init(n_position, d_pos_vec):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec)
        for pos in range(n_position)
    ])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    return torch.tensor(position_enc, dtype=torch.float)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        attn.data.masked_fill_(attn_mask, -float('inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.empty([n_head, d_model, d_k], dtype=torch.float))
        self.w_ks = nn.Parameter(torch.empty([n_head, d_model, d_k], dtype=torch.float))
        self.w_vs = nn.Parameter(torch.empty([n_head, d_model, d_v], dtype=torch.float))
        nn.init.xavier_normal_(self.w_qs)
        nn.init.xavier_normal_(self.w_ks)
        nn.init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.proj = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.proj.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        residual = q

        batch_size, len_q, d_model = q.size()
        batch_size, len_k, _ = k.size()
        batch_size, len_v, _ = v.size()

        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)

        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)

        outputs, attns = self.attention(q_s, k_s, v_s,
                                        attn_mask=attn_mask.repeat(n_head, 1, 1))

        outputs = torch.split(outputs, batch_size, dim=0)
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)
        self.layer_norm = nn.LayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)

def get_attn_padding_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(PAD).unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    attn_shape = (seq.size(1), seq.size(1))
    subsequent_mask = torch.triu(
        torch.ones(attn_shape, dtype=torch.uint8, device=seq.device),
        diagonal=1
    )
    return subsequent_mask.unsqueeze(0).expand(seq.size(0), -1, -1)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout
        )

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input,
            attn_mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    def __init__(self, n_src_vocab, max_length,
                 n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_word_vec=512, d_model=512, d_inner_hid=1024,
                 dropout=0.1):
        super(Encoder, self).__init__()
        n_position = max_length + 1
        self.max_length = max_length
        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_position, d_word_vec, padding_idx=PAD
        )
        self.position_enc.weight.data = position_encoding_init(
            n_position, d_word_vec
        )

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=PAD
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, src_seq, src_pos):
        enc_input = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)

        enc_output = enc_input
        enc_slf_attns = []
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask
            )
            enc_slf_attns += [enc_slf_attn]
        return enc_output, enc_slf_attns

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout
        )

    def forward(self, dec_input, enc_output,
                slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input,
            attn_mask=slf_attn_mask
        )
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output,
            attn_mask=dec_enc_attn_mask
        )
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self, n_tgt_vocab, max_length,
                 n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_word_vec=512, d_model=512, d_inner_hid=1024,
                 dropout=0.1):
        super(Decoder, self).__init__()
        n_position = max_length + 1
        self.max_length = max_length
        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_position, d_word_vec, padding_idx=PAD
        )
        self.position_enc.weight.data = position_encoding_init(
            n_position, d_word_vec
        )
        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=PAD
        )
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output):
        dec_input = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(
            dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0
        )

        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, src_seq)

        dec_output = dec_input
        dec_slf_attns, dec_enc_attns = [], []

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask
            )
            dec_slf_attns += [dec_slf_attn]
            dec_enc_attns += [dec_enc_attn]

        return dec_output, dec_slf_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, max_length,
                 n_layers=6, n_head=8,
                 d_word_vec=512, d_model=512, d_inner_hid=1024,
                 d_k=64, d_v=64, dropout=0.1,
                 proj_share_weight=True):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            n_src_vocab, max_length,
            n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout
        )
        self.decoder = Decoder(
            n_tgt_vocab, max_length,
            n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout
        )

        self.tgt_word_proj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_proj.weight)
        self.dropout = nn.Dropout(dropout)

        if proj_share_weight:
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src, tgt):
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        src_seq = src_seq[:, 1:]
        src_pos = src_pos[:, 1:]
        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_proj(dec_output)
        return seq_logit

# ==========================================
# 5. Metrics / Evaluation helpers
# ==========================================
def calc_bleu(refs, hyps):
    refs = [[ref[:ref.index(EOS)]] if EOS in ref else [ref] for ref in refs]
    hyps = [hyp[:hyp.index(EOS)] if EOS in hyp else hyp for hyp in hyps]
    return 100 * bleu_score.corpus_bleu(refs, hyps)

def compute_loss_eval(batch_X, batch_Y, model, criterion):
    model.eval()
    with torch.no_grad():
        pred_Y = model(batch_X, batch_Y)
        gold = batch_Y[0][:, 1:].contiguous()
        loss = criterion(
            pred_Y.view(-1, pred_Y.size(2)),
            gold.view(-1)
        )

    gold = gold.data.cpu().numpy().tolist()
    pred = pred_Y.max(dim=-1)[1].data.cpu().numpy().tolist()
    return loss.item(), gold, pred

def evaluate_model_on_valid(model, run_name="model"):
    model.eval()
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=PAD, reduction='sum'
    ).to(device)

    num_valid_batches = (len(valid_dataloader.data) + batch_size - 1) // batch_size

    total_loss = 0.0
    all_refs = []
    all_hyps = []

    print(f"\n[{run_name}] Evaluating on validation set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_dataloader, 1):
            batch_X, batch_Y = batch
            loss, gold, pred = compute_loss_eval(
                batch_X, batch_Y,
                model, criterion_eval
            )
            total_loss += loss
            all_refs += gold
            all_hyps += pred

            if (batch_idx % 50 == 0) or (batch_idx == num_valid_batches):
                print(
                    f"  [{run_name}][Valid] batch {batch_idx}/{num_valid_batches}",
                    end="\r"
                )
    print()

    avg_loss = total_loss / len(valid_dataloader.data)
    bleu = calc_bleu(all_refs, all_hyps)
    print(f"[{run_name}] valid_loss = {avg_loss:.4f}, valid_BLEU = {bleu:.2f}")
    return avg_loss, bleu

# Dense アーキ (完全 dense) としての FLOPs
def compute_dense_transformer_flops_on_dummy(model_cls, model_args,
                                             src_len=50, tgt_len=50, batch_size=1):
    if not FVCORE_AVAILABLE:
        return None

    model = model_cls(**model_args).cpu()
    model.eval()

    src_seq = torch.full(
        (batch_size, src_len), fill_value=PAD, dtype=torch.long
    )
    tgt_seq = torch.full(
        (batch_size, tgt_len), fill_value=PAD, dtype=torch.long
    )
    src_seq[:, 0] = BOS
    tgt_seq[:, 0] = BOS

    src_pos = torch.arange(
        1, src_len + 1, dtype=torch.long
    ).unsqueeze(0).repeat(batch_size, 1)
    tgt_pos = torch.arange(
        1, tgt_len + 1, dtype=torch.long
    ).unsqueeze(0).repeat(batch_size, 1)

    inputs = ((src_seq, src_pos), (tgt_seq, tgt_pos))
    flops = FlopCountAnalysis(model, inputs).total()
    return flops

# RigL のマスク（ゼロ重み）を考慮して density を推定
def estimate_density_from_state_dict(state_dict, model_args):
    model_cpu = Transformer(**model_args).cpu()
    model_cpu.load_state_dict(state_dict, strict=True)

    W, _ = get_W(model_cpu, return_linear_layers_mask=True)

    total_params = 0
    total_nonzero = 0
    for w in W:
        n = w.numel()
        nz = (w != 0).sum().item()
        total_params += n
        total_nonzero += nz

    density = (total_nonzero / total_params) if total_params > 0 else 1.0
    return density, total_nonzero, total_params

# ==========================================
# 6. Eval: Dense vs RigL （学習一切なし）
# ==========================================
print("\n========== EVAL ONLY: Dense vs RigL (mask-aware FLOPs) ==========")

# --- Dense ---
dense_state_path = "transformer_dense_best_state_dict.pth"
dense_meta_path  = "transformer_dense_best_meta.pth"
if not (os.path.exists(dense_state_path) and os.path.exists(dense_meta_path)):
    raise FileNotFoundError(
        "Dense の best_state_dict / meta が見つかりません。"
        " 元の学習スクリプトで transformer_dense_* を先に生成してください。"
    )

dense_state = torch.load(dense_state_path, map_location=device)
dense_meta  = torch.load(dense_meta_path, map_location=device)
dense_model_args = dense_meta.get("model_args", {
    'n_src_vocab': len(vocab_X.word2id),
    'n_tgt_vocab': len(vocab_Y.word2id),
    'max_length': max_length,
    'proj_share_weight': True,
    'd_k': 32,
    'd_v': 32,
    'd_model': 128,
    'd_word_vec': 128,
    'd_inner_hid': 256,
    'n_layers': 3,
    'n_head': 6,
    'dropout': 0.1,
})

dense_model_eval = Transformer(**dense_model_args).to(device)
dense_model_eval.load_state_dict(dense_state, strict=True)

dense_valid_loss, dense_valid_bleu = evaluate_model_on_valid(
    dense_model_eval, run_name="Dense"
)

dense_flops_arch = compute_dense_transformer_flops_on_dummy(
    Transformer, dense_model_args, src_len=50, tgt_len=50, batch_size=1
)
if dense_flops_arch is not None:
    print(f"[Dense] Dense FLOPs (arch, seq len 50/50) = {dense_flops_arch:.3e}")
else:
    print("[Dense] FLOPs could not be computed (fvcore not installed).")

dense_effective_flops = dense_flops_arch  # Dense は density=1.0

# --- RigL ---
rigl_state_path = "transformer_rigl_best_state_dict.pth"
rigl_meta_path  = "transformer_rigl_best_meta.pth"
if not (os.path.exists(rigl_state_path) and os.path.exists(rigl_meta_path)):
    raise FileNotFoundError(
        "RigL の best_state_dict / meta が見つかりません。"
        " 元の学習スクリプトで transformer_rigl_* を先に生成してください。"
    )

rigl_state = torch.load(rigl_state_path, map_location=device)
rigl_meta  = torch.load(rigl_meta_path, map_location=device)
rigl_model_args = rigl_meta.get("model_args", dense_model_args)

rigl_model_eval = Transformer(**rigl_model_args).to(device)
rigl_model_eval.load_state_dict(rigl_state, strict=True)

rigl_valid_loss, rigl_valid_bleu = evaluate_model_on_valid(
    rigl_model_eval, run_name="RigL"
)

# マスク（ゼロ weight）を考慮した density + 有効 FLOPs
density, nz, total = estimate_density_from_state_dict(rigl_state, rigl_model_args)
if dense_flops_arch is not None:
    rigl_effective_flops = dense_flops_arch * density
else:
    rigl_effective_flops = None

print(f"[RigL] param density ≈ {density:.4f} ({nz} / {total} non-zero params)")
if rigl_effective_flops is not None:
    print(f"[RigL] Effective FLOPs (density-scaled) = {rigl_effective_flops:.3e}")
else:
    print("[RigL] FLOPs could not be computed (fvcore not installed).")

# ==========================================
# 7. Save evaluation to CSV
# ==========================================
rows = [
    {
        "model": "dense",
        "valid_loss": dense_valid_loss,
        "valid_bleu": dense_valid_bleu,
        "flops_arch": dense_flops_arch if dense_flops_arch is not None else np.nan,
        "effective_flops": dense_effective_flops if dense_effective_flops is not None else np.nan,
        "density": 1.0,
    },
    {
        "model": "rigl",
        "valid_loss": rigl_valid_loss,
        "valid_bleu": rigl_valid_bleu,
        "flops_arch": dense_flops_arch if dense_flops_arch is not None else np.nan,
        "effective_flops": rigl_effective_flops if rigl_effective_flops is not None else np.nan,
        "density": density,
    },
]

eval_df = pd.DataFrame(rows)
csv_path = "bleu_flops_eval_masked.csv"
eval_df.to_csv(csv_path, index=False)
print(f"\n[Eval] Saved evaluation CSV to: {csv_path}")
print(eval_df)

