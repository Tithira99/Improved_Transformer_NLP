# ==========================================
# 0. Setup / 事前準備 (ターミナルで実行)
# ==========================================
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install datasets transformers sentencepiece pandarallel nltk pandas scikit-learn matplotlib
# （FLOPsを取りたい場合）:
# pip install fvcore

# ==========================================
# 1. Imports & global config / インポートと基本設定
# ==========================================
import sys
import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# HuggingFace datasets, SentencePiece, Pandarallel
from datasets import load_dataset
import sentencepiece as spm
from pandarallel import pandarallel

# BLEU (via NLTK)
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')  # 新しい NLTK では必要になる場合がある
from nltk import bleu_score

# RigL (Dynamic Sparse Training)
from rigl_scheduler import RigLScheduler  # あなたの rigl_scheduler.py

# FLOPs 計測 (fvcore)
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FlopCountAnalysis = None
    FVCORE_AVAILABLE = False
    print("[WARN] fvcore is not installed. FLOPs will not be computed.")

# Device config / デバイス設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
random_state = 42

print("Python exe:", sys.executable)
print("Torch:", torch.__version__)
print("Using device:", device)

# AMP scaler (PyTorch 2.x style)
# CUDA でのみ有効
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

# ==========================================
# 2. Load & prepare dataset (Sinhala → English)
#    データセット読み込みと整形 (シンハラ語 → 英語)
# ==========================================
print("Loading Dataset...")

# OPUS100: English–Sinhala pair ("en-si").
# We use Sinhala as source (src), English as target (tgt).
dataset = load_dataset("opus100", "en-si")

train = dataset["train"]
test = dataset["test"] if "test" in dataset else dataset["validation"]

# Map columns:
#   "si" (Sinhala) -> "src"
#   "en" (English) -> "tgt"
train_df = pd.DataFrame(train["translation"]).rename(columns={"si": "src", "en": "tgt"})
test_df  = pd.DataFrame(test["translation"]).rename(columns={"si": "src", "en": "tgt"})

# Limit dataset size (adjust for your HW)
MAX_TRAIN = 100000
MAX_TEST  = 2000

# Remove empty sentences
train_df = train_df[train_df["src"].str.strip().astype(bool)]
train_df = train_df[train_df["tgt"].str.strip().astype(bool)]

# ==========================================
# 3. トークナイザー (SentencePiece) の構築
# ==========================================
# 特殊トークンID (PAD=0, UNK=1, BOS=2, EOS=3)
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
BOS_TOKEN = '<S>'
EOS_TOKEN = '</S>'
PAD = 0
UNK = 1
BOS = 2
EOS = 3

print("Training SentencePiece...")
# 学習用テキスト書き出し
with open("train_src.txt", "w", encoding="utf-8") as f:
    for text in train_df["src"]:
        f.write(str(text) + "\n")
with open("train_tgt.txt", "w", encoding="utf-8") as f:
    for text in train_df["tgt"]:
        f.write(str(text) + "\n")

# SentencePiece学習
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

# ロード
sp_src = spm.SentencePieceProcessor(model_file='spm_src.model')
sp_tgt = spm.SentencePieceProcessor(model_file='spm_tgt.model')

# トークナイズ関数 (空白区切り文字列を返す)
def sp_tokenize(text, sp_model):
    if not isinstance(text, str):
        return ""
    return " ".join(sp_model.encode(text, out_type=str))

# ==== pandarallel を使わないシングルプロセス版 ====
print("Tokenizing Train Data (single-process apply)...")
train_df["src"] = train_df["src"].apply(lambda x: sp_tokenize(x, sp_src))
train_df["tgt"] = train_df["tgt"].apply(lambda x: sp_tokenize(x, sp_tgt))
train_df = train_df.iloc[:MAX_TRAIN].reset_index(drop=True)

print("Tokenizing Test Data (single-process apply)...")
test_df["src"] = test_df["src"].apply(lambda x: sp_tokenize(x, sp_src))
test_df["tgt"] = test_df["tgt"].apply(lambda x: sp_tokenize(x, sp_tgt))
test_df = test_df.iloc[:MAX_TEST].reset_index(drop=True)

# ==========================================
# 4. Vocabulary & ID mapping / 語彙クラスと ID 変換
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

# Initialize vocab with special tokens
initial_word2id = {
    PAD_TOKEN: PAD,
    UNK_TOKEN: UNK,
    BOS_TOKEN: BOS,
    EOS_TOKEN: EOS
}
vocab_X = Vocab(word2id=initial_word2id)
vocab_Y = Vocab(word2id=initial_word2id)

train_X_list = train_df["src"].tolist()  # Sinhala (source)
train_Y_list = train_df["tgt"].tolist()  # English (target)

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
# 5. Custom DataLoader / データローダ
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

# ==========================================
# 6. Transformer model definition
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
    """
    Subsequent mask for decoder (prevent attending to future positions).
    ここを修正: seq と同じ device 上にマスクを作る。
    """
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
# 7. Hyperparameters & DataLoaders
# ==========================================
MAX_LENGTH = 250
max_length = MAX_LENGTH + 2
batch_size = 16
num_epochs = 30
lr = 0.001

model_args = {
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
}

train_dataloader = DataLoader(train_X, train_Y, batch_size)
valid_dataloader = DataLoader(valid_X, valid_Y, batch_size, shuffle=False)

# ==========================================
# 8. FLOPs utility
# ==========================================
def compute_transformer_flops_on_dummy(model_cls, model_args,
                                       src_len=50, tgt_len=50, batch_size=1):
    """
    Transformer の FLOPs を dummy 入力で計測する。
    """
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

# ==========================================
# 9. Training & evaluation loop
# ==========================================
def calc_bleu(refs, hyps):
    refs = [[ref[:ref.index(EOS)]] if EOS in ref else [ref] for ref in refs]
    hyps = [hyp[:hyp.index(EOS)] if EOS in hyp else hyp for hyp in hyps]
    return 100 * bleu_score.corpus_bleu(refs, hyps)

def compute_loss(batch_X, batch_Y, model, criterion,
                 optimizer=None, is_train=True):
    model.train(is_train)
    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        pred_Y = model(batch_X, batch_Y)          # (B, T, vocab)
        gold = batch_Y[0][:, 1:].contiguous()
        loss = criterion(
            pred_Y.view(-1, pred_Y.size(2)),
            gold.view(-1)
        )

    if is_train and optimizer is not None:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    gold = gold.data.cpu().numpy().tolist()
    pred = pred_Y.max(dim=-1)[1].data.cpu().numpy().tolist()
    return loss.item(), gold, pred

def train_one_model(run_name,
                    use_rigl=False,
                    rigl_dense_allocation=0.2,
                    rigl_delta=100,
                    rigl_alpha=0.3,
                    rigl_grad_accum_n=4):
    """
    1 モデル (dense or DST) を学習して結果を返す。
    """
    print(f"\n===== Train run: {run_name}  (use_rigl={use_rigl}) =====")

    model = Transformer(**model_args).to(device)
    if torch.cuda.device_count() > 1:
        print(f"[{run_name}] Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion_local = nn.CrossEntropyLoss(
        ignore_index=PAD, reduction='sum'
    ).to(device)

    rigl_scheduler = None
    if use_rigl:
        num_train_batches = (len(train_dataloader.data) + batch_size - 1) // batch_size
        total_steps = num_epochs * num_train_batches

        rigl_model = model.module if isinstance(model, nn.DataParallel) else model
        rigl_scheduler = RigLScheduler(
            model=rigl_model,
            optimizer=optimizer,
            dense_allocation=rigl_dense_allocation,
            T_end=total_steps,
            sparsity_distribution="uniform",
            ignore_linear_layers=False,
            delta=rigl_delta,
            alpha=rigl_alpha,
            static_topo=False,
            grad_accumulation_n=rigl_grad_accum_n,
            state_dict=None,
        )
        print(f"[{run_name}] RigL scheduler initialized: {rigl_scheduler}")

    best_valid_bleu = 0.0
    best_epoch = 0
    best_state_dict = None

    num_train_batches = (len(train_dataloader.data) + batch_size - 1) // batch_size
    num_valid_batches = (len(valid_dataloader.data) + batch_size - 1) // batch_size

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss = 0.0
        train_refs = []
        train_hyps = []

        valid_loss = 0.0
        valid_refs = []
        valid_hyps = []

        print(f"\n[{run_name}] [Epoch {epoch}/{num_epochs}] Training...")
        for batch_idx, batch in enumerate(train_dataloader, 1):
            batch_X, batch_Y = batch
            loss, gold, pred = compute_loss(
                batch_X, batch_Y,
                model, criterion_local, optimizer,
                is_train=True
            )
            train_loss += loss
            train_refs += gold
            train_hyps += pred

            if rigl_scheduler is not None:
                rigl_scheduler()

            if (batch_idx % 100 == 0) or (batch_idx == num_train_batches):
                print(
                    f"  [{run_name}][Train] batch {batch_idx}/{num_train_batches}",
                    end='\r'
                )

        print()

        print(f"[{run_name}] [Epoch {epoch}/{num_epochs}] Validation...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_dataloader, 1):
                batch_X, batch_Y = batch
                loss, gold, pred = compute_loss(
                    batch_X, batch_Y,
                    model, criterion_local,
                    optimizer=None, is_train=False
                )
                valid_loss += loss
                valid_refs += gold
                valid_hyps += pred

                if (batch_idx % 50 == 0) or (batch_idx == num_valid_batches):
                    print(
                        f"  [{run_name}][Valid] batch {batch_idx}/{num_valid_batches}",
                        end='\r'
                    )

        print()

        scheduler_lr.step()

        train_loss /= len(train_dataloader.data)
        valid_loss /= len(valid_dataloader.data)

        train_bleu = calc_bleu(train_refs, train_hyps)
        valid_bleu = calc_bleu(valid_refs, valid_hyps)

        if valid_bleu > best_valid_bleu:
            best_valid_bleu = valid_bleu
            best_epoch = epoch
            best_state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel) else
                model.state_dict()
            )

            state_path = f"{run_name}_best_state_dict.pth"
            meta_path = f"{run_name}_best_meta.pth"

            torch.save(best_state_dict, state_path)
            torch.save(
                {"best_epoch": best_epoch,
                 "best_valid_bleu": best_valid_bleu,
                 "model_args": model_args},
                meta_path
            )
            print(
                f"  [{run_name}] New best at epoch {epoch} "
                f"(valid BLEU: {valid_bleu:.2f})"
            )
            print(f"  [{run_name}] Saved weights to: {state_path}")
            print(f"  [{run_name}] Saved meta    to: {meta_path}")

        elapsed_time = (time.time() - start) / 60.0
        print(
            f'[{run_name}] Epoch {epoch} [{elapsed_time:.1f}min]: '
            f'train_loss: {train_loss:.2f}  train_bleu: {train_bleu:.2f}  '
            f'valid_loss: {valid_loss:.2f}  valid_bleu: {valid_bleu:.2f}'
        )

    if best_state_dict is None:
        raise RuntimeError(f"[{run_name}] No best_state_dict. Training failed?")

    flops = compute_transformer_flops_on_dummy(
        Transformer, model_args,
        src_len=50, tgt_len=50, batch_size=1
    )

    print(
        f"\n[{run_name}] Training finished. "
        f"Best epoch: {best_epoch}, best valid BLEU: {best_valid_bleu:.2f}"
    )
    if flops is not None:
        print(f"[{run_name}] Estimated FLOPs (dummy seq len 50/50): {flops:.3e}")
    else:
        print(f"[{run_name}] FLOPs could not be computed (fvcore not installed).")

    return {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_valid_bleu": best_valid_bleu,
        "best_state_dict": best_state_dict,
        "flops": flops,
    }

print("Start Training (Dense & RigL)...")

dense_result = train_one_model(
    run_name="transformer_dense",
    use_rigl=False
)

rigl_result = train_one_model(
    run_name="transformer_rigl",
    use_rigl=True,
    rigl_dense_allocation=0.2,   # dense 20%, zero 80%
    rigl_delta=100,
    rigl_alpha=0.3,
    rigl_grad_accum_n=4
)

print("\n===== Summary: Dense vs RigL (DST) =====")
if dense_result['flops'] is not None:
    print(f"Dense: best BLEU = {dense_result['best_valid_bleu']:.2f}, "
          f"FLOPs = {dense_result['flops']:.3e}")
else:
    print(f"Dense: best BLEU = {dense_result['best_valid_bleu']:.2f}, FLOPs = N/A")

if rigl_result['flops'] is not None:
    print(f"RigL : best BLEU = {rigl_result['best_valid_bleu']:.2f}, "
          f"FLOPs = {rigl_result['flops']:.3e}")
else:
    print(f"RigL : best BLEU = {rigl_result['best_valid_bleu']:.2f}, FLOPs = N/A")

# ==========================================
# 10. Save results to CSV / CSV に保存
# ==========================================
results_rows = []

results_rows.append({
    "model": "dense",
    "best_epoch": dense_result["best_epoch"],
    "best_valid_bleu": dense_result["best_valid_bleu"],
    "flops": dense_result["flops"] if dense_result["flops"] is not None else np.nan,
})

results_rows.append({
    "model": "rigl",
    "best_epoch": rigl_result["best_epoch"],
    "best_valid_bleu": rigl_result["best_valid_bleu"],
    "flops": rigl_result["flops"] if rigl_result["flops"] is not None else np.nan,
})

results_df = pd.DataFrame(results_rows)
results_csv_path = "bleu_flops_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved summary CSV to: {results_csv_path}")
print(results_df)

# ==========================================
# 11. Inference (Beam Search) / 推論 (ビームサーチ)
# ==========================================
print("\n--- Loading Best RigL Model for Inference ---")

state_path = "transformer_rigl_best_state_dict.pth"
meta_path = "transformer_rigl_best_meta.pth"

if os.path.exists(state_path) and os.path.exists(meta_path):
    best_state = torch.load(state_path, map_location=device)
    meta = torch.load(meta_path, map_location=device)

    model_eval = Transformer(**meta["model_args"]).to(device)
    model_eval.load_state_dict(best_state)
    model_eval.eval()

    print(
        f"Loaded RigL model from epoch {meta['best_epoch']} "
        f"(best valid BLEU: {meta['best_valid_bleu']:.2f})"
    )
else:
    print("Best RigL checkpoint not found, falling back to dense best model.")
    model_eval = Transformer(**model_args).to(device)
    model_eval.load_state_dict(dense_result["best_state_dict"])
    model_eval.eval()

def beam_search_decode(model, src, beam_size=5, max_length=40):
    model.eval()
    src_seq, src_pos = src

    with torch.no_grad():
        enc_output, _ = model.encoder(src_seq, src_pos)

    init_seq = torch.full((1, 1), BOS, dtype=torch.long, device=device)
    beam = [(init_seq, 0.0, False)]

    for t in range(1, max_length + 1):
        new_beam = []
        all_finished = True

        for seq, score, finished in beam:
            if finished:
                new_beam.append((seq, score, True))
                continue

            all_finished = False

            pos = torch.arange(
                1, seq.size(1) + 1,
                dtype=torch.long, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                dec_output, _, _ = model.decoder(seq, pos, src_seq, enc_output)
                logits = model.tgt_word_proj(dec_output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)

            for k in range(beam_size):
                next_id = topk_ids[k].item()
                next_score = score + topk_log_probs[k].item()

                next_token = torch.tensor(
                    [[next_id]], dtype=torch.long, device=device
                )
                next_seq = torch.cat([seq, next_token], dim=1)

                next_finished = (next_id == EOS)
                new_beam.append((next_seq, next_score, next_finished))

        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]

        if all_finished:
            break

    best_seq, _, _ = max(beam, key=lambda x: x[1])
    best_ids = best_seq[0].tolist()

    if len(best_ids) > 0 and best_ids[0] == BOS:
        best_ids = best_ids[1:]

    return best_ids

def ids_to_text(ids, vocab, sp_model):
    if EOS in ids:
        ids = ids[:ids.index(EOS)]

    tokens = [vocab.id2word.get(i, UNK_TOKEN) for i in ids]
    clean_tokens = [t for t in tokens if t not in [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]]
    return sp_model.decode_pieces(clean_tokens)

print("\n--- Translation Test (Beam Search) ---")

src_text = test_df.iloc[0]["src"]
ref_text = test_df.iloc[0]["tgt"]

# src_text はすでに BPE 空白区切りなのでそのまま使う
src_ids = sentence_to_ids(vocab_X, src_text)

src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
src_pos = torch.arange(
    1, len(src_ids) + 1,
    dtype=torch.long, device=device
).unsqueeze(0)

beam_ids = beam_search_decode(model_eval, (src_tensor, src_pos), beam_size=5)
pred_str = ids_to_text(beam_ids, vocab_Y, sp_tgt)

print(f"Input (Si tokenized): {src_text}")
print(f"Ref   (En tokenized): {ref_text}")
print(f"Pred  (En decoded)  : {pred_str}")

