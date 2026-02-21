# ==========================================
# 0. Setup / Imports
# ==========================================
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install datasets transformers sentencepiece nltk pandas scikit-learn matplotlib

import sys
import os
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from datasets import load_dataset
import sentencepiece as spm

# BLEU (via NLTK)
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
from nltk import bleu_score

# Device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Python exe:", sys.executable)
print("Torch:", torch.__version__)
print("Using device:", device)

# 再現性（※ここでは評価のみなので緩め）
DATA_RANDOM_STATE = 42

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ==========================================
# 1. Checkpoint root (★ここだけ自分の runs_XXXX に変更)
# ==========================================
# 例: トレーニング時の出力ディレクトリが runs_20251127-010203 だった場合
CHECKPOINT_ROOT = "runs_20251127-014234"  # <-- 実際のディレクトリ名に書き換え

# 評価対象モデル一覧（dense/rigl × 3 seeds を想定）
MODEL_SPECS = [
    ("dense", 0),
    ("dense", 1),
    ("dense", 2),
    ("rigl", 0),
    ("rigl", 1),
    ("rigl", 2),
]

# ==========================================
# 2. Load & prepare dataset (Sinhala → English)
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

# ==========================================
# 3. SentencePiece tokenizer
# ==========================================
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
BOS_TOKEN = '<S>'
EOS_TOKEN = '</S>'
PAD = 0
UNK = 1
BOS = 2
EOS = 3

print("Training or loading SentencePiece...")

# 既存モデルがあれば再利用、なければ学習
if not (os.path.exists("spm_src.model") and os.path.exists("spm_tgt.model")):
    with open("train_src.txt", "w", encoding="utf-8") as f:
        for text in train_df["src"]:
            f.write(str(text) + "\n")
    with open("train_tgt.txt", "w", encoding="utf-8") as f:
        for text in train_df["tgt"]:
            f.write(str(text) + "\n")

    spm.SentencePieceTrainer.train(
        input='train_src.txt', model_prefix='spm_src', vocab_size=8000, model_type='bpe',
        pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
        pad_piece=PAD_TOKEN, unk_piece=UNK_TOKEN,
        bos_piece=BOS_TOKEN, eos_piece=EOS_TOKEN
    )
    spm.SentencePieceTrainer.train(
        input='train_tgt.txt', model_prefix='spm_tgt', vocab_size=8000, model_type='bpe',
        pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
        pad_piece=PAD_TOKEN, unk_piece=UNK_TOKEN,
        bos_piece=BOS_TOKEN, eos_piece=EOS_TOKEN
    )
    print("[INFO] Trained new SentencePiece models.")
else:
    print("[INFO] Found existing SentencePiece models, using them.")

sp_src = spm.SentencePieceProcessor(model_file='spm_src.model')
sp_tgt = spm.SentencePieceProcessor(model_file='spm_tgt.model')

def sp_tokenize(text, sp_model):
    if not isinstance(text, str):
        return ""
    return " ".join(sp_model.encode(text, out_type=str))

print("Tokenizing Train/Test Data (single-process apply)...")
train_df["src"] = train_df["src"].apply(lambda x: sp_tokenize(x, sp_src))
train_df["tgt"] = train_df["tgt"].apply(lambda x: sp_tokenize(x, sp_tgt))
train_df = train_df.iloc[:MAX_TRAIN].reset_index(drop=True)

test_df["src"] = test_df["src"].apply(lambda x: sp_tokenize(x, sp_src))
test_df["tgt"] = test_df["tgt"].apply(lambda x: sp_tokenize(x, sp_tgt))
test_df = test_df.iloc[:MAX_TEST].reset_index(drop=True)

# ==========================================
# 4. Vocabulary & ID mapping
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

print("Splitting Data (train/valid, for vocab reproducibility)...")
train_X, valid_X, train_Y, valid_Y = train_test_split(
    train_X_list, train_Y_list,
    test_size=0.2, random_state=DATA_RANDOM_STATE
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

print("Converting Test data to IDs...")
test_X_list = test_df["src"].tolist()
test_Y_list = test_df["tgt"].tolist()
test_X = [sentence_to_ids(vocab_X, s) for s in test_X_list]
test_Y = [sentence_to_ids(vocab_Y, s) for s in test_Y_list]

# ==========================================
# 5. Transformer model definition
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

        outputs, attns = self.attention(
            q_s, k_s, v_s,
            attn_mask=attn_mask.repeat(n_head, 1, 1)
        )

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
# 6. BLEU utility
# ==========================================
def calc_bleu(refs, hyps):
    refs = [[ref[:ref.index(EOS)]] if EOS in ref else [ref] for ref in refs]
    hyps = [hyp[:hyp.index(EOS)] if EOS in hyp else hyp for hyp in hyps]
    return 100 * bleu_score.corpus_bleu(refs, hyps)

# ==========================================
# 7. Beam search decoding
# ==========================================
def beam_search_decode(model, src, beam_size=5, max_length=40):
    """
    src: (src_seq, src_pos)  with shape [1, L]
    return: best_ids (list[int], BOS除去, EOS含む場合あり)
    """
    model.eval()
    src_seq, src_pos = src
    batch_size = src_seq.size(0)
    assert batch_size == 1, "beam_search_decode assumes batch_size = 1"

    with torch.no_grad():
        enc_output, _ = model.encoder(src_seq, src_pos)

    init_seq = torch.full((1, 1), BOS, dtype=torch.long, device=device)
    beam = [(init_seq, 0.0, False)]  # (seq, log_prob, finished)

    for t in range(1, max_length + 1):
        new_beam = []
        all_finished = True

        for seq, score, finished in beam:
            if finished:
                new_beam.append((seq, score, True))
                continue

            all_finished = False

            pos = torch.arange(1, seq.size(1) + 1, dtype=torch.long, device=device)
            pos = pos.unsqueeze(0)

            with torch.no_grad():
                dec_output, _, _ = model.decoder(seq, pos, src_seq, enc_output)
                logits = model.tgt_word_proj(dec_output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)

            for k in range(beam_size):
                next_id = topk_ids[k].item()
                next_score = score + topk_log_probs[k].item()

                next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
                next_seq = torch.cat([seq, next_token], dim=1)

                next_finished = (next_id == EOS)
                new_beam.append((next_seq, next_score, next_finished))

        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]

        if all_finished:
            break

    best_seq, best_score, _ = max(beam, key=lambda x: x[1])
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

# ==========================================
# 8. Evaluation (beam search BLEU on test set)
# ==========================================
def evaluate_model_beam(model, beam_size=5, max_length=40, max_eval_samples=None):
    refs = []
    hyps = []

    model.eval()
    n_samples = len(test_X)
    if max_eval_samples is not None:
        n_samples = min(n_samples, max_eval_samples)

    for i in range(n_samples):
        src_ids = test_X[i]
        tgt_ids = test_Y[i]

        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_pos = torch.arange(1, len(src_ids) + 1, dtype=torch.long, device=device).unsqueeze(0)

        pred_ids = beam_search_decode(model, (src_tensor, src_pos), beam_size=beam_size, max_length=max_length)

        refs.append(tgt_ids)
        hyps.append(pred_ids)

        if (i + 1) % 100 == 0 or (i + 1) == n_samples:
            print(f"  Decoding {i+1}/{n_samples}", end='\r')

    print()
    bleu = calc_bleu(refs, hyps)
    return bleu

# ==========================================
# 9. Main: load checkpoints & evaluate
# ==========================================
print("\n===== Beam Search Evaluation (Test set) =====")

# 評価設定（CSVにも書く）
EVAL_BEAM_SIZE = 5
EVAL_MAX_LENGTH = 40
EVAL_MAX_SAMPLES = None  # 全テストを使う場合は None のまま

all_results = []
all_samples = []  # サンプル翻訳をここに溜めて CSV にする

set_global_seed(0)  # 評価時のわずかな乱数要素を固定（Dropoutはevalでoff）

for (model_type, seed) in MODEL_SPECS:
    run_name = f"transformer_{model_type}_seed{seed}"
    state_path = os.path.join(CHECKPOINT_ROOT, f"{run_name}_best_state_dict.pth")
    meta_path  = os.path.join(CHECKPOINT_ROOT, f"{run_name}_best_meta.pth")

    if not (os.path.exists(state_path) and os.path.exists(meta_path)):
        print(f"[WARN] Checkpoint not found for {run_name}, skip.")
        continue

    print(f"\n--- Evaluating {run_name} ---")
    meta = torch.load(meta_path, map_location=device)
    model_args = meta["model_args"]
    best_valid_bleu = meta.get("best_valid_bleu", None)

    # モデル構築＆ロード
    model = Transformer(**model_args).to(device)
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 評価
    start_time = time.time()
    bleu_beam = evaluate_model_beam(
        model,
        beam_size=EVAL_BEAM_SIZE,
        max_length=EVAL_MAX_LENGTH,
        max_eval_samples=EVAL_MAX_SAMPLES
    )
    elapsed = time.time() - start_time

    print(f"[RESULT] {run_name}: Test BLEU (beam={EVAL_BEAM_SIZE}) = {bleu_beam:.2f}")
    if best_valid_bleu is not None:
        print(f"          (Train-time best valid BLEU: {best_valid_bleu:.2f})")

    # サンプル翻訳を3件ほど（ sanity check & CSV用 ）
    print("  Sample translations:")
    num_samples = min(3, len(test_X))
    for idx in range(num_samples):
        src_ids = test_X[idx]
        tgt_ids = test_Y[idx]
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_pos = torch.arange(1, len(src_ids) + 1, dtype=torch.long,
                               device=device).unsqueeze(0)

        pred_ids = beam_search_decode(
            model,
            (src_tensor, src_pos),
            beam_size=EVAL_BEAM_SIZE,
            max_length=EVAL_MAX_LENGTH
        )

        # 表示用
        src_tok = test_X_list[idx]  # SentencePieceトークン列（スペース区切り）
        ref_str = ids_to_text(tgt_ids, vocab_Y, sp_tgt)
        pred_str = ids_to_text(pred_ids, vocab_Y, sp_tgt)

        print(f"    [#{idx}]")
        print(f"      SRC(tok): {src_tok}")
        print(f"      REF    : {ref_str}")
        print(f"      PRED   : {pred_str}")

        # CSV用に追加（元言語を復元したい場合は SRC_TEXT を decode_pieces で復元）
        src_pieces = src_tok.split()
        src_text = sp_src.decode_pieces(src_pieces)

        all_samples.append({
            "run_name": run_name,
            "model_type": model_type,
            "seed": seed,
            "sample_idx": idx,
            "src_text": src_text,
            "src_tok": src_tok,
            "ref_text": ref_str,
            "pred_text": pred_str,
        })

    # 集計結果を保存（1行）
    n_test_total = len(test_X)
    n_used = n_test_total if EVAL_MAX_SAMPLES is None else min(n_test_total, EVAL_MAX_SAMPLES)

    all_results.append({
        "run_name": run_name,
        "model_type": model_type,
        "seed": seed,
        "checkpoint_root": CHECKPOINT_ROOT,
        "state_path": os.path.abspath(state_path),
        "meta_path": os.path.abspath(meta_path),
        "vocab_size_src": vocab_size_X,
        "vocab_size_tgt": vocab_size_Y,
        "eval_beam_size": EVAL_BEAM_SIZE,
        "eval_max_length": EVAL_MAX_LENGTH,
        "eval_max_samples": EVAL_MAX_SAMPLES if EVAL_MAX_SAMPLES is not None else n_used,
        "test_size_used": n_used,
        "test_bleu_beam": bleu_beam,
        "best_valid_bleu": best_valid_bleu,
        "eval_elapsed_sec": elapsed,
        "device": str(device),
    })

# ==========================================
# 10. Save results to CSV
# ==========================================
if len(all_results) > 0:
    results_df = pd.DataFrame(all_results)
    out_csv = "beam_search_test_bleu.csv"
    results_df.to_csv(out_csv, index=False)

    print("\nAll results (per-model summary):")
    print(results_df)
    print(f"\nSaved beam-search test BLEU summary to {out_csv}")
else:
    print("\n[ERROR] No models were evaluated. Check CHECKPOINT_ROOT and file names.")

# サンプル翻訳も CSV に保存
if len(all_samples) > 0:
    samples_df = pd.DataFrame(all_samples)
    samples_csv = "beam_search_sample_translations.csv"
    samples_df.to_csv(samples_csv, index=False)

    print(f"\nSaved sample translations to {samples_csv}")

