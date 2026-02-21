import math
import time
from typing import List, Tuple

import datasets
from datasets import load_dataset

import nltk
from nltk import bleu_score

import MeCab
import unidic  # noqa: F401  # dictionary installation side-effect in some envs

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# =============================================================================
# Config
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

RANDOM_STATE = 42

MAX_TRAIN = 2500   # データ数上限（メモリ対策用）
MAX_TEST = 250

MAX_LENGTH = 250
MAX_SEQ_LEN = MAX_LENGTH + 2  # BOS/EOS込み

BATCH_SIZE = 16
NUM_EPOCHS = 20
LR = 1e-3

CKPT_PATH = "transformer.pth"

# BLEU を計算するときに保持するサンプル数の上限（メモリ削減用）
MAX_BLEU_SAMPLES_TRAIN = 4000
MAX_BLEU_SAMPLES_VALID = 2000
MAX_BLEU_SAMPLES_TEST = 4000

# Transformer モデルのハイパラ
MODEL_ARGS = {
    "n_src_vocab": None,        # 後で設定
    "n_tgt_vocab": None,        # 後で設定
    "max_length": MAX_SEQ_LEN,
    "proj_share_weight": True,
    "d_k": 32,
    "d_v": 32,
    "d_model": 128,
    "d_word_vec": 128,
    "d_inner_hid": 256,
    "n_layers": 3,
    "n_head": 6,
    "dropout": 0.1,
}

# special tokens
PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<S>"
EOS_TOKEN = "</S>"

# -----------------------------------------------------------------------------
# NLTK / MeCab 準備
# -----------------------------------------------------------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

# MeCab: グローバルに 1 つだけ作成
MECAB_TAGGER = MeCab.Tagger("-Owakati")


def src_tokenize(text: str) -> List[str]:
    """英語トークナイズ（NLTK）"""
    return nltk.tokenize.word_tokenize(text)


def tgt_tokenize(text: str) -> List[str]:
    """日本語トークナイズ（MeCab, wakati）"""
    parsed = MECAB_TAGGER.parse(text)
    if parsed is None:
        return []
    return parsed.strip().split()


# =============================================================================
# Vocab
# =============================================================================

class Vocab:
    def __init__(self, word2id=None):
        if word2id is None:
            word2id = {}
        self.word2id = dict(word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def build_vocab(self, sentences: List[str], min_count: int = 1) -> None:
        """
        sentences: 各要素が「空白区切りトークン列」の文字列
        """
        word_counter = {}
        for sentence in sentences:
            for word in sentence.split():
                word_counter[word] = word_counter.get(word, 0) + 1

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                break
            if word not in self.word2id:
                _id = len(self.word2id)
                self.word2id[word] = _id
                self.id2word[_id] = word

    def __len__(self) -> int:
        return len(self.id2word)


def sentence_to_ids(vocab: Vocab, sentence: str, max_seq_len: int = MAX_SEQ_LEN) -> List[int]:
    """
    sentence: 空白区切りトークン列
    """
    ids = [vocab.word2id.get(word, UNK) for word in sentence.split()]
    ids = ids[: max_seq_len - 2]
    ids = [BOS] + ids + [EOS]
    return ids


def ids_to_sentence(vocab: Vocab, ids: List[int]) -> List[str]:
    return [vocab.id2word.get(_id, UNK_TOKEN) for _id in ids]


def trim_eos(ids: List[int]) -> List[int]:
    return ids[: ids.index(EOS)] if EOS in ids else ids


# =============================================================================
# Dataset / DataLoader
# =============================================================================

def preprocess_seqs(seqs: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seqs: List of [len_i]
    returns:
        data_tensor: (batch, max_len)
        position_tensor: (batch, max_len)
    """
    max_length = max(len(s) for s in seqs)
    padded = [s + [PAD] * (max_length - len(s)) for s in seqs]
    positions = [
        [pos + 1 if w != PAD else 0 for pos, w in enumerate(seq)]
        for seq in padded
    ]
    data_tensor = torch.tensor(padded, dtype=torch.long, device=device)
    position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
    return data_tensor, position_tensor


class Seq2SeqDataset(Dataset):
    def __init__(self, src_ids_list: List[List[int]], tgt_ids_list: List[List[int]]):
        assert len(src_ids_list) == len(tgt_ids_list)
        self.src_ids_list = src_ids_list
        self.tgt_ids_list = tgt_ids_list

    def __len__(self) -> int:
        return len(self.src_ids_list)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.src_ids_list[idx], self.tgt_ids_list[idx]


def collate_fn(batch: List[Tuple[List[int], List[int]]]):
    src_seqs, tgt_seqs = zip(*batch)
    src_data, src_pos = preprocess_seqs(list(src_seqs))
    tgt_data, tgt_pos = preprocess_seqs(list(tgt_seqs))
    # DataLoader からは (src_tuple, tgt_tuple) を出す
    return (src_data, src_pos), (tgt_data, tgt_pos)


# =============================================================================
# Transformer 各種モジュール（元コードを整理）
# =============================================================================

def position_encoding_init(n_position: int, d_pos_vec: int) -> torch.Tensor:
    position_enc = torch.zeros(n_position, d_pos_vec, dtype=torch.float)
    for pos in range(1, n_position):
        for j in range(0, d_pos_vec, 2):
            div_term = math.pow(10000, 2 * (j // 2) / d_pos_vec)
            position_enc[pos, j] = math.sin(pos / div_term)
            if j + 1 < d_pos_vec:
                position_enc[pos, j + 1] = math.cos(pos / div_term)
    return position_enc


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, attn_dropout: float = 0.1):
        super().__init__()
        self.temper = math.sqrt(d_model)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        attn.data.masked_fill_(attn_mask, -float("inf"))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

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
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        batch_size, len_q, d_model = q.size()
        _, len_k, _ = k.size()
        _, len_v, _ = v.size()

        q_s = q.repeat(n_head, 1, 1)
        k_s = k.repeat(n_head, 1, 1)
        v_s = v.repeat(n_head, 1, 1)

        q_s = q_s.view(n_head, -1, d_model)
        k_s = k_s.view(n_head, -1, d_model)
        v_s = v_s.view(n_head, -1, d_model)

        q_s = torch.bmm(q_s, self.w_qs)
        k_s = torch.bmm(k_s, self.w_ks)
        v_s = torch.bmm(v_s, self.w_vs)

        q_s = q_s.view(-1, len_q, d_k)
        k_s = k_s.view(-1, len_k, d_k)
        v_s = v_s.view(-1, len_v, d_v)

        if attn_mask is None:
            attn_mask = torch.zeros(batch_size, len_q, len_k, dtype=torch.bool, device=q.device)
        attn_mask = attn_mask.repeat(n_head, 1, 1)

        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)

        outputs = torch.split(outputs, batch_size, dim=0)
        outputs = torch.cat(outputs, dim=-1)

        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs + residual)

        return outputs, attns


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)
        self.layer_norm = nn.LayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(1, 2)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


def get_attn_padding_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(PAD).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    attn_shape = (seq.size(1), seq.size(1))
    subsequent_mask = torch.triu(
        torch.ones(attn_shape, dtype=torch.uint8, device=device),
        diagonal=1,
    )
    subsequent_mask = subsequent_mask.repeat(seq.size(0), 1, 1)
    return subsequent_mask


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    def __init__(self, n_src_vocab, max_length, n_layers=6, n_head=8,
                 d_k=64, d_v=64, d_word_vec=512, d_model=512,
                 d_inner_hid=1024, dropout=0.1):
        super().__init__()

        n_position = max_length + 1

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=PAD)

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, src_pos):
        enc_input = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)

        enc_slf_attns = []
        enc_output = enc_input
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)
            enc_slf_attns.append(enc_slf_attn)

        return enc_output, enc_slf_attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, n_tgt_vocab, max_length, n_layers=6, n_head=8,
                 d_k=64, d_v=64, d_word_vec=512, d_model=512,
                 d_inner_hid=1024, dropout=0.1):
        super().__init__()

        n_position = max_length + 1

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=PAD)

        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output):
        dec_input = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, src_seq)

        dec_slf_attns, dec_enc_attns = [], []

        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask,
            )
            dec_slf_attns.append(dec_slf_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_output, dec_slf_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, max_length, n_layers=6, n_head=8,
                 d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
                 dropout=0.1, proj_share_weight=True):
        super().__init__()

        self.encoder = Encoder(
            n_src_vocab, max_length, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout,
        )
        self.decoder = Decoder(
            n_tgt_vocab, max_length, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout,
        )

        self.tgt_word_proj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_proj.weight)

        assert d_model == d_word_vec

        if proj_share_weight:
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight

    def get_trainable_parameters(self):
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
        dec_freezed_param_ids = set(map(id, self.decoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, tgt):
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        # 入力側: 先頭 BOS を捨てる
        src_seq = src_seq[:, 1:]
        src_pos = src_pos[:, 1:]
        # 出力側: 末尾 EOS を捨てて右シフト
        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        enc_output, _ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_proj(dec_output)
        return seq_logit


# =============================================================================
# 学習 / 評価ユーティリティ
# =============================================================================

def calc_bleu(refs, hyps) -> float:
    """
    refs, hyps: list of list[int]
    """
    if len(refs) == 0:
        return 0.0
    refs_bleu = [[ref[: ref.index(EOS)]] if EOS in ref else [ref] for ref in refs]
    hyps_bleu = [hyp[: hyp.index(EOS)] if EOS in hyp else hyp for hyp in hyps]
    return 100.0 * bleu_score.corpus_bleu(refs_bleu, hyps_bleu)


def compute_loss(batch_X, batch_Y, model, criterion, optimizer=None, is_train=True, scaler=None):
    """
    batch_X: (src_seq, src_pos)
    batch_Y: (tgt_seq, tgt_pos)
    """
    model.train(is_train)

    src_seq, src_pos = batch_X
    tgt_seq, tgt_pos = batch_Y

    gold = tgt_seq[:, 1:].contiguous()  # 右に 1 トークンずらした教師
    gold_flat = gold.view(-1)

    # AMP: GPU のときだけ有効
    from contextlib import nullcontext
    autocast_ctx = (
        torch.cuda.amp.autocast() if (device.type == "cuda" and scaler is not None)
        else nullcontext()
    )

    with autocast_ctx:
        pred_Y = model((src_seq, src_pos), (tgt_seq, tgt_pos))
        loss = criterion(pred_Y.view(-1, pred_Y.size(2)), gold_flat)

    if is_train and optimizer is not None:
        if scaler is not None and device.type == "cuda":
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    gold_list = gold.cpu().numpy().tolist()
    pred_list = pred_Y.max(dim=-1)[1].cpu().numpy().tolist()

    # gold のトークン数（PAD 含まず）
    num_tokens = sum(len([t for t in g if t != PAD]) for g in gold_list)

    return loss.item(), num_tokens, gold_list, pred_list


def test(model, src, max_length=20):
    """
    src: (src_seq, src_pos)
    """
    model.eval()
    src_seq, src_pos = src
    batch_size = src_seq.size(0)

    enc_output, enc_slf_attns = model.encoder(src_seq, src_pos)

    tgt_seq = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    tgt_pos = torch.arange(1, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)

    dec_slf_attns = []
    dec_enc_attns = []

    for t in range(1, max_length + 1):
        dec_output, dec_slf_attn, dec_enc_attn = model.decoder(
            tgt_seq, tgt_pos, src_seq, enc_output
        )
        dec_slf_attns = dec_slf_attn
        dec_enc_attns = dec_enc_attn

        dec_output = model.tgt_word_proj(dec_output)
        out = dec_output[:, -1, :].max(dim=-1)[1].unsqueeze(1)
        tgt_seq = torch.cat([tgt_seq, out], dim=-1)
        tgt_pos = torch.arange(t + 1, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)

    return tgt_seq[:, 1:], enc_slf_attns, dec_slf_attns, dec_enc_attns


# =============================================================================
# データ読み込み・前処理
# =============================================================================

def load_and_prepare_data():
    # OPUS100 EN-JA
    dataset = load_dataset("opus100", "en-ja")

    train_split = dataset["train"]
    test_split = dataset["test"] if "test" in dataset else dataset["validation"]

    # 上限で切る（メモリ節約）
    train_split = train_split.select(range(min(MAX_TRAIN, len(train_split))))
    test_split = test_split.select(range(min(MAX_TEST, len(test_split))))

    train_src_all = [ex["translation"]["en"] for ex in train_split]
    train_tgt_all = [ex["translation"]["ja"] for ex in train_split]

    test_src_all = [ex["translation"]["en"] for ex in test_split]
    test_tgt_all = [ex["translation"]["ja"] for ex in test_split]

    # train / valid split
    train_src_texts, valid_src_texts, train_tgt_texts, valid_tgt_texts = train_test_split(
        train_src_all,
        train_tgt_all,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # トークナイズ → "空白区切り文字列" に変換
    def tokenize_to_str(texts, tokenizer):
        return [" ".join(tokenizer(t)) for t in texts]

    train_src_tok = tokenize_to_str(train_src_texts, src_tokenize)
    train_tgt_tok = tokenize_to_str(train_tgt_texts, tgt_tokenize)
    valid_src_tok = tokenize_to_str(valid_src_texts, src_tokenize)
    valid_tgt_tok = tokenize_to_str(valid_tgt_texts, tgt_tokenize)
    test_src_tok = tokenize_to_str(test_src_all, src_tokenize)
    test_tgt_tok = tokenize_to_str(test_tgt_all, tgt_tokenize)

    # vocab 構築（train のみ）
    word2id_init = {
        PAD_TOKEN: PAD,
        BOS_TOKEN: BOS,
        EOS_TOKEN: EOS,
        UNK_TOKEN: UNK,
    }

    vocab_X = Vocab(word2id=word2id_init)
    vocab_Y = Vocab(word2id=word2id_init)

    vocab_X.build_vocab(train_src_tok, min_count=1)
    vocab_Y.build_vocab(train_tgt_tok, min_count=1)

    print("vocabX size =", len(vocab_X))
    print("vocabY size =", len(vocab_Y))

    # ID 列へ変換
    train_X = [sentence_to_ids(vocab_X, s) for s in train_src_tok]
    train_Y = [sentence_to_ids(vocab_Y, s) for s in train_tgt_tok]
    valid_X = [sentence_to_ids(vocab_X, s) for s in valid_src_tok]
    valid_Y = [sentence_to_ids(vocab_Y, s) for s in valid_tgt_tok]
    test_X = [sentence_to_ids(vocab_X, s) for s in test_src_tok]
    test_Y = [sentence_to_ids(vocab_Y, s) for s in test_tgt_tok]

    return (train_X, train_Y, valid_X, valid_Y, test_X, test_Y, vocab_X, vocab_Y)


# =============================================================================
# メイン学習ループ
# =============================================================================

def main():
    (train_X, train_Y,
     valid_X, valid_Y,
     test_X, test_Y,
     vocab_X, vocab_Y) = load_and_prepare_data()

    # DataLoader
    train_dataset = Seq2SeqDataset(train_X, train_Y)
    valid_dataset = Seq2SeqDataset(valid_X, valid_Y)
    test_dataset = Seq2SeqDataset(test_X, test_Y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # モデル設定
    MODEL_ARGS["n_src_vocab"] = len(vocab_X)
    MODEL_ARGS["n_tgt_vocab"] = len(vocab_Y)

    model = Transformer(**MODEL_ARGS).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum").to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_valid_bleu = 0.0

    def get_state_dict(m: nn.Module):
        return m.module.state_dict() if isinstance(m, nn.DataParallel) else m.state_dict()

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        # train
        train_loss_sum = 0.0
        train_token_sum = 0
        train_refs = []
        train_hyps = []

        for batch in train_loader:
            batch_X, batch_Y = batch
            loss, n_tokens, gold, pred = compute_loss(
                batch_X, batch_Y,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                is_train=True,
                scaler=scaler,
            )
            train_loss_sum += loss
            train_token_sum += n_tokens

            # BLEU サンプルを上限まで蓄積
            for g, p in zip(gold, pred):
                if len(train_refs) >= MAX_BLEU_SAMPLES_TRAIN:
                    break
                train_refs.append(g)
                train_hyps.append(p)

        # valid
        valid_loss_sum = 0.0
        valid_token_sum = 0
        valid_refs = []
        valid_hyps = []

        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                batch_X, batch_Y = batch
                loss, n_tokens, gold, pred = compute_loss(
                    batch_X, batch_Y,
                    model=model,
                    criterion=criterion,
                    optimizer=None,
                    is_train=False,
                    scaler=None,
                )
                valid_loss_sum += loss
                valid_token_sum += n_tokens

                for g, p in zip(gold, pred):
                    if len(valid_refs) >= MAX_BLEU_SAMPLES_VALID:
                        break
                    valid_refs.append(g)
                    valid_hyps.append(p)

        scheduler.step()

        train_loss = train_loss_sum / max(train_token_sum, 1)
        valid_loss = valid_loss_sum / max(valid_token_sum, 1)

        train_bleu = calc_bleu(train_refs, train_hyps)
        valid_bleu = calc_bleu(valid_refs, valid_hyps)

        if valid_bleu > best_valid_bleu:
            torch.save(get_state_dict(model), CKPT_PATH)
            best_valid_bleu = valid_bleu

        elapsed = (time.time() - start) / 60
        print(
            f"Epoch {epoch} [{elapsed:.1f}min]: "
            f"train_loss: {train_loss:5.3f}  train_bleu: {train_bleu:5.2f}  "
            f"valid_loss: {valid_loss:5.3f}  valid_bleu: {valid_bleu:5.2f}"
        )
        print("-" * 80)

    # =========================================================================
    # テスト & サンプル翻訳
    # =========================================================================
    print("Loading best checkpoint...")
    model = Transformer(**MODEL_ARGS).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt)

    # 単一サンプルでのデコード
    def make_batch_from_ids(src_ids: List[int], tgt_ids: List[int]):
        # DataLoader を通さずに collate_fn を直接使用
        (src_batch, tgt_batch) = collate_fn([(src_ids, tgt_ids)])
        return src_batch, tgt_batch

    src_ids_0 = test_X[0]
    tgt_ids_0 = test_Y[0]
    src_batch, tgt_batch = make_batch_from_ids(src_ids_0, tgt_ids_0)

    src_seq_ids = src_batch[0][0].cpu().numpy()
    tgt_seq_ids = tgt_batch[0][0].cpu().numpy()
    print("src:", " ".join(ids_to_sentence(vocab_X, src_seq_ids[1:-1])).split(EOS_TOKEN)[0])
    print("tgt:", " ".join(ids_to_sentence(vocab_Y, tgt_seq_ids[1:-1])).split(EOS_TOKEN)[0])

    preds, enc_slf_attns, dec_slf_attns, dec_enc_attns = test(model, src_batch, max_length=MAX_SEQ_LEN)
    pred_ids = preds[0].cpu().numpy().tolist()
    print("out:", " ".join(ids_to_sentence(vocab_Y, trim_eos(pred_ids))).split(EOS_TOKEN)[0])

    # テスト全体で BLEU
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn,
    )

    refs_list = []
    hyp_list = []

    with torch.no_grad():
        for batch in test_loader:
            batch_X, batch_Y = batch
            preds, *_ = test(model, batch_X, max_length=MAX_SEQ_LEN)
            preds = preds.cpu().numpy().tolist()
            refs = batch_Y[0].cpu().numpy()[:, 1:].tolist()  # gold: shift
            for r, h in zip(refs, preds):
                if len(refs_list) >= MAX_BLEU_SAMPLES_TEST:
                    break
                refs_list.append(r)
                hyp_list.append(h)

    bleu = calc_bleu(refs_list, hyp_list)
    print("Test BLEU (approx):", bleu)

    # ユーザ入力から翻訳
    try:
        user_src = input("Input English sentence to translate:\n> ").strip()
        if user_src:
            print("src:", user_src)
            src_tok = " ".join(src_tokenize(user_src))
            src_ids = sentence_to_ids(vocab_X, src_tok)
            # tgt 側はダミー（中身は使わない）
            src_batch, _ = make_batch_from_ids(src_ids, src_ids)
            preds, enc_slf_attns, dec_slf_attns, dec_enc_attns = test(model, src_batch, max_length=MAX_SEQ_LEN)
            pred_ids = preds[0].cpu().numpy().tolist()
            out_seq = " ".join(ids_to_sentence(vocab_Y, trim_eos(pred_ids))).split(EOS_TOKEN)[0]
            print("out:", out_seq)

            # attention 可視化（例: encoder self-attention の 1 レイヤ 1 ヘッド）
            if enc_slf_attns:
                attn = enc_slf_attns[0][0].cpu().detach().numpy()
                plt.pcolor(attn, cmap=plt.cm.Blues)
                plt.title("Encoder self-attention (layer 0, sample 0)")
                plt.show()
    except EOFError:
        pass


if __name__ == "__main__":
    main()

