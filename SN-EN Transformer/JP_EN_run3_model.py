import sys
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import nltk
from nltk.translate import bleu_score
import MeCab
import unidic
from pandarallel import pandarallel

# RigL (Dynamic Sparse Training)
from rigl_scheduler import RigLScheduler  # あなたの rigl_scheduler.py

# ==========================================
# 0. Optional: FLOPs 計測 (fvcore)
# ==========================================
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FlopCountAnalysis = None
    FVCORE_AVAILABLE = False
    print("[WARN] fvcore is not installed. FLOPs will not be computed (effective_flops=None).")

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
class Config:
    MAX_LENGTH = 250
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LR = 0.001
    
    # Model Architecture (元コードと同じ)
    D_MODEL = 128
    D_K = 32
    D_V = 32
    D_WORD_VEC = 128
    D_INNER_HID = 256
    N_LAYERS = 3
    N_HEAD = 6
    DROPOUT = 0.1
    
    # Data limits
    MAX_TRAIN_SAMPLES = 100000
    MAX_TEST_SAMPLES  = 2000
    
    # Tokenizer settings
    MIN_COUNT = 1
    
    # Device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Random seeds
    RANDOM_STATE = 42          # train/valid split 用（固定）
    TORCH_SEED_BASE = 1        # モデル初期化用のベース
    RUN_SEEDS = [0, 1, 2]      # 各モデルで 3 run

    # Test 評価に使うサンプル数（→ テスト全件 = MAX_TEST_SAMPLES を使う）
    MAX_TEST_EVAL = MAX_TEST_SAMPLES

# Special Tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<S>"
EOS_TOKEN = "</S>"
PAD = 0
UNK = 1
BOS = 2
EOS = 3

# Setup
torch.manual_seed(Config.TORCH_SEED_BASE)
nltk.download('punkt')
try:
    pandarallel.initialize(progress_bar=True, verbose=1)
except:
    pandarallel.initialize(progress_bar=True, verbose=1, nb_workers=1)

# 一意な出力ディレクトリ（上書き防止）
EXP_ID = time.strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = f"runs_jaen_{EXP_ID}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Output dir: {OUTPUT_DIR}")
print(f"[INFO] Running on device: {Config.DEVICE}")

# ==========================================
# 2. Data Processing Classes & Functions
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

def sentence_to_ids(vocab, sentence, max_seq_len=200):
    ids = [vocab.word2id.get(word, UNK) for word in sentence.split()]
    ids = ids[:max_seq_len-2]
    ids = [BOS] + ids + [EOS]
    return ids

def ids_to_sentence(vocab, ids):
    return [vocab.id2word.get(_id, UNK_TOKEN) for _id in ids]

def trim_eos(ids):
    if EOS in ids:
        return ids[:ids.index(EOS)]
    else:
        return ids

def decode_ids_to_text(vocab, pred_ids):
    pred_ids = trim_eos(pred_ids)
    words = ids_to_sentence(vocab, pred_ids)
    return " ".join(words)

# Tokenizer functions
def src_tokenize(text: str) -> str:
    """
    日本語側トークナイザ。
    MeCab が使えない環境の場合は例外を握りつぶしてフォールバック。
    """
    text = str(text)
    try:
        tagger = MeCab.Tagger('-Owakati')
        out = tagger.parse(text)
        if out is None:
            # MeCab が返してこなかった場合はそのまま返す
            return text
        return out.strip()
    except Exception:
        # mecabrc 等の問題で MeCab が使えない環境向けフォールバック
        # 文字単位でスペース区切りにする（トークナイズの質は落ちるが学習は走る）
        return " ".join(list(text))

def tgt_tokenize(text: str) -> str:
    return " ".join(nltk.tokenize.word_tokenize(str(text)))

class DataLoader(object):
    def __init__(self, src_insts, tgt_insts, batch_size, shuffle=True):
        self.data = list(zip(src_insts, tgt_insts))
        self.batch_size = batch_size
        self.shuffle_flag = shuffle
        self.start_index = 0
        self.reset()

    def reset(self):
        if self.shuffle_flag:
            self.data = shuffle(self.data, random_state=Config.RANDOM_STATE)
        self.start_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        def preprocess_seqs(seqs):
            max_len = max([len(s) for s in seqs])
            data = [s + [PAD] * (max_len - len(s)) for s in seqs]
            positions = [[pos+1 if w != PAD else 0 for pos, w in enumerate(seq)]
                         for seq in data]
            data_tensor = torch.tensor(data, dtype=torch.long, device=Config.DEVICE)
            position_tensor = torch.tensor(positions, dtype=torch.long, device=Config.DEVICE)
            return data_tensor, position_tensor

        if self.start_index >= len(self.data):
            self.reset()
            raise StopIteration()

        src_seqs, tgt_seqs = zip(*self.data[self.start_index:self.start_index+self.batch_size])
        src_data, src_pos = preprocess_seqs(src_seqs)
        tgt_data, tgt_pos = preprocess_seqs(tgt_seqs)

        self.start_index += self.batch_size
        return (src_data, src_pos), (tgt_data, tgt_pos)

def create_dataloaders(batch_size, train_X_ids, train_Y_ids, valid_X_ids, valid_Y_ids):
    train_loader = DataLoader(train_X_ids, train_Y_ids, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_X_ids, valid_Y_ids, batch_size, shuffle=False)
    return train_loader, valid_loader

# ==========================================
# 3. Model Architecture (Transformer)
# ==========================================

def position_encoding_init(n_position, d_pos_vec):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)
    ])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    return torch.tensor(position_enc, dtype=torch.float)

def get_attn_padding_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(PAD).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    attn_shape = (seq.size(1), seq.size(1))
    subsequent_mask = torch.triu(
        torch.ones(attn_shape, dtype=torch.uint8, device=Config.DEVICE),
        diagonal=1
    )
    subsequent_mask = subsequent_mask.repeat(seq.size(0), 1, 1)
    return subsequent_mask

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
        self.proj = nn.Linear(n_head*d_v, d_model)
        nn.init.xavier_normal_(self.proj.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head
        residual = q
        batch_size, len_q, d_model = q.size()
        batch_size, len_k, d_model = k.size()
        batch_size, len_v, d_model = v.size()

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
        outputs = self.layer_norm(outputs + residual)
        return outputs, attns

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

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input,
                                                 attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    def __init__(self, n_src_vocab, max_length,
                 n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_word_vec=512, d_model=512, d_inner_hid=1024,
                 dropout=0.1):
        super(Encoder, self).__init__()
        n_position = max_length + 1
        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=PAD)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, src_seq, src_pos):
        enc_input = self.src_word_emb(src_seq)
        enc_input += self.position_enc(src_pos)
        enc_slf_attns = []
        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output,
                                                 slf_attn_mask=enc_slf_attn_mask)
            enc_slf_attns += [enc_slf_attn]
        return enc_output, enc_slf_attns

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, dec_input, enc_output,
                slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input,
                                                 attn_mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output,
                                                 attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self, n_tgt_vocab, max_length,
                 n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_word_vec=512, d_model=512, d_inner_hid=1024,
                 dropout=0.1):
        super(Decoder, self).__init__()
        n_position = max_length + 1
        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)
        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output):
        dec_input = self.tgt_word_emb(tgt_seq)
        dec_input += self.position_enc(tgt_pos)
        
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
            assert d_model == d_word_vec
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
# 4. Training & Inference Utils
# ==========================================

def calc_bleu(refs, hyps):
    refs_cleaned = []
    for ref in refs:
        if EOS in ref:
            refs_cleaned.append([ref[:ref.index(EOS)]])
        else:
            refs_cleaned.append([ref])
            
    hyps_cleaned = []
    for hyp in hyps:
        if EOS in hyp:
            hyps_cleaned.append(hyp[:hyp.index(EOS)])
        else:
            hyps_cleaned.append(hyp)
    return 100 * bleu_score.corpus_bleu(refs_cleaned, hyps_cleaned)

def compute_loss(batch_X, batch_Y, model, criterion,
                 optimizer=None, scaler=None, is_train=True):
    model.train(is_train)
    
    with torch.cuda.amp.autocast(enabled=(Config.DEVICE.type == "cuda")):
        pred_Y = model(batch_X, batch_Y)
        gold = batch_Y[0][:, 1:].contiguous()
        loss = criterion(pred_Y.view(-1, pred_Y.size(2)), gold.view(-1))

    if is_train and optimizer is not None and scaler is not None:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    gold = gold.data.cpu().numpy().tolist()
    pred = pred_Y.max(dim=-1)[1].data.cpu().numpy().tolist()

    return loss.item(), gold, pred

def beam_search_decode(model, src, beam_size=5, max_length=40, alpha=0.6):
    """Beam search with length penalty."""
    model.eval()
    src_seq, src_pos = src
    batch_size = src_seq.size(0)
    assert batch_size == 1, "beam_search_decode assumes batch_size=1"
    
    with torch.no_grad():
        enc_output, _ = model.encoder(src_seq, src_pos)

    init_seq = torch.full((1, 1), BOS, dtype=torch.long, device=Config.DEVICE)
    beam = [(init_seq, 0.0, False)]  # (seq, log_prob_sum, finished)

    def length_penalty_fn(length):
        if alpha <= 0.0:
            return 1.0
        return ((5.0 + length) ** alpha) / ((5.0 + 1.0) ** alpha)

    for t in range(1, max_length + 1):
        new_beam = []
        all_finished = True

        for seq, logp_sum, finished in beam:
            if finished:
                new_beam.append((seq, logp_sum, True))
                continue
            
            all_finished = False
            pos = torch.arange(1, seq.size(1) + 1,
                               dtype=torch.long, device=Config.DEVICE).unsqueeze(0)

            with torch.no_grad():
                dec_output, _, _ = model.decoder(seq, pos, src_seq, enc_output)
                logits = model.tgt_word_proj(dec_output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)

            for k in range(beam_size):
                next_id = topk_ids[k].item()
                next_logp_sum = logp_sum + topk_log_probs[k].item()
                
                next_token = torch.tensor([[next_id]], dtype=torch.long, device=Config.DEVICE)
                next_seq = torch.cat([seq, next_token], dim=1)
                next_finished = (next_id == EOS)
                
                new_beam.append((next_seq, next_logp_sum, next_finished))

        def scored(item):
            seq, logp_sum, finished = item
            length = seq.size(1) - 1
            lp = length_penalty_fn(max(length, 1))
            return logp_sum / lp

        new_beam.sort(key=scored, reverse=True)
        beam = new_beam[:beam_size]

        if all_finished:
            break

    best_seq, _, _ = max(beam, key=lambda x: scored(x))
    best_ids = best_seq[0].tolist()
    if len(best_ids) > 0 and best_ids[0] == BOS:
        best_ids = best_ids[1:]
    return best_ids

# ==========================================
# 5. FLOPs & Param Density Utils
# ==========================================

def compute_transformer_flops_on_dummy(model_cls, model_args,
                                       src_len=50, tgt_len=50, batch_size=1):
    """
    fvcore で FLOPs を計測するためのダミー実行。
    ここでは一時的に Config.DEVICE を CPU に切り替えて、
    マスク生成なども含めてすべて CPU 上で動かす。
    """
    if not FVCORE_AVAILABLE:
        return None

    # 元のデバイスを退避
    original_device = Config.DEVICE
    try:
        # FLOPs 計測時だけ CPU を使う
        Config.DEVICE = torch.device("cpu")

        model = model_cls(**model_args).to(Config.DEVICE)
        model.eval()

        src_seq = torch.full(
            (batch_size, src_len),
            fill_value=PAD,
            dtype=torch.long,
            device=Config.DEVICE,
        )
        tgt_seq = torch.full(
            (batch_size, tgt_len),
            fill_value=PAD,
            dtype=torch.long,
            device=Config.DEVICE,
        )
        src_seq[:, 0] = BOS
        tgt_seq[:, 0] = BOS

        src_pos = torch.arange(
            1, src_len + 1,
            dtype=torch.long,
            device=Config.DEVICE,
        ).unsqueeze(0).repeat(batch_size, 1)
        tgt_pos = torch.arange(
            1, tgt_len + 1,
            dtype=torch.long,
            device=Config.DEVICE,
        ).unsqueeze(0).repeat(batch_size, 1)

        inputs = ((src_seq, src_pos), (tgt_seq, tgt_pos))
        flops = FlopCountAnalysis(model, inputs).total()
    finally:
        # 必ず元のデバイスに戻す
        Config.DEVICE = original_device

    return flops

def count_params_and_density(state_dict):
    total = 0
    nonzero = 0
    for v in state_dict.values():
        if not torch.is_tensor(v):
            continue
        numel = v.numel()
        total += numel
        nonzero += v.ne(0).sum().item()
    density = nonzero / total
    return total, nonzero, density

# ==========================================
# 6. Helper: Seed 設定
# ==========================================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 7. Training of one model (dense or DST)
# ==========================================

def train_one_model(run_name,
                    model_args,
                    train_dataloader,
                    valid_dataloader,
                    base_arch_flops=None,
                    use_rigl=False,
                    rigl_dense_allocation=0.2,
                    rigl_delta=100,
                    rigl_alpha=0.3,
                    rigl_grad_accum_n=4):
    """
    1 モデル (dense or DST) を学習して、best checkpoint 情報と param 情報を返す。
    """
    print(f"\n===== Train run: {run_name}  (use_rigl={use_rigl}) =====")

    model = Transformer(**model_args).to(Config.DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"[{run_name}] Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion_local = nn.CrossEntropyLoss(ignore_index=PAD, reduction='sum').to(Config.DEVICE)
    scaler = torch.cuda.amp.GradScaler(enabled=(Config.DEVICE.type == "cuda"))

    # RigL scheduler 準備
    rigl_scheduler = None
    if use_rigl:
        num_train_batches = (len(train_dataloader.data) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
        total_steps = Config.NUM_EPOCHS * num_train_batches
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
    best_state_dict = None

    num_train_batches = (len(train_dataloader.data) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
    num_valid_batches = (len(valid_dataloader.data) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss = 0.0
        train_refs = []
        train_hyps = []

        valid_loss = 0.0
        valid_refs = []
        valid_hyps = []

        print(f"\n[{run_name}] [Epoch {epoch}/{Config.NUM_EPOCHS}] Training...")
        for batch_idx, batch in enumerate(train_dataloader, 1):
            batch_X, batch_Y = batch
            loss, gold, pred = compute_loss(
                batch_X, batch_Y,
                model, criterion_local,
                optimizer=optimizer, scaler=scaler,
                is_train=True
            )
            train_loss += loss
            train_refs += gold
            train_hyps += pred

            # RigL スケジューラ更新
            if rigl_scheduler is not None:
                rigl_scheduler()

            if (batch_idx % 100 == 0) or (batch_idx == num_train_batches):
                print(f"  [{run_name}][Train] batch {batch_idx}/{num_train_batches}", end='\r')

        print()
        print(f"[{run_name}] [Epoch {epoch}/{Config.NUM_EPOCHS}] Validation...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_dataloader, 1):
                batch_X, batch_Y = batch
                loss, gold, pred = compute_loss(
                    batch_X, batch_Y,
                    model, criterion_local,
                    optimizer=None, scaler=None,
                    is_train=False
                )
                valid_loss += loss
                valid_refs += gold
                valid_hyps += pred

                if (batch_idx % 50 == 0) or (batch_idx == num_valid_batches):
                    print(f"  [{run_name}][Valid] batch {batch_idx}/{num_valid_batches}", end='\r')

        print()
        scheduler_lr.step()

        train_loss /= len(train_dataloader.data)
        valid_loss /= len(valid_dataloader.data)

        train_bleu = calc_bleu(train_refs, train_hyps)
        valid_bleu = calc_bleu(valid_refs, valid_hyps)

        elapsed = (time.time() - start_time) / 60
        print(f'[{run_name}] Epoch {epoch:2d} [{elapsed:.1f}min]: '
              f'train_loss: {train_loss:.2f}  train_bleu: {train_bleu:.2f}  '
              f'valid_loss: {valid_loss:.2f}  valid_bleu: {valid_bleu:.2f}')

        if valid_bleu > best_valid_bleu:
            best_valid_bleu = valid_bleu
            best_state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel) else
                model.state_dict()
            )
            state_path = os.path.join(OUTPUT_DIR, f"{run_name}_best_state_dict.pth")
            meta_path = os.path.join(OUTPUT_DIR, f"{run_name}_best_meta.pth")
            torch.save(best_state_dict, state_path)
            torch.save(
                {
                    "best_valid_bleu": best_valid_bleu,
                    "model_args": model_args,
                    "use_rigl": use_rigl,
                },
                meta_path
            )
            print(f"  [{run_name}] New best valid BLEU: {valid_bleu:.2f}")
            print(f"  [{run_name}] Saved weights to: {state_path}")
            print(f"  [{run_name}] Saved meta    to: {meta_path}")

    if best_state_dict is None:
        raise RuntimeError(f"[{run_name}] No best_state_dict. Training failed?")

    # パラメータ数＆density
    total_params, nonzero_params, density = count_params_and_density(best_state_dict)
    param_reduction = 1.0 - density
    if base_arch_flops is not None:
        effective_flops = base_arch_flops * density
    else:
        effective_flops = None

    print(f"\n[{run_name}] Training finished. "
          f"Best valid BLEU: {best_valid_bleu:.2f}")
    print(f"[{run_name}] Params: total={total_params}, "
          f"nonzero={nonzero_params}, density={density:.4f}, "
          f"reduction={param_reduction:.4f}")
    if effective_flops is not None:
        print(f"[{run_name}] Arch FLOPs: {base_arch_flops:.3e}, "
              f"Effective FLOPs: {effective_flops:.3e}")
    else:
        print(f"[{run_name}] FLOPs could not be computed (fvcore not installed).")

    result = {
        "run_name": run_name,
        "best_valid_bleu": best_valid_bleu,
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "density": density,
        "param_reduction": param_reduction,
        "flops_arch": base_arch_flops,
        "effective_flops": effective_flops,
    }
    return result, best_state_dict

# ==========================================
# 8. Main Execution Block (dense & DST, 3 runs each)
# ==========================================

if __name__ == "__main__":
    print("Loading Dataset (opus100, en-ja)...")
    dataset = load_dataset("opus100", "en-ja")
    train = dataset["train"]
    test  = dataset["test"] if "test" in dataset else dataset["validation"]

    train_df = pd.DataFrame(train["translation"]).rename(columns={"ja": "src", "en": "tgt"})
    test_df  = pd.DataFrame(test["translation"]).rename(columns={"ja": "src", "en": "tgt"})

    train_df = train_df.iloc[:Config.MAX_TRAIN_SAMPLES].reset_index(drop=True)
    test_df  = test_df.iloc[:Config.MAX_TEST_SAMPLES].reset_index(drop=True)

    # --- Tokenization ---
    print("Tokenizing Train Data (this may take a while)...")
    src = train_df["src"].apply(src_tokenize)
    tgt = train_df["tgt"].apply(tgt_tokenize)
    train_df = pd.concat([src, tgt], axis=1)
    train_df.columns = ["src", "tgt"]

    print("Tokenizing Test Data...")
    src_test = test_df["src"].apply(src_tokenize)
    tgt_test = test_df["tgt"].apply(tgt_tokenize)
    test_df = pd.concat([src_test, tgt_test], axis=1)
    test_df.columns = ["src", "tgt"]

    # --- Splitting Train/Valid ---
    train_X = train_df.src.to_list()
    train_Y = train_df.tgt.to_list()
    train_X, valid_X, train_Y, valid_Y = train_test_split(
        train_X, train_Y,
        test_size=0.2,
        random_state=Config.RANDOM_STATE
    )

    # --- Build Vocab ---
    base_word2id = {PAD_TOKEN: PAD, BOS_TOKEN: BOS, EOS_TOKEN: EOS, UNK_TOKEN: UNK}
    vocab_X = Vocab(word2id=base_word2id)
    vocab_Y = Vocab(word2id=base_word2id)

    print("Building Vocabulary...")
    vocab_X.build_vocab(train_X, min_count=Config.MIN_COUNT)
    vocab_Y.build_vocab(train_Y, min_count=Config.MIN_COUNT)
    
    vocab_size_X = len(vocab_X.id2word)
    vocab_size_Y = len(vocab_Y.id2word)
    print(f"Vocab Sizes - Source: {vocab_size_X}, Target: {vocab_size_Y}")

    # --- Convert Sentences to IDs ---
    print("Converting sentences to IDs...")
    def batch_to_ids(text_list, vocab):
        return [sentence_to_ids(vocab, s, Config.MAX_LENGTH + 2) for s in text_list]

    train_X_ids = batch_to_ids(train_X, vocab_X)
    train_Y_ids = batch_to_ids(train_Y, vocab_Y)
    valid_X_ids = batch_to_ids(valid_X, vocab_X)
    valid_Y_ids = batch_to_ids(valid_Y, vocab_Y)

    test_X_list = test_df.src.to_list()
    test_Y_list = test_df.tgt.to_list()
    test_X_ids = batch_to_ids(test_X_list, vocab_X)
    test_Y_ids = batch_to_ids(test_Y_list, vocab_Y)

    # --- Model args & base FLOPs ---
    model_args = {
        'n_src_vocab': vocab_size_X,
        'n_tgt_vocab': vocab_size_Y,
        'max_length': Config.MAX_LENGTH + 2,
        'proj_share_weight': True,
        'd_k': Config.D_K,
        'd_v': Config.D_V,
        'd_model': Config.D_MODEL,
        'd_word_vec': Config.D_WORD_VEC,
        'd_inner_hid': Config.D_INNER_HID,
        'n_layers': Config.N_LAYERS,
        'n_head': Config.N_HEAD,
        'dropout': Config.DROPOUT,
    }

    print("Computing base architecture FLOPs (dummy input)...")
    base_arch_flops = compute_transformer_flops_on_dummy(
        Transformer, model_args, src_len=50, tgt_len=50, batch_size=1
    )
    if base_arch_flops is not None:
        print(f"[INFO] Base arch FLOPs (seq len 50/50, bs=1): {base_arch_flops:.3e}")
    else:
        print("[INFO] FLOPs will be N/A (fvcore not installed).")

    all_results = []

    # ====== Multi-run: dense & RigL (DST) ======
    for idx, seed_offset in enumerate(Config.RUN_SEEDS):
        run_seed = Config.TORCH_SEED_BASE + seed_offset
        print("\n==============================")
        print(f"== Multi-run index {idx} / seed {run_seed} ==")
        print("==============================")

        # --- Dense run ---
        set_global_seed(run_seed)
        train_loader, valid_loader = create_dataloaders(
            Config.BATCH_SIZE,
            train_X_ids, train_Y_ids,
            valid_X_ids, valid_Y_ids
        )
        dense_run_name = f"jaen_dense_seed{run_seed}"
        dense_result, dense_state = train_one_model(
            run_name=dense_run_name,
            model_args=model_args,
            train_dataloader=train_loader,
            valid_dataloader=valid_loader,
            base_arch_flops=base_arch_flops,
            use_rigl=False
        )
        dense_result["model"] = "dense"
        dense_result["seed"] = run_seed

        # Dense モデルで test beam search BLEU
        dense_model_eval = Transformer(**model_args).to(Config.DEVICE)
        dense_model_eval.load_state_dict(dense_state)
        dense_model_eval.eval()
        print(f"[{dense_run_name}] Evaluating on TEST set with beam search...")
        test_dl = DataLoader(test_X_ids, test_Y_ids, batch_size=1, shuffle=False)
        refs_list = []
        hyp_list = []
        for i, batch in enumerate(test_dl):
            batch_X, batch_Y = batch
            src_seq, src_pos = batch_X
            tgt_seq, _ = batch_Y

            best_ids = beam_search_decode(
                dense_model_eval, (src_seq, src_pos),
                beam_size=5, max_length=40, alpha=0.6
            )
            gold_ids = tgt_seq[0, 1:].cpu().numpy().tolist()  # remove BOS
            refs_list.append(gold_ids)
            hyp_list.append(best_ids)

        dense_test_bleu = calc_bleu(refs_list, hyp_list)
        dense_result["test_bleu_beam"] = dense_test_bleu
        print(f"[{dense_run_name}] TEST BLEU (beam, first {Config.MAX_TEST_EVAL}): {dense_test_bleu:.2f}")
        all_results.append(dense_result)

        # --- RigL (DST) run ---
        set_global_seed(run_seed)
        train_loader, valid_loader = create_dataloaders(
            Config.BATCH_SIZE,
            train_X_ids, train_Y_ids,
            valid_X_ids, valid_Y_ids
        )
        rigl_run_name = f"jaen_rigl_seed{run_seed}"
        rigl_result, rigl_state = train_one_model(
            run_name=rigl_run_name,
            model_args=model_args,
            train_dataloader=train_loader,
            valid_dataloader=valid_loader,
            base_arch_flops=base_arch_flops,
            use_rigl=True,
            rigl_dense_allocation=0.2,   # dense 20%, zero 80%
            rigl_delta=100,
            rigl_alpha=0.3,
            rigl_grad_accum_n=4
        )
        rigl_result["model"] = "rigl"
        rigl_result["seed"] = run_seed

        rigl_model_eval = Transformer(**model_args).to(Config.DEVICE)
        rigl_model_eval.load_state_dict(rigl_state)
        rigl_model_eval.eval()

        print(f"[{rigl_run_name}] Evaluating on TEST set with beam search...")
        test_dl = DataLoader(test_X_ids, test_Y_ids, batch_size=1, shuffle=False)
        refs_list = []
        hyp_list = []
        for i, batch in enumerate(test_dl):
            batch_X, batch_Y = batch
            src_seq, src_pos = batch_X
            tgt_seq, _ = batch_Y

            best_ids = beam_search_decode(
                rigl_model_eval, (src_seq, src_pos),
                beam_size=5, max_length=40, alpha=0.6
            )
            gold_ids = tgt_seq[0, 1:].cpu().numpy().tolist()
            refs_list.append(gold_ids)
            hyp_list.append(best_ids)

        rigl_test_bleu = calc_bleu(refs_list, hyp_list)
        rigl_result["test_bleu_beam"] = rigl_test_bleu
        print(f"[{rigl_run_name}] TEST BLEU (beam, first {Config.MAX_TEST_EVAL}): {rigl_test_bleu:.2f}")
        all_results.append(rigl_result)

    # ====== Per-run summary ======
    print("\n===== Per-run Results (Dense & RigL, 3 runs) =====")
    for res in all_results:
        print(
            f"{res['run_name']} [model={res['model']}, seed={res['seed']}]: "
            f"valid_bleu={res['best_valid_bleu']:.2f}, "
            f"test_bleu_beam={res['test_bleu_beam']:.2f}, "
            f"density={res['density']:.4f}, "
            f"reduction={res['param_reduction']:.4f}"
        )

    # ====== Aggregated summary per model type ======
    results_df = pd.DataFrame(all_results)
    summary_rows = []
    for model_name in ["dense", "rigl"]:
        sub = results_df[results_df["model"] == model_name]
        if len(sub) == 0:
            continue
        row = {
            "model": model_name,
            "num_runs": len(sub),
            "mean_valid_bleu": sub["best_valid_bleu"].mean(),
            "std_valid_bleu": sub["best_valid_bleu"].std(ddof=0),
            "mean_test_bleu_beam": sub["test_bleu_beam"].mean(),
            "std_test_bleu_beam": sub["test_bleu_beam"].std(ddof=0),
            "flops_arch": sub["flops_arch"].iloc[0] if base_arch_flops is not None else np.nan,
            "mean_effective_flops": sub["effective_flops"].mean() if base_arch_flops is not None else np.nan,
            "mean_density": sub["density"].mean(),
            "mean_param_reduction": sub["param_reduction"].mean(),
            "mean_total_params": sub["total_params"].mean(),
            "mean_nonzero_params": sub["nonzero_params"].mean(),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("\n===== Aggregated Summary (mean over 3 runs per model) =====")
    print(summary_df)

    # CSV 保存
    runs_csv_path = os.path.join(OUTPUT_DIR, "jaen_runs_dense_vs_rigl.csv")
    summary_csv_path = os.path.join(OUTPUT_DIR, "jaen_summary_dense_vs_rigl.csv")
    results_df.to_csv(runs_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved per-run CSV to   : {runs_csv_path}")
    print(f"Saved summary CSV to   : {summary_csv_path}")
    print(f"All checkpoints & meta : {OUTPUT_DIR}/ ...")

