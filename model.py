import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from attention import Attention


class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):

    def __init__(self, n_embd, n_head, context_dim=None, causal=False, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = Attention(n_embd, n_head, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), key_padding_mask=mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderBlock(nn.Module):

    def __init__(self, n_embd, n_head, context_dim=None, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.self_attn = Attention(n_embd, n_head, causal=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.cross_attn = Attention(
            n_embd, n_head, context_dim=context_dim, dropout=dropout
        )
        self.ffd = FeedForward(n_embd, dropout)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x, memory, memory_mask=None):
        x = x + self.self_attn(self.ln1(x))
        a, w = self.cross_attn(
            self.ln2(x), context=memory, key_padding_mask=memory_mask, need_weights=True
        )
        x = x + a
        x = x + self.ffd(self.ln3(x))
        return x, w


class AlwaysDropout(nn.Dropout):

    def forward(self, x):
        return F.dropout(x, self.p, training=True, inplace=self.inplace)


class PreNet(nn.Module):

    def __init__(self, n_mels, n_hidden, d_model, dropout=0.5):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_mels, n_hidden),
            nn.ReLU(),
            AlwaysDropout(dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            AlwaysDropout(dropout),
        )
        self.proj = nn.Linear(n_hidden, d_model)

    def forward(self, x):
        x = self.model(x)
        return self.proj(x)


class PostNet(nn.Module):

    def __init__(self, n_mels, n_hidden, p):
        super().__init__()

        self.p = p
        layers = []
        for i in range(5):
            in_ch = n_mels if i == 0 else n_hidden
            out_ch = n_mels if i == 4 else n_hidden
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2, bias=False)
            )
            layers.append(nn.BatchNorm1d(out_ch))
            if i != 4:
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(p))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x is (B, T-mels, N-mels)
        x = x.transpose(-2, -1)  # conver to (B, N-mels, T-mels)

        x = self.model(x)
        return x.transpose(-2, -1)


def make_pad_mask(lens, max_len):
    return torch.arange(max_len, device=lens.device)[None, :] >= lens[:, None]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TransformerTTS(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_mels,
        n_embd=256,
        n_head=4,
        n_enc_layers=3,
        n_dec_layers=3,
        dropout=0.1,
        prenet_hidden=256,
        prenet_dropout=0.5,
        postnet_hidden=512,
    ):
        super().__init__()
        self.n_mels = n_mels

        # encoder
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.enc_alpha = nn.Parameter(torch.ones(1))  # learned scale on pos-enc
        self.enc_blocks = nn.ModuleList(
            [EncoderBlock(n_embd, n_head, dropout=dropout) for _ in range(n_enc_layers)]
        )
        self.ln_enc = nn.LayerNorm(n_embd)

        # decoder
        self.prenet = PreNet(n_mels, prenet_hidden, n_embd, prenet_dropout)
        self.dec_alpha = nn.Parameter(torch.ones(1))  # learned scale on pos-enc
        self.dec_blocks = nn.ModuleList(
            [DecoderBlock(n_embd, n_head, dropout=dropout) for _ in range(n_dec_layers)]
        )
        self.ln_dec = nn.LayerNorm(n_embd)

        # ---- shared ----
        self.pos = PositionalEncoding(n_embd)
        self.drop = nn.Dropout(dropout)

        # ---- heads ----
        self.mel_linear = nn.Linear(n_embd, n_mels)
        self.stop_linear = nn.Linear(n_embd, 1)

        # ---- postnet ----
        self.postnet = PostNet(n_mels, postnet_hidden, dropout)

    def encode(self, text, text_lens):
        x = self.embedding(text)
        x = x + self.enc_alpha * self.pos(x)
        x = self.drop(x)
        text_mask = make_pad_mask(text_lens, text.size(1))
        for block in self.enc_blocks:
            x = block(x, mask=text_mask)
        return self.ln_enc(x), text_mask

    def decode(self, mel_in, enc_out, text_mask):
        x = self.prenet(mel_in)
        x = x + self.dec_alpha * self.pos(x)
        x = self.drop(x)
        attns = []
        for block in self.dec_blocks:
            x, a = block(x, enc_out, memory_mask=text_mask)
            attns.append(a)
        x = self.ln_dec(x)
        mel_pre = self.mel_linear(x)
        stop = self.stop_linear(x).squeeze(-1)
        return (mel_pre, stop, attns)

    def forward(self, text, text_lens, mel_in):
        enc_out, text_mask = self.encode(text, text_lens)
        mel_pre, stop, attns = self.decode(mel_in, enc_out, text_mask)
        mel_post = mel_pre + self.postnet(mel_pre)
        return mel_pre, mel_post, stop, attns
