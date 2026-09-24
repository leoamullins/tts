import torch
import torch.nn.functional as F

from torch import nn


class Attention(nn.Module):
    def __init__(self, n_embd, n_head, context_dim=None, causal=False, dropout=0.0):
        super().__init__()
        context_dim = context_dim or n_embd
        self.n_head, self.causal, self.dropout = n_head, causal, dropout

        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.kv = nn.Linear(context_dim, 2 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, context=None, key_padding_mask=None, need_weights=False):
        context = x if context is None else context
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_head, -1).transpose(1, 2)
        k, v = self.kv(context).view(B, context.shape[1], 2, self.n_head, -1).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        attn_mask = (
            None if key_padding_mask is None else ~key_padding_mask[:, None, None, :]
        )
        if need_weights:
            att = (q @ k.transpose(-2, -1)) * q.size(-1) ** -0.5
            if attn_mask is not None:
                att = att.masked_fill(~attn_mask, float("-inf"))
            att = att.softmax(dim=-1)
            y = F.dropout(att, self.dropout, self.training) @ v
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=self.causal and attn_mask is None,
                dropout_p=self.dropout if self.training else 0.0,
            )
            att = None
        out = self.proj(y.transpose(1, 2).reshape(B, T, C))
        return (out, att) if need_weights else out
