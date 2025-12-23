import math

import torch.nn as nn
import torch
import torch.nn.functional as F

from ModelConfig import ModelConfig
from function import apply_rotary_emb, repeat_kv


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

    def forward(self, x):
        out = self._norm(x.float()).type_as(x)
        return out * self.weight


class Attention(nn.Module):#分组注意力
    def __init__(self, args: ModelConfig):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads#没定义kv头的数量就不分组

        assert args.n_heads % args.n_kv_heads == 0

        model_parallel_size = 1#分布式的参数

        self.n_local_heads = args.n_heads // model_parallel_size

        self.n_local_kv_heads = args.n_kv_heads // model_parallel_size

        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        self.head_dim = args.dim // args.n_heads


        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')#TODO 这里这句是什么意思

        if not self.flash:
            print("WARNING:using slow attention.Flash Attention requires PyTorch >= 2.")

            self.mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len ), float("-inf"))
            self.mask = torch.triu(self.mask, diagonal=1)

            self.register_buffer("mask", self.mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):

        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask= None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)

        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask') #TODO hasattr是啥？
            scores = scores + self.mask[:, :, seqlen, seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()

        self.n_heads = args.n_heads

        self.dim = args.dim

        self.head_dim = args.dim // args.n_heads

        self.attention = Attention(args)

        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout
        )

        self.layer_id = layer_id

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):

        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)

        out = h + self.feed_forward(self.ffn_norm(h))

        return out
