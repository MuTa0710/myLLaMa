import math
from operator import index
from typing import Optional

import torch
import torch.nn as nn
from numpy.ma.core import logical_or
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F

from ModelConfig import ModelConfig
from Modules import DecoderLayer, RMSNorm
from function import precompute_freqs_cis


class Transformer(PreTrainedModel):
    config_class = ModelConfig
    last_loss:Optional[torch.Tensor]

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)

        self.args = args

        self.vocab_size = args.vocab_size

        self.n_layer = args.n_layers

        #Embedding
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        #Dropout
        self.dropout = nn.Dropout(args.dropout)

        #Decoderlayers
        self.layers = nn.ModuleList()
        for layer in range(args.n_layers):
            self.layers.append(DecoderLayer(layer, args))

        #Normalize
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        #Output
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        #sharing weight of Embedding and Output
        self.tok_embeddings.weight = self.output.weight

        #initial frequency matrix
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer('freq_cos', self.freqs_cos, persistent=False)
        self.register_buffer('freq_sin', self.freqs_sin, persistent=False)

        self.apply(self._init_weights)#TODO 初始化所有权重，但是什么意思呢

        #TODO 以下代码均需要搞清楚是什么意思
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()
        self._no_split_modules  = [name for name, _ in self.named_modules()]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'attention_mask' in kwargs:
            targets = kwargs['attention_mask']

        _bsz, seqlen = tokens.shape

        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = 0, reduction='none')

        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return  self.OUT

    @torch.inference_mode()
    def generate(self, idx, stop_id = None, max_new_tokens=256, temperature=1.0, top_k=None):
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len]

            logits = self(idx_cond).logits
            logits = logits[:, -1, :]

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k = 1, dim=-1)

            else:
                logits = logits/temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == stop_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:]



