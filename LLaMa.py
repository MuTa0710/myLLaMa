from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

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
        freq_cos, freq_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer('freq_cos', freq_cos, persistent=False)
        self.register_buffer('freq_sin', freq_sin, persistent=False)

        self.apply(self._init_weights)#TODO 初始化所有权重，但是什么意思呢


