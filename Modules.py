import torch.nn as nn
import torch

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


