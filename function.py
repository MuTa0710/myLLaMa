import torch
from typing import Tuple

def repeat_kv(x: torch.Tensor, n_rep: int):
    bs, slen, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


#TODO 这块旋转编码的代码不清楚啥意思，需要结合公式理解
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    ##freqs = [1/10000^(2 * (i-1))]

    t = torch.arange(end, device=freqs.device)

    freqs = torch.outer(t, freqs).float()

    '''
        |t0*freqs0 t1*freqs0 t2*freqs0 t3*freqs0 ... |
        |t0*freqs1                                   |
        |t0*freqs2                                   |                              
        |t0*freqs3                                   |
    '''

    freqs_cos = torch.cos(freqs)

    freqs_sin = torch.sin(freqs)

    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):

    ndim = x.ndim #x.shape = bs, head, seqlen, embedding

    assert 0 <= 1 < ndim

    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    return freqs_cis.view(shape)

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xq.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
    ## shape = [bs, head, seqlen, D/2]

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
#TODO 至此，我已经完成了旋转位置编码要用到的所有工具函数的编写，但是仍然有一些问题：torch在对张量进行处理的时候，维度的变化究竟是如何规定的，比如分离x的奇偶维度时，为什么按照上面的书写,维度就会按照奇偶被分别拆成两半，而不是前一半后一半，为什么后面对维度进行处理的时候，维度会按照奇偶交叉堆叠，而不是这一堆在另一堆上面
    return xq_out.type_as(xq), xk_out.type_as(xk)
