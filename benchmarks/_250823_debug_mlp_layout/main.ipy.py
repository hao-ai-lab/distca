
import torch

def prepend_zero_fn(tensor: torch.Tensor, dim: int=0):
    zero = torch.zeros_like(tensor.select(dim, 0)).unsqueeze(dim)
    return torch.cat([zero, tensor], dim=dim)


def mlp_layout_packed_params(seq_lens: torch.Tensor):
    """
    Compute the MLP layout packed_seq_params. MLP layout guarantees seqlens_q == seqlens_kv.
    This is mainly for RoPE.
    NOTE: this is the seq lens on the local rank.
    """
    cu_seqlens = prepend_zero_fn(seq_lens.cumsum(dim=0))
    max_seqlen = seq_lens.max()
    

a = torch.tensor([[12790,  3594,  3594, 12790,     0,     0,     0,     0,     0,     0,
             0,     0,     0],
        [12790,  3594,  3594, 12790,     0,     0,     0,     0,     0,     0,
             0,     0,     0],
        [12790,  3594,  3594, 12790,     0,     0,     0,     0,     0,     0,
             0,     0,     0],
        [ 1496,  1496,  2560,  2096,  2592,  2160,  4240,  5264,   272,   432,
          3840,  3488,  2832]])

seq_lens = a