import torch
from torch import Tensor
from typing import Optional
from megatron.core.models.transformer.config import TransformerConfig
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
import rich

def apply_rotary_pos_emb_d2(
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    shard_logical_range: Optional[Tensor] = None, 
    mscale: float = 1.0
) -> Tensor:
    """    
        cu_seqlens (Tensor): [Batch + 1], cumulative sequence lengths for t.
        shard_logical_range (Tensor): [Batch, 2], logical [start, end) index in global document for each shard.
    """
    if cu_seqlens is None or shard_logical_range is None or freqs is None:
        rich.print("[red]Warning:[/red] Skip RoPE. Need cu_seqlens, shard_logical_range and freqs for apply_rotary_pos_emb_d2.")
        return t
    
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    
    t_shards = torch.split(t, seqlens)
    assert len(t_shards) == len(shard_logical_range)

    results = []
    assert shard_logical_range.shape[0] == len(t_shards), f"shard_logical_range must have the same length as t_shards, but got {shard_logical_range.shape[0]} and {len(t_shards)}"
    for i, shard in enumerate(t_shards):
        start_idx, end_idx = shard_logical_range[i]
        expected_len = end_idx - start_idx
        if shard.size(0) != expected_len:
             raise ValueError(f"Shard {i} physical length ({shard.size(0)}) matches logical range ({expected_len})")

        curr_freqs = freqs[start_idx:end_idx]
        shard_bshd = shard.unsqueeze(1)
        
        shard_out = _apply_rotary_pos_emb_bshd(
            shard_bshd,
            curr_freqs,
            rotary_interleaved=config.rotary_interleaved,
            multi_latent_attention=config.multi_latent_attention,
            mscale=mscale
        )
        
        results.append(shard_out.squeeze(1))

    return torch.cat(results, dim=0)