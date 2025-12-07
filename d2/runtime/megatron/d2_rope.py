import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd


def apply_rotary_pos_emb_d2(
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    shard_logical_range: Optional[Tensor] = None, 
    mscale: float = 1.0
) -> Tensor:
    with torch.cuda.nvtx.range("RoPE_d2.apply_rotary_pos_emb_d2_opt"):
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        offsets = shard_logical_range[:, 0] - cu_seqlens[:-1]
        
        token_offsets = torch.repeat_interleave(offsets, seqlens)
        final_indices = torch.arange(t.shape[0], device=t.device) + token_offsets
        
        curr_freqs = freqs[final_indices]
        with torch.cuda.nvtx.range("RoPE_d2.apply_rotary_pos_emb_bshd"):
            t_out = _apply_rotary_pos_emb_bshd(
                t.unsqueeze(1),
                curr_freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=config.multi_latent_attention,
                mscale=mscale
            )
        
        return t_out.squeeze(1)


@triton.jit
def _rope_embedding_kernel(
    Q_ptr,
    Freqs_ptr,
    CuSeqlens_ptr,
    LogicalStarts_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_f_seq,
    stride_f_dummy1,
    stride_f_dummy2,
    stride_f_dim,
    stride_logical_start,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MSCALE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_idx_in_seq = tl.program_id(2)
    
    start_token_idx = tl.load(CuSeqlens_ptr + batch_idx)
    end_token_idx = tl.load(CuSeqlens_ptr + batch_idx + 1)
    seq_len = end_token_idx - start_token_idx
    
    current_seq_offset = block_idx_in_seq * BLOCK_SIZE
    
    if current_seq_offset >= seq_len:
        return

    logical_start = tl.load(LogicalStarts_ptr + batch_idx * stride_logical_start)
    
    offs_block = tl.arange(0, BLOCK_SIZE)
    
    mask = current_seq_offset + offs_block < seq_len
    
    physical_token_indices = start_token_idx + current_seq_offset + offs_block
    logical_indices = logical_start + current_seq_offset + offs_block

    offs_dim = tl.arange(0, HEAD_DIM // 2)
    
    q_ptr_base = Q_ptr + (physical_token_indices[:, None] * stride_q_token) + (head_idx * stride_q_head)
    
    q1_ptrs = q_ptr_base + (offs_dim[None, :] * stride_q_dim)
    q2_ptrs = q_ptr_base + ((offs_dim[None, :] + (HEAD_DIM // 2)) * stride_q_dim)
    
    q1 = tl.load(q1_ptrs, mask=mask[:, None])
    q2 = tl.load(q2_ptrs, mask=mask[:, None])
    
    f_ptr_base = Freqs_ptr + (logical_indices[:, None] * stride_f_seq)
    
    cos_ptrs = f_ptr_base + (offs_dim[None, :] * stride_f_dim)
    sin_ptrs = f_ptr_base + ((offs_dim[None, :] + (HEAD_DIM // 2)) * stride_f_dim)
    
    cos = tl.load(cos_ptrs, mask=mask[:, None])
    sin = tl.load(sin_ptrs, mask=mask[:, None])
    
    if MSCALE != 1.0:
        cos = cos * MSCALE
        sin = sin * MSCALE

    out1 = q1 * cos - q2 * sin
    out2 = q1 * sin + q2 * cos
    
    tl.store(q1_ptrs, out1, mask=mask[:, None])
    tl.store(q2_ptrs, out2, mask=mask[:, None])


def apply_rotary_pos_emb_d2_triton(
    t: torch.Tensor,
    freqs: torch.Tensor,
    config: object,
    cu_seqlens: Optional[torch.Tensor] = None,
    shard_logical_range: Optional[torch.Tensor] = None, 
    mscale: float = 1.0,
    max_seq_len: Optional[int] = None,
    check_args: bool = False
) -> torch.Tensor:
    """
    Triton kernel version of apply_rotary_pos_emb_d2.
    Requires all tensors to be on GPU and t to have shape [TotalTokens, NumHeads, HeadDim].
    """
    if check_args:
        if cu_seqlens is None or shard_logical_range is None:
            raise ValueError("cu_seqlens and shard_logical_range are required")
    
    if not t.is_cuda:
        raise ValueError(f"Triton kernel requires GPU tensors, but t is on {t.device}")
    if not freqs.is_cuda:
        raise ValueError(f"Triton kernel requires GPU tensors, but freqs is on {freqs.device}")
    if not cu_seqlens.is_cuda:
        raise ValueError(f"Triton kernel requires GPU tensors, but cu_seqlens is on {cu_seqlens.device}")
    if not shard_logical_range.is_cuda:
        raise ValueError(f"Triton kernel requires GPU tensors, but shard_logical_range is on {shard_logical_range.device}")

    batch_size = int(cu_seqlens.shape[0] - 1)
    num_heads = int(t.shape[1])
    head_dim = int(t.shape[2])
    mscale_val = float(mscale)
    
    if freqs.is_complex():
        freqs = torch.view_as_real(freqs).flatten(-2)
    
    if not freqs.is_contiguous():
        freqs = freqs.contiguous()

    if max_seq_len is None:
        with torch.no_grad():
            max_seq_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    else:
        max_seq_len = int(max_seq_len)
    
    BLOCK_SIZE = 64
    num_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (batch_size, num_heads, num_blocks)
    
    logical_starts = shard_logical_range[:, 0]
    
    _rope_embedding_kernel[grid](
        Q_ptr=t,
        Freqs_ptr=freqs,
        CuSeqlens_ptr=cu_seqlens,
        LogicalStarts_ptr=logical_starts,
        
        stride_q_token=int(t.stride(0)),
        stride_q_head=int(t.stride(1)),
        stride_q_dim=int(t.stride(2)),
        
        stride_f_seq=int(freqs.stride(0)),
        stride_f_dummy1=int(freqs.stride(1)),
        stride_f_dummy2=int(freqs.stride(2)),
        stride_f_dim=int(freqs.stride(3)),
        
        stride_logical_start=int(logical_starts.stride(0)),
        
        HEAD_DIM=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        MSCALE=mscale_val
    )
    
    return t
