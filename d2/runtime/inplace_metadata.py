from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams

"""
NOTE:
There is an implicit global index for all sequence shards.
The index is a flatten (rank, seq_shard_id, dispatch_id) on FFN layout.
For query, dispatch_id is 0.

For query, Attention layout, the local order of the shards is determined by
their global index.

For key-value, Attention layout, the local order is determined by the query's
order. Hence, it needs to know a Q-shard and KV-shard correlation.
"""

def exclusive_cumsum(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Cumsum but excluding itself."""
    cumsum = tensor.cumsum(dim=dim)
    zero = torch.zeros_like(tensor.select(dim, 0))
    return torch.cat([zero.unsqueeze(dim), cumsum.narrow(dim, 0, cumsum.size(dim) - 1)], dim=dim)


@dataclass
class Metadata:
    # Rank, offset, and length described local sequence index.
    dst_rank: torch.Tensor
    dst_offset: torch.Tensor
    seq_len: torch.Tensor
    # num tokens received from each rank.
    num_recv_tokens: torch.Tensor
    # NOTE: used for kv gradient communication.
    # This is a sequence-scale mask, recording the number of sequences received
    # from each rank.
    seq_recv_mask: Optional[torch.Tensor] = None
    # NOTE: used for kv gradient communication.
    # This records the length of all sequences received to this rank.
    recv_seq_lens: Optional[torch.Tensor] = None
    # Number of sequences on each rank. This is used when this Metadata
    # describes all ranks in the world, and thus is padded to the one with
    # the most sequences. Otherwise, this should be None.
    num_seqs: Optional[torch.Tensor] = None
    world_size: int = None
    normalized: bool = False
    num_total_recv_tokens: Union[int, tuple[int]] = None

    def get_slice(self, rank: int):
        assert self.world_size is not None
        assert self.num_seqs is not None

        num_seqs = self.num_seqs[rank]
        return Metadata(
            dst_rank=self.dst_rank[rank][:num_seqs],
            dst_offset=self.dst_offset[rank][:num_seqs],
            seq_len=self.seq_len[rank][:num_seqs],
            num_recv_tokens=self.num_recv_tokens[rank], # this is of shape (world_size,)
            seq_recv_mask=self.seq_recv_mask[rank] if self.seq_recv_mask is not None else None,
            recv_seq_lens=self.recv_seq_lens[rank] if self.recv_seq_lens is not None else None,
            normalized=self.normalized,
            num_total_recv_tokens=self.num_total_recv_tokens[rank],
        )

    def normalize_dtype(self):
        return Metadata(
            dst_rank=self.dst_rank.to(torch.int32),
            dst_offset=self.dst_offset.to(torch.uint32),
            seq_len=self.seq_len.to(torch.uint32),
            num_recv_tokens=self.num_recv_tokens.to(torch.uint64),
            seq_recv_mask=self.seq_recv_mask.to(torch.uint32) if self.seq_recv_mask is not None else None,
            recv_seq_lens=self.recv_seq_lens.to(torch.uint32) if self.recv_seq_lens is not None else None,
            num_seqs=self.num_seqs.to(torch.int32) if self.num_seqs is not None else None,
            world_size=self.world_size,
            normalized=True,
            num_total_recv_tokens=self.num_total_recv_tokens,
        )

    def cuda(self):
        return Metadata(
            dst_rank=self.dst_rank.cuda().contiguous(),
            dst_offset=self.dst_offset.cuda().contiguous(),
            seq_len=self.seq_len.cuda().contiguous(),
            num_recv_tokens=self.num_recv_tokens.cuda().contiguous(),
            seq_recv_mask=self.seq_recv_mask.cuda().contiguous() if self.seq_recv_mask is not None else None,
            recv_seq_lens=self.recv_seq_lens.cuda().contiguous() if self.recv_seq_lens is not None else None,
            num_seqs=self.num_seqs.cuda().contiguous() if self.num_seqs is not None else None,
            world_size=self.world_size,
            normalized=self.normalized,
            num_total_recv_tokens=self.num_total_recv_tokens,
        )


@torch.no_grad()
def compute_metadata(
    seq_len: torch.Tensor,  # shape of (world_size, max_num_local_seqs)
    global_dispatch: torch.Tensor,  # shape of (world_size, max_num_local_seqs, max_cp_degree)
    return_intermediate: bool = False,
) -> Tuple[Metadata, Metadata]:
    """
    Given a dispatch plan, this function assigns the query tensor's attention layout.
    Args:
        seq_len (Tensor): shape (world_size, max_num_local_seq_shards). The length of each sequence.
        global_dispatch (Tensor): shape (world_size, max_num_local_seq_shards, max_cp_degree). value -1 is padding.
        The decision tensor. Recording the ranks that each sequence is dispatched to.
    Returns:
        fwd_metadata, rev_metadata: Metadata for forward and reverse communication.
    """
    world_size = global_dispatch.shape[0]
    max_num_local_seqs = global_dispatch.shape[1]
    assert seq_len.shape == (world_size, max_num_local_seqs)
    assert global_dispatch.dtype == torch.int64
    assert seq_len.dtype == torch.int64

    # Query. No max cp degree dimension.
    dispatch_shape = global_dispatch.shape
    if global_dispatch.dim() == 2:
        global_dispatch = global_dispatch.unsqueeze(-1)
    # max_cp_degree = global_dispatch.shape[2]

    ######## Forward side metadata
    seq_to_dst = F.one_hot(global_dispatch + 1, num_classes=world_size + 1)[:, :, :, 1:]
    # (world_size, max_num_local_seqs, max_cp_degree, world_size) [i][j][k][l]:
    # for each sequence ([i][j]), for each dispatch ([k]), the number of sequence sent to rank [l] (either 0 or seq_len[i][j])
    tokens_to_dst_per_dispatch = seq_to_dst * seq_len.unsqueeze(-1).unsqueeze(-1)
    # all tokens by global index i, sending to rank j
    tokens_to_dst = tokens_to_dst_per_dispatch.reshape(-1, world_size)
    seq_begin_offset = tokens_to_dst.cumsum(dim=0) - tokens_to_dst
    # masking and back to the original shape
    seq_begin_offset = seq_begin_offset.reshape(*seq_to_dst.shape) * seq_to_dst
    seq_begin_offset = seq_begin_offset.sum(dim=-1)
    # compute the sequence level offset
    # (do not consider number of tokens. This is only used for reverse communication's indexing in scatter_)
    seq_to_dst_flatten = seq_to_dst.reshape(-1, world_size)
    seq_offset = seq_to_dst_flatten.cumsum(dim=0) - seq_to_dst_flatten
    seq_offset = seq_offset.reshape(*seq_to_dst.shape) * seq_to_dst
    seq_offset = seq_offset.sum(dim=-1)

    # number of tokens received from each rank ([i][j] means tokens received at rank i from rank j), last column is the total number.
    num_recv_tokens = tokens_to_dst_per_dispatch.reshape(world_size, -1, world_size).sum(dim=1).transpose(0, 1)
    num_recv_tokens = torch.cat([num_recv_tokens, num_recv_tokens.sum(dim=1, keepdim=True)], dim=1)

    ######## reverse side metadata:
    seq_rank_expanded = torch.arange(world_size, device=global_dispatch.device).reshape(world_size, 1, 1).expand_as(global_dispatch)
    # seq_offset = seq_len.cumsum(dim=1) - seq_len # NOTE: this is the final offset after deduplication.
    # expanding by max_cp_degree to allow concurrently receiving from multiple ranks
    seq_len_expanded = seq_len.unsqueeze(2).expand_as(global_dispatch).transpose(1, 2)
    seq_offset_expanded = seq_len_expanded.reshape(world_size, -1).cumsum(dim=1).reshape(seq_len_expanded.shape) - seq_len_expanded
    seq_offset_expanded = seq_offset_expanded.transpose(1, 2)
    seq_len_expanded = seq_len_expanded.transpose(1, 2)
    # reverse side communication
    num_received_seqs = seq_to_dst.reshape(-1, world_size).sum(0)
    max_rev_seqs = num_received_seqs.max()
    rev_dst_rank = torch.zeros(world_size, max_rev_seqs, dtype=torch.int64, device=global_dispatch.device)
    rev_dst_offset = torch.zeros(world_size, max_rev_seqs, dtype=torch.int64, device=global_dispatch.device)
    rev_seq_len = torch.zeros(world_size, max_rev_seqs, dtype=torch.int64, device=global_dispatch.device)

    # Create valid mask for non-padding entries
    valid_mask = global_dispatch != -1
    # Flatten all tensors for vectorized processing
    # flatten to [world_size * max_num_local_seqs * max_cp_degree]
    valid_mask_flat = valid_mask.flatten()
    global_dispatch_flat = global_dispatch.flatten()
    seq_rank_expanded_flat = seq_rank_expanded.flatten()
    seq_offset_expanded_flat = seq_offset_expanded.flatten()
    seq_len_expanded_flat = seq_len_expanded.flatten()

    # Use flatten valid indices to choose the valid sequence dispatch entries
    valid_indices = torch.where(valid_mask_flat)[0] # ranging from 0 to world_size * max_num_local_seqs * max_cp_degree - 1
    # from global dispatch id to the dst rank
    valid_dst_ranks = global_dispatch_flat[valid_indices]
    # from global dispatch id to the src rank
    valid_src_ranks = seq_rank_expanded_flat[valid_indices]
    valid_src_offsets = seq_offset_expanded_flat[valid_indices]
    valid_src_seq_lens = seq_len_expanded_flat[valid_indices]
    dst_seq_local_offset = seq_offset.flatten()[valid_indices]

    # Compute global destination (flatten) indices for scatter operation
    # For each valid entry, compute where it should go in the reverse tensors
    global_dst_indices = valid_dst_ranks * max_rev_seqs + dst_seq_local_offset

    rev_dst_rank.view(-1).scatter_(0, global_dst_indices, valid_src_ranks)
    rev_dst_offset.view(-1).scatter_(0, global_dst_indices, valid_src_offsets)
    rev_seq_len.view(-1).scatter_(0, global_dst_indices, valid_src_seq_lens)

    # masking
    rev_dst_rank = rev_dst_rank.masked_fill(
        num_received_seqs.unsqueeze(1) <= torch.arange(max_rev_seqs, device=rev_dst_rank.device), -1)
    rev_dst_offset = rev_dst_offset.masked_fill(
        num_received_seqs.unsqueeze(1) <= torch.arange(max_rev_seqs, device=rev_dst_offset.device), 0)
    rev_seq_len = rev_seq_len.masked_fill(
        num_received_seqs.unsqueeze(1) <= torch.arange(max_rev_seqs, device=rev_seq_len.device), 0)
    # number of tokens received in the reverse communication. This equals the number of tokens sent in the forward communication.
    rev_num_received_tokens = tokens_to_dst_per_dispatch.reshape(world_size, -1, world_size).sum(dim=1)
    rev_num_received_tokens = torch.cat([rev_num_received_tokens, rev_num_received_tokens.sum(dim=1, keepdim=True)], dim=1)

    # If this is kv (has cp degree), we add the sequence-cp mask to the reverse communication metadata.
    seq_recv_mask = None
    if len(dispatch_shape) == 3:
        seq_recv_mask = global_dispatch.reshape(dispatch_shape) >= 0
    num_seqs = (seq_len > 0).sum(dim=1)
    num_seqs_rev = (rev_seq_len > 0).sum(dim=1)

    fwd_metadata = Metadata(
        dst_rank=global_dispatch.reshape(dispatch_shape),
        dst_offset=seq_begin_offset.reshape(dispatch_shape),
        seq_len=seq_len,
        num_recv_tokens=num_recv_tokens,
        num_seqs=num_seqs,
        world_size=world_size,
        num_total_recv_tokens=num_recv_tokens[:, -1].tolist(),
    )
    rev_metadata = Metadata(
        dst_rank=rev_dst_rank,
        dst_offset=rev_dst_offset,
        seq_len=rev_seq_len,
        num_recv_tokens=rev_num_received_tokens,
        seq_recv_mask=seq_recv_mask,
        recv_seq_lens=seq_len if seq_recv_mask is not None else None,
        num_seqs=num_seqs_rev,
        world_size=world_size,
        num_total_recv_tokens=rev_num_received_tokens[:, -1].tolist(),
    )

    # NOTE: use this for the fast alltoall dispatch layout.
    intermediates = (
        tokens_to_dst_per_dispatch, seq_to_dst, num_received_seqs
    )
    if return_intermediate:
        return fwd_metadata, rev_metadata, intermediates
    return fwd_metadata, rev_metadata


@torch.no_grad()
def compute_attn_layout_seqlens(
    seq_shard_len: torch.Tensor, seq_shard_cumsum: torch.Tensor,
    dispatch: torch.Tensor,
):
    """
    Compute the cu_seqlens_q and cu_seqlens_kv for the attention layout.
    NOTE: both inputs and outputs are the global value, so it has `world_size` dimension.
    Args:
        seq_shard_len: shape (world_size, max_num_local_seqs), length of each sequence shard.
          A sequence shard is the unit sending to a rank.
        seq_shard_cumsum: shape (world_size, max_num_local_seqs). Cumulative number of tokens
          for the sequence shard's context. This is to compute the KV seqlens of the shard.
        dispatch: shape (world_size, max_num_local_seqs). The rank that each sequence
          shard is dispatched to. The value is -1 for padding.
    """
    world_size = dispatch.shape[0]
    max_num_local_seqs = dispatch.shape[1]
    assert dispatch.dim() == 2
    assert seq_shard_len.shape == (world_size, max_num_local_seqs)
    assert seq_shard_cumsum.shape == (world_size, max_num_local_seqs)
    assert dispatch.dtype == torch.int64
    assert seq_shard_len.dtype == torch.int64
    assert seq_shard_cumsum.dtype == torch.int64
    # dispatch[i, j] = the rank that sequence [i,j] is dispatched to.
    flatten_dispatch = dispatch.flatten()

    flatten_dispatch_one_hot = F.one_hot(flatten_dispatch + 1, num_classes=world_size + 1)[:, 1:]
    # shape: (world_size, seq_len, world_size)
    local_indices_flat = (
        # cumsum: the id of this sequence at the dst rank.
        (flatten_dispatch_one_hot.cumsum(dim=0) - 1) * flatten_dispatch_one_hot
    ).sum(dim=1).reshape(-1)
    # if dispatch[i, j] = k, then local_indices_flat[i, j, k] = l means sequence [i,j] is at out_sequence [k,l]
    # out_seqlens_q[k, l] = seq_shard_len[i, j]
    max_num_seq = int(local_indices_flat.max().item() + 1)
    scatter_index = (flatten_dispatch * (flatten_dispatch >= 0)) * max_num_seq + local_indices_flat

    src_seqlens = seq_shard_len.flatten()
    src_seq_lens_kv = seq_shard_cumsum.flatten()

    out_seqlens_q = torch.zeros(world_size * max_num_seq, dtype=torch.int64, device=dispatch.device)
    out_seqlens_kv = torch.zeros(world_size * max_num_seq, dtype=torch.int64, device=dispatch.device)
    out_seqlens_q.scatter_(0, scatter_index, src_seqlens)
    out_seqlens_kv.scatter_(0, scatter_index, src_seq_lens_kv)
    out_seqlens_q = out_seqlens_q.reshape(world_size, max_num_local_seqs)
    out_seqlens_kv = out_seqlens_kv.reshape(world_size, max_num_local_seqs)

    num_local_seqs_recv = local_indices_flat.reshape(-1, world_size).max(dim=0)[0] + 1

    cu_seqlens_q = out_seqlens_q.cumsum(dim=1)
    cu_seqlens_kv = out_seqlens_kv.cumsum(dim=1)
    max_seqlen_q = out_seqlens_q.max(dim=1)[0]
    max_seqlen_kv = out_seqlens_kv.max(dim=1)[0]

    return cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, num_local_seqs_recv


def mlp_layout_packed_params(seq_lens: torch.Tensor):
    """
    Compute the MLP layout packed_seq_params. MLP layout guarantees seqlens_q == seqlens_kv.
    This is mainly for RoPE.
    """
    cu_seqlens = torch.cat([
        torch.zeros((1,), dtype=seq_lens.dtype, device=seq_lens.device),
        seq_lens.cumsum(dim=0)
    ])
    max_seqlen = seq_lens.max()
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    return packed_seq_params
