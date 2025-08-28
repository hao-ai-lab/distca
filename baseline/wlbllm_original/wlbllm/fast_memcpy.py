import torch

from wlbllm.utils import doc_shard


def _exclusive_cumsum(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Cumsum but excluding itself."""
    cumsum = tensor.cumsum(dim=dim)
    zero = torch.zeros_like(tensor.select(dim, 0))
    return torch.cat([zero.unsqueeze(dim), cumsum.narrow(dim, 0, cumsum.size(dim) - 1)], dim=dim)


def prepare_metadata(doc_shards: list[list[doc_shard]], device: torch.device, cp_size: int, chunk_size: int):
    assert len(doc_shards) == cp_size * 2
    # shape (num_chunks, num_docs)
    chunk_shard_lens = torch.tensor([
        [ds.shard_len if ds is not None else 0 for ds in cds]
        for cds in doc_shards
    ], dtype=torch.uint64, device=device)

    # compute the current shard's offset at dst layout
    intra_chunk_offset = _exclusive_cumsum(chunk_shard_lens, dim=1)
    # here we use chunk_size instead of sum(shard_lens_on_rank) because there are padding values.
    chunk_offset = chunk_size * torch.arange(0, cp_size * 2, device=device, dtype=torch.uint64)
    shard_src_offset = intra_chunk_offset + chunk_offset.unsqueeze(1)

    # return value is in the (docs, chunks) layout
    chunk_shard_lens = chunk_shard_lens.transpose(0, 1).contiguous()
    shard_src_offset = shard_src_offset.transpose(0, 1).contiguous()
    return chunk_shard_lens, shard_src_offset


def prepare_tensor(k_tensor_list: list[torch.Tensor], context_length: int, cp_size: int):
    chunk_size = context_length // (2 * cp_size)
    assert len(k_tensor_list) == cp_size
    k_tensors = [torch.split(k, chunk_size, dim=0) for k in k_tensor_list]
    # sort by the chunk index
    k_tensors = [ks[0] for ks in k_tensors] + list(reversed([ks[1] for ks in k_tensors]))
    # concat to shape (num_chunks, chunk_size, hidden)
    k_tensors = [k.unsqueeze(0) for k in k_tensors]
    k_tensor = torch.cat(k_tensors, dim=0).flatten(start_dim=2).contiguous()
    return k_tensor
