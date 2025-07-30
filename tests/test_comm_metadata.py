"""Test whether forward and backward metadata generation works correctly."""

import torch

from d2.runtime.inplace_metadata import Metadata, compute_metadata

from test_util import gen_seq_lens


@torch.no_grad()
def orchestrate_simulate(tensor: torch.Tensor, output_tensor: torch.Tensor, metadata: Metadata):
    assert tensor.dim() == 3    # (world_size, num_tokens, hidden_dim)
    world_size = tensor.shape[0]
    # handle sending rank-by-rank:
    for src_rank in range(world_size):
        dst_rank = metadata.dst_rank[src_rank]
        dst_offset = metadata.dst_offset[src_rank]
        seq_lens = metadata.seq_len[src_rank]
        acu_tokens = 0
        for j, rs in enumerate(dst_rank):
            seq_len = seq_lens[j]
            seq = tensor[src_rank][acu_tokens:acu_tokens + seq_len]
            if dst_rank.dim() == 1:
                rank = rs
                if rank >= 0:
                    try:
                        output_tensor[rank][dst_offset[j]: dst_offset[j] + seq_len] = seq
                    except RuntimeError as e:
                        print(f"{src_rank=}, {rank=}, {dst_offset[j]=}, {dst_offset[j] + seq_len=}, {seq_len=}, {output_tensor.shape, seq.shape, acu_tokens, tensor.shape}")
                        raise e
            else:
                for k, rank in enumerate(rs):
                    if rank >= 0:
                        output_tensor[rank][dst_offset[j][k]: dst_offset[j][k] + seq_len] = seq
            acu_tokens += seq_len
    return output_tensor


def test_query_dispatch():
    torch.manual_seed(0)
    WORLD_SIZE = 2
    NUM_SEQS = 3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    HIDDEN_SIZE = 4
    TOTAL_SEQ_LEN = 1024
    seq_len = gen_seq_lens(WORLD_SIZE, NUM_SEQS, TOTAL_SEQ_LEN).long().to(DEVICE)
    global_dispatch = torch.randint(0, WORLD_SIZE, (WORLD_SIZE, NUM_SEQS),
                                    device=DEVICE)

    tensor = torch.rand((WORLD_SIZE, TOTAL_SEQ_LEN, HIDDEN_SIZE), device=DEVICE) + 1    # + 1 to avoid zeros
    fwd_metadata, rev_metadata = compute_metadata(seq_len, global_dispatch)

    # forward
    max_recv_tokens = fwd_metadata.num_recv_tokens.max()
    output_tensor = torch.zeros((WORLD_SIZE, max_recv_tokens, HIDDEN_SIZE),
                                device=DEVICE, dtype=tensor.dtype)
    output_tensor = orchestrate_simulate(tensor, output_tensor, fwd_metadata)

    # reverse
    rev_metadata.num_recv_tokens.max()
    rev_tensor = torch.zeros((WORLD_SIZE, TOTAL_SEQ_LEN, HIDDEN_SIZE),
                             device=DEVICE, dtype=output_tensor.dtype)
    rev_tensor = orchestrate_simulate(output_tensor, rev_tensor, rev_metadata)
    rev_tensor = rev_tensor.reshape(WORLD_SIZE, TOTAL_SEQ_LEN, HIDDEN_SIZE)
    torch.testing.assert_close(tensor, rev_tensor)
    print("test metadata passed")


if __name__ == '__main__':
    test_query_dispatch()
