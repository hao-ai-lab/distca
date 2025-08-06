"""
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 2 test_megatron_layer_bwd_resend.py \
    --world-size 2
"""

from megatron.core.packed_seq_params import PackedSeqParams
import torch

from d2.runtime.inplace_metadata import mlp_layout_packed_params
from d2.runtime.megatron_patch.packed_seq_params import PingPangSingleStepPackedSeqParams

from test_util import create_qkv_dispath_with_backward
from test_megatron_layer import MegatronLayerWorker, init_megatron_test


def test_forward(
    seed, world_size, total_seq_len, num_seqs, max_cp_degree,
    worker: MegatronLayerWorker, hidden_size_q: int, hidden_size_k: int, num_heads: int
):
    torch.manual_seed(seed)
    dtype = torch.float16
    element_size = dtype.itemsize
    (
        # fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
        # attention_metadata_attn_layout, intermediates, seq_lens
        fwd_metadata_q, bwd_metadata_q, fwd_metadata_kv, bwd_metadata_kv,
        fa_fwd_params, fa_bwd_params,
        qkv_fwd_fa2a_metadata, qkv_bwd_fa2a_metadata,
        attn_out_fwd_fa2a_metadata, attn_out_qkv_bwd_fa2a_metadata,
        seq_lens,
    ) = create_qkv_dispath_with_backward(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        hidden_size_q, hidden_size_k, element_size, num_heads * torch.float32.itemsize // element_size,
        return_mlp_no_shard_seq_lens=True
    )

    # thd layout's hidden size input is "t,1,h"
    torch.manual_seed(seed)
    tensors = torch.randn(
        (world_size, total_seq_len, 1, hidden_size_q), dtype=dtype
    )
    rank = worker.rank
    tensor_shard = tensors[rank]

    # NOTE: this already adds prepended zeros and is sharded to tuples (remove padding seqs)
    (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, *_) = fa_bwd_params
    bwd_packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens_q[rank],
        cu_seqlens_kv=cu_seqlens_kv[rank],
        max_seqlen_q=max_seqlen_q[rank],
        max_seqlen_kv=max_seqlen_kv[rank],
    )
    (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, *_) = fa_fwd_params

    # 1. normal forward. Need to provide the PackedSeqParams
    seq_lens_local = seq_lens[rank][:num_seqs]
    packed_seq_params = mlp_layout_packed_params(seq_lens_local)
    normal_forward_out, debug_ref = worker.forward_normal(
        tensor_shard, packed_seq_params
    )

    ping_pang_params = PingPangSingleStepPackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q[rank],
        cu_seqlens_kv=cu_seqlens_kv[rank],
        max_seqlen_q=max_seqlen_q[rank],
        max_seqlen_kv=max_seqlen_kv[rank],
        qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(rank),
        qkv_bwd_metadata=qkv_bwd_fa2a_metadata.get_slice(rank),
        attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(rank),
        attn_out_bwd_metadata=attn_out_qkv_bwd_fa2a_metadata.get_slice(rank),
        bwd_packed_seq_params=bwd_packed_seq_params,
    )
    ping_pang_out, debug_out = worker.forward_ping_pang_one_stage(
        tensor_shard, ping_pang_params
    )

    torch.testing.assert_close(normal_forward_out, ping_pang_out)
    print("pass final result")


def test(args):
    # only test multi-head-attention. For GQA, update get_gpt_config
    world_size = args.world_size
    max_tokens_query = args.num_tokens * world_size
    max_tokens_key_value = args.num_tokens * world_size
    hidden_size = args.hidden_size
    max_cp_degree = args.max_cp_degree
    seed = args.seed
    tp_size = 1
    num_heads = args.num_heads
    dtype = torch.float16
    num_query_heads = num_heads
    hidden_size_kv = (hidden_size * num_query_heads) // num_heads

    worker = init_megatron_test(
        world_size, hidden_size, num_heads, num_query_heads, dtype,
        max_tokens_query, max_tokens_key_value, max_cp_degree, tp_size, seed,
    )

    test_forward(
        args.seed, world_size, args.num_tokens, args.num_seqs,
        max_cp_degree, worker, hidden_size, hidden_size_kv, num_heads,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--max-cp-degree", type=int, default=2)
    # NOTE: when increasing this value, remember to increase num-heads as well
    # because FA2 only supports head_dim_qk <= 256.
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    args = parser.parse_args()
    test(args)
