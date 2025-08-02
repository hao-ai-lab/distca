"""
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 2 test_megatron_layer.py \
    --world-size 2
"""

from typing import Optional

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
import torch

from d2.runtime.inplace_metadata import mlp_layout_packed_params
from d2.runtime.megatron_patch.model_patch import get_gpt_layer_with_transformer_engine_spec, get_gpt_config
from d2.runtime.megatron_patch.packed_seq_params import PingPangSingleStepPackedSeqParams
from d2.runtime.megatron_patch.transformer_layer import TransformerLayer as PingPangTransformerLayer
from d2.runtime.fast_alltoall_metadata import compute_fa2a_metadata_from_logical_metadata

from test_util import (
    MegatronBaseWorker, ParallelConfig, create_qkv_dispatch,
    init_worker_torch_distributed,
)


class MegatronLayerWorker(MegatronBaseWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.layer: Optional[PingPangTransformerLayer] = None

    def init_layer(self, config: TransformerConfig, spec: ModuleSpec,
                   seed: int):
        torch.manual_seed(seed)
        self.layer = build_module(spec, config)

    def forward_normal(self, tensor_input: torch.Tensor, packed_seq_params: PackedSeqParams):
        packed_seq_params = PackedSeqParams(
            qkv_format=packed_seq_params.qkv_format,
            cu_seqlens_q=packed_seq_params.cu_seqlens_q.cuda().to(torch.int32),
            cu_seqlens_kv=packed_seq_params.cu_seqlens_kv.cuda().to(torch.int32),
            max_seqlen_q=packed_seq_params.max_seqlen_q.cuda().to(torch.int32),
            max_seqlen_kv=packed_seq_params.max_seqlen_kv.cuda().to(torch.int32),
        )
        tensor_input = tensor_input.cuda()
        self.layer.train()
        mlp_output, context, debug = self.layer.forward_no_switch(tensor_input, packed_seq_params=packed_seq_params)
        torch.cuda.synchronize()
        print(self.rank, "normal forward done")
        return (mlp_output, context), debug

    def forward_ping_pang_one_stage(
        self, tensor_input: torch.Tensor,
        packed_seq_params: PingPangSingleStepPackedSeqParams,
    ):
        packed_seq_params = packed_seq_params.to_device()
        tensor_input = tensor_input.cuda()
        self.layer.train()
        mlp_output, context, debug_tensors = self.layer.forward_one_stage(tensor_input, packed_seq_params=packed_seq_params)
        torch.cuda.synchronize()
        print(self.rank, "ping-pong one stage forward done")
        return (mlp_output, context), debug_tensors


def test_forward(
    seed, world_size, total_seq_len, num_seqs, max_cp_degree,
    worker: MegatronLayerWorker, hidden_size_q: int, hidden_size_k: int
):
    torch.manual_seed(seed)
    dtype = torch.float16
    element_size = dtype.itemsize
    (
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
        attention_metadata_attn_layout, intermediates, seq_lens
    ) = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        return_intermediate=True, return_mlp_no_shard_seq_lens=True
    )
    # NOTE: this already adds prepended zeros and is sharded to tuples (remove padding seqs)
    (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
     num_local_seqs_recv) = attention_metadata_attn_layout

    (qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
     attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
    ) = compute_fa2a_metadata_from_logical_metadata(
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
        intermediates, total_seq_len, hidden_size_q, hidden_size_k,
        element_size,
    )

    # thd layout's hidden size input is "t,1,h"
    tensors = torch.randn(
        (world_size, total_seq_len, 1, hidden_size_q), dtype=dtype
    )
    rank = worker.rank
    tensor_shard = tensors[rank]
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
        qkv_bwd_metadata=qkv_rev_fa2a_metadata.get_slice(rank),
        attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(rank),
        attn_out_bwd_metadata=attn_out_rev_fa2a_metadata.get_slice(rank),
    )
    ping_pang_out, debug_out = worker.forward_ping_pang_one_stage(
        tensor_shard, ping_pang_params
    )
    torch.testing.assert_close(normal_forward_out, ping_pang_out)
    print("pass final result")
    # TODO: this does not check debug tensors. Implement it later.
    # Ref: attn/playground/deprecated/test_megatron_layer.py


def test(args):
    # only test multi-head-attention. For GQA, update get_gpt_config
    token_bytes_q = args.hidden_size * torch.float16.itemsize
    token_bytes_kv = args.hidden_size * torch.float16.itemsize
    world_size = args.world_size
    max_tokens_query = args.num_tokens * world_size
    max_tokens_key_value = args.num_tokens * world_size
    max_cp_degree = args.max_cp_degree

    buffer_size = (
        token_bytes_q * max_tokens_query +
        token_bytes_kv * max_tokens_key_value * max_cp_degree * 2
    )
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=args.tp_size
    )
    worker: MegatronLayerWorker = init_worker_torch_distributed(
        world_size, buffer_size, MegatronLayerWorker, parallel_config
    )
    spec = get_gpt_layer_with_transformer_engine_spec()
    config = get_gpt_config(
        num_layers=1,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        ffn_hidden_size=args.hidden_size * 4,
        fp16=True,
        deterministic_mode=True,
        params_dtype=torch.float16,
    )
    worker.init_layer(config, spec, seed=args.seed)
    test_forward(
        args.seed, world_size, args.num_tokens, args.num_seqs,
        max_cp_degree, worker, args.hidden_size, args.hidden_size
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
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=2)
    args = parser.parse_args()
    test(args)
