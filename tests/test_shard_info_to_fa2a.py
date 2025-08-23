import argparse
import os
import random

import torch

from d2.runtime.compute_metadata import from_shard_info
from d2.runtime.shard_info import ShardInfo

from test_util import set_random_seed
from test_fa2a_metadata import (
    simulate_fa2a, simulate_fa2a_send_qkv, simulate_fa2a_send_qkv_rev,
    simulate_fa2a_recv_qkv, simulate_fa2a_recv_qkv_rev
)

def create_random_shard_info(
    seed: int, world_size: int, num_doc: int,
    max_num_shard: int, max_shard_len: int, min_shard_len: int=8
):
    set_random_seed(seed)
    scheduler_output = []
    num_shards = torch.randint(1, max_num_shard + 1, (num_doc,)).tolist()
    has_shard_src = [False] * world_size
    has_shard_dst = [False] * world_size
    src_num_token = [0] * world_size
    for doc_id in range(num_doc):
        num_shard = num_shards[doc_id]
        doc_schedule = []
        for shard_id in range(num_shard):
            rid = random.randint(0, world_size - 1)
            d_rid = random.randint(0, world_size - 1)
            has_shard_src[rid] = True
            has_shard_dst[d_rid] = True
            shard_len = random.randint(min_shard_len, max_shard_len)
            src_num_token[rid] += shard_len
            doc_schedule.append(
                ShardInfo(rid=rid, dispatch_rid=d_rid, logical_sid=shard_id, shard_len=shard_len)
            )
        scheduler_output.append(doc_schedule)
    for rank in range(world_size):
        if not has_shard_src[rank]:
            scheduler_output.append([ShardInfo(rid=rank, dispatch_rid=rank, logical_sid=0, shard_len=min_shard_len)])
            has_shard_src[rank] = True
            has_shard_dst[rank] = True
            src_num_token[rank] += min_shard_len
        if not has_shard_dst[rank]:
            scheduler_output.append([ShardInfo(rid=rank, dispatch_rid=rank, logical_sid=0, shard_len=min_shard_len)])
            has_shard_src[rank] = True
            has_shard_dst[rank] = True
            src_num_token[rank] += min_shard_len
    return scheduler_output, src_num_token


def create_redispatch_info(
    seed: int, fwd_dispatch: list[list[ShardInfo]], world_size: int,
    max_num_shard: int, max_shard_len: int, min_shard_len: int=8,
):
    """
    TODO: fix the doc length from the given fwd_dispatch, create a new bwd dispatch.
    However, it should be guaranteed that each src and dst must have at least one shard.
    """
    raise NotImplementedError("")


def test(args):
    seed = args.seed
    num_doc = args.num_doc
    max_num_shard = args.max_num_shard
    max_shard_len = args.max_shard_len
    min_shard_len = args.min_shard_len
    simulate = args.simulate_world_size > 0
    world_size = args.simulate_world_size if simulate else os.environ.get("WORLD_SIZE")

    hidden_size_q = args.hidden_size_q
    hidden_size_k = args.hidden_size_k
    is_pp = args.is_pp
    num_head = args.num_head
    element_size = torch.bfloat16.itemsize
    lse_size = num_head * torch.float32.itemsize // element_size

    scheduler_output, src_num_token = create_random_shard_info(
        seed, world_size, num_doc, max_num_shard, max_shard_len, min_shard_len
    )

    assert not is_pp, "not supported yet"
    if is_pp:
        scheduler_output_bwd = create_redispatch_info(seed, scheduler_output, world_size)
    else:
        scheduler_output_bwd = None
    fwd_qkv_metadata, bwd_qkv_metadata, fwd_attn_out_metadata, bwd_attn_out_metadata = from_shard_info(
        world_size, scheduler_output, hidden_size_q, hidden_size_k, lse_size, element_size,
        is_pp, scheduler_output_bwd
    )

    src_q = [
        torch.rand((src_num_token[rank], hidden_size_q), dtype=torch.bfloat16)
        for rank in range(world_size)
    ]
    src_k = [
        torch.rand((src_num_token[rank], hidden_size_k), dtype=torch.bfloat16)
        for rank in range(world_size)
    ]
    src_v = [
        torch.rand((src_num_token[rank], hidden_size_k), dtype=torch.bfloat16)
        for rank in range(world_size)
    ]
    # TODO: verify this by:
    # without pp, send with fwd_qkv_metadata and bwd_qkv_metadata, examine if received is the same as sent
    # without pp, send with fwd_attn_out_metadata and bwd_att_out_metadata, examine if received is the same as sent
    # with pp, send with the four metadata, examine if received is the same as sent.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--simulate-world_size", type=int, default=-1)
    parser.add_argument("--num-doc", type=int, default=4)
    parser.add_argument("--max-num-shard", type=int, default=2)
    parser.add_argument("--max-shard-len", type=int, default=128)
    parser.add_argument("--min-shard-len", type=int, default=2)
    parser.add_argument("--hidden-size-q", type=int, default=512)
    parser.add_argument("--hidden-size-k", type=int, default=128)
    parser.add_argument("--num-head", type=int, default=2)
    parser.add_argument("--is-pp", type=bool, default=False)
    args = parser.parse_args()

    test(args)
