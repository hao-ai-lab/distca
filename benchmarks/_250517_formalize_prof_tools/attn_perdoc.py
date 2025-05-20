import numpy as np
import torch
from itertools import accumulate

import torch.distributed as dist

from torch.distributed import (
    get_world_size,
    get_rank,
)

from torch.cuda.nvtx import range_push as nvtx_range_push
from torch.cuda.nvtx import range_pop  as nvtx_range_pop


from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward
)

from utils import (
    kv_shuffle_for_per_doc_cp,
    kv_unshuffle_for_per_doc_cp,
    compute_per_doc_cp_shard_doc_len,
)
from utils import (
    compute_per_doc_metadate_combined,
    compute_per_doc_cp_shard_doc_len
)
import time

def cat_slices(tensor, starts, lens):
    if starts.numel() == 1:
        s, l = int(starts[0]), int(lens[0])
        return tensor[s:s+l]
    return torch.cat([
        tensor[int(s):int(s)+int(l)]
        for s, l in zip(starts.tolist(), lens.tolist())
    ],dim=0)   

def all_gather(global_tensor, local_tensor, world_size):
    # sleep a bit and return
    pass

def reduce_scatter_tensor(local_tensor, global_tensor, op, world_size):
    # sleep a bit and return
    pass

class PerDocumentCPAttention(torch.autograd.Function):
    """
    Attention with per-document context parallelism.

    - local_q / k / v should contain every shard that this rank owns:
    [Doc1-Shard1, Doc2-Shard1, ..., DocN-Shard1, Doc1-Shard2, Doc2-Shard2, ..., DocN-Shard2]

    - cu_seqlens_q / kv are built in the same order as the local qkv tensor
    - q_len_left is the number of tokens of the concatenated front chunks (Shard1's) of the documents
    """

    @staticmethod
    def forward(
        ctx,
        local_q, local_k, local_v,
        cu_seqlens_q_list, cu_seqlens_kv_list,
        max_seqlen_q_list, max_seqlen_kv_list,
        doc_lens,
        doc_shards,
        kv_idx_list,                 # global offsets of the K shards after shuffle
        dropout_p,
        softmax_scale,
        attn_mask_type,
        cp_size,
        rank,
        cp_stream,
    ):
        nvtx_range_push("PerDocumentCPAttention.fwd")
        assert attn_mask_type == "causal", "Only causal attention is supported"
        assert cp_stream is not None, "cp_stream must be provided"

        context_length = local_q.shape[0] * cp_size
        
        # allgather kv, then shuffle back to global order
        gather_k_list = [torch.randn_like(local_q) for _ in range(cp_size)]
        gather_v_list = [torch.randn_like(local_q) for _ in range(cp_size)]
        with torch.cuda.stream(cp_stream):
            local_k = local_k.contiguous()
            local_v = local_v.contiguous()
            all_gather(gather_k_list, local_k, world_size=cp_size)
            all_gather(gather_v_list, local_v, world_size=cp_size)

        k_global, v_global = kv_shuffle_for_per_doc_cp(context_length, gather_k_list, gather_v_list, doc_lens, doc_shards, cp_size)

        # compute fwd of two chunks
        outputs, lses = [], []
        local_ks, local_vs = [], []
        q_chunks = local_q.chunk(2, dim=0)
        for chunk_id in range(2):
            local_k_list = []
            local_v_list = []
            for start, end in kv_idx_list[chunk_id]:
                local_k_list.append(k_global[start:end, :, :])
                local_v_list.append(v_global[start:end, :, :])

            local_k = torch.cat(local_k_list, dim=0)
            local_v = torch.cat(local_v_list, dim=0)
            local_ks.append(local_k)
            local_vs.append(local_v)

            out, lse, _ = flash_attn_varlen_func(
                q=q_chunks[chunk_id],
                k=local_k,
                v=local_v,
                cu_seqlens_q=cu_seqlens_q_list [chunk_id],
                cu_seqlens_k=cu_seqlens_kv_list[chunk_id],
                max_seqlen_q=max_seqlen_q_list [chunk_id],
                max_seqlen_k=max_seqlen_kv_list[chunk_id],
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,
                return_attn_probs=True
            )
            
            outputs.append(out)
            lses.append(lse)

        final_out = torch.cat(outputs, dim=0)
        ctx.save_for_backward(
            local_q,
            k_global, v_global,
            *outputs,       
            *lses,
            *local_ks,
            *local_vs,   
            *cu_seqlens_q_list, *cu_seqlens_kv_list,
            *max_seqlen_q_list, *max_seqlen_kv_list,
        )
        ctx.kv_idx_list    = kv_idx_list
        ctx.q_chunk_sizes  = [c.shape[0] for c in q_chunks]
        ctx.dropout_p      = dropout_p
        ctx.doc_lens       = doc_lens
        ctx.doc_shards     = doc_shards
        ctx.softmax_scale  = softmax_scale
        ctx.attn_mask_type = attn_mask_type
        ctx.cp_size        = cp_size
        ctx.rank           = rank
        ctx.cp_stream      = cp_stream
        
        nvtx_range_pop()
        return final_out

    @staticmethod
    def backward(ctx, d_out_cat):
        """
        Backward pass for PerDocumentCPAttention.
        """
        nvtx_range_push("PerDocumentCPAttention.bwd")

        (
            local_q,
            gathered_k, gathered_v,
            out_L, out_R, 
            lse_L, lse_R,
            k_L, k_R,
            v_L, v_R,
            cu_q_L, cu_q_R, cu_k_L, cu_k_R,
            maxq_L, maxq_R, maxk_L, maxk_R,
        ) = ctx.saved_tensors

        cp_size   = ctx.cp_size
        rank      = ctx.rank
        kv_idx_list = ctx.kv_idx_list 
        doc_lens = ctx.doc_lens
        doc_shards = ctx.doc_shards
        (qlen_L, qlen_R) = ctx.q_chunk_sizes
        world_size = cp_size

        # split grad_out into two chunks
        dq_local = torch.zeros_like(local_q)
        dk_global = torch.zeros_like(gathered_k)
        dv_global = torch.zeros_like(gathered_v)

        # split incoming d_out into the two logical chunks
        d_out_L, d_out_R   = d_out_cat.split([qlen_L, qlen_R], dim=0)

        chunk_size = d_out_L.size(0)
        context_length = chunk_size * 2 * cp_size

        for i, (d_out, q_len, out, lse, kv_k, kv_v, cu_q, cu_k, max_q, max_k) in enumerate([
            (d_out_L, qlen_L, out_L, lse_L, k_L, v_L, cu_q_L, cu_k_L, maxq_L, maxk_L),
            (d_out_R, qlen_R, out_R, lse_R, k_R, v_R, cu_q_R, cu_k_R, maxq_R, maxk_R),
        ]):
            if i == 0:
                chunk_index = rank
            else:
                chunk_index = 2 * cp_size - 1 - rank

            dq_chunk = torch.zeros_like(local_q[:q_len])
            dk_chunk = torch.zeros_like(kv_k)
            dv_chunk = torch.zeros_like(kv_v)

            _ = _flash_attn_varlen_backward(
                d_out,
                local_q[ sum(ctx.q_chunk_sizes[:i]) : sum(ctx.q_chunk_sizes[:i+1]) ],
                kv_k, kv_v,
                out,
                lse,
                dq_chunk, dk_chunk, dv_chunk,
                cu_q, cu_k, int(max_q), int(max_k),
                # 0.0, ctx.softmax_scale, True, (-1,-1), None, False, None
                0.0, ctx.softmax_scale, True, -1, -1, 0, None, False, None
            )

            dq_local[ sum(ctx.q_chunk_sizes[:i]) : sum(ctx.q_chunk_sizes[:i+1]) ] = dq_chunk
            local_idx = 0
            for start, end in kv_idx_list[i]:
                chunk_len = end - start
                dk_global[start:end] += dk_chunk[local_idx:local_idx + chunk_len] 
                dv_global[start:end] += dv_chunk[local_idx:local_idx + chunk_len] 
                local_idx += chunk_len

        dk_global, dv_global = kv_unshuffle_for_per_doc_cp(context_length, dk_global, dv_global, doc_lens, doc_shards, cp_size)
 
        # now do reduce_scatter for dk/dv
        dk_local = torch.empty_like(dq_local)
        dv_local = torch.empty_like(dq_local)
        with torch.cuda.stream(ctx.cp_stream):
            dk_global = dk_global.contiguous()
            dv_global = dv_global.contiguous()

            reduce_scatter_tensor(dk_local, dk_global, op=dist.ReduceOp.SUM, world_size=cp_size)
            reduce_scatter_tensor(dv_local, dv_global, op=dist.ReduceOp.SUM, world_size=cp_size)

        nvtx_range_pop()

        return (
            dq_local,        # grad w.r.t. local_q
            dk_local,        # grad w.r.t. local_k
            dv_local,        # grad w.r.t. local_v
            None,  # cu_seqlens_q_list
            None,  # cu_seqlens_kv_list
            None,  # max_seqlen_q_list
            None,  # max_seqlen_kv_list
            None,  # doc_lens
            None,  # doc_shards
            None,  # kv_idx_list
            None,  # dropout_p
            None,  # softmax_scale
            None,  # attn_mask_type
            None,  # cp_size
            None,  # rank
            None,  # cp_stream
        )


def gen_events(n):
    events = []
    for i in range(n):
        events.append(torch.cuda.Event(enable_timing=True))
    return events




def test_doc_cp_attn(
    doc_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    cp_size,
):
    rank = 0
    device = torch.device("cuda", rank)

    softmax_scale = head_dim ** -0.5
    batch_size = len(doc_lens)
    context_length = sum(doc_lens)

    n = batch_size * context_length
    dtype = torch.bfloat16
    q_global = torch.randn(n, num_qo_heads, head_dim, device=device, dtype=dtype)
    k_global = torch.randn(n, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_global = torch.randn(n, num_kv_heads, head_dim, device=device, dtype=dtype)
    d_out_global = torch.randn(n, num_qo_heads, head_dim, device=device, dtype=dtype)

    doc_shards = compute_per_doc_cp_shard_doc_len(doc_lens, context_length, cp_size)
    
    n_warmup = 10
    n_repeat = 10
    warmup_timeout = 10.0
    iter_timeout = 30.0
    iter_min_repeat = 3
    iter_std_threshold = 0.05


    fwd_times = []
    bwd_times = []
    
    start_time = time.time()
    for i in range(n_warmup + n_repeat):
        cur_time = time.time()
        if i < n_warmup and (cur_time - start_time) > warmup_timeout:
            continue
        (
            local_q_doc, local_k_doc, local_v_doc, 
            cu_seqlens_q, cu_seqlens_k, 
            max_seqlen_q, max_seqlen_k, 
            kv_idx_list, local_d_out,
        ) = compute_per_doc_metadate_combined(    
            context_length, q_global, k_global, v_global, 
            doc_lens, doc_shards,
            cp_size, rank, 
            d_out=d_out_global,
        )
        local_q_doc.retain_grad()
        local_k_doc.retain_grad()
        local_v_doc.retain_grad()
        fwd_st_event, fwd_ed_event = gen_events(2)
        bwd_st_event, bwd_ed_event = gen_events(2)
        
        torch.cuda.synchronize()
        fwd_st_event.record()
        out = PerDocumentCPAttention.apply(
            local_q_doc, 
            local_k_doc, 
            local_v_doc,
            cu_seqlens_q, 
            cu_seqlens_k,
            max_seqlen_q, 
            max_seqlen_k,
            doc_lens,
            doc_shards,
            kv_idx_list, 
            0.0, # dropout_p
            softmax_scale, 
            "causal",
            cp_size,
            rank,
            torch.cuda.current_stream(device) 
        )
        fwd_ed_event.record()
        torch.cuda.synchronize()

        bwd_st_event.record()
        out.backward(local_d_out)
        bwd_ed_event.record()
        torch.cuda.synchronize()

        fwd_time = fwd_st_event.elapsed_time(fwd_ed_event)
        bwd_time = bwd_st_event.elapsed_time(bwd_ed_event)
        # print(f"FWD time: {fwd_time:.2f}ms, BWD time: {bwd_time:.2f}ms")
        if i >= n_warmup:
            fwd_times.append(fwd_time)
            bwd_times.append(bwd_time)
        
        if i >= n_warmup:
            if len(fwd_times) >= iter_min_repeat:
                fwd_mean, fwd_std = np.mean(fwd_times), np.std(fwd_times)
                bwd_mean, bwd_std = np.mean(bwd_times), np.std(bwd_times)
                if fwd_std / fwd_mean < iter_std_threshold and bwd_std / bwd_mean < iter_std_threshold:
                    cur_time = time.time()
                    if cur_time - start_time > iter_timeout:
                        break
                    pass
                pass
            pass


        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    
    fwd_median, fwd_std, fwd_min, fwd_max, fwd_25p, fwd_75p = np.median(fwd_times), np.std(fwd_times), np.min(fwd_times), np.max(fwd_times), np.percentile(fwd_times, 25), np.percentile(fwd_times, 75)
    bwd_median, bwd_std, bwd_min, bwd_max, bwd_25p, bwd_75p = np.median(bwd_times), np.std(bwd_times), np.min(bwd_times), np.max(bwd_times), np.percentile(bwd_times, 25), np.percentile(bwd_times, 75)
    # print(f"{doc_lens}: FWD: {fwd_median:.2f}ms ± {fwd_std:.2f} ({fwd_min:.2f}ms~{fwd_25p:.2f}ms~{fwd_75p:.2f}ms~{fwd_max:.2f}ms), BWD: {bwd_median:.2f}ms ± {bwd_std:.2f} ({bwd_min:.2f}ms~{bwd_25p:.2f}ms~{bwd_75p:.2f}ms~{bwd_max:.2f}ms)")

    return (
        fwd_median, bwd_median
    )


def test_case():
    K = 1024
    num_qo_heads = 32
    num_kv_heads = 32
    head_dim = 128
    cp_size = 8

    first = K * 12
    second = K * 4
    a = test_doc_cp_attn([first, second], num_qo_heads, num_kv_heads, head_dim, cp_size)
    b = test_doc_cp_attn([first], num_qo_heads, num_kv_heads, head_dim, cp_size)
    c = test_doc_cp_attn([second], num_qo_heads, num_kv_heads, head_dim, cp_size)
    print(f"FWD error: {(a[0] - b[0] - c[0]):.2f}ms, BWD error: {(a[1] - b[1] - c[1]):.2f}ms")

K = 1024

# Cache for test_doc_cp_attn results
_test_cache = {}

def _get_cache_key(doc_lens, num_qo_heads, num_kv_heads, head_dim, cp_size):
    doc_lens = tuple(sorted(doc_lens))
    return (doc_lens, num_qo_heads, num_kv_heads, head_dim, cp_size)

def print_doc_cp_attn_table(
        head_dim = 128,
    num_qo_heads = 32,
    num_kv_heads = 32,
):
    output_file = f"per_doc_cp_attn.table.qo-{num_qo_heads}_kv-{num_kv_heads}_d-{head_dim}.txt"

    header = f"doc_cp_attn_table(head_dim={head_dim}, num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads})\n-----\ntp,cp,sf,fwd,bwd\n"
    print(header)
    with open(output_file, "a") as f:
        if f.tell() == 0:
            f.write(header)
        for tp in [1, 2, 4, 8]:
            for cp in [1, 2, 4, 8]:
                for sf in [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 48, 64, 96, 128]:
                    doc_lens = [int(sf * K)]
                    qo_heads = num_qo_heads // tp
                    kv_heads = num_kv_heads // tp
                    
                    # Check cache first
                    cache_key = _get_cache_key(doc_lens, qo_heads, kv_heads, head_dim, cp)
                    if cache_key in _test_cache:
                        a = _test_cache[cache_key]
                    else:
                        # Run in separate process to ensure CUDA memory cleanup
                        import multiprocessing as mp
                        ctx = mp.get_context('spawn')
                        with ctx.Pool(1) as pool:
                            a = pool.apply(test_doc_cp_attn, (doc_lens, qo_heads, kv_heads, head_dim, cp))
                        _test_cache[cache_key] = a
                        
                    line = f"{tp},{cp},{sf},{a[0]:.2f},{a[1]:.2f}\n"
                    print(line, end="")
                    f.write(line)
                    f.flush()

if __name__ == "__main__":
    print_doc_cp_attn_table(
        head_dim = 128,
        num_qo_heads = 32,
        num_kv_heads = 32,
    )

    # print_doc_cp_attn_table(
    #     head_dim = 128,
    #     num_qo_heads = 64,
    #     num_kv_heads = 64,
    # )

    # print_doc_cp_attn_table(
    #     head_dim = 128,
    #     num_qo_heads = 128,
    #     num_kv_heads = 128,
    # )

# if __name__ == "__main__":
#     test_case()