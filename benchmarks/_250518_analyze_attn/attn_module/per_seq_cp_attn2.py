import numpy as np
import torch
from itertools import accumulate

import torch.distributed as dist

from torch.cuda.nvtx import range_push as nvtx_range_push
from torch.cuda.nvtx import range_pop  as nvtx_range_pop


from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func, 
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward
)

import time
from utils import compute_per_seq_metadate_combined


def per_seq_kv_shuffle(k_tensor, v_tensor, cp_size):
    chunk_k = k_tensor.chunk(2 * cp_size, dim=0)
    chunk_v = v_tensor.chunk(2 * cp_size, dim=0)

    new_k, new_v = [], []
    for r in range(cp_size):
        new_k.append(chunk_k[r])
        new_k.append(chunk_k[2 * cp_size - 1 - r])
        new_v.append(chunk_v[r])
        new_v.append(chunk_v[2 * cp_size - 1 - r])
    return torch.cat(new_k, dim=0), torch.cat(new_v, dim=0)

def per_seq_kv_unshuffle(k_tensor, v_tensor, cp_size):
     # Split into 2·cp_size equal chunks along dim-0
    k_chunks = k_tensor.chunk(2 * cp_size, dim=0)
    v_chunks = v_tensor.chunk(2 * cp_size, dim=0)

    orig_k, orig_v = [None] * (2 * cp_size), [None] * (2 * cp_size)

    # Let Li be the i-th chunk, p = cp_size, restore from [L0, L(2p-1), L1, L(2p-2), L2, … ]
    for r in range(cp_size):
        even_idx = 2 * r           # position of L r 
        odd_idx = 2 * r + 1       # position of L(2p-1-r)

        orig_k[r] = k_chunks[even_idx]
        orig_k[2 * cp_size - 1 - r] = k_chunks[odd_idx]

        orig_v[r] = v_chunks[even_idx]
        orig_v[2 * cp_size - 1 - r] = v_chunks[odd_idx]

    return torch.cat(orig_k, dim=0), torch.cat(orig_v, dim=0)

def gen_events(n):
    events = []
    for i in range(n):
        events.append(torch.cuda.Event(enable_timing=True))
    return events


def all_gather_into_tensor(global_tensor, local_tensor, world_size):
    # sleep a bit and return
    pass

def reduce_scatter_tensor(local_tensor, global_tensor, op, world_size):
    # sleep a bit and return
    pass
    

class PerSequenceCPAttention(torch.autograd.Function):
    """
    Attention with per‑sequence context parallelism.
    
    Forward path:
      1. allgather local K / V across pipeline ranks
      2. shuffle KV back to global order with `per_seq_kv_shuffle`
      3. for chunk id 0 and 1 : slice KV for this chunk
         and call `flash_attn_varlen_func`
      4. concatenate outputs and store everything for backward
    """

    @staticmethod
    def forward(
        ctx,
        local_q, local_k, local_v,
        cu_seqlens_q_list, cu_seqlens_kv_list,
        max_seqlen_q_list, max_seqlen_kv_list,
        k_offsets, 
        dropout_p,
        softmax_scale,
        attn_mask_type,
        cp_size,
        rank,
        cp_stream
    ):
        nvtx_range_push("PerSequenceCPAttention.fwd")
        assert attn_mask_type == "causal", "Only causal attention is supported"
        assert cp_stream is not None, "cp_stream must be provided"

        chunk_size = local_k.size(0) // 2

        # allgather kv, then shuffle back to global order
        with torch.cuda.stream(cp_stream):
            # gather k and v
            local_k = local_k.contiguous()
            local_v = local_v.contiguous()

            # all_gather into global k and v
            world = cp_size
            gathered_k = torch.randn(
                (world * local_k.size(0), *local_k.shape[1:]),
                dtype  = local_k.dtype,
                device = local_k.device,
            )
            gathered_v = torch.randn(
                (world * local_v.size(0), *local_v.shape[1:]),
                dtype  = local_v.dtype,
                device = local_v.device,
            )

            all_gather_into_tensor(gathered_k, local_k, world)
            all_gather_into_tensor(gathered_v, local_v, world)
            
            gathered_k = gathered_k.contiguous()
            gathered_v = gathered_v.contiguous()
        k_global, v_global = per_seq_kv_unshuffle(gathered_k, gathered_v, cp_size)

        # compute forward pass
        outputs, lses = [], []
        local_ks, local_vs = [], []
        q_chunks = local_q.chunk(2, dim=0)
        for chunk_id in range(2):  # 0, 1
            if chunk_id == 0:
                chunk_index = rank
            else:
                chunk_index = 2 * cp_size - 1 - rank
            k_start = int(k_offsets[chunk_id])
            k_end = (chunk_index+1) * chunk_size
            local_k_slice = k_global[k_start:k_end]
            local_v_slice = v_global[k_start:k_end]

            local_ks.append(local_k_slice)
            local_vs.append(local_v_slice)

            out, lse, _ = flash_attn_varlen_func(
                q=q_chunks[chunk_id],
                k=local_k_slice,
                v=local_v_slice,
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

        # concatenate chunk-results
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
        ctx.k_offsets      = k_offsets
        # ctx.k_lens         = k_lens
        ctx.q_chunk_sizes  = [c.shape[0] for c in q_chunks]
        ctx.dropout_p      = dropout_p
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
        Backward pass for PerSequenceCPAttention.
        """
        nvtx_range_push("PerSequenceCPAttention.bwd")

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

        k_offsets  = ctx.k_offsets
        (qlen_L, qlen_R) = ctx.q_chunk_sizes
        world_size = ctx.cp_size
        rank       = ctx.rank

        # split grad_out into two chunks
        dq_local = torch.zeros_like(local_q)
        dk_global = torch.zeros_like(gathered_k)
        dv_global = torch.zeros_like(gathered_v)
        
        # split grad-out
        d_out_L, d_out_R   = d_out_cat.split([qlen_L, qlen_R], dim=0)

        cp_size = ctx.cp_size
        chunk_size = d_out_L.size(0)

        # compute dq and dk/dv for each chunk
        # for i, (d_out, q_len, out, lse, cu_q, cu_k, max_q, max_k) in enumerate([
        #     (d_out_L, qlen_L, out_L, lse_L, cu_q_L, cu_k_L, maxq_L, maxk_L),
        #     (d_out_R, qlen_R, out_R, lse_R, cu_q_R, cu_k_R, maxq_R, maxk_R),
        # ]):
        for i, (d_out, q_len, out, lse, kv_k, kv_v, cu_q, cu_k, max_q, max_k) in enumerate([
            (d_out_L, qlen_L, out_L, lse_L, k_L, v_L, cu_q_L, cu_k_L, maxq_L, maxk_L),
            (d_out_R, qlen_R, out_R, lse_R, k_R, v_R, cu_q_R, cu_k_R, maxq_R, maxk_R),
        ]):
            if i == 0:
                chunk_index = rank
            else:
                chunk_index = 2 * cp_size - 1 - rank
            k_start = int(k_offsets[i])
            k_end = (chunk_index+1) * chunk_size

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
            dk_global[k_start : k_end] += dk_chunk
            dv_global[k_start : k_end] += dv_chunk
        
        # shuffle dk_global, dv_global
        dk_global, dv_global = per_seq_kv_shuffle(dk_global, dv_global, cp_size)

        # now do reduce_scatter for dk/dv
        dk_local = torch.empty_like(dq_local)
        dv_local = torch.empty_like(dq_local)
        with torch.cuda.stream(ctx.cp_stream):
            dk_global = dk_global.contiguous()
            dv_global = dv_global.contiguous()

            # print(f"{rank}, {dk_local.shape}, {dk_global.shape}, {dv_local.shape}, {dv_global.shape}")
            reduce_scatter_tensor(dk_local, dk_global, op=dist.ReduceOp.SUM, world_size=world_size)
            reduce_scatter_tensor(dv_local, dv_global, op=dist.ReduceOp.SUM, world_size=world_size)

        nvtx_range_pop()

        return (
            dq_local,        # grad w.r.t. local_q
            dk_local,        # grad w.r.t. local_k
            dv_local,        # grad w.r.t. local_v
            None,  # cu_seqlens_q_list
            None,  # cu_seqlens_kv_list
            None,  # max_seqlen_q_list
            None,  # max_seqlen_kv_list
            None,  # k_offsets
            None,  # dropout_p
            None,  # softmax_scale
            None,  # attn_mask_type
            None,  # cp_size
            None,  # rank
            None,  # cp_stream
        )


def test_seq_cp_attn(
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

    q_global = torch.randn(batch_size * context_length, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_global = torch.randn(batch_size * context_length, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_global = torch.randn(batch_size * context_length, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    d_out_global = torch.randn(batch_size * context_length, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16)

    
    n_warmup = 5
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
            local_q, local_k, local_v, 
            cu_seqlens_q, cu_seqlens_k, 
            max_seqlen_q, max_seqlen_k, 
            k_offsets, local_d_out,
        ) = compute_per_seq_metadate_combined(
            context_length, q_global, k_global, v_global, 
            doc_lens, cp_size, rank, 
            d_out=d_out_global
        )
        
        local_q.retain_grad() 
        local_k.retain_grad()
        local_v.retain_grad()

        fwd_st_event, fwd_ed_event = gen_events(2)
        bwd_st_event, bwd_ed_event = gen_events(2)
        
        torch.cuda.synchronize()
        fwd_st_event.record()
        out = PerSequenceCPAttention.apply(
            local_q, 
            local_k, 
            local_v,
            cu_seqlens_q, 
            cu_seqlens_k,
            max_seqlen_q, 
            max_seqlen_k,
            k_offsets, 
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
        fwd_median, bwd_median, (fwd_times, bwd_times)
    )

def test_case():
    K = 1024
    num_qo_heads = 32
    num_kv_heads = 32
    head_dim = 128
    cp_size = 8

    first = K * 12
    second = K * 4
    a = test_seq_cp_attn([first, second], num_qo_heads, num_kv_heads, head_dim, cp_size)
    b = test_seq_cp_attn([first], num_qo_heads, num_kv_heads, head_dim, cp_size)
    c = test_seq_cp_attn([second], num_qo_heads, num_kv_heads, head_dim, cp_size)
    print(f"FWD error: {(a[0] - b[0] - c[0]):.2f}ms, BWD error: {(a[1] - b[1] - c[1]):.2f}ms")