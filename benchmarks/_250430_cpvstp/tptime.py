"""
mamba activate sglang
"""
import json
from typing import List, Tuple
import gc
import torch
import flashinfer
from rich.progress import Progress

allreduce_byte2timeus = [
    (1024, 15.89),
    (2048, 16.82),
    (4096, 16.96),
    (8192, 17.04),
    (16384, 16.69),
    (32768, 18.02),
    (65536, 18.38),
    (131072, 18.43),
    (262144, 18.96),
    (524288, 23.44),
    (1048576, 31.29),
    (2097152, 52.66),
    (4194304, 78.68),
    (8388608, 115.8),
    (16777216, 190.1),
    (33554432, 301.2),
    (67108864, 557.5),
    (134217728, 1021.0),
    (268435456, 1981.5),
    (536870912, 3726.6),
    (1073741824, 7273.4),
]
allreduce_byte2timems = [(x, t / 1000) for x, t in allreduce_byte2timeus]

def get_allreduce_time(byte, tp_size):
    """Find the range where the byte falls into, and then use linear interpolation to get the time"""
    factor = tp_size / 4
    for i in range(len(allreduce_byte2timems)):
        if allreduce_byte2timems[i][0] >= byte:
            if i == 0:
                return allreduce_byte2timems[i][1] * factor
            else:
                prev_byte, prev_time = allreduce_byte2timems[i-1]
                curr_byte, curr_time = allreduce_byte2timems[i]
                result = prev_time + (byte - prev_byte) * (curr_time - prev_time) / (curr_byte - prev_byte)
                return result * factor
    return allreduce_byte2timems[-1][1] * factor

def get_tensor_size(x: torch.Tensor) -> int:
    return x.element_size() * x.nelement()

device: str = "cuda:0"
workspace_size_mb: int = 128
workspace_bytes = workspace_size_mb * 1024 * 1024
workspace_buffer = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")


def get_attention_tp_time(
    tp_size: int = 4,
    batch_size: int = 8,
    seq_len: int = 1024,
    num_layers: int = 1,
    head_dim: int = 128,
    num_qo_heads = 32,
    num_kv_heads = 8,
    page_size: int = 16,
    device: str = "cuda:0",
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], float, float]:
    """
    Runs one Shot of FlashInfer BatchPrefillWithPagedKVCacheWrapper and returns
    per-layer (LSE, logsumexp) outputs plus elapsed time in ms.

    Returns:
        outputs: List of length num_layers, each a tuple (lse, logsumexp) Tensor
        duration_ms: float, GPU elapsed time for the run
    """
    # derive heads from tp_size
    num_qo_heads = num_qo_heads // tp_size
    num_kv_heads = num_kv_heads  // tp_size

    # flatten queries
    nnz_qo = batch_size * seq_len
    qo_indptr = torch.arange(0, nnz_qo + 1, seq_len, dtype=torch.int32, device=device)

    # compute paging
    pages_per_sample     = (seq_len + page_size - 1) // page_size
    total_pages          = pages_per_sample * batch_size
    paged_kv_indptr      = torch.arange(0, total_pages + 1, pages_per_sample,
                                        dtype=torch.int32, device=device)
    paged_kv_indices     = torch.arange(total_pages, dtype=torch.int32, device=device)
    last_page_len        = seq_len - (pages_per_sample - 1) * page_size
    paged_kv_last_len    = torch.full((batch_size,),
                                      last_page_len,
                                      dtype=torch.int32,
                                      device=device)

    # fake data
    q_at_layer = torch.randn(
        num_layers, nnz_qo, num_qo_heads, head_dim
    ).half().to(device)
    kv_cache_at_layer = torch.randn(
        num_layers, total_pages, 2, page_size, num_kv_heads, head_dim
    ).half().to(device)

    # plan once
    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
    )

    # benchmark run_return_lse
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for layer in range(num_layers):
        lse, logsumexp = wrapper.run_return_lse(
            q_at_layer[layer],
            kv_cache_at_layer[layer]
        )
        outputs.append((lse, logsumexp))
    end.record()

    torch.cuda.synchronize()
    compute_duration_ms = start.elapsed_time(end)

    # get the tp communication overhead
    comm_size = get_tensor_size(lse) + get_tensor_size(logsumexp)
    comm_duration_ms = get_allreduce_time(comm_size, tp_size=tp_size)

    return outputs, compute_duration_ms, comm_duration_ms

configs = dict(
    llama7b=dict(
        head_dim=128,
        num_qo_heads=32,
        num_kv_heads=8,   
    ),
    llama70b=dict(
        head_dim=128,
        num_qo_heads=64,
        num_kv_heads=8,
    ),
)


results = []
batch_sizes = [1, 2, 4, 8, 16]
seq_len_factors = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

total_tasks = len(configs) * len(batch_sizes) * len(seq_len_factors)

import os

# Load existing results to check for completed runs
result_file_name = "tp_results.jsonl"
completed_runs = set()
if os.path.exists(result_file_name):
    with open(result_file_name, "r") as f:
        for line in f:
            item = json.loads(line)
            completed_runs.add((item['tp_size'], item['config_name'], item['batch_size'], item['seq_len_factor']))

with Progress() as progress:
    task = progress.add_task("[cyan]Processing...", total=total_tasks)

    tp_size = 4
    for config_name, config in configs.items():
        for batch_size in batch_sizes:
            for seq_len_factor in seq_len_factors:
                if (tp_size, config_name, batch_size, seq_len_factor) in completed_runs:
                    progress.console.print(f"Skipping completed run: {config_name}, {batch_size}, {seq_len_factor}")
                    progress.advance(task)
                    continue

                seq_len = 2 ** seq_len_factor
                result_one_run = get_attention_tp_time(
                    tp_size=tp_size,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_layers=1,
                    **config,
                )
                item = dict(
                    tp_size=tp_size,
                    config_name=config_name,
                    config=config,
                    batch_size=batch_size,
                    seq_len_factor=seq_len_factor,
                    seq_len=seq_len,
                    compute_duration_ms=result_one_run[1],
                    comm_duration_ms=result_one_run[2],
                )
                results.append(item)
                with open(result_file_name, "a") as f:
                    json.dump(item, f)
                    f.write("\n")
                progress.console.print(f"{batch_size = }, {seq_len = }, {result_one_run[1]}, {result_one_run[2]}")

                gc.collect()
                progress.advance(task)

            pass