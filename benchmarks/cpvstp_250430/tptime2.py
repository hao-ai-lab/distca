"""
mamba activate sglang
"""
import os
import json
from typing import List, Tuple
import gc
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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

def run_tp_worker(
    rank: int,
    tp_size: int,
    seq_len: int,
    batch_size: int,
    config: dict,
    return_dict: dict,
):
    # Initialize NCCL
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=tp_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Warm up comm
    xs = [torch.zeros(1, device=device) for _ in range(tp_size)]
    dist.all_gather(xs, xs[0])

    # Setup workspace and wrapper
    workspace_bytes = 128 * 1024 * 1024
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

    # Derive heads from tp_size
    head_dim = config["head_dim"]
    num_qo_heads = config["num_qo_heads"] // tp_size
    num_kv_heads = config["num_kv_heads"] // tp_size
    page_size = config.get("page_size", 16)
    num_layers = config.get("num_layers", 1)

    # Setup tensors
    nnz_qo = batch_size * seq_len
    qo_indptr = torch.arange(0, nnz_qo + 1, seq_len, dtype=torch.int32, device=device)

    # Compute paging
    pages_per_sample = (seq_len + page_size - 1) // page_size
    total_pages = pages_per_sample * batch_size
    paged_kv_indptr = torch.arange(0, total_pages + 1, pages_per_sample, dtype=torch.int32, device=device)
    paged_kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    last_page_len = seq_len - (pages_per_sample - 1) * page_size
    paged_kv_last_len = torch.full((batch_size,), last_page_len, dtype=torch.int32, device=device)

    # Generate fake data
    q_at_layer = torch.randn(
        num_layers, nnz_qo, num_qo_heads, head_dim,
        dtype=torch.float16, device=device
    )
    kv_cache_at_layer = torch.randn(
        num_layers, total_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=torch.float16, device=device
    )

    # Plan once
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

    # Warm up forward
    lse, logsumexp = wrapper.forward_return_lse(
        q_at_layer[0],
        kv_cache_at_layer[0],
    )

    # Time local compute
    torch.cuda.synchronize(device)
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for layer in range(num_layers):
        lse, logsumexp = wrapper.forward_return_lse(
            q_at_layer[layer],
            kv_cache_at_layer[layer],
        )
    t1.record()
    torch.cuda.synchronize(device)
    compute_ms = t0.elapsed_time(t1)

    # Time NCCL all-gather of both outputs
    torch.cuda.synchronize(device)
    c0 = torch.cuda.Event(enable_timing=True)
    c1 = torch.cuda.Event(enable_timing=True)
    c0.record()
    # gather lse
    gathered_lse = [torch.empty_like(lse) for _ in range(tp_size)]
    dist.all_gather(gathered_lse, lse)
    # gather logsumexp
    gathered_lse2 = [torch.empty_like(logsumexp) for _ in range(tp_size)]
    dist.all_gather(gathered_lse2, logsumexp)
    c1.record()
    torch.cuda.synchronize(device)
    comm_ms = c0.elapsed_time(c1)

    # Record results and cleanup
    return_dict[f"compute_{rank}"] = compute_ms
    return_dict[f"comm_{rank}"] = comm_ms
    dist.destroy_process_group()

def get_attention_tp_time(
    tp_size: int = 4,
    batch_size: int = 8,
    seq_len: int = 1024,
    **config,  # must include head_dim, num_qo_heads, num_kv_heads, optionally page_size, num_layers
) -> Tuple[List, float, float]:
    """
    Returns:
      [],           # dummy placeholder for per-layer outputs
      max compute latency across ranks (ms),
      max comm latency across ranks    (ms)
    """
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        run_tp_worker,
        args=(tp_size, seq_len, batch_size, config, return_dict),
        nprocs=tp_size,
        join=True,
    )

    compute_times = [return_dict[f"compute_{r}"] for r in range(tp_size)]
    comm_times = [return_dict[f"comm_{r}"] for r in range(tp_size)]

    return dict(compute_times=compute_times, comm_times=comm_times), max(compute_times), max(comm_times)

def main():
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

    result_file_name = "tp_results2.jsonl"
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
                        allcomputeandcomm=result_one_run[0],
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

if __name__ == "__main__":
    main()
