from typing import Iterable, List, Optional
from d2.simulator.optimizers.samples import (
    sample_wlbllm_docs_upsample,
    sample_prolong_docs, 
    batch_documents,
)

ITERATION_ID = 0
iterated_samples = []
GLOBAL_BATCH: Optional[Iterable[List[int]]] = None

def setup_global_batch(
    total_seq_len, 
    up_sample_factor=2,
    elongate_factor=1,
    filter_threshold=64 * 1024,
    filter_ratio=0.90,
    # should_add_debug_cases=True,
    should_add_debug_cases=False,
    change_long_doc_ratio=0.0,
    sample_name='wlbllm',
):
    global GLOBAL_BATCH
    if GLOBAL_BATCH is not None:
        return

    assert elongate_factor > 0, f"elongate_factor: {elongate_factor} must be greater than 0"

    if sample_name == 'wlbllm':
        sample_func = sample_wlbllm_docs_upsample
    elif sample_name == 'prolong':
        sample_func = sample_prolong_docs
    else:
        raise ValueError(f"Invalid sample_name: {sample_name}")

    GLOBAL_BATCH = batch_documents(
        sample_func(
            size=10000,
            filter_threshold=filter_threshold,
            filter_ratio=filter_ratio,
            upsample_long_factor=up_sample_factor,
            elongate_factor=elongate_factor,
            change_long_doc_ratio=change_long_doc_ratio,
        ), max_ctx_length=total_seq_len
    )
    if should_add_debug_cases:
        GLOBAL_BATCH = list(GLOBAL_BATCH)
        manual_case = [
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            [total_seq_len], 
            # [total_seq_len // 8] * 8,
            # [total_seq_len], [total_seq_len // 8] * 8,
            # [total_seq_len], [total_seq_len // 8] * 8,
            # [total_seq_len], [total_seq_len // 8] * 8,
        ]
        GLOBAL_BATCH = manual_case * 100 + GLOBAL_BATCH
    GLOBAL_BATCH = iter(GLOBAL_BATCH)
    return


def get_next_batch(dp_size):
    global GLOBAL_BATCH
    global ITERATION_ID
    global iterated_samples
    
    batches = []
    if GLOBAL_BATCH is None:
        raise RuntimeError("GLOBAL_BATCH has not been initialized. Call setup_global_batch() first.")
        
    for _ in range(dp_size):    
        batches.append(next(GLOBAL_BATCH))
    ITERATION_ID += 1
    iterated_samples.append(batches)
    return batches


def balance_ping_pong(seq_lens: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    def batch_flops(batch):
        return sum(y ** 2 // 2 for y in batch)

    assert len(seq_lens) % 2 == 0, f"ping pong should have even number of batches, but got {len(seq_lens)} batches, seq_lens={seq_lens}"
    sorted_batches = sorted(seq_lens, key=batch_flops, reverse=True)
    ping, pong = [], []
    ping_flops, pong_flops = 0, 0
    avg_num_batches = len(seq_lens) // 2

    for batch in sorted_batches:
        if (ping_flops <= pong_flops and len(ping) < avg_num_batches) or len(pong) >= avg_num_batches:
            ping.append(batch)
            ping_flops += batch_flops(batch)
        else:
            pong.append(batch)
            pong_flops += batch_flops(batch)

    assert len(ping) == len(pong) == avg_num_batches, f"ping batches={ping}, pong batches={pong}"
    return ping, pong
