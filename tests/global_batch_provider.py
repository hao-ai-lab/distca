from typing import Iterable, List, Optional
from d2.simulator.optimizers.samples import (
    sample_wlbllm_docs_upsample,
    sample_prolong_docs, 
    batch_documents,
)
from collections import deque


def balance_ping_pong_with_mb(
    seq_lens: list[list[int]],
    mb: int, batch_size: int,
) -> list[list[int]]:
    
    def batch_flops(batch):
        return sum(y ** 2 // 2 for y in batch)

    assert len(seq_lens) % 2 == 0, f"ping pong should have even number of batches, but got {len(seq_lens)} batches, seq_lens={seq_lens}"
    assert len(seq_lens) == (mb * batch_size * 2), f"seq_lens should be divisible by {mb * batch_size * 2}, but got {len(seq_lens)}"
    sorted_batches = sorted(seq_lens, key=batch_flops, reverse=True)
    sorted_batches_deque = deque(sorted_batches)
    
    # Now sorted_batches_deque is sorted by flops >
    print(f"sorted_batches_deque: {sorted_batches_deque}")
    microbatch_results: 'list[list[list[int]]]' = []

    for _ in range(mb):
        batches = []
        for _ in range(batch_size * 2):
            batches.append(sorted_batches_deque.popleft())

        # Now do ping/pong balance with the batches
        ping, pong = [], []
        ping_flops, pong_flops = 0, 0
        avg_num_batches = len(batches) // 2

        for batch in batches:
            if (ping_flops <= pong_flops and len(ping) < avg_num_batches) or len(pong) >= avg_num_batches:
                ping.append(batch)
                ping_flops += batch_flops(batch)
            else:
                pong.append(batch)
                pong_flops += batch_flops(batch)
        
        # Feed the ping and pong back to the results
        microbatch_results.append(
            (ping, pong)
        )

    print(f"microbatch_results: {microbatch_results}")
    results: deque[list[int]] = deque()
    flip = False
    while microbatch_results:
        # Pop the smallest flop items.
        ping, pong = microbatch_results.pop()
        if flip:
            for b in ping:
                results.appendleft(b)
            for b in pong:
                results.appendleft(b)
        else:
            for b in ping:
                results.append(b)
            for b in pong:
                results.append(b)
        flip = not flip

    assert len(results) == len(seq_lens), f"results should have the same length as seq_lens, but got {len(results)} != {len(seq_lens)}"
    print(f"results: {results}")
    return list(results)


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
    balance_ping_pong_batch_size: None | dict[str, int] =None,
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
    GLOBAL_BATCH = list(GLOBAL_BATCH)
    
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
    
    if balance_ping_pong_batch_size:
        # Construct a new global batch array such that
        # we take `balance_ping_pong_batch_size x 2` every time, 
        # and then balance the ping/pong.
        new_global_batch = []
        mb = balance_ping_pong_batch_size['mb']
        batch_size = balance_ping_pong_batch_size['batch_size']
        for star_idx in range(0, len(GLOBAL_BATCH), mb * batch_size * 2):
            end_idx = star_idx + mb * batch_size * 2
            if end_idx > len(GLOBAL_BATCH):
                new_global_batch.extend(GLOBAL_BATCH[star_idx:])
                continue
            else:
                new_data_chunk = balance_ping_pong_with_mb(
                    GLOBAL_BATCH[star_idx:end_idx],
                    mb=mb,
                    batch_size=batch_size,
                )
                new_global_batch.extend(new_data_chunk)
        GLOBAL_BATCH = new_global_batch
    
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