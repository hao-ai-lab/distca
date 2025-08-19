# %%
# Implementing the varlen packing heuristic in WLBLLM paper
# Algorithm 1
# 
# Their algorithm is essentially do a "pre-reordering" of batch.

from collections import deque

K = 1024
N = 4 # microbatch per iteration
Lmax = 64 * K
outlier_threshold = 8 * K
GLOBAL_BATCH = [
    [    
        [56, 12, 14, 156, 1024 * 32],
        [56, 12, 14, 156, 1024 * 64],
        [56, 12, 14, 156, 1024 * 20],
        [56, 12, 14, 156, 1024 * 48],
    ],
]

def is_outlier(doc: int) -> bool:
    return doc > outlier_threshold

def get_workload(micro_batch: list[int]) -> int:
    return sum(i ** 2 for i in micro_batch)

def get_length(micro_batch: list[int]) -> int:
    return sum(micro_batch)


Q = deque()
B = []
remained_docs = []
for batch in GLOBAL_BATCH:
    doc_set = remained_docs
    remained_docs = []

    # Delay execution of outlier docs
    for doc in batch:
        if is_outlier(doc):
            Q.append(doc)
        else:
            doc_set.append(doc)
    pass
    
    # Pop outliers for the current batch.
    for q in Q:
        if q >= N:
            for i, doc in enumerate(doc_set):
                if i <= N: 
                    l = Q.popleft()
                    doc_set.append(l)
                pass
        pass
    
    # Sort the doc_set in descending order and do greedy packing.
    doc_set.sort(reverse=True) 

    # Start packing
    new_batch = []
    for _ in range(N):
        new_batch.append([])

    for doc in doc_set:
        # Get the microbatch with the least workload and length
        workloads = [get_workload(batch) for batch in new_batch]
        lengths = [get_length(batch) for batch in new_batch]
        
        min_workload_idx = workloads.index(min(workloads))
        min_length_idx = lengths.index(min(lengths))
        
        if lengths[min_length_idx] + doc <= Lmax:
            new_batch[min_length_idx].append(doc)
        else:
            if lengths[min_workload_idx] + doc <= Lmax:
                new_batch[min_workload_idx].append(doc)
            else:
                remained_docs.append(doc)
            pass
    
    B.append(new_batch)
    pass