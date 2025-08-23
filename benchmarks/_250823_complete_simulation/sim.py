"""
Simulate pipeline parallel for WLBLLM (no defer)

# Given a distribution, we take DP*PP batches every time.
# We first perform a balancing algorithm on this particular batch,
# and then calcualte the time for this particular stage

"""

# %%
K = 1024
megatron_batch_size = 2 # because we do pingpong.
base_seq_len = K * 64
attn_base_time = 12.5020
# linear_base_time = (13.5 + 8.5) # mlp + qkvo
# linear_base_time = (13.5 + 8.5)  # mlp + qkvo
mlp_base_time = 13.5  # assume expert parallel
qkvo_base_time = 8.5 
linear_base_time = (mlp_base_time + qkvo_base_time)  # mlp + qkvo

# %%

def get_attn_time(batch):
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += attn_base_time * (ratio ** 2)
    return total_time

def get_network_time(batch, cp_size):
    # Get the all-gather network time.
    # 128k batch=2*8=16 then we take 20ms to do AG.
    pass

def flatten(batch):
    return [item for sublist in batch for item in sublist]

def take(it, n):
    results = []
    for _ in range(n):
        results.append(next(it))
    return results


# %%
from d2.simulator.optimizers.samples import sample_wlbllm_docs_upsample, batch_documents

# group = tp group. CP will scale the seq_len_per_group up.
seq_len_per_group = K * 256

N = 100
up_sample_factor = 16
elongate_factor = 1
filter_threshold = 64 * K
filter_ratio = 0.10
dp_size = 4
cp_size = 2
pp_size = 1
take_size = dp_size * pp_size * 2

GLOBAL_BATCH = batch_documents(
    sample_wlbllm_docs_upsample(
        size=20000,
        upsample_long_factor=up_sample_factor,
        filter_threshold=filter_threshold,
        filter_ratio=filter_ratio,
        elongate_factor=elongate_factor,
    ), max_ctx_length=seq_len_per_group * cp_size
)

# %%

# Calculate WLBLLM Time
batch = take(GLOBAL_BATCH, take_size)

# 1. Rebalance the batch across different devices