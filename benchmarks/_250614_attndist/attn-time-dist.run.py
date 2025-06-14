# %%
import modal
from d2.simulator.optimizers.samples import (
    sample_multimodal_gaussian, 
    sample_random_docs,
    sample_wlbllm_docs,
    batch_documents,
)
from d2.timemodule import get_attn_time, get_allreduce_time, get_allgather_time

K = 1024

# name = "dist_wlbllm"
# name = "dist_random"

func = modal.Function.from_name("attn-prof", "attn_flash_attn")


name = "dist_multimodal"
seed = 42
size = 1000
max_ctx_length = 64 * K

# %%


if name == "dist_wlbllm":
    docs = sample_wlbllm_docs(
        size = size, seed = seed,
    )

elif name == "dist_multimodal":
    # means = [1*K,  8*K, 16*K]
    means = [8*K, 64*K]
    sigmas = [0.1*K, 0.2*K]
    weights = [0.35, 0.65]

    docs = sample_multimodal_gaussian(
        means=means, 
        sigmas=sigmas, 
        weights=weights, 
        size=size,
        seed=seed,
    )

elif name == "dist_random":
    docs = sample_random_docs(
        max_ctx_length = max_ctx_length,
        size = size, seed=seed,
    )

batches = batch_documents(
    docs, max_ctx_length = max_ctx_length,
)
batches = list(batches)


print("Running benchmark: Attention Time Distribution")
print(f"- Dataset: {name}")
print(f"- Size: {size}")
print(f"- Max Context Length: {max_ctx_length}")
print(f"- Number of Batches: {len(batches)}")

with open(f"attn-{name}.psv", "w") as f:
    f.write("tp|cp|total_len|bs|real_attn_time_ms|sim_attn_time_ms|allreduce_time_ms|allgather_time_ms|batch\n")
    for batch in batches:
        for cp in [8, 4, 2, 1]:
            for tp in [8, 4, 2, 1]:

                real_time = func.remote(
                    batch = batch,
                    num_qo_heads = 64,
                    num_kv_heads = 4,
                    head_dim = 128,
                    cp = cp,
                    tp = tp,
                )

                sim_time = 0
                for x in batch:
                    sim_time += get_attn_time(
                        x = x, tp = tp, cp = cp,
                        hqo = 64, hkv = 4, d = 128,
                    )

                total_len = sum(batch)
                
                allreduce_time = get_allreduce_time(
                    x = total_len, tp = tp,
                )
                allgather_time = get_allgather_time(
                    x = total_len, cp = cp,
                )

                bs = len(batch)
                f.write(f"{tp}|{cp}|{total_len}|{bs}|{real_time:.2f}|{sim_time:.2f}|{allreduce_time:.2f}|{allgather_time:.2f}|{batch}\n")
                f.flush()
                
