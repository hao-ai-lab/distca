from wlbllm.attn_module import (
    flash_attn_varlen_func,
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)

import torch
import time
import argparse

# ---- CLI ----
K = 1024
parser = argparse.ArgumentParser(description='Context Parallel Attention Benchmark')
parser.add_argument('--nhead', type=int, default=1)
parser.add_argument('--head_dim', type=int, default=128)
parser.add_argument('--total_tokens', type=int, default=128 * K)
parser.add_argument('--single_seq_len', type=int, default=2 * K)
parser.add_argument('--longest_seq_len', type=int, default=2 * K)
parser.add_argument('--cp_degree', type=int, default=4)
parser.add_argument('--data_dist', type=str, default='homo')  # ['homo', '1lns']
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()

torch.cuda.set_device("cuda:0")
device = torch.device("cuda:0")

nhead = args.nhead
head_dim = args.head_dim
total_tokens = args.total_tokens
single_seq_len = args.single_seq_len
cp_degree = args.cp_degree
data_dist = args.data_dist
longest_seq_len = args.longest_seq_len
batch_size = args.batch_size

# ---- Build per-sequence lengths for each "chunk" (2 chunks) ----
if data_dist == "homo":
    num_seq = total_tokens // single_seq_len
    local_seq_len = single_seq_len // cp_degree // 2  # your original intent
    # For each chunk, build a list of per-seq lengths (not cumulative yet)
    seqlens_q_list = [
        [local_seq_len] * num_seq,
        [local_seq_len] * num_seq,
    ]
    # Model: chunk 0 attends to local KV; chunk 1 attends to full KV
    seqlens_k_list = [
        [local_seq_len] * num_seq,
        [single_seq_len] * num_seq,
    ]
    seqlens_q_list[0] *= batch_size
    seqlens_q_list[1] *= batch_size
    seqlens_k_list[0] *= batch_size
    seqlens_k_list[1] *= batch_size

elif data_dist == "1lns":
    local_long_seq_len = longest_seq_len // cp_degree // 2
    short_seq_len = single_seq_len
    local_short_seq_len = short_seq_len // cp_degree // 2
    remaining_length = (total_tokens - longest_seq_len) // cp_degree // 2
    num_seq = (total_tokens - longest_seq_len) // short_seq_len
    print("- local_long_seq_len", local_long_seq_len)
    print("- remaining_length", remaining_length)
    print("- short_seq_len", short_seq_len)
    print("- num_seq", num_seq)
    print("- local_short_seq_len", local_short_seq_len)

    seqlens_q_list = [
        [local_long_seq_len] + [local_short_seq_len] * num_seq,
        [local_long_seq_len] + [local_short_seq_len] * num_seq,
    ]
    seqlens_q_list[0] *= batch_size
    seqlens_q_list[1] *= batch_size

    seqlens_k_list = [
        [local_long_seq_len] + [local_short_seq_len] * num_seq,
        [longest_seq_len] + [short_seq_len] * num_seq,
    ]
    seqlens_k_list[0] *= batch_size
    seqlens_k_list[1] *= batch_size

else:
    raise ValueError(f"Invalid data distribution: {data_dist}")


# ---- Convert per-seq lengths -> cumulative offsets (prefix sums starting at 0) ----
def to_cu(lens):
    cu = [0]
    s = 0
    for x in lens:
        s += int(x)
        cu.append(s)
    return cu

cu_seqlens_q_list = [to_cu(l) for l in seqlens_q_list]
cu_seqlens_k_list = [to_cu(l) for l in seqlens_k_list]

max_seqlen_q_list = [max(l) if len(l) > 0 else 0 for l in seqlens_q_list]
max_seqlen_k_list = [max(l) if len(l) > 0 else 0 for l in seqlens_k_list]

print("cu_seqlens_q_list[0]", cu_seqlens_q_list[0])
print("cu_seqlens_k_list[0]", cu_seqlens_k_list[0])
print("cu_seqlens_q_list[1]", cu_seqlens_q_list[1])
print("cu_seqlens_k_list[1]", cu_seqlens_k_list[1])

print("max_seqlen_q_list[0]", max_seqlen_q_list[0])
print("max_seqlen_k_list[0]", max_seqlen_k_list[0])
print("max_seqlen_q_list[1]", max_seqlen_q_list[1])
print("max_seqlen_k_list[1]", max_seqlen_k_list[1])

# ---- Allocate Q/K/V with the correct TOTAL lengths for EACH chunk ----
q_chunks = []
k_chunks = []
v_chunks = []
cu_seqlens_qs = []
cu_seqlens_ks = []

for chunk_id in range(2):
    total_q = cu_seqlens_q_list[chunk_id][-1]
    total_k = cu_seqlens_k_list[chunk_id][-1]
    assert total_q >= 0 and total_k >= 0

    q_chunks.append(torch.randn(total_q, nhead, head_dim, device=device, dtype=torch.bfloat16))
    k_chunks.append(torch.randn(total_k, nhead, head_dim, device=device, dtype=torch.bfloat16))
    v_chunks.append(torch.randn(total_k, nhead, head_dim, device=device, dtype=torch.bfloat16))

    cu_seqlens_qs.append(torch.tensor(cu_seqlens_q_list[chunk_id], device=device, dtype=torch.int32))
    cu_seqlens_ks.append(torch.tensor(cu_seqlens_k_list[chunk_id], device=device, dtype=torch.int32))

    # Sanity checks: API expects these exact totals
    assert q_chunks[chunk_id].shape[0] == cu_seqlens_qs[chunk_id][-1].item()
    assert k_chunks[chunk_id].shape[0] == cu_seqlens_ks[chunk_id][-1].item()

# ---- Warmup ----
torch.cuda.synchronize()
for _ in range(3):
    for chunk_id in range(2):
        out = flash_attn_varlen_func(
            q=q_chunks[chunk_id],
            k=k_chunks[chunk_id],
            v=v_chunks[chunk_id],
            cu_seqlens_q=cu_seqlens_qs[chunk_id],
            cu_seqlens_k=cu_seqlens_ks[chunk_id],
            max_seqlen_q=max_seqlen_q_list[chunk_id],
            max_seqlen_k=max_seqlen_k_list[chunk_id],
            dropout_p=0.0,
            softmax_scale=1.0,
            causal=True,
            return_attn_probs=False,   # you weren't using attn probs; this saves memory/time
            deterministic=False,
        )
torch.cuda.synchronize()

# ---- Timed run ----
N = 5
start_time = time.time()
for _ in range(N):
    for chunk_id in range(2):
        out = flash_attn_varlen_func(
            q=q_chunks[chunk_id],
            k=k_chunks[chunk_id],
            v=v_chunks[chunk_id],
            cu_seqlens_q=cu_seqlens_qs[chunk_id],
            cu_seqlens_k=cu_seqlens_ks[chunk_id],
            max_seqlen_q=max_seqlen_q_list[chunk_id],
            max_seqlen_k=max_seqlen_k_list[chunk_id],
            dropout_p=0.0,
            softmax_scale=1.0,
            causal=True,
            return_attn_probs=False,
            deterministic=False,
        )
torch.cuda.synchronize()
duration_ms = (time.time() - start_time) * 1000.0 / N

print(f"total_tokens = {total_tokens}, single_seq_len = {single_seq_len}, cp_degree = {cp_degree}, batch_size = {batch_size}, time = {duration_ms:.2f} ms")