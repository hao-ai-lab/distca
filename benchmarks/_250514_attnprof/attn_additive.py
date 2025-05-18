import time
from flash_attn.flash_attn_interface import (
    flash_attn_func, 
    flash_attn_varlen_func,
)
import torch

def get_qkv(
    batch: list[int] = None,
    head_dim: int = 128,
    n_qo_head: int = 32,
    n_kv_head: int = 8,
    device: str = 'cuda',
):
    assert batch is not None, "batch is required"

    q = []
    k = []
    v = []
    cu_seqlens_q = [0,]
    cu_seqlens_k = [0,]
    max_seqlen_q = max(batch)
    max_seqlen_k = max(batch)
    for idx, i in enumerate(batch):
        q.append(torch.randn(i, n_qo_head, head_dim, dtype=torch.float16))
        k.append(torch.randn(i, n_kv_head, head_dim, dtype=torch.float16))
        v.append(torch.randn(i, n_kv_head, head_dim, dtype=torch.float16))
        cu_seqlens_q.append(sum(batch[:idx+1]))
        cu_seqlens_k.append(sum(batch[:idx+1]))
    

    q = torch.cat(q, dim=0).to(device)
    k = torch.cat(k, dim=0).to(device)
    v = torch.cat(v, dim=0).to(device)
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
    max_seqlen_q = torch.tensor(max_seqlen_q, dtype=torch.int32, device=device)
    max_seqlen_k = torch.tensor(max_seqlen_k, dtype=torch.int32, device=device)
    return q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k

llama8b_config = dict(
    head_dim=128,
    n_qo_head=32,
    n_kv_head=8,
)

K = 2 ** 10
batches = [
    [2 ** 10, 2 ** 10],
    [2 ** 10],

    [2 ** 10, 2 ** 5],
    [2 ** 10],
    [2 ** 5],


    [2 ** 10, 2 ** 10, 2 ** 10, 2 ** 4],
    [2 ** 10,  2 ** 10,],
    [2 ** 10, 2 ** 4],
]

for batch in batches:
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = get_qkv(
        **llama8b_config,
        batch=batch,
    )

    warmup = 10
    n_iter = 30
    for idx in range(n_iter + warmup):
        if idx == warmup:
            start = time.time()
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),  # -1 means infinite context window
            softcap=0.0, # 0.0 means deactivated
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
            block_table=None,
        )
    torch.cuda.synchronize()
    end = time.time()
    duration = (end - start) * 1000 * 1000 / n_iter
    print(f"{batch = }, {llama8b_config = }, {duration:.2f = } us")
    