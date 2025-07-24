import torch

from flash_attn import flash_attn_varlen_func

num_tokens = 1024
seqlens = [500, 524]
num_heads = 16
hidden_size = 128

seq_shards = [250, 250, 124, 400]

q = torch.randn(num_tokens, num_heads, hidden_size).cuda().to(torch.bfloat16)
k = torch.randn(num_tokens, num_heads, hidden_size).cuda().to(torch.bfloat16)
v = torch.randn(num_tokens, num_heads, hidden_size).cuda().to(torch.bfloat16)

q0 = torch.concat((q[:250], q[500:624]), dim=0)
q1 = torch.concat((q[250:500], q[624:]), dim=0)

k0 = torch.concat((k[:250], k[500:624]), dim=0)
k1 = k.clone()

v0 = torch.concat((v[:250], v[500:624]), dim=0)
v1 = v.clone()

seq_lens_q = torch.tensor(seqlens, dtype=torch.int32).cuda()
seq_lens_k = seq_lens_q.clone()
seq_lens_q0 = torch.tensor([250, 124], dtype=torch.int32).cuda()
seq_lens_k0 = seq_lens_q0.clone()
seq_lens_q1 = torch.tensor([250, 400], dtype=torch.int32).cuda()
seq_lens_k1 = seq_lens_k.clone()

def cumsum_fn(x):
    return torch.concat(
        (torch.zeros(1, dtype=torch.int32, device=x.device), x.cumsum(dim=0)),
        dim=0,
    ).to(torch.int32)

cu_seqlens_q = cumsum_fn(seq_lens_q)
cu_seqlens_k = cumsum_fn(seq_lens_k)
cu_seqlens_q0 = cumsum_fn(seq_lens_q0)
cu_seqlens_k0 = cumsum_fn(seq_lens_k0)
cu_seqlens_q1 = cumsum_fn(seq_lens_q1)
cu_seqlens_k1 = cumsum_fn(seq_lens_k1)

max_seqlens_q = seq_lens_q.max()
max_seqlens_k = seq_lens_k.max()
max_seqlens_q0 = seq_lens_q0.max()
max_seqlens_k0 = seq_lens_k0.max()
max_seqlens_q1 = seq_lens_q1.max()
max_seqlens_k1 = seq_lens_k1.max()

ans = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k, dropout_p=0, softmax_scale=1.0, causal=True)
res_0 = flash_attn_varlen_func(q0, k0, v0, cu_seqlens_q0, cu_seqlens_k0, max_seqlens_q0, max_seqlens_k0, dropout_p=0, softmax_scale=1.0, causal=True)
res_1 = flash_attn_varlen_func(q1, k1, v1, cu_seqlens_q1, cu_seqlens_k1, max_seqlens_q1, max_seqlens_k1, dropout_p=0, softmax_scale=1.0, causal=True)

res = torch.concat((res_0[:250], res_1[:250], res_0[250:], res_1[250:]), dim=0)

torch.testing.assert_close(res, ans, atol=1e-4, rtol=1e-4)
print("flash_attn_varlen_func test passed")
print(max_seqlens_q1, max_seqlens_k1)
