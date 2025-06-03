"""
Profiling the attention time of the attention server.
Use Modal to deploy the attention server.
"""

import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm-flash-attn")
    .pip_install("transformers==4.51.3")
)


image = vllm_image

K = 1024

app = modal.App("profiling-attn", image=image)

@app.function(gpu="H100:1")
def attn_flash_attn(
    batch,
    num_qo_heads = 64,
    num_kv_heads = 4,
    head_dim = 128,
    cp = 1,
    tp = 1,
):
    import time

    import torch
    device = torch.device("cuda")
    # print(torch)

    # # Test basic torch matmul
    # a = torch.randn(1024, 1024, device=device)
    # b = torch.randn(1024, 1024, device=device)
    # c = torch.matmul(a, b)
    # torch.cuda.synchronize()
    # print("Successfully finished matmul")

    import vllm_flash_attn
    # print(vllm_flash_attn)
    # print(dir(vllm_flash_attn))

    import vllm_flash_attn.flash_attn_interface
    # print(dir(vllm_flash_attn.flash_attn_interface))

    from vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    # Qwen3 253B activate attention data
    num_qo_heads = num_qo_heads // tp
    num_kv_heads = max(num_kv_heads // tp, 1)

    kv_lens = [(1/2 + 1/(2 * cp)) * i for i in batch]
    
    batch = [int(i // cp) for i in batch]
    kv_lens = [int(i) for i in kv_lens]

    total_tokens = sum(batch)
    total_kv_tokens = sum(kv_lens)

    q = torch.randn(total_tokens, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(total_kv_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(total_kv_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    max_seqlen_q = max(batch)
    max_seqlen_k = max(kv_lens)

    cu_seqlens_q = [0,]
    cu_seqlens_k = [0,]
    for idx, _ in enumerate(batch):
        cu_seqlens_q.append(sum(batch[:idx+1]))
        cu_seqlens_k.append(sum(kv_lens[:idx+1]))
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
    max_seqlen_q = torch.tensor(max_seqlen_q, dtype=torch.int32, device=device)
    max_seqlen_k = torch.tensor(max_seqlen_k, dtype=torch.int32, device=device)
    

    def test_flash_attn():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = flash_attn_varlen_func(
            q, k, v, 
            cu_seqlens_q, cu_seqlens_k, 
            max_seqlen_q, max_seqlen_k,
            dropout_p=0.0, causal=True,
        )
        end_event.record()
        torch.cuda.synchronize()
        duration = start_event.elapsed_time(end_event)
        return duration
    
    # warmup
    for _ in range(5):
        test_flash_attn()
    
    # benchmark
    num_iters = 10
    durations = []
    for _ in range(num_iters):
        duration = test_flash_attn()
        durations.append(duration)

    avg_duration = sum(durations) / len(durations)

    # print(f"TP: {tp}, CP: {cp}, Result: {avg_duration:.2f} ms")

    return avg_duration


@app.function(gpu="H100:1")
def mlp_gemm(
    batch,
    num_qo_heads = 64,
    num_kv_heads = 4,
    head_dim = 128,
    mlp_dim = 4096,
    tp = 1,
    cp = 1,
):
    import torch
    import time

    num_qo_heads = num_qo_heads // tp
    num_kv_heads = max(num_kv_heads // tp, 1)

    total_tokens = sum(batch)
    total_kv_tokens = sum(kv_lens)

    device = torch.device("cuda")
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device)
    for _ in range(10):
        _ = torch.matmul(a, b)
    
    start = time.time()
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()

    duration = end - start
    duration *= 1000
    return duration


@app.function(gpu="B200:1")
def mlp_benchmark():
    import time
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    model_id = "Qwen/Qwen3-235B-A22B"
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)  # 94 layers originally  [oai_citation:0‡Hugging Face](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json?utm_source=chatgpt.com)
    cfg.num_hidden_layers = 1          # ← the only change strictly required
    torch.set_default_device("cuda")                # PyTorch 2.1+
    torch.manual_seed(42)              # reproducible randomness


    with torch.no_grad():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    profile = {}  # name → dict(start, end, params)

    def want_hook(name, module):
        # leaf = no children; skip attention classes (FlashAttention2, QwenAttention, …)
        is_leaf = len(list(module.children())) == 0
        is_attn = "attn" in module.__class__.__name__.lower() or \
                "attention" in module.__class__.__name__.lower()
        return is_leaf 

    for name, m in model.named_modules():
        if not want_hook(name, m):
            continue

        # forward-pre & forward hooks share the closure variable 'name'
        def pre_hook(mod, inp, name=name):
            profile[name] = {"start": time.perf_counter(),
                            "params": sum(p.numel() for p in mod.parameters())}

        def post_hook(mod, inp, out, name=name):
            torch.cuda.synchronize()
            profile[name]["end"] = time.perf_counter()

        m.register_forward_pre_hook(pre_hook, prepend=True)
        m.register_forward_hook(post_hook)


    def run_model_profile(ctx_len):
        inp = tok(" a" * ctx_len, return_tensors="pt").to(model.device)
        ctx_len = len(inp.input_ids[0])

        with torch.no_grad():
            model(**inp)                        # warm-up (CUDA kernels/JIT)
            torch.cuda.synchronize()

            start = time.perf_counter()
            model(**inp)
            torch.cuda.synchronize()
            total = time.perf_counter() - start
            total *= 1e3

        rows = []
        for name, rec in profile.items():
            dur_ms = (rec["end"] - rec["start"]) * 1e3
            rows.append((dur_ms, name, rec["params"]))

        print(f"{'module':45}  latency  params")
        print("-"*70)

        total_layer0_time = 0
        expert_time = 0
        expert_params = 0
        for dur, name, nparam in rows:
            if "layers.0" not in name:
                continue
                
            if "experts" in name:
                expert_time += dur
                expert_params += nparam
            else:
                print(f"{name:45}  {dur:7.3f} ms  {nparam/1e6:7.2f} M")
            total_layer0_time += dur

        print(f"{'model.layers.0.*expert*':45}  {expert_time:7.3f} ms  {expert_params/1e6:7.2f} M")
        print("-"*70)
        print(f"model: {model_id}")
        device_name = torch.cuda.get_device_name(model.device)
        print(f"device: {device_name}")
        print(f"ctx_len: {ctx_len}")
        print(f"total layer 0 time: {total_layer0_time:.3f} ms")
        print(f"full forward time: {total:.3f} ms")
        
        return total_layer0_time, total


    K = 1024
    ctx_lengths = [K * i for i in [1, 2, 4, 8, 16, 32, 48, 64, 96]]
    for ctx_len in ctx_lengths:
        print(f"\nMLP testing context length: {ctx_len}")
        layer0_time, total_time = run_model_profile(ctx_len)
        print(f"layer 0 time: {layer0_time:.3f} ms")
        print(f"total time: {total_time:.3f} ms")
        torch.cuda.empty_cache()

    pass



# @app.local_entrypoint()
# def main():
#     model_config = dict(
#         num_qo_heads = 64,
#         num_kv_heads = 4,
#         head_dim = 128,
#     )
#     batch = " [i * K for i in [16] + [2] * 8 ] "
#     print(f"Batch: {batch}")
#     print("-" * 10)
#     for tp in [1, 2, 4, 8]:
#         for cp in [1, 2, 4, 8]:
#             avg_duration = attn_flash_attn.remote(
#                 batch = eval(batch),
#                 cp = cp,
#                 tp = tp,
#                 **model_config,
#             )
#             print(f"TP: {tp}, CP: {cp}, Result: {avg_duration:.2f} ms")
#     print("-" * 10)

@app.local_entrypoint()
def main():
    mlp_benchmark.remote()
    pass