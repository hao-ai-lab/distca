

def llama_latency(
    attn_tp: int, attn_cp: int,
    other_tp: int, other_cp: int,
    n_qo_head: int,
    n_kv_head: int,
    head_dim: int,
    mlp_in_dim: int,
    mlp_out_dim: int,
    batch: list[int],
):
    # attn latency:
    def get_attn_time(head_dim, n_qo_head, n_kv_head, batch):
        pass

    pass