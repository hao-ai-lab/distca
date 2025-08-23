# %%
from d2.simulator.optimizers.samples import sample_wlbllm_docs_upsample, batch_documents

K = 1024
total_seq_len = 128 * K

GLOBAL_BATCH = batch_documents(
    sample_wlbllm_docs_upsample(
        size=10000,
        upsample_long_factor=2,
        filter_threshold=10000,
        filter_ratio=0.09,
    ), max_ctx_length=total_seq_len
)

# %%
sorted(next(GLOBAL_BATCH), reverse=True)