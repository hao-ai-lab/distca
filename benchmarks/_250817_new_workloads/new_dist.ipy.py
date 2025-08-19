# %%
from d2.simulator.optimizers.samples import sample_wlbllm_docs_upsample, batch_documents

K = 1024

def calculate_ratio(
    batch_generator, limit = 100,
    small_large_threshold = 8 * K,
):
    short_doc_tokens_cnt_list = []
    long_doc_tokens_cnt_list = [] 
    for i, batch in enumerate(batch_generator):
        if limit is not None and i > limit: break
        short_doc_tokens_cnts = 0
        long_doc_tokens_cnts = 0

        for doc in batch:
            if doc < small_large_threshold:
                short_doc_tokens_cnts += doc
            else:
                long_doc_tokens_cnts += doc
        ratio = long_doc_tokens_cnts / (long_doc_tokens_cnts + short_doc_tokens_cnts)
        
        # print(f"Batch {i}: short: {short_doc_tokens_cnts}, long: {long_doc_tokens_cnts}, ratio: {ratio:.2%}")
        short_doc_tokens_cnt_list.append(short_doc_tokens_cnts)
        long_doc_tokens_cnt_list.append(long_doc_tokens_cnts)

    print(f"Average short doc tokens cnt: {sum(short_doc_tokens_cnt_list) / len(short_doc_tokens_cnt_list):.2f}")
    print(f"Average long doc tokens cnt: {sum(long_doc_tokens_cnt_list) / len(long_doc_tokens_cnt_list):.2f}")

    # ratio of avg long doc token cnt / avg long doc token cnt + avg short doc token cnt
    ratio = sum(long_doc_tokens_cnt_list) / (sum(long_doc_tokens_cnt_list) + sum(short_doc_tokens_cnt_list))
    print(f"Ratio of long / (long + short) token counts: {ratio:.2%}")
    return ratio
    
# %% 
# 
# 60.32%
# 

GLOBAL_BATCH = batch_documents(
    sample_wlbllm_docs_upsample(
        size=10000,
        upsample_long_factor = 4,
        filter_threshold = 64 * K,
        filter_ratio = 0.50,
        elongate_factor = 1,
    ), max_ctx_length=(K * 128),
)
ratio = calculate_ratio(GLOBAL_BATCH, limit=100, small_large_threshold=8 * K)
print(f"Ratio: {ratio:.2%}")

# %%


# %%

GLOBAL_BATCH = batch_documents(
    sample_wlbllm_docs_upsample(
        size=10000,
        upsample_long_factor = 8,
        filter_threshold = 64 * K,
        filter_ratio = 0.50,
        elongate_factor = 1,
    ), max_ctx_length=(K * 128),
)
ratio = calculate_ratio(GLOBAL_BATCH, limit=None, small_large_threshold=8 * K)
print(f"Ratio: {ratio:.2%}")

"""
size=10000
upsample = 1,2,4,8 
filter_threshold = 64 * K
filter_ratio = 0.50
elongate_factor = 1
max_ctx_length=(K * 128)

=-> ratio: 30, 45, 60, 73%
"""

# %%

# Using the original distribution, but upsample long docs by 6x will get to 60% long.

GLOBAL_BATCH = batch_documents(
    sample_wlbllm_docs_upsample(
        size=10000,
        upsample_long_factor = 6,
        filter_threshold = 16 * K,
        filter_ratio = 1.00,
        elongate_factor = 1,
    ), max_ctx_length=(K * 128),
)
ratio = calculate_ratio(GLOBAL_BATCH, limit=None, small_large_threshold=8 * K)
print(f"Ratio: {ratio:.2%}")


# %%
GLOBAL_BATCH = batch_documents(
    sample_wlbllm_docs_upsample(
        size=10000,
        upsample_long_factor = 6,
        filter_threshold = 16 * K,
        filter_ratio = 1.00,
        elongate_factor = 1,
    ), max_ctx_length=(K * 128),
)
ratio = calculate_ratio(GLOBAL_BATCH, limit=None, small_large_threshold=6 * K)
print(f"Ratio: {ratio:.2%}")

"""
small_large_threshold = 2, 4, 6, 8 * K

2: 95
4: 83
6: 60
8: 58
"""
# %%
