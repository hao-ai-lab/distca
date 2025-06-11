#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>

// Forward declaration of the CUDA kernel
template <typename T_q, typename T_kv>
__global__ void qkv_dispatch_kernel(
    T_q* query_out,
    T_kv* key_value_out,
    const T_q* query_in,
    const T_kv* key_value_in,
    const int32_t* query_dst_id,
    const int32_t* query_dst_offset,
    const int32_t* key_value_dst_id,
    const int32_t* key_value_dst_offset,
    size_t token,
    size_t hidden_q,
    size_t hidden_kv,
    uint32_t cp_degree
);

// The main dispatch function
template <typename T_q, typename T_kv>
void qkv_dispatch(
    T_q* query_out,
    T_kv* key_value_out,
    const T_q* query_in,
    const T_kv* key_value_in,
    const int32_t* query_dst_id,
    const int32_t* query_dst_offset,
    const int32_t* key_value_dst_id,
    const int32_t* key_value_dst_offset,
    size_t token,
    size_t hidden_q,
    size_t hidden_kv,
    uint32_t my_rank,
    uint32_t world_size,
    uint32_t cp_degree
) {
    // A grid for each token
    int numSMs;
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));

    constexpr unsigned NUM_WARPS = 10;
    const unsigned numBlocks = std::min(
        std::max(
            ceil_div<unsigned>(token, NUM_WARPS), (unsigned)(token * cp_degree)
        ),
        static_cast<unsigned>(numSMs)
    );
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(NUM_WARPS * 32, 1, 1);

    qkv_dispatch_kernel<<<dimGrid, dimBlock>>>(
        query_out,
        key_value_out,
        query_in,
        key_value_in,
        query_dst_id,
        query_dst_offset,
        key_value_dst_id,
        key_value_dst_offset,
        token,
        hidden_q,
        hidden_kv,
        cp_degree
    );
}