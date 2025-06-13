#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>

// Forward declaration of the CUDA kernel
namespace attn {

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
    uint32_t cp_degree,
    cudaStream_t stream
);
};  // namespace attn
