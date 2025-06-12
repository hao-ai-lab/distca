#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>

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
) {
    // --- Calculate thread/warp IDs based on the new launch grid ---
    const unsigned warp_size = 32;
    // The local warp and lane index for the current thread
    const unsigned warp_id_in_block = threadIdx.x / warp_size;
    const unsigned lane_id = threadIdx.x % warp_size;
    
    // The number of warps per block is determined by the launch configuration
    const unsigned warps_per_block = blockDim.x / warp_size;

    // The globally unique ID for the current warp across the entire grid
    const unsigned global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;
    // The total number of warps launched in the grid
    const unsigned total_warps_in_grid = gridDim.x * warps_per_block;

    // --- SENDER-SIDE LOGIC with Warp-Level Grid-Stride Loop ---
    
    // Each warp processes one token at a time and strides through the entire dataset.
    // This allows a grid of any size to process all 'token' items.
    for (int token_idx = global_warp_id; token_idx < token; token_idx += total_warps_in_grid) {
        
        // --- 1. Dispatch the query tensor ---
        // The first lane of the assigned warp is responsible for dispatching the query.
        if (lane_id == 0) {
            int query_dest_rank = query_dst_id[token_idx];
            int query_dest_offset = query_dst_offset[token_idx];
            const T_q* query_src_ptr = query_in + token_idx * hidden_q;
            T_q* query_dest_ptr = query_out + query_dest_offset * hidden_q;
            
            nvshmem_putmem_nbi(query_dest_ptr, query_src_ptr, hidden_q * sizeof(T_q), query_dest_rank);
        }

        // --- 2. Dispatch the key_value tensor, if any ---
        // attn_out -> mlp only uses the query_tensor's part: each token is sent to only one rank.
        // The lanes of the warp cooperate to dispatch the `cp_degree` copies.
        // This loop strides by `warp_size`, so if cp_degree > 32, all lanes stay busy.
        if (key_value_in != nullptr) {
            for (int i = lane_id; i < cp_degree; i += warp_size) {
                int kv_idx = token_idx * cp_degree + i;
                
                int kv_dest_rank = key_value_dst_id[kv_idx];
                if (kv_dest_rank == -1) {
                    continue;
                }
                int kv_dest_offset = key_value_dst_offset[kv_idx];
                const T_kv* kv_src_ptr = key_value_in + token_idx * hidden_kv;
                T_kv* kv_dest_ptr = key_value_out + kv_dest_offset * hidden_kv;

                nvshmem_putmem_nbi(kv_dest_ptr, kv_src_ptr, hidden_kv * sizeof(T_kv), kv_dest_rank);
            }
        }
    }

    // --- RECEIVER-SIDE SYNCHRONIZATION ---

    // The cooperative group sync ensures all threads in the grid finish their loops.
    cooperative_groups::this_grid().sync();

    // per grid barrier
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvshmem_quiet();
    }
}