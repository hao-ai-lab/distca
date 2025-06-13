#include "bindings/all_to_all.h"
#include "core/in_place_attn_switch.h"

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

using namespace attn;

namespace {

#define _CHECK_TENSOR(ndim, x) \
    TORCH_CHECK(x.ndimension() == ndim, "tensor " #x " must have " #ndim " dimensions"); \
    TORCH_CHECK(x.is_cuda(), "tensor " #x " must be on CUDA"); \
    TORCH_CHECK(x.is_contiguous(), "tensor " #x " must be contiguous");

template <typename T_q, typename T_kv>
void dispatch_kernel(
    at::Tensor &query_out,
    std::optional<const at::Tensor> &key_value_out,
    const at::Tensor &query_in,
    std::optional<const at::Tensor> &key_value_in,
    const at::Tensor &query_dst_id,
    const at::Tensor &query_dst_offset,
    std::optional<const at::Tensor> &key_value_dst_id,
    std::optional<const at::Tensor> &key_value_dst_offset,
    int64_t token,
    int64_t hidden_q,
    int64_t hidden_kv,
    int64_t cp_degree,
    cudaStream_t stream
) {
    const int32_t* query_dst_id_ptr = query_dst_id.data_ptr<int32_t>();
    const int32_t* query_dst_offset_ptr = query_dst_offset.data_ptr<int32_t>();
    const int32_t* key_value_dst_id_ptr = key_value_dst_id.has_value() ? 
        key_value_dst_id.value().data_ptr<int32_t>() : nullptr;
    const int32_t* key_value_dst_offset_ptr = key_value_dst_offset.has_value() ? 
        key_value_dst_offset.value().data_ptr<int32_t>() : nullptr;
    // FIXME(yonghao): query_out and key_value_out are nvshmem registered buffers when sending to the qkv_dispatch.
    qkv_dispatch<T_q, T_kv>(
        (T_q*)query_out.data_ptr(),
        key_value_out.has_value() ? (T_kv*)key_value_out.value().data_ptr() : nullptr,
        (const T_q*)query_in.data_ptr(),
        key_value_in.has_value() ? (const T_kv*)key_value_in.value().data_ptr() : nullptr,
        query_dst_id_ptr,
        query_dst_offset_ptr,
        key_value_dst_id_ptr,
        key_value_dst_offset_ptr,
        token,
        hidden_q,
        hidden_kv,
        cp_degree,
        stream
    );
}

void dispatch(
    at::Tensor &query_out,
    std::optional<const at::Tensor> &key_value_out,
    const at::Tensor &query_in,
    std::optional<const at::Tensor> &key_value_in,
    const at::Tensor &query_dst_id,
    const at::Tensor &query_dst_offset,
    std::optional<const at::Tensor> &key_value_dst_id,
    std::optional<const at::Tensor> &key_value_dst_offset,
    int64_t token
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    // Get hidden dimension size of the tensors to send
    const size_t hidden_q = query_in.size(1);
    size_t hidden_kv;
    size_t cp_degree;

    if (key_value_out.has_value()) {
        hidden_kv = key_value_out.value().size(1);
        TORCH_CHECK(key_value_dst_id.has_value(), 
            "key_value tensor, dst_id, dst_offset must be provided together");
        cp_degree = key_value_dst_id.value().size(2);
    } else {
        hidden_kv = 0;
        cp_degree = 0;
    }
    // Get dtype for tensors to send
    const c10::ScalarType query_dtype = query_in.scalar_type();
    const c10::ScalarType key_value_dtype = key_value_in.has_value() ? 
        key_value_in.value().scalar_type() : query_dtype;

    // Set device
    at::cuda::OptionalCUDAGuard const device_guard(device_of(query_out));

    switch (query_dtype) {
        case c10::ScalarType::Float: {
            switch (key_value_dtype) {
                case c10::ScalarType::Float: {
                    dispatch_kernel<float, float>(query_out, key_value_out, query_in, 
                        key_value_in, query_dst_id, query_dst_offset, key_value_dst_id,
                        key_value_dst_offset, token, hidden_q, hidden_kv, cp_degree, stream);
                }
                default: {
                    TORCH_CHECK(false, "key_value dtype must be the same as query dtype");
                }
            };
            break;
        };
        case c10::ScalarType::BFloat16: {
            switch (key_value_dtype) {
                case c10::ScalarType::BFloat16: {
                    dispatch_kernel<nv_bfloat16, nv_bfloat16>(query_out, key_value_out,
                        query_in, key_value_in, query_dst_id, query_dst_offset,
                        key_value_dst_id, key_value_dst_offset, token, hidden_q,
                        hidden_kv, cp_degree, stream);
                }
                default: {
                    TORCH_CHECK(false, "key_value dtype must be the same as query dtype");
                }
            };
            break;
        };
        case c10::ScalarType::Half: {
            switch (key_value_dtype) {
                case c10::ScalarType::Half: {
                    dispatch_kernel<half, half>(query_out, key_value_out, query_in,
                        key_value_in, query_dst_id, query_dst_offset, key_value_dst_id,
                        key_value_dst_offset, token, hidden_q, hidden_kv, cp_degree, stream);
                }
                default: {
                    TORCH_CHECK(false, "key_value dtype must be the same as query dtype");
                }
            };
            break;
        };
        default: {
            TORCH_CHECK(false, "Unsupported query dtype");
        }
    };

}

}; // namespace


namespace attn {
void register_all_to_all_ops(torch::Library &m) {
    m.def("dispatch", &dispatch);
}
}; // namespace attn
