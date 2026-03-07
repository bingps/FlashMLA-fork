#include "splitkv_mla.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cutlass/numeric_types.h>

#include "utils.h"

namespace smxx::decode::sparse_bf16 {

using bf16 = cutlass::bfloat16_t;

namespace {

constexpr int TOPK_BLOCK_SIZE = 64;
constexpr int NUM_THREADS = 128;
constexpr int MAX_DV_PER_THREAD = 4;

struct MainloopArgs {
    int start_block_idx;
    int end_block_idx;
    bool is_no_split;
    int topk_length;
    int extra_topk_length;
    int num_orig_kv_blocks;
};

__host__ __device__ __forceinline__ int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ MainloopArgs get_cur_req_info(
    const SparseAttnDecodeParams &params,
    const DecodingSchedMeta &sched_meta,
    int batch_idx
) {
    MainloopArgs args;

    int orig_topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
    int orig_topk_nonzero = orig_topk_length > 0 ? orig_topk_length : 1;
    int orig_topk_padded = ceil_div_int(orig_topk_nonzero, TOPK_BLOCK_SIZE) * TOPK_BLOCK_SIZE;
    int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
    int extra_topk_padded = ceil_div_int(extra_topk_length, TOPK_BLOCK_SIZE) * TOPK_BLOCK_SIZE;
    int total_topk_padded = orig_topk_padded + extra_topk_padded;

    args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
    args.end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / TOPK_BLOCK_SIZE;
    args.is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);
    args.topk_length = orig_topk_length;
    args.extra_topk_length = extra_topk_length;
    args.num_orig_kv_blocks = orig_topk_padded / TOPK_BLOCK_SIZE;
    return args;
}

__device__ __forceinline__ int load_index(
    const SparseAttnDecodeParams &params,
    bool is_extra,
    int batch_idx,
    int s_q_idx,
    int topk_idx
) {
    const int *base = is_extra ? params.extra_indices : params.indices;
    const int stride_b = is_extra ? params.stride_extra_indices_b : params.stride_indices_b;
    const int stride_s_q = is_extra ? params.stride_extra_indices_s_q : params.stride_indices_s_q;
    return __ldg(base + batch_idx * stride_b + s_q_idx * stride_s_q + topk_idx);
}

__device__ __forceinline__ const bf16* get_token_ptr(
    const SparseAttnDecodeParams &params,
    bool is_extra,
    int token_idx
) {
    const bf16 *base = is_extra ? params.extra_kv : params.kv;
    const int page_block_size = is_extra ? params.extra_page_block_size : params.page_block_size;
    const int stride_block = is_extra ? params.stride_extra_kv_block : params.stride_kv_block;
    const int stride_row = is_extra ? params.stride_extra_kv_row : params.stride_kv_row;
    const int block_idx = token_idx / page_block_size;
    const int row_idx = token_idx % page_block_size;
    const int64_t offset = int64_t(block_idx) * int64_t(stride_block) + int64_t(row_idx) * int64_t(stride_row);
    return base + offset;
}

__global__ void __launch_bounds__(NUM_THREADS)
flash_fwd_splitkv_mla_bf16_sparse_kernel(__grid_constant__ const SparseAttnDecodeParams params) {
    const int head_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int tid = threadIdx.x;

    FLASH_DEVICE_ASSERT(params.h_kv == 1);
    FLASH_DEVICE_ASSERT(params.d_v == 512);
    FLASH_DEVICE_ASSERT(params.d_qk == 512 || params.d_qk == 576);
    FLASH_DEVICE_ASSERT(head_idx < params.h_q);
    FLASH_DEVICE_ASSERT(s_q_idx < params.s_q);
    FLASH_DEVICE_ASSERT(partition_idx < params.num_sm_parts);

    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) {
        return;
    }

    extern __shared__ float shared_buf[];
    float *dot_buf = shared_buf;

    float o_local[MAX_DV_PER_THREAD] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int i = 0; i < MAX_DV_PER_THREAD; ++i) {
        o_local[i] = 0.f;
    }

    float m = -CUDART_INF_F;
    float l = 0.f;

    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        MainloopArgs args = get_cur_req_info(params, sched_meta, batch_idx);
        const int64_t q_offset = int64_t(batch_idx) * int64_t(params.stride_q_b) + int64_t(s_q_idx) * int64_t(params.stride_q_s_q) + int64_t(head_idx) * int64_t(params.stride_q_h_q);
        const bf16 *q_ptr = params.q + q_offset;

        m = -CUDART_INF_F;
        l = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_DV_PER_THREAD; ++i) {
            o_local[i] = 0.f;
        }

        for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
            const bool is_extra_block = block_idx >= args.num_orig_kv_blocks;
            const int block_base = block_idx * TOPK_BLOCK_SIZE - (is_extra_block ? args.num_orig_kv_blocks * TOPK_BLOCK_SIZE : 0);
            const int active_topk_length = is_extra_block ? args.extra_topk_length : args.topk_length;
            const int max_topk = is_extra_block ? params.extra_topk : params.topk;
            const int max_token_idx = (is_extra_block ? params.extra_num_blocks * params.extra_page_block_size : params.num_blocks * params.page_block_size);

            #pragma unroll 1
            for (int offset = 0; offset < TOPK_BLOCK_SIZE; ++offset) {
                const int topk_idx = block_base + offset;
                const bool topk_in_range = topk_idx < max_topk;
                const bool topk_is_active = topk_idx < active_topk_length;

                int token_idx = -1;
                bool valid = false;
                if (topk_in_range && topk_is_active) {
                    token_idx = load_index(params, is_extra_block, batch_idx, s_q_idx, topk_idx);
                    valid = token_idx >= 0 && token_idx < max_token_idx;
                }

                float partial_dot = 0.f;
                if (valid) {
                    const bf16 *kv_ptr = get_token_ptr(params, is_extra_block, token_idx);
                    for (int d = tid; d < params.d_qk; d += NUM_THREADS) {
                        partial_dot += float(q_ptr[d]) * float(kv_ptr[d]);
                    }
                }
                dot_buf[tid] = partial_dot;
                __syncthreads();

                for (int stride = NUM_THREADS / 2; stride > 0; stride /= 2) {
                    if (tid < stride) {
                        dot_buf[tid] += dot_buf[tid + stride];
                    }
                    __syncthreads();
                }

                const float score = valid ? dot_buf[0] * params.sm_scale_div_log2 : -CUDART_INF_F;
                const float new_m = fmaxf(m, score);
                const float alpha = (m == -CUDART_INF_F) ? 0.f : exp2f(m - new_m);
                const float p = valid ? exp2f(score - new_m) : 0.f;

                if (valid) {
                    const bf16 *kv_ptr = get_token_ptr(params, is_extra_block, token_idx);
                    for (int d = tid; d < params.d_v; d += NUM_THREADS) {
                        const int slot = d / NUM_THREADS;
                        o_local[slot] = o_local[slot] * alpha + p * float(kv_ptr[d]);
                    }
                } else {
                    for (int d = tid; d < params.d_v; d += NUM_THREADS) {
                        const int slot = d / NUM_THREADS;
                        o_local[slot] *= alpha;
                    }
                }

                l = l * alpha + p;
                m = new_m;
                __syncthreads();
            }
        }

        if (args.is_no_split) {
            const int64_t out_offset = int64_t(batch_idx) * int64_t(params.stride_o_b) + int64_t(s_q_idx) * int64_t(params.stride_o_s_q) + int64_t(head_idx) * int64_t(params.stride_o_h_q);
            const int64_t lse_offset = int64_t(batch_idx) * int64_t(params.stride_lse_b) + int64_t(s_q_idx) * int64_t(params.stride_lse_s_q) + int64_t(head_idx);
            bf16 *out_ptr = params.out + out_offset;
            float *lse_ptr = params.lse + lse_offset;
            float inv = 0.f;
            if (l != 0.f) {
                float denom = l;
                if (params.attn_sink != nullptr) {
                    const float sink_base2 = __ldg(params.attn_sink + head_idx) * CUDART_L2E_F;
                    denom += exp2f(sink_base2 - m);
                }
                inv = 1.f / denom;
            }
            for (int d = tid; d < params.d_v; d += NUM_THREADS) {
                const int slot = d / NUM_THREADS;
                out_ptr[d] = bf16(o_local[slot] * inv);
            }
            if (tid == 0) {
                *lse_ptr = l == 0.f ? CUDART_INF_F : (logf(l) + m / CUDART_L2E_F);
            }
        } else {
            const int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
            const int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
            const int64_t o_accum_offset = int64_t(split_idx) * int64_t(params.stride_o_accum_split) + int64_t(s_q_idx) * int64_t(params.stride_o_accum_s_q) + int64_t(head_idx) * int64_t(params.stride_o_accum_h_q);
            const int64_t lse_accum_offset = int64_t(split_idx) * int64_t(params.stride_lse_accum_split) + int64_t(s_q_idx) * int64_t(params.stride_lse_accum_s_q) + int64_t(head_idx);
            float *o_accum_ptr = params.o_accum + o_accum_offset;
            float *lse_accum_ptr = params.lse_accum + lse_accum_offset;
            const float inv = l == 0.f ? 0.f : (1.f / l);
            for (int d = tid; d < params.d_v; d += NUM_THREADS) {
                const int slot = d / NUM_THREADS;
                o_accum_ptr[d] = o_local[slot] * inv;
            }
            if (tid == 0) {
                *lse_accum_ptr = l == 0.f ? -CUDART_INF_F : (log2f(l) + m);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

} // namespace

void run_flash_splitkv_mla_bf16_sparse_kernel(const SparseAttnDecodeParams &params) {
    FLASH_ASSERT(params.h_kv == 1);
    FLASH_ASSERT(params.d_v == 512);
    FLASH_ASSERT(params.d_qk == 512 || params.d_qk == 576);

    auto kernel = &flash_fwd_splitkv_mla_bf16_sparse_kernel;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config = {
        dim3(params.h_q, params.s_q, params.num_sm_parts),
        dim3(NUM_THREADS, 1, 1),
        sizeof(float) * NUM_THREADS,
        params.stream,
        attribute,
        1
    };
    CHECK_CUDA(cudaLaunchKernelEx(&config, kernel, params));
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
