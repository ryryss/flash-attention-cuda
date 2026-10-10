/*
 * flash_attn_kernel.cu
 * Custom FlashAttention-style SDPA, targeting 2 blocks/SM (16 warps) to hide
 * compute/shared/dependency latency (the real bottleneck per ncu).
 *
 * Changes vs v5:
 *   - Q stored in shared WITHOUT padding (stride 128) to save 2KB.
 *   - K/V single-buffered (stride 136, padded) to halve K/V shared.
 *   - BLOCK_N=32 to cut register pressure (s_frag[2]/p_frag[2]).
 *   => shared ~49KB -> 2 blocks/SM -> 512 threads -> 16 warps.
 *
 * Core stays: tiling + online softmax, WMMA bf16 tensor cores, register S/P/acc.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <math.h>

using bf16 = __nv_bfloat16;
using namespace nvcuda;

#define NEG_INF (-INFINITY)

__device__ __forceinline__ int row_off(int i) { const int r[8] = {0,0,8,8,0,0,8,8}; return r[i]; }
__device__ __forceinline__ int col_off(int i) { const int c[8] = {0,1,0,1,8,9,8,9}; return c[i]; }

__device__ __forceinline__ float warp_reduce_max4(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}
__device__ __forceinline__ float warp_reduce_sum4(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

template <int HEAD_DIM, int BLOCK_N, int KSTRIDE>
__device__ __forceinline__ void load_kv_tile(
    const bf16* __restrict__ k_ptr, const bf16* __restrict__ v_ptr,
    bf16* K_dst, bf16* V_dst, int valid, int tid, int nthreads)
{
    // async for full tiles, sync guarded for partial
    const int chunks_per_row = HEAD_DIM / 8;
    const int total_chunks    = BLOCK_N * chunks_per_row;
    for (int c = tid; c < total_chunks; c += nthreads) {
        int r    = c / chunks_per_row;
        int cc   = c % chunks_per_row;
        int elem = cc * 8;
        int valid_r = r < valid;
        if (valid_r) {
            __pipeline_memcpy_async(&K_dst[r * KSTRIDE + elem], &k_ptr[r * HEAD_DIM + elem], 16);
            __pipeline_memcpy_async(&V_dst[r * KSTRIDE + elem], &v_ptr[r * HEAD_DIM + elem], 16);
        } else {
            K_dst[r * KSTRIDE + elem] = __float2bfloat16(0.0f);
            K_dst[r * KSTRIDE + elem + 1] = __float2bfloat16(0.0f);
            K_dst[r * KSTRIDE + elem + 2] = __float2bfloat16(0.0f);
            K_dst[r * KSTRIDE + elem + 3] = __float2bfloat16(0.0f);
            K_dst[r * KSTRIDE + elem + 4] = __float2bfloat16(0.0f);
            K_dst[r * KSTRIDE + elem + 5] = __float2bfloat16(0.0f);
            K_dst[r * KSTRIDE + elem + 6] = __float2bfloat16(0.0f);
            K_dst[r * KSTRIDE + elem + 7] = __float2bfloat16(0.0f);
            V_dst[r * KSTRIDE + elem] = __float2bfloat16(0.0f);
            V_dst[r * KSTRIDE + elem + 1] = __float2bfloat16(0.0f);
            V_dst[r * KSTRIDE + elem + 2] = __float2bfloat16(0.0f);
            V_dst[r * KSTRIDE + elem + 3] = __float2bfloat16(0.0f);
            V_dst[r * KSTRIDE + elem + 4] = __float2bfloat16(0.0f);
            V_dst[r * KSTRIDE + elem + 5] = __float2bfloat16(0.0f);
            V_dst[r * KSTRIDE + elem + 6] = __float2bfloat16(0.0f);
            V_dst[r * KSTRIDE + elem + 7] = __float2bfloat16(0.0f);
        }
    }
}

template <int HEAD_DIM, int BLOCK_M, int BLOCK_N, int KSTRIDE>
__global__ void flash_attn_bf16_kernel(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    const bf16* __restrict__ V,
    bf16* __restrict__ O,
    int S,
    int causal)
{
    const int NT = BLOCK_N / 16;
    const int DT = HEAD_DIM / 16;

    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> A_frag;
    typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> B_fragK;
    typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> B_fragV;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> C_frag;

    // shared: Q_s (stride HEAD_DIM, unpadded) | K_s (stride KSTRIDE) | V_s (stride KSTRIDE)
    extern __shared__ char smem_raw[];
    bf16* Q_s = reinterpret_cast<bf16*>(smem_raw);
    bf16* K_s = Q_s + BLOCK_M * HEAD_DIM;
    bf16* V_s = K_s + BLOCK_N * KSTRIDE;

    const int bh       = blockIdx.y;
    const int q_start  = blockIdx.x * BLOCK_M;
    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;
    const int warp     = tid / 32;
    const int lane     = tid % 32;
    const int g        = lane / 4;
    const int p        = lane % 4;

    // ---- load Q tile (unpadded stride HEAD_DIM) ----
    {
        const bf16* q_ptr = Q + (size_t)bh * S * HEAD_DIM + (size_t)q_start * HEAD_DIM;
        for (int i = tid; i < BLOCK_M * HEAD_DIM; i += nthreads) {
            int r  = i / HEAD_DIM;
            int d  = i % HEAD_DIM;
            int gr = q_start + r;
            Q_s[i] = (gr < S) ? q_ptr[i] : __float2bfloat16(0.0f);
        }
    }

    float m_lo = NEG_INF, m_hi = NEG_INF;
    float l_lo = 0.0f,      l_hi = 0.0f;

    C_frag acc_frag[DT];
    #pragma unroll
    for (int d = 0; d < DT; d++) wmma::fill_fragment(acc_frag[d], 0.0f);

    const float scale = rsqrtf((float)HEAD_DIM);
    const int num_kv  = (S + BLOCK_N - 1) / BLOCK_N;

    __syncthreads();

    for (int kv = 0; kv < num_kv; kv++) {
        const int kv_start = kv * BLOCK_N;

        // ---- load K/V tile (single buffer) ----
        {
            const bf16* k_ptr = K + (size_t)bh * S * HEAD_DIM + (size_t)kv_start * HEAD_DIM;
            const bf16* v_ptr = V + (size_t)bh * S * HEAD_DIM + (size_t)kv_start * HEAD_DIM;
            int valid = min(BLOCK_N, S - kv_start);
            load_kv_tile<HEAD_DIM, BLOCK_N, KSTRIDE>(k_ptr, v_ptr, K_s, V_s, valid, tid, nthreads);
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // ---- QK^T -> S ----
        C_frag s_frag[NT];
        #pragma unroll
        for (int n = 0; n < NT; n++) wmma::fill_fragment(s_frag[n], 0.0f);
        #pragma unroll
        for (int k0 = 0; k0 < HEAD_DIM; k0 += 16) {
            A_frag a;
            wmma::load_matrix_sync(a, &Q_s[warp * 16 * HEAD_DIM + k0], HEAD_DIM);
            #pragma unroll
            for (int n = 0; n < NT; n++) {
                B_fragK b;
                wmma::load_matrix_sync(b, &K_s[n * 16 * KSTRIDE + k0], KSTRIDE);
                wmma::mma_sync(s_frag[n], a, b, s_frag[n]);
            }
        }
        #pragma unroll
        for (int n = 0; n < NT; n++)
            #pragma unroll
            for (int i = 0; i < 8; i++) s_frag[n].x[i] *= scale;

        if (causal) {
            const int row_base = q_start + warp * 16;
            #pragma unroll
            for (int n = 0; n < NT; n++)
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int qidx = row_base + g + row_off(i);
                    int kidx = kv_start + n * 16 + 2 * p + col_off(i);
                    if (kidx > qidx) s_frag[n].x[i] = NEG_INF;
                }
        }

        float s_lo = NEG_INF, s_hi = NEG_INF;
        #pragma unroll
        for (int n = 0; n < NT; n++) {
            s_lo = fmaxf(s_lo, fmaxf(fmaxf(s_frag[n].x[0], s_frag[n].x[1]), fmaxf(s_frag[n].x[4], s_frag[n].x[5])));
            s_hi = fmaxf(s_hi, fmaxf(fmaxf(s_frag[n].x[2], s_frag[n].x[3]), fmaxf(s_frag[n].x[6], s_frag[n].x[7])));
        }
        s_lo = warp_reduce_max4(s_lo);
        s_hi = warp_reduce_max4(s_hi);

        float a_lo = (s_lo > NEG_INF) ? expf(m_lo - s_lo) : 1.0f;
        float a_hi = (s_hi > NEG_INF) ? expf(m_hi - s_hi) : 1.0f;

        #pragma unroll
        for (int d = 0; d < DT; d++) {
            acc_frag[d].x[0] *= a_lo; acc_frag[d].x[1] *= a_lo;
            acc_frag[d].x[4] *= a_lo; acc_frag[d].x[5] *= a_lo;
            acc_frag[d].x[2] *= a_hi; acc_frag[d].x[3] *= a_hi;
            acc_frag[d].x[6] *= a_hi; acc_frag[d].x[7] *= a_hi;
        }

        A_frag p_frag[NT];
        float sum_lo = 0.0f, sum_hi = 0.0f;
        #pragma unroll
        for (int n = 0; n < NT; n++)
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                bool is_lo = (i == 0 || i == 1 || i == 4 || i == 5);
                float r  = is_lo ? s_lo : s_hi;
                float pv = (r > NEG_INF) ? expf(s_frag[n].x[i] - r) : 0.0f;
                p_frag[n].x[i] = __float2bfloat16(pv);
                if (is_lo) sum_lo += pv; else sum_hi += pv;
            }
        sum_lo = warp_reduce_sum4(sum_lo);
        sum_hi = warp_reduce_sum4(sum_hi);

        l_lo = l_lo * a_lo + sum_lo;
        l_hi = l_hi * a_hi + sum_hi;
        m_lo = s_lo;
        m_hi = s_hi;

        #pragma unroll
        for (int d = 0; d < DT; d++)
            #pragma unroll
            for (int n = 0; n < NT; n++) {
                B_fragV v_frag;
                wmma::load_matrix_sync(v_frag, &V_s[n * 16 * KSTRIDE + d * 16], KSTRIDE);
                wmma::mma_sync(acc_frag[d], p_frag[n], v_frag, acc_frag[d]);
            }

        __syncthreads();
    }

    // ---- finalize ----
    {
        bf16* o_ptr = O + (size_t)bh * S * HEAD_DIM + (size_t)q_start * HEAD_DIM;
        const int row_base = warp * 16;
        #pragma unroll
        for (int d = 0; d < DT; d++)
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                bool is_lo = (i == 0 || i == 1 || i == 4 || i == 5);
                float l = is_lo ? l_lo : l_hi;
                int r = row_base + g + row_off(i);
                int qidx = q_start + r;
                int dcol = d * 16 + 2 * p + col_off(i);
                if (qidx < S) {
                    o_ptr[r * HEAD_DIM + dcol] = __float2bfloat16(acc_frag[d].x[i] / l);
                }
            }
    }
}

// ---- host launcher ----
extern "C" __declspec(dllexport) int flash_attn_bf16(
    const void* q, const void* k, const void* v, void* o,
    int B, int H, int S, int D, int causal,
    void* stream_ptr)
{
    if (D != 128) return -1;

    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 32;
    constexpr int KSTRIDE = 136;          // D(128) + 8, padded stride for K/V
    const int smem = BLOCK_M * D * 2      // Q_s (unpadded)
                   + BLOCK_N * KSTRIDE * 2  // K_s
                   + BLOCK_N * KSTRIDE * 2; // V_s

    cudaStream_t stream = (cudaStream_t)stream_ptr;
    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, B * H);
    const int threads = (BLOCK_M / 16) * 32;

    static bool inited = false;
    if (!inited) {
        cudaFuncSetAttribute(flash_attn_bf16_kernel<128, BLOCK_M, BLOCK_N, KSTRIDE>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        inited = true;
    }

    flash_attn_bf16_kernel<128, BLOCK_M, BLOCK_N, KSTRIDE><<<grid, threads, smem, stream>>>(
        (const bf16*)q, (const bf16*)k, (const bf16*)v, (bf16*)o, S, causal);

    return (int)cudaGetLastError();
}
