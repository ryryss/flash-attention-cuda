/*
 * flash_attn_kernel.cu
 * Custom FlashAttention-style SDPA, register-resident + cp.async double-buffered
 * + shared-memory padding to eliminate bank conflicts.
 *
 * Techniques:
 *   - Tiling (BLOCK_M query rows / BLOCK_N kv rows) + online softmax.
 *   - WMMA bf16 tensor cores for QK^T and PV.
 *   - S, P, acc in registers (WMMA fragments).
 *   - cp.async double-buffered K/V prefetch.
 *   - Shared stride padded to HEAD_DIM+8 to avoid 8-way bank conflicts
 *     (ldm=128 is a multiple of 32 -> same bank; 136 breaks the aliasing).
 *
 * WMMA m16n16k16 fragment layout (accumulator f32 == matrix_a bf16):
 *   lane -> g = lane/4, p = lane%4
 *   x[0]=(g,2p) x[1]=(g,2p+1) x[2]=(g+8,2p) x[3]=(g+8,2p+1)
 *   x[4]=(g,2p+8) x[5]=(g,2p+9) x[6]=(g+8,2p+8) x[7]=(g+8,2p+9)
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

// async (cp.async) load of a full K/V tile into padded shared buffers
template <int HEAD_DIM, int BLOCK_N>
__device__ __forceinline__ void cp_async_load_tile(
    const bf16* __restrict__ k_ptr, const bf16* __restrict__ v_ptr,
    bf16* K_dst, bf16* V_dst, int tid, int nthreads)
{
    const int PAD = HEAD_DIM + 8;
    const int chunks_per_row = HEAD_DIM / 8;   // 16-byte chunks per row (8 bf16 each)
    const int total_chunks    = BLOCK_N * chunks_per_row;
    for (int c = tid; c < total_chunks; c += nthreads) {
        int r    = c / chunks_per_row;
        int cc   = c % chunks_per_row;
        int elem = cc * 8;
        __pipeline_memcpy_async(&K_dst[r * PAD + elem], &k_ptr[r * HEAD_DIM + elem], 16);
        __pipeline_memcpy_async(&V_dst[r * PAD + elem], &v_ptr[r * HEAD_DIM + elem], 16);
    }
}

// synchronous guarded load (for the possibly-partial last tile)
template <int HEAD_DIM, int BLOCK_N>
__device__ __forceinline__ void sync_load_tile(
    const bf16* __restrict__ k_ptr, const bf16* __restrict__ v_ptr,
    bf16* K_dst, bf16* V_dst, int valid, int tid, int nthreads)
{
    const int PAD = HEAD_DIM + 8;
    for (int i = tid; i < BLOCK_N * HEAD_DIM; i += nthreads) {
        int r = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        if (r < valid) {
            K_dst[r * PAD + d] = k_ptr[r * HEAD_DIM + d];
            V_dst[r * PAD + d] = v_ptr[r * HEAD_DIM + d];
        } else {
            K_dst[r * PAD + d] = __float2bfloat16(0.0f);
            V_dst[r * PAD + d] = __float2bfloat16(0.0f);
        }
    }
}

template <int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__global__ void flash_attn_bf16_kernel(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    const bf16* __restrict__ V,
    bf16* __restrict__ O,
    int S,
    int causal)
{
    const int PAD = HEAD_DIM + 8;
    const int NT  = BLOCK_N / 16;
    const int DT  = HEAD_DIM / 16;
    const int TILE_ELEMS = BLOCK_N * PAD;

    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> A_frag;
    typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> B_fragK;
    typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> B_fragV;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> C_frag;

    // shared: Q_s | K_s[2] | V_s[2]  (padded stride PAD)
    extern __shared__ char smem_raw[];
    bf16* Q_s = reinterpret_cast<bf16*>(smem_raw);
    bf16* K_s = Q_s + BLOCK_M * PAD;
    bf16* V_s = K_s + 2 * TILE_ELEMS;

    const int bh       = blockIdx.y;
    const int q_start  = blockIdx.x * BLOCK_M;
    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;
    const int warp     = tid / 32;
    const int lane     = tid % 32;
    const int g        = lane / 4;
    const int p        = lane % 4;

    // ---- load Q tile (padded) ----
    {
        const bf16* q_ptr = Q + (size_t)bh * S * HEAD_DIM + (size_t)q_start * HEAD_DIM;
        for (int i = tid; i < BLOCK_M * HEAD_DIM; i += nthreads) {
            int r  = i / HEAD_DIM;
            int d  = i % HEAD_DIM;
            int gr = q_start + r;
            Q_s[r * PAD + d] = (gr < S) ? q_ptr[r * HEAD_DIM + d] : __float2bfloat16(0.0f);
        }
    }

    float m_lo = NEG_INF, m_hi = NEG_INF;
    float l_lo = 0.0f,      l_hi = 0.0f;

    C_frag acc_frag[DT];
    #pragma unroll
    for (int d = 0; d < DT; d++) wmma::fill_fragment(acc_frag[d], 0.0f);

    const float scale = rsqrtf((float)HEAD_DIM);
    const int num_kv  = (S + BLOCK_N - 1) / BLOCK_N;

    // ---- prefetch tile 0 ----
    {
        const bf16* k0 = K + (size_t)bh * S * HEAD_DIM;
        const bf16* v0 = V + (size_t)bh * S * HEAD_DIM;
        if (S >= BLOCK_N) {
            cp_async_load_tile<HEAD_DIM, BLOCK_N>(k0, v0, K_s, V_s, tid, nthreads);
        } else {
            sync_load_tile<HEAD_DIM, BLOCK_N>(k0, v0, K_s, V_s, S, tid, nthreads);
        }
    }
    __pipeline_commit();
    __syncthreads();

    for (int kv = 0; kv < num_kv; kv++) {
        const int buf      = kv & 1;
        const int kv_start = kv * BLOCK_N;
        const bool has_next = (kv + 1 < num_kv);

        if (has_next) {
            const bf16* kn = K + (size_t)bh * S * HEAD_DIM + (size_t)(kv + 1) * BLOCK_N * HEAD_DIM;
            const bf16* vn = V + (size_t)bh * S * HEAD_DIM + (size_t)(kv + 1) * BLOCK_N * HEAD_DIM;
            const int nbuf   = (kv + 1) & 1;
            const int nvalid = min(BLOCK_N, S - (kv + 1) * BLOCK_N);
            if (nvalid == BLOCK_N) {
                cp_async_load_tile<HEAD_DIM, BLOCK_N>(kn, vn, K_s + nbuf * TILE_ELEMS, V_s + nbuf * TILE_ELEMS, tid, nthreads);
            } else {
                sync_load_tile<HEAD_DIM, BLOCK_N>(kn, vn, K_s + nbuf * TILE_ELEMS, V_s + nbuf * TILE_ELEMS, nvalid, tid, nthreads);
            }
        }
        __pipeline_commit();
        __pipeline_wait_prior(has_next ? 1 : 0);
        __syncthreads();

        bf16* Kb = K_s + buf * TILE_ELEMS;
        bf16* Vb = V_s + buf * TILE_ELEMS;

        // ---- QK^T -> S ----
        C_frag s_frag[NT];
        #pragma unroll
        for (int n = 0; n < NT; n++) wmma::fill_fragment(s_frag[n], 0.0f);
        #pragma unroll
        for (int k0 = 0; k0 < HEAD_DIM; k0 += 16) {
            A_frag a;
            wmma::load_matrix_sync(a, &Q_s[warp * 16 * PAD + k0], PAD);
            #pragma unroll
            for (int n = 0; n < NT; n++) {
                B_fragK b;
                wmma::load_matrix_sync(b, &Kb[n * 16 * PAD + k0], PAD);
                wmma::mma_sync(s_frag[n], a, b, s_frag[n]);
            }
        }
        #pragma unroll
        for (int n = 0; n < NT; n++)
            #pragma unroll
            for (int i = 0; i < 8; i++) s_frag[n].x[i] *= scale;

        // ---- causal mask ----
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

        // ---- online softmax ----
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

        // ---- PV -> acc ----
        #pragma unroll
        for (int d = 0; d < DT; d++)
            #pragma unroll
            for (int n = 0; n < NT; n++) {
                B_fragV v_frag;
                wmma::load_matrix_sync(v_frag, &Vb[n * 16 * PAD + d * 16], PAD);
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

    const int BLOCK_M = 128;
    const int BLOCK_N = 48;
    const int PAD = D + 8;
    const int smem = BLOCK_M * PAD * 2      // Q_s
                   + 2 * BLOCK_N * PAD * 2  // K_s[2]
                   + 2 * BLOCK_N * PAD * 2; // V_s[2]

    cudaStream_t stream = (cudaStream_t)stream_ptr;
    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, B * H);
    const int threads = (BLOCK_M / 16) * 32;

    static bool inited = false;
    if (!inited) {
        cudaFuncSetAttribute(flash_attn_bf16_kernel<128, BLOCK_M, BLOCK_N>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        inited = true;
    }

    flash_attn_bf16_kernel<128, BLOCK_M, BLOCK_N><<<grid, threads, smem, stream>>>(
        (const bf16*)q, (const bf16*)k, (const bf16*)v, (bf16*)o, S, causal);

    return (int)cudaGetLastError();
}
