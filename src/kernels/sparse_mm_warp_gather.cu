/***********************************************************************
 *  spmm_rowGather.cu    Sparse (rowgather) matrix multiplication
 *  ---------------------------------------------------------------
 *  Two kernels:
 *    (1)  Warplevel rowgather SPMM (any CC  7.0)
 *    (2)  TensorCore version using TF32 (CC  8.0, shape 16168)
 *
 *  20250627    Fixed TF32 WMMA shape; cleaned warnings.
 **********************************************************************/
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda/pipeline>
#include <mma.h>                     // WMMA / Tensor Core intrinsics
#include <cooperative_groups.h>      // cg::this_thread_block
#include <cuda_fp16.h>               // half precision helpers

using namespace nvcuda;              // for wmma
namespace cg = cooperative_groups;

/* ------------------------------------------------------------------ *
 *  General utility macros                                             *
 * ------------------------------------------------------------------ */
#define WARP_SIZE     32
#define FULL_MASK     0xffffffffu
#define CHECK_CUDA(x) do {                                              \
    cudaError_t e = (x);                                                \
    if (e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d  : %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e));                                 \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

static inline double to_ms(cudaEvent_t s, cudaEvent_t e) {
    float ms = 0.f;
    cudaEventElapsedTime(&ms, s, e);
    return static_cast<double>(ms);
}

/* ================================================================== *
 *  1.  Warplevel rowgather kernel (float)                           *
 * ================================================================== */
#define WARPS_CTA         4
#define THREADS_CTA       (WARPS_CTA * WARP_SIZE)
#define COLS_PER_CTA      WARPS_CTA
#define PREFETCH_DIST     WARP_SIZE          // hide HBML1 latency
#define VEC_WIDTH         4                  // float4 = 128bit
using  A_vec_t = float4;

/**
 * Rowgather based sparse matrix multiplication (float  float).
 *
 *   P = W  A  +  B
 *   W : (M  K)  rowmajor, generally sparse
 *   A : (K  N)  columnmajor (i.e. contiguous in K for each column)
 *   B,P follow A's layout.
 */
__global__ __launch_bounds__(THREADS_CTA, 2)
void spmm_rowGather(const float* __restrict__ W,
                    const float* __restrict__ A,
                    const float* __restrict__ B,
                          float* __restrict__ P,
                    int M, int K, int N)
{
    /* ---- perthread / warp ids ---- */
    const int lane      = threadIdx.x & 31;        // 031
    const int warpInCTA = threadIdx.x >> 5;        // 03
    const int col       = blockIdx.x * COLS_PER_CTA + warpInCTA;
    if (col >= N) return;

    const int warpRowGrp = blockIdx.y * WARPS_CTA + warpInCTA;
    const int row        = warpRowGrp * WARP_SIZE + lane;
    const bool valid_row = (row < M);

    /* ---- accumulator & activation ptr ---- */
    float acc = 0.f;
    const float* __restrict__ A_col = A + static_cast<size_t>(col) * K;

    /* ---- software prefetch registers ---- */
    float a_reg  = (lane < K) ? __ldg(&A_col[lane]) : 0.f;
    float a_next = 0.f;

    /* ---- loop over K dimension, 32 values / iter ---- */
    const int kTiles = (K + WARP_SIZE - 1) >> 5;
    for (int t = 0; t < kTiles; ++t) {
        const int kBase = t * WARP_SIZE;

        /* prefetch next */
        const int kPref = kBase + PREFETCH_DIST + lane;
        a_next = (kPref < K) ? __ldg(&A_col[kPref]) : 0.f;

        /* warp ballot & gather */
        unsigned mask = __ballot_sync(FULL_MASK, a_reg != 0.f);
        while (mask) {
            const int nzLane = __ffs(mask) - 1;
            mask &= mask - 1;

            const float a_val = __shfl_sync(FULL_MASK, a_reg, nzLane);
            const int   kIdx  = kBase + nzLane;

            if (valid_row) {
                const float w = __ldg(&W[row + static_cast<size_t>(kIdx) * M]);
                acc += w * a_val;
            }
        }
        a_reg = a_next;              // advance
    }

    /* ---- writeback ---- */
    if (valid_row) {
        const size_t off = row + static_cast<size_t>(col) * M;
        P[off] = acc + __ldg(&B[off]);
    }
}

/* ================================================================== *
 *  2.  TensorCore (TF32) rowgather kernel                           *
 * ================================================================== */
/* --- WMMA tile params : 16168 TF32 --- */
#define TC_TILE_M   16
#define TC_TILE_N   16
#define TC_TILE_K    8               // *** TF32 requires K == 8 ***

#define TC_WARPS     8               // 8 warps = 256 threads / CTA
#define TC_THREADS  (TC_WARPS * WARP_SIZE)

/* shared memory elements (float) : Wtile + Atile + Ctile            */
#define TC_SMEM_F32   (TC_TILE_M * TC_TILE_K + \
                       TC_TILE_K * TC_TILE_N + \
                       TC_TILE_M * TC_TILE_N)
#define TC_SMEM_BYTES (TC_SMEM_F32 * sizeof(float))

__global__ __launch_bounds__(TC_THREADS, 2)
void spmm_rowGather_tc(const float* __restrict__ W,
                       const float* __restrict__ A,
                       const float* __restrict__ B,
                             float* __restrict__ P,
                       int M, int K, int N)
{
#if (__CUDA_ARCH__ >= 800)
    extern __shared__ float smem[];         // dynamic shared memory
    float* smemW = smem;                                            // 0-(M*K-1)
    float* smemA = smemW + TC_TILE_M * TC_TILE_K;                   // offset
    float* smemC = smemA + TC_TILE_K * TC_TILE_N;

    /* ---- CTA coordinates ---- */
    const int lane   = threadIdx.x & 31;
    const int wid    = threadIdx.x >> 5;            // warp id 0-7

    const int tile_m = blockIdx.y * TC_TILE_M;
    const int tile_n = blockIdx.x * TC_TILE_N;

    /* ---- accumulator fragment ---- */
    wmma::fragment<wmma::accumulator,
                   TC_TILE_M, TC_TILE_N, TC_TILE_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    /* ---- iterate over K tiles ---- */
    const int kTiles = (K + TC_TILE_K - 1) / TC_TILE_K;
    for (int kt = 0; kt < kTiles; ++kt) {
        const int kBase = kt * TC_TILE_K;

        /* === 1)  Load Wtile & Atile to shared memory =============== */
        const int tid = threadIdx.x;

        /* Wtile : (M,K) -> smemW[row * TC_TILE_K + col]               */
        if (tid < TC_TILE_M * TC_TILE_K) {
            const int r   = tid % TC_TILE_M;       // 015
            const int c   = tid / TC_TILE_M;       // 07
            const int g_r = tile_m + r;
            const int g_k = kBase   + c;

            float v = (g_r < M && g_k < K)
                      ? __ldg(&W[g_r + static_cast<size_t>(g_k) * M])
                      : 0.f;
            smemW[r * TC_TILE_K + c] = v;
        }

        /* Atile : (K,N) -> smemA[row * TC_TILE_N + col]               */
        if (tid < TC_TILE_K * TC_TILE_N) {
            const int r   = tid % TC_TILE_K;       // Kdim (07)
            const int c   = tid / TC_TILE_K;       // colintile (015)
            const int g_k = kBase   + r;
            const int g_c = tile_n  + c;

            float v = (g_k < K && g_c < N)
                      ? __ldg(&A[g_c * static_cast<size_t>(K) + g_k])
                      : 0.f;
            smemA[r * TC_TILE_N + c] = v;
        }
        __syncthreads();

        /* === 2)  TF32 Tensor Core MMA ================================= */
        wmma::fragment<wmma::matrix_a,
                       TC_TILE_M, TC_TILE_N, TC_TILE_K,
                       wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b,
                       TC_TILE_M, TC_TILE_N, TC_TILE_K,
                       wmma::precision::tf32, wmma::row_major> b_frag;

        wmma::load_matrix_sync(a_frag, smemW, TC_TILE_K);
        wmma::load_matrix_sync(b_frag, smemA, TC_TILE_N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    /* === 3)  Write Ctile (+B) back to global ======================= */
    if (tile_m < M && tile_n < N) {
        /* dump fragment to shared to ensure each element visible       */
        wmma::store_matrix_sync(smemC, c_frag,
                                TC_TILE_N, wmma::mem_row_major);
        __syncthreads();

        const int rowInTile = threadIdx.x %  TC_TILE_M;   // 015
        const int colInTile = threadIdx.x /  TC_TILE_M;   // 015

        const int g_r = tile_m + rowInTile;
        const int g_c = tile_n + colInTile;

        if (g_r < M && g_c < N) {
            const float out = smemC[rowInTile * TC_TILE_N + colInTile] +
                              __ldg(&B[g_r + static_cast<size_t>(g_c) * M]);
            P[g_r + static_cast<size_t>(g_c) * M] = out;
        }
    }
#endif  // __CUDA_ARCH__ >= 800
}

/* ================================================================== *
 *  Host wrappers                                                      *
 * ================================================================== */
double runWarpGatherSparse(const float* dW, const float* dA,
                           const float* dB,       float* dP,
                           int M, int K, int N)
{
    const dim3 thr(THREADS_CTA, 1, 1);
    const int rowsPerCTA = WARPS_CTA * WARP_SIZE;
    const dim3 grid((N + COLS_PER_CTA - 1) / COLS_PER_CTA,
                    (M + rowsPerCTA  - 1) / rowsPerCTA);

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));

    spmm_rowGather<<<grid, thr>>>(dW, dA, dB, dP, M, K, N);

    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaEventSynchronize(t1));

    double ms = to_ms(t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

double runWarpGatherSparseTC(const float* dW, const float* dA,
                             const float* dB,       float* dP,
                             int M, int K, int N)
{
    const dim3 thr(TC_THREADS, 1, 1);
    const dim3 grid((N + TC_TILE_N - 1) / TC_TILE_N,
                    (M + TC_TILE_M - 1) / TC_TILE_M);

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));

    spmm_rowGather_tc<<<grid, thr, TC_SMEM_BYTES>>>(dW, dA, dB, dP, M, K, N);

    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaEventSynchronize(t1));

    double ms = to_ms(t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}
/*******************************  END  ********************************/
