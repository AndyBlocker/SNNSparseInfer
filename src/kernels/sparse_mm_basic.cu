/***********************************************************************
*  basic_spmm_tc.cu — Sparse(ish) Matrix × Matrix with optional TC    *
*                                                                    *
*  • 自动生成 A 的稀疏掩码并按掩码跳过零‑tile                         *
*  • 若 tile 大小能被 16 整除，且硬件 Compute Capability ≥ 75，      *
*    则在每个被访问的 tile 上使用 Tensor Core (WMMA + TF32)          *
*  • 其余情况退化为改进后的 shared‑memory SPMM 路径（无正确性缺陷）  *
*                                                                    *
*  编译示例（Ampere / Hopper 等支持 TF32 TensorCore 的 GPU）：        *
*      nvcc -O3 -arch=sm_80 -DBLK_X=16 -DBLK_Y=16 -DPAD=1 \          *
*           -DUSE_TENSOR_CORE basic_spmm_tc.cu -o spmm               *
***********************************************************************/

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef BLK_X
#   define BLK_X 16      // output tile 宽度（软件路径）
#endif
#ifndef BLK_Y
#   define BLK_Y 16      // output tile 高度（软件路径）
#endif
#ifndef PAD
#   define PAD   1       // 共享内存行末 padding → 避免 bank conflict
#endif

/* ===================== 工具宏 & util ====================== */
#define CHECK_CUDA(call)                                                      \
do {                                                                          \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr,"CUDA error %s:%d : %s\n",                             \
                __FILE__,__LINE__, cudaGetErrorString(_e));                   \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

static inline double to_ms(cudaEvent_t beg, cudaEvent_t end)
{
    float ms = 0.f;
    cudaEventElapsedTime(&ms, beg, end);
    return static_cast<double>(ms);
}

/* ========================================================== *
 *                     Kernel 1 : buildMask                   *
 * ========================================================== */
__global__ void buildMask(const float *__restrict__ A,
                          uint8_t     *__restrict__ mask,
                          int K, int N, int tile)
{
    const int nt = blockIdx.x;
    const int kt = blockIdx.y;
    const int nTiles = gridDim.x;

    const int k0   = kt * tile;
    const int n0   = nt * tile;
    const int kEnd = min(k0 + tile, K);
    const int nEnd = min(n0 + tile, N);

    bool hasNZ = false;
    for (int k = k0 + threadIdx.x; k < kEnd && !hasNZ; k += BLK_X)
        for (int n = n0 + threadIdx.y; n < nEnd && !hasNZ; n += BLK_Y)
            if (__ldg(&A[k + (size_t)n * K]) != 0.f) { hasNZ = true; break; }

    __syncthreads();
    hasNZ = __syncthreads_or(hasNZ);
    if (threadIdx.x == 0 && threadIdx.y == 0)
        mask[kt * nTiles + nt] = static_cast<uint8_t>(hasNZ);
}

/* ========================================================== *
 *              Kernel 2 : shared‑mem SPMM (改进版)            *
 * ========================================================== */
__launch_bounds__(BLK_X * BLK_Y, 2)
__global__ void spmmTile_sw(const float *__restrict__ W,
                            const float *__restrict__ A,
                            const float *__restrict__ B,
                                  float *__restrict__ P,
                            const uint8_t *__restrict__ mask,
                            int M, int K, int N, int tile)
{
    /* 本线程处理的全局 (m,n) */
    const int m = blockIdx.y * BLK_Y + threadIdx.y;
    const int n = blockIdx.x * BLK_X + threadIdx.x;
    const bool valid_m = (m < M);
    const bool valid_n = (n < N);

    const int nTiles  = (N + tile - 1) / tile;
    const int kTiles  = (K + tile - 1) / tile;
    const int nTileId = n / tile;

    /* ----------- 动态共享内存布局 ----------- *
     *  shW : tile × BLK_Y      (行步长 = BLK_Y+PAD)
     *  shA : tile × tile       (行步长 = tile+PAD)
     */
    extern __shared__ float sh[];
    const int strideW = BLK_Y + PAD;
    const int strideA = tile   + PAD;
    float *shW = sh;
    float *shA = sh + tile * strideW;

    float acc = 0.f;

    for (int kt = 0; kt < kTiles; ++kt)
    {
        if (!mask[kt * nTiles + nTileId])      // 整个 tile 全零
            continue;

        const int kBase = kt * tile;
        const int rows  = min(tile, K - kBase);

        /* ---- 1. W 子块读入共享内存 (col‑major → row in shmem) ---- */
        for (int kk = threadIdx.x; kk < rows; kk += BLK_X)
        {
            const int kIdx = kBase + kk;
            float w = 0.f;
            if (valid_m)  w = __ldg(&W[m + (size_t)kIdx * M]);
            shW[kk * strideW + threadIdx.y] = w;
        }

        /* ---- 2. 清零 A‑tile 有效区 ---- */
        for (int idx = threadIdx.y * BLK_X + threadIdx.x;
             idx < rows * strideA;
             idx += BLK_X * BLK_Y)
            shA[idx] = 0.f;
        __syncthreads();

        /* ---- 3. 把 A 子块搬进共享内存 (col‑major → row‑major in shmem) ---- */
        const int n0       = nTileId * tile;
        const int colsIn   = min(tile, N - n0);
        for (int kk = threadIdx.y; kk < rows; kk += BLK_Y)
            for (int nn = threadIdx.x; nn < colsIn; nn += BLK_X)
            {
                size_t gOff = (kBase + kk) + (size_t)(n0 + nn) * K;
                shA[kk * strideA + nn] = __ldg(&A[gOff]);
            }
        __syncthreads();

        /* ---- 4. 本线程完成 rows × colsIn 的乘加 ---- */
        if (valid_m && valid_n)
        {
            const int local_n = n - n0;
#pragma unroll 4
            for (int kk = 0; kk < rows; ++kk)
            {
                float a = (local_n < colsIn) ?
                          shA[kk * strideA + local_n] : 0.f;
                acc += shW[kk * strideW + threadIdx.y] * a;
            }
        }
        __syncthreads();
    }

    /* ---- 5. 写回 ---- */
    if (valid_m && valid_n)
    {
        const size_t off = m + (size_t)n * M;
        P[off] = acc + __ldg(&B[off]);
    }
}

/* ========================================================== *
 *       (可选) Kernel 3 : Tensor Core (WMMA‑TF32) 路径         *
 * ========================================================== */
#if defined(USE_TENSOR_CORE) && (__CUDA_ARCH__ >= 750)
#include <mma.h>
using namespace nvcuda;

__launch_bounds__(32, 4)      // 一个 warp 即一个 16×16 输出 tile
__global__ void spmmTile_tc(const float *__restrict__ W,
                            const float *__restrict__ A,
                            const float *__restrict__ B,
                                  float *__restrict__ P,
                            const uint8_t *__restrict__ mask,
                            int M, int K, int N, int tile)   // tile 必须是 16 的倍数
{
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    const int mTile = blockIdx.y;                 // 16 行块
    const int nTile = blockIdx.x;                 // 16 列块

    const int row_base = mTile * WMMA_M;
    const int col_base = nTile * WMMA_N;

    const int nTiles   = (N + tile - 1) / tile;
    const int kTiles   = (K + tile - 1) / tile;
    const int nTileId  = col_base / tile;         // 对应 buildMask 的列‑tile 号

    /* 累加器 fragment 初始化为 0 */
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    /* 每次循环处理 K 方向一个 “tile”(用户指定大小, 多为 16/32/64) */
    for (int kt = 0; kt < kTiles; ++kt)
    {
        if (!mask[kt * nTiles + nTileId])         // 整 tile 为 0 → 跳过
            continue;

        const int kBase  = kt * tile;
        const int rowsK  = min(tile, K - kBase);

        /* tile 里按 16 分块做 mma */
        for (int ks = 0; ks < rowsK; ks += WMMA_K)
        {
            const int kSub = kBase + ks;
            /* 共有 rowsK /16 (向上取整) 次子 mma */

            /* ===================================================== *
             *          1. 装入 W / A 到共享内存 (row‑major)           *
             * ===================================================== */
            __shared__ float shW[WMMA_M * WMMA_K];   // 256B
            __shared__ float shA[WMMA_K * WMMA_N];   // 256B

            const int lane = threadIdx.x;            // 0..31

            /* 让 32 线程一次性 copy 两个 16×16 字板块 */
            for (int i = lane; i < WMMA_M * WMMA_K; i += 32)
            {
                int r = i / WMMA_K;
                int c = i % WMMA_K;
                int gR = row_base + r;
                int gC = kSub     + c;
                shW[i] = (gR < M && gC < K) ? __ldg(&W[gR + (size_t)gC * M]) : 0.0f;
            }
            for (int i = lane; i < WMMA_K * WMMA_N; i += 32)
            {
                int r = i / WMMA_N;
                int c = i % WMMA_N;
                int gR = kSub     + r;
                int gC = col_base + c;
                shA[i] = (gR < K && gC < N) ? __ldg(&A[gR + (size_t)gC * K]) : 0.0f;
            }
            __syncthreads();

            /* ===================================================== *
             *               2. WMMA 计算 (row_major)                 *
             * ===================================================== */
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> b_frag;

            wmma::load_matrix_sync(a_frag, shW, WMMA_K);
            wmma::load_matrix_sync(b_frag, shA, WMMA_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            __syncthreads();
        } /* --- end ks --- */
    }     /* --- end kt --- */

    /* ===================================================== *
     *             3. store 结果到全局内存                   *
     * ===================================================== */
    __shared__ float shC[WMMA_M * WMMA_N];   // 256B
    wmma::store_matrix_sync(shC, c_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    const int lane = threadIdx.x;
    for (int i = lane; i < WMMA_M * WMMA_N; i += 32)
    {
        int r = i / WMMA_N;
        int c = i % WMMA_N;
        int gR = row_base + r;
        int gC = col_base + c;
        if (gR < M && gC < N)
        {
            size_t off = gR + (size_t)gC * M;
            P[off] = shC[i] + __ldg(&B[off]);
        }
    }
}
#endif  /* USE_TENSOR_CORE && arch>=75 */

/* ========================================================== *
 *                  Host wrapper : runBasicSparse             *
 * ========================================================== */
double runBasicSparse(const float *dW, const float *dA,
                      const float *dB,       float *dP,
                      int M, int K, int N, int tile)
{
    const int nTiles = (N + tile - 1) / tile;
    const int kTiles = (K + tile - 1) / tile;

    /* ---- 0. 生成稀疏掩码 ---- */
    uint8_t *dMask = nullptr;
    CHECK_CUDA(cudaMalloc(&dMask, nTiles * kTiles * sizeof(uint8_t)));

    dim3 thrMask(BLK_X, BLK_Y);
    dim3 gridMask(nTiles, kTiles);
    buildMask<<<gridMask, thrMask>>>(dA, dMask, K, N, tile);
    CHECK_CUDA(cudaGetLastError());

    /* ---- 1. 决定使用哪条计算路径 ---- */
    bool useTC =
#if defined(USE_TENSOR_CORE)
        (tile % 16 == 0) && (cudaDeviceProp{}.major >= 8 /* conservative */);
#else
        false;
#endif

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    if (useTC)
    {
#if defined(USE_TENSOR_CORE)
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        spmmTile_tc<<<grid, 32>>>(dW, dA, dB, dP,
                                  dMask, M, K, N, tile);
        CHECK_CUDA(cudaGetLastError());
#else
        (void)0;   // 不会走到
#endif
    }
    else
    {
        dim3 thr(BLK_X, BLK_Y);
        dim3 grid((N + BLK_X - 1) / BLK_X,
                  (M + BLK_Y - 1) / BLK_Y);

        size_t shBytes = (size_t)tile * (BLK_Y + PAD) +
                         (size_t)tile * (tile   + PAD);
        shBytes *= sizeof(float);

        spmmTile_sw<<<grid, thr, shBytes>>>(dW, dA, dB, dP,
                                            dMask, M, K, N, tile);
        CHECK_CUDA(cudaGetLastError());
    }

    cudaEventRecord(t1);
    CHECK_CUDA(cudaEventSynchronize(t1));

    double ms = to_ms(t0, t1);
    cudaFree(dMask);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}
