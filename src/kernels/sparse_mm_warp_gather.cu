#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda/pipeline>
#include <mma.h>
using namespace nvcuda;  

#define WARP_SIZE   32
#define WARPS_CTA    4                
#define THREADS_CTA (WARPS_CTA * WARP_SIZE)
#define COLS_PER_CTA WARPS_CTA
#define FULL_MASK   0xffffffffu

#define PREFETCH_DIST   WARP_SIZE   /* 单位:scalar，选 32 足以隐藏 HBM→L1 延迟      */
#define VEC_WIDTH       4           /* 128‑bit vector 化读取 A                      */
using  A_vec_t = float4;            /* 方便改成 int4 / uint4 做 pack‑int8 时复用     */

// W matrix tiling parameters for coalesced access
#define TILE_ROWS      32           /* Tile height matches warp size               */
#define TILE_K_WIDTH   4            /* 4 consecutive K values per tile             */
#define SMEM_W_SIZE    (TILE_ROWS * TILE_K_WIDTH * WARPS_CTA)  /* Per-CTA SMEM size */

// Async pipeline parameters  
#define PIPELINE_STAGES 2           /* Double buffering: current + next            */
#define SMEM_TOTAL_SIZE (SMEM_W_SIZE * PIPELINE_STAGES)
#define BYTES_PER_LDG   16          /* cp.async.ca.shared.global [addr], [addr], 16 */

#define CHECK_CUDA(x)  do{ cudaError_t e=(x); if(e!=cudaSuccess){\
    fprintf(stderr,"CUDA error %s:%d : %s\n",__FILE__,__LINE__,\
            cudaGetErrorString(e)); exit(EXIT_FAILURE);} }while(0)

static inline double to_ms(cudaEvent_t s, cudaEvent_t e){
    float ms=0; cudaEventElapsedTime(&ms,s,e); return (double)ms;
}

__global__ __launch_bounds__(THREADS_CTA, 2)
void spmm_rowGather(const float* __restrict__ W,
                    const float* __restrict__ A,
                    const float* __restrict__ B,
                          float* __restrict__ P,
                    int M,int K,int N)
{
    const int lane      = threadIdx.x & 31; 
    const int warpInCTA = threadIdx.x >> 5;     
    const int col       = blockIdx.x * COLS_PER_CTA + warpInCTA;
    if (col >= N) return;

    const int warpRowGrp = blockIdx.y * WARPS_CTA + warpInCTA;
    const int row        = warpRowGrp * WARP_SIZE + lane;   
    const bool valid_row = (row < M);

    float  acc = 0.f;

    const float* __restrict__ A_col = A + (size_t)col * K;

    float a_reg  = 0.f;  
    float a_next = 0.f;  
    if (lane < K)                 
        a_reg = __ldg(&A_col[lane]);

    const int kTiles = (K + WARP_SIZE - 1) >> 5; 
    for (int t = 0; t < kTiles; ++t)
    {
        const int kBase = t * WARP_SIZE;

        {
            const int kPref = kBase + PREFETCH_DIST + lane;
            a_next = (kPref < K) ? __ldg(&A_col[kPref]) : 0.f;
        }

        unsigned mask = __ballot_sync(FULL_MASK, a_reg != 0.f);
        while (mask)
        {
            int nzLane = __ffs(mask) - 1;
            mask &= mask - 1;

            float a_val = __shfl_sync(FULL_MASK, a_reg, nzLane);
            int kIdx = kBase + nzLane;

            if (valid_row)
            {
                float w = __ldg(&W[row + (size_t)kIdx * M]);
                acc += w * a_val;
            }
        }

        a_reg = a_next; 
    }

    if (valid_row)
    {
        size_t off = row + (size_t)col * M;
        P[off] = acc + __ldg(&B[off]);
    }
}


double runWarpGatherSparse(const float *dW, const float *dA,
                           const float *dB,       float *dP,
                           int M, int K, int N, int /*tile_unused*/)
{
    dim3 thr(THREADS_CTA, 1, 1);
    const int rowsPerCTA = WARPS_CTA * WARP_SIZE; 
    dim3 grid( (N + COLS_PER_CTA - 1) / COLS_PER_CTA,
               (M + rowsPerCTA - 1) / rowsPerCTA );

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    spmm_rowGather<<<grid, thr>>>(
        dW, dA, dB, dP, M, K, N);
    cudaEventRecord(t1);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(t1));
    double ms = to_ms(t0,t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}