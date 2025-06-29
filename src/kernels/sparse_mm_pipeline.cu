/**
 * @file sparse_mm_pipeline.cu
 * @brief Pipeline-based sparse matrix multiplication with vectorized processing
 * 
 * Features:
 * - Vectorized 4-element processing per thread for improved memory bandwidth
 * - Shared memory optimization with zero-tile skipping
 * - Lambda-based tile loading for code reuse
 * - Pipeline design for overlapping computation and data movement
 */

#include "../../include/kernels.h"
#include "../../include/common.h"
#include <cuda_runtime.h>
#include <assert.h>

// block and vectorization parameters
#ifndef BLK_X        // threads.x (columns in one CTA)
#   define BLK_X 32
#endif
#ifndef BLK_Y        // threads.y (rows in one CTA)
#   define BLK_Y 4
#endif
#ifndef VEC_N        // vector width per thread (4 columns)
#   define VEC_N 4
#endif
#ifndef PAD          // shared memory row padding to avoid bank conflicts
#   define PAD 8
#endif
#ifndef SH_TILE_MAX  // maximum supported tile size in K dimension
#   define SH_TILE_MAX 32
#endif

// alignment utility macros
#define ALIGN4(x)   (((x)+3)&~3u)            // align to 4 floats = 16 bytes
template<int BYTES>
__device__ __forceinline__ void* align_up(void* p){
    return reinterpret_cast<void*>(
        ((uintptr_t)p + (BYTES-1)) & ~(BYTES-1)
    );
}

// error checking macro
#ifndef CHECK_CUDA
#   define CHECK_CUDA(x)  do{ cudaError_t _e=(x);                 \
        if(_e!=cudaSuccess){                                      \
            printf("CUDA error %s %d : %s\n",__FILE__,__LINE__,     \
                   cudaGetErrorString(_e));                       \
            asm("trap;"); } }while(0)
#endif

/**
 * @brief Pipeline-based sparse matrix multiplication kernel
 * 
 * Processes matrices in tiles with vectorized 4-element computation per thread.
 * Uses shared memory for W and A matrices with optimized loading patterns.
 */
__launch_bounds__(128,2)
__global__
void spmm_pipeline(const float* __restrict__ W,
                   const float* __restrict__ A,
                   const float* __restrict__ B,
                   float*       __restrict__ P,
                   int  M, int  K, int  N, int  tile)    // tile: K-tile size
{
    // calc thread-related indexes
    const int m  = blockIdx.y*BLK_Y + threadIdx.y;                  // output row
    const int n0 = (blockIdx.x*BLK_X + threadIdx.x) * VEC_N;        // starting output column
    const bool valid_m = (m < M);

    // calc tile dimensions
    const int kTiles = (K + tile - 1) / tile;
    const int nBase  = blockIdx.x * BLK_X * VEC_N;                  // CTA column base
    const int cols   = min(BLK_X*VEC_N, N - nBase);

    // shared memory allocation
    extern __shared__ float shm_raw[];
    float* shm = reinterpret_cast<float*>(align_up<16>(shm_raw));

    const size_t STR_W = ALIGN4(BLK_Y + PAD);                       // per-k row stride for W
    const size_t STR_A = ALIGN4(BLK_X*VEC_N + PAD);                 // per-k row stride for A

    const size_t SZ_W  = ALIGN4(SH_TILE_MAX * STR_W);
    const size_t SZ_A  = ALIGN4(SH_TILE_MAX * STR_A);

    float* shW = shm;
    float* shA = reinterpret_cast<float*>(align_up<16>(shW + SZ_W)); // single buffer

    // initialize accumulator
    float4 acc = {0.f,0.f,0.f,0.f};

    // lambda function for loading a tile
    auto load_tile = [&](int kBase, int rows){
        // load W matrix (column vectors, transpose to row layout in shared memory)
        for(int r = threadIdx.x; r < rows; r += BLK_X){
            float* dst = &shW[r*STR_W + threadIdx.y];
            float  val = valid_m ? __ldg(&W[m + size_t(kBase+r)*M]) : 0.f;
            *dst = val;
        }

        // load A matrix (same row, consecutive 4 columns)
        for(int r = threadIdx.y; r < rows; r += BLK_Y){
            const int gRow = kBase + r;               // global K-index
            float*   dst   = &shA[r*STR_A + threadIdx.x*VEC_N];

        #pragma unroll
            for(int j = 0; j < VEC_N; ++j){
                const int gCol = nBase + threadIdx.x*VEC_N + j;
                float v = (gCol < N)
                         ? __ldg(&A[gRow + size_t(gCol)*K])
                         : 0.f;
                dst[j] = v;
            }
        }
    };

    // main computation loop
    for(int kt = 0; kt < kTiles; ++kt){
        const int kBase = kt*tile;
        const int rows  = min(tile, K - kBase);

        // load K-tile to shared memory
        load_tile(kBase, rows);
        __syncthreads();            // ensure W/A blocks are ready

        // fast zero-tile skipping
        unsigned int nz = 0u;
        for(int r = threadIdx.y; r < rows; r += BLK_Y){
            float4 v = *reinterpret_cast<float4*>(
                          &shA[r*STR_A + threadIdx.x*VEC_N]);
            nz |= __float_as_uint(v.x)|__float_as_uint(v.y)|
                  __float_as_uint(v.z)|__float_as_uint(v.w);
        }
        bool hasData = __syncthreads_or(nz != 0);

        // compute matrix multiplication
        if(hasData){
            const bool c0 = (n0+0) < N;
            const bool c1 = (n0+1) < N;
            const bool c2 = (n0+2) < N;
            const bool c3 = (n0+3) < N;

            #pragma unroll 4
            for(int k = 0; k < rows; ++k){
                float  wv = shW[k*STR_W + threadIdx.y];
                const float4 av = *reinterpret_cast<float4*>(
                                    &shA[k*STR_A + threadIdx.x*VEC_N]);

                if(c0) acc.x = __fmaf_rn(wv, av.x, acc.x);
                if(c1) acc.y = __fmaf_rn(wv, av.y, acc.y);
                if(c2) acc.z = __fmaf_rn(wv, av.z, acc.z);
                if(c3) acc.w = __fmaf_rn(wv, av.w, acc.w);
            }
        }
        __syncthreads();            // protect shW/shA
    }

    // write back result
    if(valid_m){
        const size_t base = m + size_t(n0)*M;          // column-major layout
        if((n0+0) < N) P[base         ] = acc.x + B[base         ];
        if((n0+1) < N) P[base +   M   ] = acc.y + B[base +   M   ];
        if((n0+2) < N) P[base + 2ul*M ] = acc.z + B[base + 2ul*M ];
        if((n0+3) < N) P[base + 3ul*M ] = acc.w + B[base + 3ul*M ];
    }
}

/**
 * @brief Host wrapper function for pipeline sparse matrix multiplication
 * 
 * Configures shared memory requirements and launches the pipeline kernel
 * with optimal grid dimensions for the given problem size.
 */
double runPipelineSparse(const float* dW,const float* dA,
                         const float* dB,      float* dP,
                         int M,int K,int N,int tile=SH_TILE_MAX)
{
    // calc grid dimensions
    dim3 grid((N + BLK_X*VEC_N - 1) / (BLK_X*VEC_N),
              (M + BLK_Y        - 1) /  BLK_Y);

    // calc shared memory requirements
    const size_t STR_W   = ALIGN4(BLK_Y + PAD);
    const size_t STR_A   = ALIGN4(BLK_X*VEC_N + PAD);
    const size_t SZ_W    = ALIGN4(SH_TILE_MAX * STR_W);
    const size_t SZ_A    = ALIGN4(SH_TILE_MAX * STR_A);

    const size_t shmFloats = SZ_W + SZ_A + 4;   // single buffer
    const size_t shmBytes  = shmFloats * sizeof(float);

    // launch kernel and measure timing
    cudaEvent_t s0,s1;
    CHECK_CUDA(cudaEventCreate(&s0));
    CHECK_CUDA(cudaEventCreate(&s1));
    CHECK_CUDA(cudaEventRecord(s0));

    spmm_pipeline<<<grid, dim3(BLK_X,BLK_Y), shmBytes>>>(
        dW,dA,dB,dP, M,K,N,tile );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(s1));
    CHECK_CUDA(cudaEventSynchronize(s1));

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, s0, s1));
    CHECK_CUDA(cudaEventDestroy(s0));
    CHECK_CUDA(cudaEventDestroy(s1));
    return static_cast<double>(ms);
}