// --------  sparse_mm_pipeline.cu  (20250623  final) -------------
#include "../../include/kernels.h"
#include "../../include/common.h"
#include <cuda_runtime.h>
#include <assert.h>

/* ------------------------------------------------------------------ */
/*        shape -DBLK_X=        */
/* ------------------------------------------------------------------ */
#ifndef BLK_X        /* threads.x  (columns in one CTA)  */
#   define BLK_X 32
#endif
#ifndef BLK_Y        /* threads.y  (rows   in one CTA)  */
#   define BLK_Y 4
#endif
#ifndef VEC_N        /*         */
#   define VEC_N 4
#endif
#ifndef PAD          /* sharedmem  padding          */
#   define PAD 8
#endif
#ifndef SH_TILE_MAX  /*  tile(K)           */
#   define SH_TILE_MAX 32
#endif

/* ----------------  ---------------- */
#define ALIGN4(x)   (((x)+3)&~3u)            /* 4 floats = 16 B */
template<int BYTES>
__device__ __forceinline__ void* align_up(void* p){
    return reinterpret_cast<void*>(
        ((uintptr_t)p + (BYTES-1)) & ~(BYTES-1)
    );
}

/* ---------------- CHECK_CUDA ---------------- */
#ifndef CHECK_CUDA
#   define CHECK_CUDA(x)  do{ cudaError_t _e=(x);                 \
        if(_e!=cudaSuccess){                                      \
            printf("CUDAerr %s %d : %s\n",__FILE__,__LINE__,     \
                   cudaGetErrorString(_e));                       \
            asm("trap;"); } }while(0)
#endif

/* ================================================================== */
/*                        PIPELINE sparse kernel                       */
/* ================================================================== */
__launch_bounds__(128,2)
__global__
void spmm_pipeline(const float* __restrict__ W,
                   const float* __restrict__ A,
                   const float* __restrict__ B,
                   float*       __restrict__ P,
                   int  M, int  K, int  N, int  tile)    // tile : Ktile size
{
    /* ----------------   ---------------- */
    const int m  = blockIdx.y*BLK_Y + threadIdx.y;                  // 
    const int n0 = (blockIdx.x*BLK_X + threadIdx.x) * VEC_N;        // 
    const bool valid_m = (m < M);

    /* ----------------  tile  ---------------- */
    const int kTiles = (K + tile - 1) / tile;
    const int nBase  = blockIdx.x * BLK_X * VEC_N;                  // CTA 
    const int cols   = min(BLK_X*VEC_N, N - nBase);

    /* ----------------  sharedmem  ---------- */
    extern __shared__ float shm_raw[];
    float* shm = reinterpret_cast<float*>(align_up<16>(shm_raw));

    const size_t STR_W = ALIGN4(BLK_Y + PAD);                       // perk row stride
    const size_t STR_A = ALIGN4(BLK_X*VEC_N + PAD);                 // perk row stride

    const size_t SZ_W  = ALIGN4(SH_TILE_MAX * STR_W);
    const size_t SZ_A  = ALIGN4(SH_TILE_MAX * STR_A);

    float* shW = shm;
    float* shA = reinterpret_cast<float*>(align_up<16>(shW + SZ_W));/*  */

    /* ----------------   --------------- */
    float4 acc = {0.f,0.f,0.f,0.f};

    /* ----------------   :  ------------- */
    auto load_tile = [&](int kBase, int rows){
        /* ---- 5.1  W ---- ( shared) */
        for(int r = threadIdx.x; r < rows; r += BLK_X){
            float* dst = &shW[r*STR_W + threadIdx.y];
            float  val = valid_m ? __ldg(&W[m + size_t(kBase+r)*M]) : 0.f;
            *dst = val;
        }

        /* ---- 5.2  A ---- ( row 4 )     */
        for(int r = threadIdx.y; r < rows; r += BLK_Y){
            const int gRow = kBase + r;               //  Kindex
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

    /* ----------------   ------------------- */
    for(int kt = 0; kt < kTiles; ++kt){
        const int kBase = kt*tile;
        const int rows  = min(tile, K - kBase);

        /* 6.1  Ktile  shared */
        load_tile(kBase, rows);
        __syncthreads();            //  W / A 

        /* 6.2  */
        unsigned int nz = 0u;
        for(int r = threadIdx.y; r < rows; r += BLK_Y){
            float4 v = *reinterpret_cast<float4*>(
                          &shA[r*STR_A + threadIdx.x*VEC_N]);
            nz |= __float_as_uint(v.x)|__float_as_uint(v.y)|
                  __float_as_uint(v.z)|__float_as_uint(v.w);
        }
        bool hasData = __syncthreads_or(nz != 0);

        /* 6.3  */
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
        __syncthreads();            //  shW / shA
    }

    /* ----------------   --------------------- */
    if(valid_m){
        const size_t base = m + size_t(n0)*M;          // columnmajor
        if((n0+0) < N) P[base         ] = acc.x + B[base         ];
        if((n0+1) < N) P[base +   M   ] = acc.y + B[base +   M   ];
        if((n0+2) < N) P[base + 2ul*M ] = acc.z + B[base + 2ul*M ];
        if((n0+3) < N) P[base + 3ul*M ] = acc.w + B[base + 3ul*M ];
    }
}

/* ================================================================== */
/*   Host helper : run the pipeline kernel                            */
/* ================================================================== */
double runPipelineSparse(const float* dW,const float* dA,
                         const float* dB,      float* dP,
                         int M,int K,int N,int tile=SH_TILE_MAX)
{
    dim3 grid((N + BLK_X*VEC_N - 1) / (BLK_X*VEC_N),
              (M + BLK_Y        - 1) /  BLK_Y);

    const size_t STR_W   = ALIGN4(BLK_Y + PAD);
    const size_t STR_A   = ALIGN4(BLK_X*VEC_N + PAD);
    const size_t SZ_W    = ALIGN4(SH_TILE_MAX * STR_W);
    const size_t SZ_A    = ALIGN4(SH_TILE_MAX * STR_A);

    const size_t shmFloats = SZ_W + SZ_A + 4;   // 
    const size_t shmBytes  = shmFloats * sizeof(float);

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
// -------------------  END OF FILE  ------------------------------
