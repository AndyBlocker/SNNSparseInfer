// --------  sparse_mm_pipeline.cu  (2025‑06‑23 • final) -------------
#include "../../include/kernels.h"
#include "../../include/common.h"
#include <cuda_runtime.h>
#include <assert.h>

/* ------------------------------------------------------------------ */
/*     默认参数 —— 如需其他 shape，可在编译命令行里 -DBLK_X=… 覆盖       */
/* ------------------------------------------------------------------ */
#ifndef BLK_X        /* threads.x  (columns in one CTA)  */
#   define BLK_X 32
#endif
#ifndef BLK_Y        /* threads.y  (rows   in one CTA)  */
#   define BLK_Y 4
#endif
#ifndef VEC_N        /* 每线程一次计算的列向量长度        */
#   define VEC_N 4
#endif
#ifndef PAD          /* shared‑mem 行尾 padding          */
#   define PAD 8
#endif
#ifndef SH_TILE_MAX  /* 支持的最大 tile(K) 大小          */
#   define SH_TILE_MAX 32
#endif

/* ---------------- 对齐辅助 ---------------- */
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
            printf("CUDA‑err %s %d : %s\n",__FILE__,__LINE__,     \
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
                   int  M, int  K, int  N, int  tile)    // tile : K‑tile size
{
    /* ---------------- ① 线程坐标 ---------------- */
    const int m  = blockIdx.y*BLK_Y + threadIdx.y;                  // 输出行
    const int n0 = (blockIdx.x*BLK_X + threadIdx.x) * VEC_N;        // 起始列
    const bool valid_m = (m < M);

    /* ---------------- ② tile 分块 ---------------- */
    const int kTiles = (K + tile - 1) / tile;
    const int nBase  = blockIdx.x * BLK_X * VEC_N;                  // CTA 起始列
    const int cols   = min(BLK_X*VEC_N, N - nBase);

    /* ---------------- ③ shared‑mem 布局 ---------- */
    extern __shared__ float shm_raw[];
    float* shm = reinterpret_cast<float*>(align_up<16>(shm_raw));

    const size_t STR_W = ALIGN4(BLK_Y + PAD);                       // per‑k row stride
    const size_t STR_A = ALIGN4(BLK_X*VEC_N + PAD);                 // per‑k row stride

    const size_t SZ_W  = ALIGN4(SH_TILE_MAX * STR_W);
    const size_t SZ_A  = ALIGN4(SH_TILE_MAX * STR_A);

    float* shW = shm;
    float* shA = reinterpret_cast<float*>(align_up<16>(shW + SZ_W));/* 单缓冲即可 */

    /* ---------------- ④ 累加寄存器 --------------- */
    float4 acc = {0.f,0.f,0.f,0.f};

    /* ---------------- ⑤ λ : 加载一块 ------------- */
    auto load_tile = [&](int kBase, int rows){
        /* ---- 5.1 读 W ---- (列向量，转存到 shared) */
        for(int r = threadIdx.x; r < rows; r += BLK_X){
            float* dst = &shW[r*STR_W + threadIdx.y];
            float  val = valid_m ? __ldg(&W[m + size_t(kBase+r)*M]) : 0.f;
            *dst = val;
        }

        /* ---- 5.2 读 A ---- (同一 row，连续 4 列)     */
        for(int r = threadIdx.y; r < rows; r += BLK_Y){
            const int gRow = kBase + r;               // 全局 K‑index
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

    /* ---------------- ⑥ 主循环 ------------------- */
    for(int kt = 0; kt < kTiles; ++kt){
        const int kBase = kt*tile;
        const int rows  = min(tile, K - kBase);

        /* 6.1 载入 K‑tile 至 shared */
        load_tile(kBase, rows);
        __syncthreads();            // 保证 W / A 块就绪

        /* 6.2 零块快速跳过 */
        unsigned int nz = 0u;
        for(int r = threadIdx.y; r < rows; r += BLK_Y){
            float4 v = *reinterpret_cast<float4*>(
                          &shA[r*STR_A + threadIdx.x*VEC_N]);
            nz |= __float_as_uint(v.x)|__float_as_uint(v.y)|
                  __float_as_uint(v.z)|__float_as_uint(v.w);
        }
        bool hasData = __syncthreads_or(nz != 0);

        /* 6.3 计算 */
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
        __syncthreads();            // 保护 shW / shA
    }

    /* ---------------- ⑦ 写回 --------------------- */
    if(valid_m){
        const size_t base = m + size_t(n0)*M;          // column‑major
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

    const size_t shmFloats = SZ_W + SZ_A + 4;   // 单缓冲
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
