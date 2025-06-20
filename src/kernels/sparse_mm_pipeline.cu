// --------  sparse_mm_opt_pipeline_fix.cu  (2025‑06‑20) -------------
#include "../../include/kernels.h"
#include "../../include/common.h"
#include <cuda_runtime.h>
#include <assert.h>

/* ---------------- compile‑time helpers ---------------- */
#ifndef CHECK_CUDA
#   define CHECK_CUDA(x)  do{ cudaError_t _e=(x);                 \
        if(_e!=cudaSuccess){                                      \
            printf("CUDA‑err %s %d : %s\n",__FILE__,__LINE__,     \
                   cudaGetErrorString(_e));                       \
            asm("trap;"); } }while(0)
#endif

/* NOTE: CP_ASYNC macros are now defined in common.h */

/* ================================================================= */
/*   PIPELINE sparse kernel (fixed)                                   */
/* ================================================================= */
__launch_bounds__(128,2)
__global__
void spmm_pipeline(const float* __restrict__ W,
                   const float* __restrict__ A,
                   const float* __restrict__ B,
                   float*       __restrict__ P,
                   int  M, int  K, int  N, int  tile)
{
    /* ---------------- ① 线程坐标 ---------------- */
    const int m  = blockIdx.y*BLK_Y + threadIdx.y;            // 输出行
    const int n0 = (blockIdx.x*BLK_X + threadIdx.x)*VEC_N;    // 起始列
    const bool valid_m = (m < M);

    /* ---------------- ② tile 尺寸 ---------------- */
    const int kTiles = (K + tile - 1) / tile;
    const int nTiles = (N + tile - 1) / tile;
    const int nTileId= n0 / tile;           // 本线程块固定的 n‑tile

    /* ---------------- ③ shared‑mem 双缓冲 -------- */
    extern __shared__ float shm[];
    const size_t STR_W = BLK_Y + PAD;                       // W 行 stride
    const size_t STR_A = BLK_X*VEC_N + PAD;                 // A 行 stride
    const size_t SZ_W  = SH_TILE_MAX * STR_W;               // 单缓冲大小
    const size_t SZ_A  = SH_TILE_MAX * STR_A;

    float* shW[2]; float* shA[2];
    shW[0]=shm;             shA[0]=shW[0]+SZ_W;
    shW[1]=shA[0]+SZ_A;     shA[1]=shW[1]+SZ_W;

    /* ---------------- ④ 累加寄存器 --------------- */
    float4 acc = {0.f,0.f,0.f,0.f};

    /* ---------------- ⑤ λ : mask  ---------------- */
    auto has_nz = [&](float* buf,int rows,int cols)->bool{
        bool nz=false;
        for(int r=threadIdx.y; r<rows; r+=BLK_Y){
            for(int c=threadIdx.x*VEC_N; c<cols; c+=BLK_X*VEC_N){
                float4 v = *reinterpret_cast<float4*>(&buf[r*STR_A + c]);
                nz |= (v.x!=0.f)|(v.y!=0.f)|(v.z!=0.f)|(v.w!=0.f);
            }
        }
        nz = __syncthreads_or(nz);
        return nz;
    };

    /* ---------------- ⑥ λ : async‑load ----------- */
    auto load_tile = [&](int kt,int buf){
        const int kBase = kt*tile;
        const int rows  = min(tile, K - kBase);
        const int nBase = nTileId*tile;
        const int cols  = min(tile, N - nBase);

        /* ---- 6.1 copy W ---- */
        for(int r=threadIdx.x; r<rows; r+=BLK_X){
            float* dst=&shW[buf][r*STR_W + threadIdx.y];
            float  val = valid_m ? __ldg(&W[m + size_t(kBase+r)*M]) : 0.f;
            *dst = val;
        }

    #if defined(USE_CP_ASYNC) && (__CUDA_ARCH__ >= 800)
        /* ---- 6.2 copy A with cp.async & 3‑group pipeline ---- */
        int reqCnt = 0;
    #endif
        for(int r=threadIdx.y*4; r<rows; r+=BLK_Y*4){
            const int rem = rows - r;
            for(int c=threadIdx.x*VEC_N; c<cols; c+=BLK_X*VEC_N){
                float*       dst0 = &shA[buf][r*STR_A + c];
                const float* src0 = &A[(kBase+r) + size_t(nBase+c)*K];

                /* 每次搬 16B */
                cp_async_cg(dst0, src0);                       /* r+0 */
            #if defined(USE_CP_ASYNC) && (__CUDA_ARCH__ >= 800)
                if(++reqCnt == 16){ cp_async_commit(); reqCnt=0; }
            #endif
                if(rem>1){
                    cp_async_cg(dst0+STR_A, src0+1);          /* r+1 */
                #if defined(USE_CP_ASYNC) && (__CUDA_ARCH__ >= 800)
                    if(++reqCnt == 16){ cp_async_commit(); reqCnt=0; }
                #endif
                }
                if(rem>2){
                    cp_async_cg(dst0+2*STR_A, src0+2);      /* r+2 */
                #if defined(USE_CP_ASYNC) && (__CUDA_ARCH__ >= 800)
                    if(++reqCnt == 16){ cp_async_commit(); reqCnt=0; }
                #endif
                }
                if(rem>3){
                    cp_async_cg(dst0+3*STR_A, src0+3);      /* r+3 */
                #if defined(USE_CP_ASYNC) && (__CUDA_ARCH__ >= 800)
                    if(++reqCnt == 16){ cp_async_commit(); reqCnt=0; }
                #endif
                }
            }
        }
    #if defined(USE_CP_ASYNC) && (__CUDA_ARCH__ >= 800)
        cp_async_commit();                  // flush残余请求
    #endif
    };

    /* ---------------- ⑦ λ : GEMM  ---------------- */
    auto compute = [&](int buf,int rows,int cols){
        const bool c0 = (n0+0)<N;
        const bool c1 = (n0+1)<N;
        const bool c2 = (n0+2)<N;
        const bool c3 = (n0+3)<N;

        #pragma unroll 4
        for(int k=0;k<rows;++k){
            float  wv = shW[buf][k*STR_W + threadIdx.y];

            const float4 av = *reinterpret_cast<float4*>(
                               &shA[buf][k*STR_A + threadIdx.x*VEC_N]);

            if(c0) acc.x = __fmaf_rn(wv, av.x, acc.x);
            if(c1) acc.y = __fmaf_rn(wv, av.y, acc.y);
            if(c2) acc.z = __fmaf_rn(wv, av.z, acc.z);
            if(c3) acc.w = __fmaf_rn(wv, av.w, acc.w);
        }
    };

    /* ---------------- ⑧ pipeline main loop ------- */
    int cur=0, nxt=1;
    if(kTiles>0){
        load_tile(0,cur);
        cp_async_wait();  __syncthreads();
    }

    for(int kt=0; kt<kTiles; ++kt){
        const int rows = min(tile, K - kt*tile);
        const int cols = min(tile, N - nTileId*tile);

        if(kt+1<kTiles) load_tile(kt+1, nxt);   // 预取下一 tile

        bool nz = has_nz(shA[cur], rows, cols); // mask 生成

        if(nz) compute(cur, rows, cols);        // 计算

        if(kt+1<kTiles){ cp_async_wait(); __syncthreads(); }

        cur^=1; nxt^=1;                         // 双缓冲翻转
    }

    /* ---------------- ⑨ 写回 --------------------- */
    if(valid_m){
        const size_t base = m + size_t(n0)*M;
        if((n0+0)<N) P[base            ] = acc.x + B[base            ];
        if((n0+1)<N) P[base +     M    ] = acc.y + B[base +     M    ];
        if((n0+2)<N) P[base + 2ul*M    ] = acc.z + B[base + 2ul*M    ];
        if((n0+3)<N) P[base + 3ul*M    ] = acc.w + B[base + 3ul*M    ];
    }
}

/* ================================================================= */
/*   Host helper : run the pipeline kernel                           */
/* ================================================================= */
double runPipelineSparse(const float* dW,const float* dA,
                         const float* dB,      float* dP,
                         int M,int K,int N,int tile)
{
    dim3 grid((N + BLK_X*VEC_N - 1)/(BLK_X*VEC_N),
              (M + BLK_Y       - 1)/(BLK_Y));
    const size_t shmBytes = 2 * ( SH_TILE_MAX*(BLK_Y+PAD) +
                                  SH_TILE_MAX*(BLK_X*VEC_N + PAD) )
                            * sizeof(float);

    cudaEvent_t s0,s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
    cudaEventRecord(s0);
    spmm_pipeline<<<grid, dim3(BLK_X,BLK_Y), shmBytes>>>(
        dW,dA,dB,dP, M,K,N,tile );
    cudaEventRecord(s1); cudaEventSynchronize(s1);

    double ms = to_ms(s0,s1);
    cudaEventDestroy(s0); cudaEventDestroy(s1);
    return ms;
}
// -------------------  END OF FILE  ------------------------------
