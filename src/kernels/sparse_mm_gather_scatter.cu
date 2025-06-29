/**********************************************************************
*  sparse_mm_gather_scatter.cu
*  Dense (row major) W [MK]    BlockCSR Sparse A [KN]  +  B [MN]
*   P [MN] (row major)
*
*   (Ampere+Tensor Core TF32)
*    nvcc -O3 -arch=sm_80 -DUSE_TENSOR_CORE \
*         -Xptxas -v -maxrregcount=128 \
*         sparse_mm_gather_scatter.cu -o spmm_gs
**********************************************************************/
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>

/* ======== TUNEABLE MACROS ======== */
#ifndef NB
#   define NB  16          /* block edge16/32/64 <=32            */
#endif
#ifndef M_TILE
#   define M_TILE 128      /*  / CTA128  A/B100 SMEM  */
#endif
#ifndef PAD
#   define PAD  1          /* sharedmem row padding to kill bankconflict         */
#endif

/* ======== CHECK CUDA ======== */
#define CHECK_CUDA(x)                                            \
do { cudaError_t _e=(x);                                         \
     if(_e!=cudaSuccess){                                        \
         fprintf(stderr,"CUDA error %s:%d : %s\n",               \
                 __FILE__,__LINE__,cudaGetErrorString(_e));      \
         exit(EXIT_FAILURE);} } while(0)

/* ======== util ======== */
static inline double to_ms(cudaEvent_t s,cudaEvent_t e)
{ float ms=0.f;  cudaEventElapsedTime(&ms,s,e); return double(ms); }

/*  */
#define IS_ALIGNED_16(ptr) (((uintptr_t)(ptr) & 0xF) == 0)

/*  -  */
template<typename T>
__device__ __forceinline__
void global_async_copy_var(T* smem, const T* gmem, int bytes)
{
#if CP_ASYNC
    constexpr int STEP16 = 16 / sizeof(T);   // 4  float
    int idx = threadIdx.x * STEP16;          //  16B 

    if (IS_ALIGNED_16(gmem)) {
        /* ===== 16byte fast path ===== */
        for (int i = idx; i < bytes / sizeof(T); i += blockDim.x * STEP16) {
            asm volatile(
                "{ .reg .u64 smp, gptr;"
                "  cvta.to.shared.u64 smp, %0;"
                "  cvta.to.global.u64 gptr, %1;"
                "  cp.async.ca.shared.global [smp], [gptr], 16; }"
                :: "l"(smem + i),
                   "l"(gmem + i));
        }
    } else {
        /* ===== 4byte safe path ===== */
        for (int i = threadIdx.x; i < bytes / sizeof(T); i += blockDim.x) {
            asm volatile(
                "{ .reg .u64 smp, gptr;"
                "  cvta.to.shared.u64 smp, %0;"
                "  cvta.to.global.u64 gptr, %1;"
                "  cp.async.shared.global [smp], [gptr], 4; }"
                :: "l"(smem + i),
                   "l"(gmem + i));
        }
    }
#else
    /* plain fallback copy */
    for (int i = threadIdx.x; i < bytes / sizeof(T); i += blockDim.x)
        smem[i] = gmem[i];
#endif
}

/* ------------------------------------------------------------------ *
 *                   1.  BlockCSR                             *
 * ------------------------------------------------------------------ */
struct BCSR
{
    int    nBlockRows;          /* = ceil(K/NB)                       */
    int    nnzb;                /*                          */
    int   *rowPtr;              /* [nBlockRows+1]                     */
    int   *colIdx;              /* [nnzb]                             */
    float *vals;                /* [nnzb * NB * NB]  rowmajor dense  */
};

/*    constant symbol buildkernel  */
__device__ int d_gNNZB;

/* ------------------------------------------------------------------ *
 *                   2.  Kernel0  :  buildBCSR                        *
 * ------------------------------------------------------------------ */
__device__ __forceinline__
void copyDenseTile(const float *__restrict__ src,
                         float *__restrict__ dst,
                         int ldm, int tile)        /* ldm = K stride */
{
    /* 32  (1 warp)  tiletile float 128bit  */
    int lane = threadIdx.x;
    for (int i=lane;i<tile*tile;i+=32)
    {
        int r = i / tile;
        int c = i % tile;
        if((ldm & 3) == 0 && (c & 3) == 0) {        /* 16B aligned fastpath */
            const float4* p4 = reinterpret_cast<const float4*>(src + r*ldm + c);
            float4 v = *p4;
            float4* q4 = reinterpret_cast<float4*>(dst + r*tile + c);
            *q4 = v;
        } else {                                    /* misaligned fallback */
            dst[r*tile + c] = src[r*ldm + c];
        }
    }
}

__global__ void buildBCSR(const float *__restrict__ A,
                          BCSR bcsr,
                          int K, int N, int tile)
{
    const int tile2 = tile*tile;
    const int kblk = blockIdx.x;                   /* one Ktile / CTA */
    const int nTiles = (N + tile - 1)/tile;

    __shared__ unsigned long long nzMask;          /* up to 64 cols/row */
    if(threadIdx.x==0) nzMask = 0ull;
    __syncthreads();

    /* --- step1 :  coltile --- */
    int nblk = threadIdx.x;   /* 031 */
    if (nblk < nTiles)
    {
        int k0 = kblk*tile, kEnd=min(k0+tile,K);
        int n0 = nblk*tile, nEnd=min(n0+tile,N);
        bool has = false;
        for(int k=k0; k<kEnd && !has; ++k)
            for(int n=n0; n<nEnd; ++n)
                if (A[k + size_t(n)*K] != 0.f) { has = true; break; }
        if (has) atomicOr(&nzMask, 1ull<<nblk);
    }
    __syncthreads();

    /* --- step2 :  nnzb prefix idx/vals --- */
    if(threadIdx.x==0)
    {
        unsigned long long bits = nzMask;
        int base = atomicAdd(&d_gNNZB, __popcll(bits));
        bcsr.rowPtr[kblk] = base;

        int loc=0;
        while(bits)
        {
            int nb = __ffsll(bits)-1;    /* lowest set */
            bcsr.colIdx[base+loc] = nb;

            const float *src = A + kblk*tile + size_t(nb*tile)*K;
            float *dst = bcsr.vals + (base+loc)*tile2;
            copyDenseTile(src,dst,K,tile);

            bits &= bits-1; ++loc;
        }
    }
    /* rowPtr[nBlockRows]  host  exclusivescan  */
}

/* ------------------------------------------------------------------ *
 *                   3.  Kernel1  :  spmmBCSR                         *
 * ------------------------------------------------------------------ */
#if defined(USE_TENSOR_CORE) && (__CUDA_ARCH__>=800)
#include <mma.h>
using namespace nvcuda;
#endif

struct Task { int kblk; int beg; int end; };

__device__ Task  *g_tasks;
__device__ int    g_head;

#if __CUDA_ARCH__>=800
#define CP_ASYNC 1
#else
#define CP_ASYNC 0
#endif

template<int TILE_M>
__launch_bounds__(256,2)            /* 256 threads / CTA ,  2 CTA / SM */
__global__ void spmmBCSR(const float *__restrict__ W,
                         const BCSR bcsr,
                         const float *__restrict__ B,
                               float *__restrict__ P,
                         int  M, int K, int N, int tile, int nTasks)
{
    /* -------------------- sharedmem layout -------------------- *
     *   Wbuf[2] : (TILE_M  tile) ,  Ablock[2] : (tile  tile)
     * ----------------------------------------------------------- */
    extern __shared__ float sh[];               /*  */
    const int WBUF_ELE = TILE_M * tile;
    const int ABUF_ELE = tile * tile;
    float *shW[2] = { sh,                     sh +   WBUF_ELE };
    float *shA[2] = { sh + 2*WBUF_ELE,        sh + 2*WBUF_ELE + ABUF_ELE };

    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;        /* 0  7 (for 256 thr) */

    /* persistentCTA  */
    for(;;)
    {
        int tid = (threadIdx.x==0)? atomicAdd(&g_head,1):0;
        tid = __shfl_sync(0xffffffff,tid,0);
        if (tid >= nTasks) break;

        Task t = g_tasks[tid];
        const int kblk = t.kblk;
        int nbBeg=t.beg, nbEnd=t.end;
        const int k0 = kblk*tile;

        /*  id == blockIdx.y */
        const int mBase = blockIdx.y * TILE_M;
        const bool validRowBlk = mBase < M;
        const int rowsThisCTA = min(TILE_M, M - mBase);

        /* === 1.  W stripe () === */
        if(validRowBlk)
        {
            global_async_copy_var(shW[0],
                W + mBase + size_t(k0)*M, rowsThisCTA * tile * sizeof(float));
#if CP_ASYNC
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
#else
            __syncthreads();
#endif
        }

        int buf=0;
        /* === 2.  Ablock === */
        if (nbBeg < nbEnd) {
            global_async_copy_var(shA[buf],
                bcsr.vals + nbBeg*tile*tile, tile * tile * sizeof(float));
#if CP_ASYNC
            asm volatile("cp.async.commit_group;");
#endif
        }

        /* === 3.  Ablock === */
        for(int off=nbBeg; off<nbEnd; ++off)
        {
            /* 3.1  Ablock */
            int nxt = buf^1;
            if (off+1 < nbEnd) {
                global_async_copy_var(shA[nxt],
                    bcsr.vals + (off+1)*tile*tile, tile * tile * sizeof(float));
#if CP_ASYNC
                asm volatile("cp.async.commit_group;");
#endif
            }

            /*  Ablock index / col offset */
            int nblk = bcsr.colIdx[off];

            /* 3.2  shA[buf]  */
#if CP_ASYNC
            asm volatile("cp.async.wait_group 0;");
#else
            __syncthreads();
#endif

            /* 2.3    warp  16tile (Rowmajor) */
#if defined(USE_TENSOR_CORE) && (__CUDA_ARCH__>=800)
            if (tile == 16) {  /* Tensor Core  16x16 */
                const int warpRow = warpId*16;
                if (warpRow < rowsThisCTA)
                {
                    /* load fragments */
                    wmma::fragment<wmma::matrix_a,16,16,16,
                                   wmma::precision::tf32,wmma::row_major>  a_frag;
                    wmma::fragment<wmma::matrix_b,16,16,16,
                                   wmma::precision::tf32,wmma::row_major>  b_frag;
                    wmma::fragment<wmma::accumulator,16,16,16,float>      c_frag;
                    wmma::fill_fragment(c_frag, 0.0f);

                    wmma::load_matrix_sync(a_frag,
                                           shW[0]+warpRow*tile, tile);
                    wmma::load_matrix_sync(b_frag,
                                           shA[buf], tile);
                    wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);

                    /* store rowmajor : ldm = N */
                    int rowOut = mBase + warpRow;
                    int colOut = nblk*tile;
                    float *Cptr = P + rowOut*N + colOut;
                    wmma::store_matrix_sync(Cptr, c_frag, N, wmma::mem_row_major);

                    /* bias 32  */
                    const float *Bptr = B + rowOut*N + colOut;
                    for(int i=lane;i<16*tile;i+=32)
                        Cptr[i] += Bptr[i];
                }
            } else
#endif
            {   /* =====  Soft GEMM  ===== */
                const int rowLocal = (warpId<<4) + (lane>>3); 
                const int colLocal = (lane & 7);
                const bool valid = (rowLocal < rowsThisCTA);
                
                /*  -  8  */
                const int maxCols = min(8, tile - colLocal);
                float acc[8] = {0.f};

                for(int kk=0; kk<tile; ++kk)
                {
                    float a = valid? shW[0][rowLocal*tile+kk] : 0.f;
                    for(int jj=0; jj<maxCols; ++jj)
                        if(colLocal+jj < tile)
                            acc[jj] += a * shA[buf][kk*tile + (colLocal+jj)];
                }
                if(valid)
                {
                    int outRow = mBase + rowLocal;
                    int outCol = nblk*tile + colLocal;
                    float *out = P + outRow*N + outCol;
                    for(int jj=0; jj<maxCols; ++jj)
                        if(colLocal+jj < tile && outCol+jj < N)
                            out[jj] = acc[jj] + B[outRow*N + outCol+jj];
                }
            }
            buf ^=1;
#if !CP_ASYNC
            __syncthreads();
#endif
        } /* for off */
#if CP_ASYNC
        asm volatile("cp.async.wait_group 0;");
#endif
    } /* persistent loop */
}

/* ------------------------------------------------------------------ *
 *                   4.  Host wrapper  : runGatherScatterSparse        *
 * ------------------------------------------------------------------ */
double runGatherScatterSparse(const float *dW,const float *dA,
                              const float *dB,      float *dP,
                              int M,int K,int N,int tile/*==NB*/)
{
    /* ============ 0. build BCSR ============ */
    BCSR bcsr;  bcsr.nBlockRows = (K + tile - 1)/tile;
    int maxNNZ = bcsr.nBlockRows * ((N+tile-1)/tile);   /*  */
    CHECK_CUDA(cudaMalloc(&bcsr.rowPtr,(bcsr.nBlockRows+1)*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&bcsr.colIdx,maxNNZ*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&bcsr.vals,  maxNNZ*tile*tile*sizeof(float)));
    int zero = 0;
    CHECK_CUDA(cudaMemcpyToSymbol(d_gNNZB, &zero, sizeof(int)));

    dim3 gridBuild(bcsr.nBlockRows);
    buildBCSR<<<gridBuild,32>>>(dA,bcsr,K,N,tile);
    CHECK_CUDA(cudaGetLastError());

    /*  device  nnzb rowPtr[nBlockRows] */
    int nnzb;
    CHECK_CUDA(cudaMemcpyFromSymbol(&nnzb, d_gNNZB, sizeof(int)));
    bcsr.nnzb = nnzb;
    CHECK_CUDA(cudaMemcpy(bcsr.rowPtr + bcsr.nBlockRows,
                          &nnzb, sizeof(int), cudaMemcpyHostToDevice));

    /* ============ 1.  Task  ============ */
    const int nTasks = bcsr.nBlockRows;           /*  kblk  */
    Task *hTask = new Task[nTasks];
    
    /*  rowPtr  host  task */
    int *hRowPtr = new int[bcsr.nBlockRows+1];
    CHECK_CUDA(cudaMemcpy(hRowPtr, bcsr.rowPtr, 
                          (bcsr.nBlockRows+1)*sizeof(int), cudaMemcpyDeviceToHost));
    for(int k=0;k<nTasks;++k)
        hTask[k] = {k, hRowPtr[k], hRowPtr[k+1]};
    delete[] hRowPtr;
    Task *dTask; CHECK_CUDA(cudaMalloc(&dTask,nTasks*sizeof(Task)));
    CHECK_CUDA(cudaMemcpy(dTask,hTask,nTasks*sizeof(Task),
                          cudaMemcpyHostToDevice));
    delete[] hTask;
    CHECK_CUDA(cudaMemcpyToSymbol(g_tasks,&dTask,sizeof(Task*)));

    CHECK_CUDA(cudaMemcpyToSymbol(g_head,&zero,sizeof(int)));

    /* ============ 2. launch spmmBCSR ============ */
    dim3 grid(nTasks, (M+M_TILE-1)/M_TILE);
    size_t shBytes = 2*M_TILE*tile*sizeof(float) + 2*tile*tile*sizeof(float);

    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);

    // spmmBCSR<M_TILE><<<grid,256,shBytes>>>(
    //     dW,bcsr,dB,dP,M,K,N,tile,nTasks);
    CHECK_CUDA(cudaGetLastError());

    cudaEventRecord(e); CHECK_CUDA(cudaEventSynchronize(e));
    double ms = to_ms(s,e);
    cudaEventDestroy(s); cudaEventDestroy(e);

    /* ============ 3. cleanup ============ */
    cudaFree(bcsr.rowPtr); cudaFree(bcsr.colIdx); cudaFree(bcsr.vals);
    cudaFree(dTask);
    return ms;
}
