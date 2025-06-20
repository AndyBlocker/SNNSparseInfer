// -------- sparse_mm_opt.cu  (pipeline version, 2025‑06‑20) ----------
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <cmath>

/* ===== compile‑time switches  ===== */
#ifndef BLK_X
#   define BLK_X 32        // threads‑x
#endif
#ifndef BLK_Y
#   define BLK_Y 4         // threads‑y  -> 32*4 = 128 thr / block
#endif
#ifndef VEC_N
#   define VEC_N 4         // columns produced per thread
#endif
#ifndef PAD
#   define PAD   4         // SM bank‑conflict padding
#endif
#ifndef SH_TILE_MAX
#   define SH_TILE_MAX 32  // <= 32  (k‑tile & n‑tile upper‑bound)
#endif

/* ---- check assumptions ---- */
static_assert(BLK_X % 32 == 0,            "BLK_X must be warp multiple");
static_assert(VEC_N == 4,                 "code assumes 4‑way vector");
static_assert(SH_TILE_MAX <= 32,          "shared‑mem bound violated");

#ifndef CHECK_CUDA
#   define CHECK_CUDA(call) do{                                        \
        cudaError_t e_=(call);                                         \
        if(e_!=cudaSuccess){                                           \
            std::cerr<<__FILE__<<":"<<__LINE__<<"  "                   \
                     <<cudaGetErrorString(e_)<<std::endl; std::exit(1);}\
    }while(0)
#endif

/* ---------------- host random ---------------- */
__host__ float frand(){
    static thread_local std::mt19937 g{std::random_device{}()};
    static thread_local std::uniform_real_distribution<float> d(-1.f,1.f);
    return d(g);
}

/* ---------------- device helpers ------------- */
__device__ __forceinline__ float4 ld4(const float* __restrict__ g){
    float4 v;
    if(((uintptr_t)g & 0xF)==0)
        v=*reinterpret_cast<const float4*>(g);
    else{
        v.x=__ldg(g+0); v.y=__ldg(g+1);
        v.z=__ldg(g+2); v.w=__ldg(g+3);
    }
    return v;
}

/* optional cp.async intrinsics (sm_80+) */
#if defined(USE_CP_ASYNC) && (__CUDA_ARCH__ >= 800)
#   define CP_ASYNC(dst, src)  asm volatile(                           \
        "cp.async.cg.shared.global [%0], [%1], 16;\n" ::               \
        "r"(dst), "l"(src))
#   define CP_WAIT()          asm volatile("cp.async.wait_group 0;\n")
#   define CP_COMMIT()        asm volatile("cp.async.commit_group;\n")
#else
#   define CP_ASYNC(dst, src)  (*reinterpret_cast<float4*>(dst)=ld4(src))
#   define CP_WAIT()
#   define CP_COMMIT()
#endif

/* ============================================================= */
/*   HUGE kernel : triple‑stream pipeline                         */
/*   stage‑0 : mask‑gen   (detect non‑zero tile)                  */
/*   stage‑1 : async load (cp.async or fall‑back)                 */
/*   stage‑2 : compute                                            */
/* ============================================================= */
__launch_bounds__(128,2)
__global__ void spmm_pipeline(const float* __restrict__ W,
                              const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ P,
                              int M,int K,int N,int tile)
{
    /* ---- coordinates ---- */
    const int m  = blockIdx.y*BLK_Y + threadIdx.y;          // row
    const int n0 = (blockIdx.x*BLK_X + threadIdx.x)*VEC_N;  // first col of thread
    const bool valid_m = (m < M);

    /* ---- tile counts ---- */
    const int kTiles  = (K + tile - 1) / tile;
    const int nTiles  = (N + tile - 1) / tile;
    const int nTileId = n0 / tile;   // fixed for this thread‑block

    /* ---- shared memory: double buffer ---- */
    extern __shared__ float shm[];
    const size_t szW = SH_TILE_MAX*(BLK_Y+PAD);
    const size_t szA = SH_TILE_MAX*(BLK_X*VEC_N + PAD);

    float* shW[2];
    float* shA[2];
    shW[0] = shm;
    shA[0] = shW[0] + szW;
    shW[1] = shA[0] + szA;
    shA[1] = shW[1] + szW;

    /* ---- private accumulators ---- */
    float4 acc; acc.x=acc.y=acc.z=acc.w=0.f;

    /* ---- helper lambdas (device) ---- */
    auto compute_mask = [&](float* aBuf,int rows,int cols)->bool{
        bool nz=false;
        for(int kk=threadIdx.x*VEC_N; kk<rows*cols; kk+=BLK_X*VEC_N){
            if(aBuf[kk]!=0.f){nz=true;break;}
        }
        /* reduction across 128 threads */
        nz = __syncthreads_or(nz);
        return nz;
    };

    auto async_load_tile = [&](int kt,int buf){
        const int kBase = kt*tile;
        const int rows  = min(tile, K - kBase);
        const int nBase = nTileId*tile;
        const int cols  = min(tile, N - nBase);

        /* --- load W (M×K, col‑major) into shW --- */
        for(int kk=threadIdx.x; kk<rows; kk+=BLK_X){
            float* dst=&shW[buf][kk*(BLK_Y+PAD)+threadIdx.y];
            const float val = (valid_m) ? __ldg(&W[m + size_t(kBase+kk)*M]) : 0.f;
            *dst = val;
        }

        /* --- load A (K×N) into shA ---  each iteration loads 16B == 4 fp32 */
        for(int kk=threadIdx.y*4; kk<rows; kk+=BLK_Y*4){
            const int remain = rows - kk;
            for(int nn=threadIdx.x*VEC_N; nn<cols; nn+=BLK_X*VEC_N){
                float* dst=&shA[buf][(kk+0)*(BLK_X*VEC_N+PAD)+nn];
                const float* src=&A[(kBase+kk) + size_t(nBase+nn)*K];
                /* vector copy 16B */
                CP_ASYNC(dst, src);
                if(remain>1) CP_ASYNC(dst + (BLK_X*VEC_N+PAD), src + K);
                if(remain>2) CP_ASYNC(dst + 2*(BLK_X*VEC_N+PAD), src + 2*K);
                if(remain>3) CP_ASYNC(dst + 3*(BLK_X*VEC_N+PAD), src + 3*K);
            }
        }
        CP_COMMIT();
    };

    auto compute_tile = [&](int buf,int rows,int cols){
        #pragma unroll 4
        for(int kk=0; kk<rows; ++kk){
            const float wv = shW[buf][kk*(BLK_Y+PAD)+threadIdx.y];
            const float4 av =
                *reinterpret_cast<float4*>(&shA[buf][kk*(BLK_X*VEC_N+PAD)
                                                      + threadIdx.x*VEC_N]);
            acc.x = __fmaf_rn(wv, av.x, acc.x);
            acc.y = __fmaf_rn(wv, av.y, acc.y);
            acc.z = __fmaf_rn(wv, av.z, acc.z);
            acc.w = __fmaf_rn(wv, av.w, acc.w);
        }
    };

    /* ============================================================ */
    /*   pipeline : preload tile‑0, then loop                       */
    /* ============================================================ */
    int bufCur = 0, bufNxt = 1;
    /* preload tile‑0 */
    if(kTiles>0){
        async_load_tile(0,bufCur);
        CP_WAIT();           // ensure tile‑0 data visible
        __syncthreads();
    }

    for(int kt=0; kt<kTiles; ++kt){
        const int rows = min(tile, K - kt*tile);
        const int cols = min(tile, N - nTileId*tile);

        /* launch async load for next tile while computing current */
        if(kt+1<kTiles){
            async_load_tile(kt+1, bufNxt);
        }

        /* ----- mask generation stage (logical stream‑0) ----- */
        bool nz = compute_mask(shA[bufCur], rows, cols);
        __syncthreads();   // ensure mask decision visible to all threads

        /* ----- compute stage (stream‑2) ----- */
        if(nz)
            compute_tile(bufCur, rows, cols);

        /* ----- wait until next tile is ready (stream‑1) ------ */
        if(kt+1<kTiles){
            CP_WAIT();
            __syncthreads();
        }

        /* swap buffers */
        bufCur ^=1; bufNxt ^=1;
    }

    /* ---- epilogue : write result ---- */
    if(valid_m){
        const size_t base = m + size_t(n0)*M;
        if(n0+0 < N) P[base]                 = acc.x + B[base];
        if(n0+1 < N) P[base +     M]         = acc.y + B[base +     M];
        if(n0+2 < N) P[base + 2ul*M]         = acc.z + B[base + 2ul*M];
        if(n0+3 < N) P[base + 3ul*M]         = acc.w + B[base + 3ul*M];
    }
}

/* =================== host helpers (unchanged) =================== */
void gen_matrices(std::vector<float>& W,
                  std::vector<float>& A,
                  std::vector<float>& B,
                  int M,int K,int N,int tile,float sparsity)
{
    std::generate(W.begin(),W.end(),frand);
    std::generate(B.begin(),B.end(),frand);

    const int nTiles=(N+tile-1)/tile, kTiles=(K+tile-1)/tile;
    std::bernoulli_distribution bern(1.f-sparsity);
    std::mt19937 rng{std::random_device{}()};
    for(int kt=0;kt<kTiles;++kt)
        for(int nt=0;nt<nTiles;++nt){
            const bool active=bern(rng);
            for(int k=kt*tile;k<std::min((kt+1)*tile,K);++k)
                for(int n=nt*tile;n<std::min((nt+1)*tile,N);++n)
                    A[k + size_t(n)*K] = active?frand():0.f;
        }
}

inline double to_ms(cudaEvent_t s,cudaEvent_t e){
    float t=0.f; cudaEventElapsedTime(&t,s,e); return t;
}

/* ---- cuBLAS‑Lt dense baseline (保持不变) ---- */
double denseLtTF32(cublasLtHandle_t lt,const float* dW,const float* dA,
                   const float* dB,float* dP,int M,int K,int N)
{
    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a,b,c;
    cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&a,CUDA_R_32F,K,N,K);
    cublasLtMatrixLayoutCreate(&b,CUDA_R_32F,M,K,M);
    cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,M,N,M);

    CHECK_CUDA(cudaMemcpy(dP,dB,size_t(M)*N*sizeof(float),
                          cudaMemcpyDeviceToDevice));
    const float alpha=1.f,beta=1.f;
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    cublasLtMatmul(lt,op,&alpha,
                   dW,b,dA,a,
                   &beta,
                   dP,c,dP,c,
                   /* algo */nullptr,nullptr,0,0);
    cudaEventRecord(e); cudaEventSynchronize(e);
    const double ms=to_ms(s,e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c); cublasLtMatmulDescDestroy(op);
    return ms;
}

/* ============================== main ============================== */
int main(int argc,char* argv[])
{
    const int M   = argc>1?atoi(argv[1]):1024;
    const int K   = argc>2?atoi(argv[2]):1024;
    const int N   = argc>3?atoi(argv[3]):1024;
    const int tile= argc>4?atoi(argv[4]):32;
    const float sp= argc>5?atof(argv[5]):0.9f;
    assert(tile>0 && tile<=SH_TILE_MAX);

    const size_t szW=size_t(M)*K, szA=size_t(K)*N, szB=size_t(M)*N;
    std::vector<float> hW(szW),hA(szA),hB(szB),hPd(szB),hPs(szB);
    gen_matrices(hW,hA,hB,M,K,N,tile,sp);

    float *dW,*dA,*dB,*dP;
    CHECK_CUDA(cudaMalloc(&dW,szW*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dA,szA*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB,szB*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dP,szB*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dW,hW.data(),szW*sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA,hA.data(),szA*sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,hB.data(),szB*sizeof(float),
                          cudaMemcpyHostToDevice));

    /* baseline */
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    const double ms_dense=denseLtTF32(lt,dW,dA,dB,dP,M,K,N);
    CHECK_CUDA(cudaMemcpy(hPd.data(),dP,szB*sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* huge kernel launch */
    dim3 grid((N+BLK_X*VEC_N-1)/(BLK_X*VEC_N),
              (M+BLK_Y-1)/BLK_Y);
    const size_t shBytes = 2*(SH_TILE_MAX*(BLK_Y+PAD)+
                              SH_TILE_MAX*(BLK_X*VEC_N+PAD))*sizeof(float);

    cudaEvent_t s0,s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
    cudaEventRecord(s0);
    spmm_pipeline<<<grid,dim3(BLK_X,BLK_Y),shBytes>>>(
        dW,dA,dB,dP,M,K,N,tile);
    cudaEventRecord(s1); cudaEventSynchronize(s1);
    const double ms_spmm = to_ms(s0,s1);

    CHECK_CUDA(cudaMemcpy(hPs.data(),dP,szB*sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* error */
    double diff2=0,ref2=0;
    for(size_t i=0;i<szB;++i){
        const double d=hPd[i]-hPs[i];
        diff2+=d*d; ref2+=hPd[i]*hPd[i];
    }

    /* report */
    std::cout<<"=== Config ===  M="<<M<<" K="<<K<<" N="<<N
             <<" tile="<<tile<<" sparsity="<<sp<<"\n\n";
    std::cout<<"Dense_TF32 : "<<ms_dense<<" ms\n";
    std::cout<<"Sparse_pipeline : "<<ms_spmm<<" ms\n";
    std::cout<<"Speed‑up (dense / sparse) : "
             <<ms_dense/ms_spmm<<"x\n";
    std::cout<<"Relative‑RMS error : "
             <<std::sqrt(diff2/ref2)<<"\n";

    /* clean */
    cublasLtDestroy(lt);
    cudaFree(dW); cudaFree(dA); cudaFree(dB); cudaFree(dP);
    return 0;
}
// ------------------------- END FILE -------------------------------
