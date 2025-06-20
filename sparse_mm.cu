// ---------------  sparse_mm_opt.cu  ---------------

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <cmath>

constexpr int BLK_X = 32;        // threads‑x
constexpr int BLK_Y =  4;        // threads‑y  (32*4 = 128 thr / block)
constexpr int PAD    = 4;        // avoid bank conflict
constexpr int SH_TILE_MAX = 32;  // max tile dim

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)  do{                           \
    cudaError_t e = (call);                             \
    if (e != cudaSuccess){                              \
        std::cerr<<__FILE__<<":"<<__LINE__<<"  "<<       \
        cudaGetErrorString(e)<<"\n"; std::exit(1);} }while(0)
#endif

__host__ float frand(){
    static thread_local std::mt19937 g{std::random_device{}()};
    static thread_local std::uniform_real_distribution<float> d(-1.f,1.f);
    return d(g);
}

__device__ __forceinline__ float4 load4_safe(const float* gptr){
    float4 v;
    if(((uintptr_t)gptr & 0xF)==0)
        v=*reinterpret_cast<const float4*>(gptr);
    else{
        v.x=__ldg(gptr+0); v.y=__ldg(gptr+1);
        v.z=__ldg(gptr+2); v.w=__ldg(gptr+3);
    }
    return v;
}

__global__ void buildMask(const float* __restrict__ A,
                                uint8_t* __restrict__ mask,
                                int K,int N,int tile)
{
    const int nt = blockIdx.x,  kt = blockIdx.y;
    const int nTiles = gridDim.x;

    const int k0=kt*tile, n0=nt*tile;
    const int kEnd=min(k0+tile,K), nEnd=min(n0+tile,N);

    bool hasNZ=false;
    for(int k=k0+threadIdx.x;k<kEnd && !hasNZ;k+=BLK_X)
        for(int n=n0+threadIdx.y;n<nEnd && !hasNZ;n+=BLK_Y)
            if(__ldg(&A[k+(size_t)n*K])!=0.f){hasNZ=true;break;}

    __syncthreads();
    hasNZ=__syncthreads_or(hasNZ);
    if(threadIdx.x==0 && threadIdx.y==0)
        mask[kt*nTiles+nt]=static_cast<uint8_t>(hasNZ);
}

__launch_bounds__(128,2)
__global__ void spmmTile(const float* __restrict__ W,
                               const float* __restrict__ A,
                               const float* __restrict__ B,
                               float*       __restrict__ P,
                               const uint8_t* __restrict__ mask,
                               int M,int K,int N,int tile)
{
    const int m = blockIdx.y*BLK_Y + threadIdx.y;
    const int n = blockIdx.x*BLK_X + threadIdx.x;
    if(m>=M || n>=N) return;

    const int nTiles=(N+tile-1)/tile, kTiles=(K+tile-1)/tile;
    const int nTileIdx = n / tile;

    extern __shared__ float shmem[];
    float* shW = shmem;                              
    float* shA = shmem + SH_TILE_MAX*(BLK_Y+PAD);    
    float acc=0.f;

    for(int kt=0; kt<kTiles; ++kt){
        if(!mask[kt*nTiles+nTileIdx]) continue;

        const int kBase=kt*tile;
        const int rows = min(tile, K-kBase);

        for(int kk=threadIdx.x; kk<rows; kk+=BLK_X){
            int gIdx = m + (kBase+kk)*M;
            shW[kk*(BLK_Y+PAD)+threadIdx.y] = (m<M)? W[gIdx] : 0.f;
        }

        const int n0=nTileIdx*tile;
        const int colsInTile=min(tile,N-n0);

        for(int kk=threadIdx.y*4; kk<rows; kk+=BLK_Y*4){
            int remain = rows-kk;
            for(int nn=threadIdx.x; nn<colsInTile; nn+=BLK_X){
                size_t gOff = (kBase+kk) + (size_t)(n0+nn)*K;
                if(remain>=4){
                    float4 v = load4_safe(&A[gOff]);
                    float *dst0=&shA[(kk+0)*(BLK_X+PAD)+nn];
                    dst0[0]=v.x;
                    shA[(kk+1)*(BLK_X+PAD)+nn]=v.y;
                    shA[(kk+2)*(BLK_X+PAD)+nn]=v.z;
                    shA[(kk+3)*(BLK_X+PAD)+nn]=v.w;
                }else{                                       
                    for(int r=0;r<remain;++r)
                        shA[(kk+r)*(BLK_X+PAD)+nn] =
                            __ldg(&A[(kBase+kk+r)+(size_t)(n0+nn)*K]);
                }
            }
        }
        __syncthreads();

        /* ---- compute ---- */
        #pragma unroll 4
        for(int kk=0; kk<rows; ++kk)
            acc += shW[kk*(BLK_Y+PAD)+threadIdx.y] *
                   shA[kk*(BLK_X+PAD)+threadIdx.x];
        __syncthreads();
    }
    if(m<M && n<N)
        P[m + (size_t)n*M] = acc + B[m + (size_t)n*M];
}

void gen_matrices(std::vector<float>& W,
                  std::vector<float>& A,
                  std::vector<float>& B,
                  int M,int K,int N,int tile,float sparsity)
{
    std::generate(W.begin(),W.end(),frand);
    std::generate(B.begin(),B.end(),frand);

    int nTiles=(N+tile-1)/tile, kTiles=(K+tile-1)/tile;
    std::bernoulli_distribution bern(1.f-sparsity);
    std::mt19937 rng{std::random_device{}()};
    for(int kt=0;kt<kTiles;++kt)
        for(int nt=0;nt<nTiles;++nt){
            bool active=bern(rng);
            for(int k=kt*tile;k<min((kt+1)*tile,K);++k)
                for(int n=nt*tile;n<min((nt+1)*tile,N);++n)
                    A[k + (size_t)n*K] = active?frand():0.f;
        }
}

inline double to_ms(cudaEvent_t s,cudaEvent_t e){
    float t=0.f; cudaEventElapsedTime(&t,s,e); return t;
}

double denseLtTF32(cublasLtHandle_t lt,const float* dW,const float* dA,
                   const float* dB,float* dP,int M,int K,int N)
{
    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a,b,c;
    cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&a,CUDA_R_32F,K,N,K);
    cublasLtMatrixLayoutCreate(&b,CUDA_R_32F,M,K,M);
    cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,M,N,M);

    CHECK_CUDA(cudaMemcpy(dP,dB,(size_t)M*N*sizeof(float),
                          cudaMemcpyDeviceToDevice));
    float alpha=1.f,beta=1.f;
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    cublasLtMatmul(lt,op,&alpha,dW,b,dA,a,&beta,dP,c,dP,c,
                   nullptr,nullptr,0,0);
    cudaEventRecord(e); cudaEventSynchronize(e);
    double ms=to_ms(s,e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c);
    cublasLtMatmulDescDestroy(op);
    return ms;
}

/* ---------- main ---------- */
int main(int argc,char* argv[])
{
    int M=argc>1?atoi(argv[1]):1024;
    int K=argc>2?atoi(argv[2]):1024;
    int N=argc>3?atoi(argv[3]):1024;
    int tile=argc>4?atoi(argv[4]):32;
    float sp=argc>5?atof(argv[5]):0.9f;
    assert(tile>0 && tile<=SH_TILE_MAX);

    size_t szW=(size_t)M*K, szA=(size_t)K*N, szB=(size_t)M*N;
    std::vector<float> hW(szW),hA(szA),hB(szB),hPd(szB),hPs(szB);
    gen_matrices(hW,hA,hB,M,K,N,tile,sp);

    float *dW,*dA,*dB,*dP; uint8_t* dMask;
    CHECK_CUDA(cudaMalloc(&dW,szW*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dA,szA*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB,szB*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dP,szB*sizeof(float)));
    int nTiles=(N+tile-1)/tile, kTiles=(K+tile-1)/tile;
    CHECK_CUDA(cudaMalloc(&dMask,nTiles*kTiles*sizeof(uint8_t)));

    CHECK_CUDA(cudaMemcpy(dW,hW.data(),szW*sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA,hA.data(),szA*sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,hB.data(),szB*sizeof(float),
                          cudaMemcpyHostToDevice));

    cublasLtHandle_t lt; cublasLtCreate(&lt);

    /* warm‑up & dense timing */
    double ms_dense=denseLtTF32(lt,dW,dA,dB,dP,M,K,N);
    CHECK_CUDA(cudaMemcpy(hPd.data(),dP,szB*sizeof(float),
                          cudaMemcpyDeviceToHost));

    dim3 gridMask(nTiles,kTiles);
    dim3 gridSp((N+BLK_X-1)/BLK_X,(M+BLK_Y-1)/BLK_Y);
    size_t shBytes=(SH_TILE_MAX*(BLK_Y+PAD)+
                    SH_TILE_MAX*(BLK_X+PAD))*sizeof(float);

    /* sparse run timing */
    cudaEvent_t t0,t1,t2; cudaEventCreate(&t0);cudaEventCreate(&t1);cudaEventCreate(&t2);
    cudaEventRecord(t0);
    buildMask<<<gridMask,dim3(BLK_X,BLK_Y)>>>(dA,dMask,K,N,tile);
    cudaEventRecord(t1);
    spmmTile<<<gridSp,dim3(BLK_X,BLK_Y),shBytes>>>(
        dW,dA,dB,dP,dMask,M,K,N,tile);
    cudaEventRecord(t2); cudaEventSynchronize(t2);
    double ms_mask = to_ms(t0,t1);
    double ms_spmm = to_ms(t1,t2);
    double ms_total= to_ms(t0,t2);

    CHECK_CUDA(cudaMemcpy(hPs.data(),dP,szB*sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* error */
    double diff2=0,ref2=0;
    for(size_t i=0;i<szB;++i){
        double d=hPd[i]-hPs[i];
        diff2+=d*d; ref2+=hPd[i]*hPd[i];
    }

    std::cout<<"=== Config ===\nM="<<M<<" K="<<K<<" N="<<N
             <<" tile="<<tile<<" sparsity="<<sp<<"\n\n";
    std::cout<<"Dense_TF32        : "<<ms_dense<<" ms\n";
    std::cout<<"Sparse_buildMask  : "<<ms_mask <<" ms\n";
    std::cout<<"Sparse_SPMM       : "<<ms_spmm <<" ms\n";
    std::cout<<"Sparse_total      : "<<ms_total<<" ms\n";
    std::cout<<"Speed‑up (dense/total‑sparse): "
             <<ms_dense/ms_total<<"x\n";
    std::cout<<"Relative‑RMS error: "
             <<std::sqrt(diff2/ref2)<<"\n";

    cublasLtDestroy(lt);
    cudaFree(dW); cudaFree(dA); cudaFree(dB);
    cudaFree(dP); cudaFree(dMask);
    return 0;
}
