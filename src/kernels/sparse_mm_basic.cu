#include "../../include/kernels.h"

// Basic sparse matrix multiplication with separate mask generation
__global__ void buildMask(const float* __restrict__ A,
                         uint8_t* __restrict__ mask,
                         int K, int N, int tile)
{
    const int nt = blockIdx.x, kt = blockIdx.y;
    const int nTiles = gridDim.x;

    const int k0 = kt * tile, n0 = nt * tile;
    const int kEnd = min(k0 + tile, K), nEnd = min(n0 + tile, N);

    bool hasNZ = false;
    for(int k = k0 + threadIdx.x; k < kEnd && !hasNZ; k += BLK_X)
        for(int n = n0 + threadIdx.y; n < nEnd && !hasNZ; n += BLK_Y)
            if(__ldg(&A[k + (size_t)n * K]) != 0.f) { hasNZ = true; break; }

    __syncthreads();
    hasNZ = __syncthreads_or(hasNZ);
    if(threadIdx.x == 0 && threadIdx.y == 0)
        mask[kt * nTiles + nt] = static_cast<uint8_t>(hasNZ);
}

__launch_bounds__(128, 2)
__global__ void spmmTile(const float* __restrict__ W,
                        const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ P,
                        const uint8_t* __restrict__ mask,
                        int M, int K, int N, int tile)
{
    const int m = blockIdx.y * BLK_Y + threadIdx.y;
    const int n = blockIdx.x * BLK_X + threadIdx.x;
    if(m >= M || n >= N) return;

    const int nTiles = (N + tile - 1) / tile, kTiles = (K + tile - 1) / tile;
    const int nTileIdx = n / tile;

    extern __shared__ float shmem[];
    float* shW = shmem;                              
    float* shA = shmem + SH_TILE_MAX * (BLK_Y + PAD);    
    float acc = 0.f;

    for(int kt = 0; kt < kTiles; ++kt) {
        if(!mask[kt * nTiles + nTileIdx]) continue;

        const int kBase = kt * tile;
        const int rows = min(tile, K - kBase);

        for(int kk = threadIdx.x; kk < rows; kk += BLK_X) {
            int gIdx = m + (kBase + kk) * M;
            shW[kk * (BLK_Y + PAD) + threadIdx.y] = (m < M) ? W[gIdx] : 0.f;
        }

        const int n0 = nTileIdx * tile;
        const int colsInTile = min(tile, N - n0);

        for(int kk = threadIdx.y * 4; kk < rows; kk += BLK_Y * 4) {
            int remain = rows - kk;
            for(int nn = threadIdx.x; nn < colsInTile; nn += BLK_X) {
                size_t gOff = (kBase + kk) + (size_t)(n0 + nn) * K;
                if(remain >= 4) {
                    float4 v = ld4(&A[gOff]);
                    float *dst0 = &shA[(kk + 0) * (BLK_X + PAD) + nn];
                    dst0[0] = v.x;
                    shA[(kk + 1) * (BLK_X + PAD) + nn] = v.y;
                    shA[(kk + 2) * (BLK_X + PAD) + nn] = v.z;
                    shA[(kk + 3) * (BLK_X + PAD) + nn] = v.w;
                } else {                                       
                    for(int r = 0; r < remain; ++r)
                        shA[(kk + r) * (BLK_X + PAD) + nn] =
                            __ldg(&A[(kBase + kk + r) + (size_t)(n0 + nn) * K]);
                }
            }
        }
        __syncthreads();

        #pragma unroll 4
        for(int kk = 0; kk < rows; ++kk)
            acc += shW[kk * (BLK_Y + PAD) + threadIdx.y] *
                   shA[kk * (BLK_X + PAD) + threadIdx.x];
        __syncthreads();
    }
    if(m < M && n < N)
        P[m + (size_t)n * M] = acc + B[m + (size_t)n * M];
}

// Host interface for basic sparse implementation
double runBasicSparse(const float* dW, const float* dA, const float* dB, float* dP,
                     int M, int K, int N, int tile)
{
    int nTiles = (N + tile - 1) / tile, kTiles = (K + tile - 1) / tile;
    
    uint8_t* dMask;
    CHECK_CUDA(cudaMalloc(&dMask, nTiles * kTiles * sizeof(uint8_t)));

    dim3 gridMask(nTiles, kTiles);
    dim3 gridSp((N + BLK_X - 1) / BLK_X, (M + BLK_Y - 1) / BLK_Y);
    size_t shBytes = (SH_TILE_MAX * (BLK_Y + PAD) +
                     SH_TILE_MAX * (BLK_X + PAD)) * sizeof(float);

    cudaEvent_t t0, t1, t2; 
    cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventCreate(&t2);
    
    cudaEventRecord(t0);
    buildMask<<<gridMask, dim3(BLK_X, BLK_Y)>>>(dA, dMask, K, N, tile);
    cudaEventRecord(t1);
    spmmTile<<<gridSp, dim3(BLK_X, BLK_Y), shBytes>>>(
        dW, dA, dB, dP, dMask, M, K, N, tile);
    cudaEventRecord(t2); 
    cudaEventSynchronize(t2);
    
    double ms_total = to_ms(t0, t2);
    
    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaEventDestroy(t2);
    cudaFree(dMask);
    
    return ms_total;
}