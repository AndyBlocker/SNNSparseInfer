#include <mma.h>
#include "../../include/common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef TILE_SIZE
    #define TILE_SIZE 32
#endif
#ifndef SPLIT_K
    #define SPLIT_K 1
#endif

#if (__CUDA_ARCH__ >= 800)
# define CP_ASYNC_CG_16(dst,src)     asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"((unsigned)(uintptr_t)(dst)),"l"(src))
# define CP_ASYNC_CG_4(dst,src)      (*((float*)(dst)) = __ldg((const float*)(src)))
# define CP_COMMIT()                 asm volatile("cp.async.commit_group;")
# define CP_WAIT()                   asm volatile("cp.async.wait_all;")
#else
# define CP_ASYNC_CG_16(dst,src)     memcpy((dst),(src),16)
# define CP_ASYNC_CG_4(dst,src)      memcpy((dst),(src),4)
# define CP_COMMIT()
# define CP_WAIT()                   __syncthreads()
#endif

#define VEC_BYTES 16
template<bool USE_VEC = true>
static __device__ __forceinline__ void cp_async_vec(void *dst, const void *src, int bytes=sizeof(float))
{
    if constexpr(USE_VEC) {
        if(bytes == 16 && ((uintptr_t)dst & (VEC_BYTES-1)) == 0 && ((uintptr_t)src & (VEC_BYTES-1)) == 0) {
            CP_ASYNC_CG_16(dst, src);
            return;
        }
    }
    // For non-vectorized or unaligned access, always use single element copy
    CP_ASYNC_CG_4(dst, src);
}

__device__ __forceinline__ bool tile_has_nonzero(const float *A, int K, int kBase, int rowsK, int nBase, int colsN)
{
    bool nz_local = false;
    for(int kk = threadIdx.y; kk < rowsK; kk += BLK_Y)
        for(int nn = threadIdx.x;nn < colsN; nn += BLK_X)
            nz_local |= (__ldg(&A[(kBase + kk) + (nBase + nn) * K]) != 0.f);
    
    bool warp_any = __ballot_sync(0xFFFFFFFF, nz_local);
    return __any_sync(0xFFFFFFFF, warp_any != 0);
}

template<int TILE>
__device__ __forceinline__ void load_tile_W(float *sh, const float *W, int M, int m, int kBase, int rowsK, int strideW)
{
    if (!(m<M)) return;

    for(int kk = threadIdx.x;kk < rowsK; kk += BLK_X)
    {
        int gK = kBase + kk;
        const void *src = &W[m + (size_t)gK * M];
        void *dst = &sh[kk * strideW + threadIdx.y];
        cp_async_vec<false>(dst, src, 4);
    }

    return;
}

template<int TILE>
__device__ __forceinline__ void load_tile_A(float *sh, const float *A, int K, int kBase, int rowsK, int nBase, int colsN, int strideA)
{
    for(int kk = threadIdx.y; kk < rowsK; kk += BLK_Y) 
        for(int nn = threadIdx.x; nn < colsN; nn += BLK_X)
        {
            // A is K x N matrix in column-major order, so A[k,n] = A[k + n*K]
            size_t gOff = (kBase + kk) + (size_t)(nBase + nn) * K;
            void *dst = &sh[kk * strideA + nn];
            cp_async_vec<false>(dst, &A[gOff], 4);
        }
    return;
}

template<int TILE=TILE_SIZE, bool USE_TC=false, int SK=(SPLIT_K > 1 ? SPLIT_K : 1)>
__launch_bounds__(BLK_X * BLK_Y, 2)
__global__ void spmm_fused(const float * __restrict__ W,
                           const float * __restrict__ A,
                           const float * __restrict__ B,
                                 float * __restrict__ P,
                                 float * __restrict__ PBuf,
                           int M, int K, int N)
{
    // calc thread-related indexes
    const int tidx = threadIdx.x, tidy = threadIdx.y;
    const int m = blockIdx.y * BLK_Y + tidy, n = blockIdx.x * BLK_X + tidx;
    const int valid_m = (m < M), valid_n = (n < N);

    // calc split K index
    const int sk_id = blockIdx.z;
    const int kTilesTot = (K + TILE - 1) / TILE;
    const int kStart = sk_id * kTilesTot / SK;
    const int kEnd = (sk_id + 1) * kTilesTot / SK;
    const int kTiles = kEnd - kStart;

    // shared memory allocation
    extern __shared__ float sh[];
    const int strideW = BLK_Y + PAD, strideA = TILE + PAD;
    float *shW = sh, *shA = sh + 2 * TILE * strideW;

    // initialize accumulator
    float acc = 0.f;
#if (__CUDA_ARCH__ >= 800)
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> tc_frag;
    if constexpr(USE_TC)
        nvcuda::wmma::fill_fragment(tc_frag, 0.f);
#endif

    bool nzTile[2] = {false, false};
    int buf = 0;

    // prefetch first tile
    if(kTiles > 0)
    {
        const int kBase = kStart * TILE;
        const int rowsK = min(TILE, K - kBase);
        const int nBase = (n / TILE) * TILE;
        const int colsN = min(TILE, N - nBase);

        nzTile[buf] = tile_has_nonzero(A, K, kBase, rowsK, nBase, colsN);
        if(nzTile[buf])
        {
            load_tile_W<TILE>(&shW[buf * TILE * strideW], W, M, m, kBase, rowsK, strideW);
            load_tile_A<TILE>(&shA[buf * TILE * strideA], A, K, kBase, rowsK, nBase, colsN, strideA);
            CP_COMMIT();
        }
    }
    CP_WAIT(); __syncthreads();

    // main loop over split K tiles
    for(int kt = 0;kt < kTiles; kt++)
    {
        // prefetch next tile if available
        int nextBuf = buf ^1, nextKt = kt +1;
        if(nextKt < kTiles)
        {
            const int kBase = (kStart + nextKt) * TILE;
            const int rowsK = min(TILE, K - kBase);
            const int nBase = (n / TILE) * TILE;
            const int colsN = min(TILE, N - nBase);

            nzTile[nextBuf] = tile_has_nonzero(A, K, kBase, rowsK, nBase, colsN);
            if(nzTile[nextBuf])
            {
                load_tile_W<TILE>(&shW[nextBuf * TILE * strideW], W, M, m, kBase, rowsK, strideW);
                load_tile_A<TILE>(&shA[nextBuf * TILE * strideA], A, K, kBase, rowsK, nBase, colsN, strideA);
                CP_COMMIT();
            }
        }

        // calc this tile
        if(nzTile[buf])
        {
            const int kBase = (kStart + kt) * TILE;
            const int rowsK = min(TILE, K - kBase);
            const int nBase = (n / TILE) * TILE;
            const int colsN = min(TILE, N - nBase);

#if defined (__CUDA_ARCH__) &&  (__CUDA_ARCH__ >= 800)
            if constexpr(USE_TC)
            {
                using namespace nvcuda;
                for(int ks = 0; ks < rowsK; ks += 8)
                {
                    const float * tileW = &shW[buf * TILE * strideW + ks * strideW];
                    const float * tileA = &shA[buf * TILE * strideA + ks * strideA];
                    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, tileW, strideW);
                    wmma::load_matrix_sync(b_frag, tileA, strideA);
                    wmma::mma_sync(tc_frag, a_frag, b_frag, tc_frag);
                }
            } else
#endif          
            {
                if(valid_m && valid_n)
                {
                    const int local_n = n - nBase;
                    float tmp = 0.f;
                    for(int kk = 0; kk < rowsK;kk++)
                    {
                        float a = (local_n < colsN) ? shA[buf * TILE * strideA + kk * strideA + local_n] : 0.f;
                        tmp += shW[buf * TILE * strideW + kk * strideW + tidy] * a;
                    }
                    acc += tmp;
                }
            }
        }
        CP_WAIT(); __syncthreads();
        buf ^= 1; // switch buffer
    }

    // write back result
    if(valid_m && valid_n)
    {
        size_t off = m + (size_t)n * M;
        float out = acc;
#if (__CUDA_ARCH__ >= 800)
        if constexpr(USE_TC)
        {
            __shared__ float shOut[16 * 16];
            nvcuda::wmma::store_matrix_sync(shOut, tc_frag, 16, nvcuda::wmma::mem_row_major);
            int idx = (tidy % 16) * 16 + (tidx % 16);
            out = shOut[idx];
        }
#endif
        if constexpr(SK > 1)
        {
            atomicAdd(&PBuf[off + (size_t)sk_id * M * N], out);
            if(sk_id == SK - 1 && tidx == 0 && tidy == 0)
            {
                float sum = 0.f;
                for(int s = 0; s < SK; s++)
                    sum += PBuf[off + (size_t)s * M * N];
                P[off] = sum + __ldg(&B[off]);
            }
        }
        else
            P[off] = out + __ldg(&B[off]);
    }
    return;
}

double runSpmmFused(const float *dW, const float *dA, const float *dB, float *dP, int M, int K, int N,
                    int tile = TILE_SIZE, int splitk = SPLIT_K)
{
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    const int smemCap = prop.sharedMemPerBlockOptin;
    const int shmBytes = 2 * ((size_t)tile * (BLK_Y + PAD) + (size_t)tile * (tile + PAD)) * sizeof(float);

    float *dPbuf = nullptr;
    if(splitk > 1)
    {
        CHECK_CUDA(cudaMalloc(&dPbuf, (size_t)M * N * splitk * sizeof(float)));
        CHECK_CUDA(cudaMemset(dPbuf, 0, (size_t)M * N * splitk * sizeof(float)));
    }

    dim3 thr(BLK_X, BLK_Y);
    dim3 grid((N + BLK_X - 1) / BLK_X, (M + BLK_Y - 1) / BLK_Y, splitk);

    cudaFuncSetAttribute(spmm_fused<>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmBytes > smemCap ? smemCap : shmBytes);
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    cudaEventRecord(t0);
    spmm_fused<<<grid, thr, shmBytes>>>(dW, dA, dB, dP, dPbuf, M, K, N);
    CHECK_CUDA(cudaGetLastError());

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    double ms = to_ms(t0, t1);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    if(dPbuf) cudaFree(dPbuf);
    return ms;
}                     