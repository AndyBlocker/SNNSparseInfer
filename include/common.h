#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <nvtx3/nvToolsExt.h>

// Compile-time configuration
#ifndef BLK_X
#define BLK_X 32        // threads-x
#endif
#ifndef BLK_Y  
#define BLK_Y 4         // threads-y -> 32*4 = 128 threads/block
#endif
#ifndef VEC_N
#define VEC_N 4         // columns produced per thread (pipeline version)
#endif
#ifndef PAD
#define PAD 16           // shared memory bank-conflict padding
#endif
#ifndef SH_TILE_MAX
#define SH_TILE_MAX 32  // max tile dimension
#endif

// Compile-time assertions
static_assert(BLK_X % 32 == 0, "BLK_X must be warp multiple");
static_assert(VEC_N == 4, "code assumes 4-way vector");
static_assert(SH_TILE_MAX <= 32, "shared-mem bound violated");

// Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do{                                        \
    cudaError_t e_=(call);                                          \
    if(e_!=cudaSuccess){                                            \
        std::cerr<<__FILE__<<":"<<__LINE__<<"  "                    \
                 <<cudaGetErrorString(e_)<<std::endl; std::exit(1); \
    } \
}while(0)
#endif

// NVTX helpers
#define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
#define NVTX_RANGE_POP() nvtxRangePop()

// Timing helper
inline double to_ms(cudaEvent_t s, cudaEvent_t e) {
    float t = 0.f; 
    cudaEventElapsedTime(&t, s, e); 
    return t;
}

// Host random number generator
__host__ float frand();

// Device helper functions
__device__ __forceinline__ float4 ld4(const float* __restrict__ g) {
    float4 v;
    if(((uintptr_t)g & 0xF) == 0)
        v = *reinterpret_cast<const float4*>(g);
    else {
        v.x = __ldg(g + 0); v.y = __ldg(g + 1);
        v.z = __ldg(g + 2); v.w = __ldg(g + 3);
    }
    return v;
}

