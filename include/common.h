/**
 * @file common.h
 * @brief Common definitions and utilities for sparse matrix multiplication benchmark
 * 
 * Provides compile-time configuration, error checking macros, timing utilities,
 * and common device helper functions used across all kernels.
 */

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

// compile-time configuration parameters
#ifndef BLK_X
#define BLK_X 32        // thread block width (x-dimension)
#endif
#ifndef BLK_Y  
#define BLK_Y 4         // thread block height (y-dimension) -> 32*4 = 128 threads/block
#endif
#ifndef VEC_N
#define VEC_N 4         // vector width: columns processed per thread (pipeline version)
#endif
#ifndef PAD
#define PAD 16          // shared memory bank-conflict padding
#endif
#ifndef SH_TILE_MAX
#define SH_TILE_MAX 32  // maximum supported tile dimension
#endif

// compile-time assertions for configuration validation
static_assert(BLK_X % 32 == 0, "BLK_X must be multiple of warp size");
static_assert(VEC_N == 4, "current implementation assumes 4-way vectorization");
static_assert(SH_TILE_MAX <= 32, "shared memory capacity constraint violated");

// CUDA error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do{                                        \
    cudaError_t e_=(call);                                          \
    if(e_!=cudaSuccess){                                            \
        std::cerr<<__FILE__<<":"<<__LINE__<<"  "                    \
                 <<cudaGetErrorString(e_)<<std::endl; std::exit(1); \
    } \
}while(0)
#endif

// NVTX profiling helpers
#define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
#define NVTX_RANGE_POP() nvtxRangePop()

// CUDA event timing utility
inline double to_ms(cudaEvent_t s, cudaEvent_t e) {
    float t = 0.f; 
    cudaEventElapsedTime(&t, s, e); 
    return t;
}

// host-side random number generator
__host__ float frand();

// device-side utility functions
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

