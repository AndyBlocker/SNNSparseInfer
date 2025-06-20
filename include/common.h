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

#define DEBUG_CP 1
// ===== helper ====================================================
__device__ __forceinline__
void cp_async_cg(void* smem_dst, const void* gmem_src)
{
#ifdef DEBUG_CP
    if(((uintptr_t)smem_dst & 0xF) != 0){
        printf("!! misaligned smem addr %p  block=(%d,%d)  thread=(%d,%d,%d)\n",
               smem_dst, blockIdx.x, blockIdx.y,
               threadIdx.x, threadIdx.y, threadIdx.z);
        asm("trap;");       // 立刻终止，触发相同的 716 错误，但已打印坐标
    }
#endif
#if defined(USE_CP_ASYNC) && (__CUDA_ARCH__ >= 800)
    unsigned int       sdst = __cvta_generic_to_shared(smem_dst);
    unsigned long long gsrc = (unsigned long long)gmem_src;
    asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                  "r"(sdst), "l"(gsrc));
#else
    *reinterpret_cast<float4*>(smem_dst) =
        *reinterpret_cast<const float4*>(gmem_src);
#endif
}
#if defined(USE_CP_ASYNC) && (__CUDA_ARCH__ >= 800)
    // #define cp_async_cg(dst, src) do {                                \
    //     unsigned int       _s = __cvta_generic_to_shared(dst);        \
    //     unsigned long long _g = (unsigned long long)(src);            \
    //     asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;\n"   \
    //                   :: "r"(_s), "l"(_g));                           \
    // } while(0)
    #define cp_async_commit() asm volatile("cp.async.commit_group;\n")
    #define cp_async_wait()   asm volatile("cp.async.wait_group 0;\n")
#else
    // #define cp_async_cg(dst, src)                                     \
        // (*reinterpret_cast<float4*>(dst)=*reinterpret_cast<const float4*>(src))
    #define cp_async_commit() ((void)0)
    #define cp_async_wait()   ((void)0)
#endif

// =================================================================
