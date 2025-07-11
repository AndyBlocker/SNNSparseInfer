/**
 * @file kernels.h
 * @brief Kernel function declarations for sparse matrix multiplication variants
 * 
 * Declares all CUDA kernels and host wrapper functions for different
 * sparse matrix multiplication approaches including basic, pipeline,
 * warp-gather, fused, and gather-scatter implementations.
 */

#pragma once

#include "common.h"

// basic sparse matrix multiplication kernels
__global__ void buildMask(const float* __restrict__ A,
                         uint8_t* __restrict__ mask,
                         int K, int N, int tile);

__global__ void spmmTile(const float* __restrict__ W,
                        const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ P,
                        const uint8_t* __restrict__ mask,
                        int M, int K, int N, int tile);

// pipeline sparse matrix multiplication kernel
__global__ void spmm_pipeline(const float* __restrict__ W,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ P,
                             int M, int K, int N, int tile);

// host wrapper functions for different kernel variants
double runBasicSparse(const float* dW, const float* dA, const float* dB, float* dP,
                     int M, int K, int N, int tile);

double runPipelineSparse(const float* dW, const float* dA, const float* dB, float* dP,
                        int M, int K, int N, int tile);

double runWarpGatherSparse(const float* dW, const float* dA, const float* dB, float* dP,
                          int M, int K, int N, int tile);

double runWarpGatherSparseTC(const float* dW, const float* dA, const float* dB, float* dP,
                            int M, int K, int N);

double runSpmmFused(const float* dW, const float* dA, const float* dB, float* dP,
                   int M, int K, int N, int tile, int splitk);

double runGatherScatterSparse(const float* dW, const float* dA, const float* dB, float* dP,
                             int M, int K, int N, int tile);