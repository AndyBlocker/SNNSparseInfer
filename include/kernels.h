#pragma once

#include "common.h"

// Basic sparse matrix multiplication kernels
__global__ void buildMask(const float* __restrict__ A,
                         uint8_t* __restrict__ mask,
                         int K, int N, int tile);

__global__ void spmmTile(const float* __restrict__ W,
                        const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ P,
                        const uint8_t* __restrict__ mask,
                        int M, int K, int N, int tile);

// Pipeline sparse matrix multiplication kernel
__global__ void spmm_pipeline(const float* __restrict__ W,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ P,
                             int M, int K, int N, int tile);

// Host interface functions for kernel variants
double runBasicSparse(const float* dW, const float* dA, const float* dB, float* dP,
                     int M, int K, int N, int tile);

double runPipelineSparse(const float* dW, const float* dA, const float* dB, float* dP,
                        int M, int K, int N, int tile);