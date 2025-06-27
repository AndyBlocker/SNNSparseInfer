#pragma once

#include "common.h"

// Matrix generation
void gen_matrices(std::vector<float>& W,
                 std::vector<float>& A,
                 std::vector<float>& B,
                 int M, int K, int N, int tile, float sparsity);

// Dense baselines
double denseSGEMM(cublasHandle_t handle, const float* dW, const float* dA,
                 const float* dB, float* dP, int M, int K, int N);

double denseLtTF32(cublasLtHandle_t lt, const float* dW, const float* dA,
                  const float* dB, float* dP, int M, int K, int N);

double denseLtOptimal(cublasLtHandle_t lt, const float* dW, const float* dA,
                     const float* dB, float* dP, int M, int K, int N);

// Benchmark utilities
double calculateRMSError(const std::vector<float>& ref,
                        const std::vector<float>& test);

struct BenchmarkResult {
    double ms_sgemm;
    double ms_lt_tf32;
    double ms_lt_optimal;
    double ms_sparse_basic;
    double ms_sparse_warp_gather;
    double best_dense_ms;
    double rms_error_basic;
    double rms_error_warp_gather;
};

BenchmarkResult runCompleteBenchmark(
    const float* dW, const float* dA, const float* dB, float* dP,
    int M, int K, int N, int tile);