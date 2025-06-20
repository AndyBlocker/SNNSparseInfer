#include "../../include/benchmark.h"

// Standard cuBLAS SGEMM baseline
double denseSGEMM(cublasHandle_t handle, const float* dW, const float* dA,
                 const float* dB, float* dP, int M, int K, int N)
{
    CHECK_CUDA(cudaMemcpy(dP, dB, (size_t)M * N * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    float alpha = 1.f, beta = 1.f;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                dW, M, dA, K, &beta, dP, M);
    cudaEventRecord(e); cudaEventSynchronize(e);
    double ms = to_ms(s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

// cuBLASLt with TF32
double denseLtTF32(cublasLtHandle_t lt, const float* dW, const float* dA,
                  const float* dB, float* dP, int M, int K, int N)
{
    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, K, N, K);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, M, K, M);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, M, N, M);

    CHECK_CUDA(cudaMemcpy(dP, dB, (size_t)M * N * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    float alpha = 1.f, beta = 1.f;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    cublasLtMatmul(lt, op, &alpha, dW, b, dA, a, &beta, dP, c, dP, c,
                   nullptr, nullptr, 0, 0);
    cudaEventRecord(e); cudaEventSynchronize(e);
    double ms = to_ms(s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c);
    cublasLtMatmulDescDestroy(op);
    return ms;
}

// cuBLASLt with heuristic algorithm selection
double denseLtOptimal(cublasLtHandle_t lt, const float* dW, const float* dA,
                     const float* dB, float* dP, int M, int K, int N)
{
    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, K, N, K);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, M, K, M);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, M, N, M);

    // Find best algorithm using heuristics
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t workspaceSize = 1024 * 1024 * 32; // 32MB workspace
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &workspaceSize, sizeof(workspaceSize));
    
    cublasLtMatmulHeuristicResult_t heuristic;
    int returnedAlgoCount;
    cublasLtMatmulAlgoGetHeuristic(lt, op, a, b, c, c, pref, 1, &heuristic, &returnedAlgoCount);
    
    void* workspace = nullptr;
    if(returnedAlgoCount > 0) {
        cudaMalloc(&workspace, workspaceSize);
    }

    CHECK_CUDA(cudaMemcpy(dP, dB, (size_t)M * N * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    float alpha = 1.f, beta = 1.f;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    
    if(returnedAlgoCount > 0) {
        cublasLtMatmul(lt, op, &alpha, dW, b, dA, a, &beta, dP, c, dP, c,
                       &heuristic.algo, workspace, workspaceSize, 0);
    } else {
        cublasLtMatmul(lt, op, &alpha, dW, b, dA, a, &beta, dP, c, dP, c,
                       nullptr, nullptr, 0, 0);
    }
    
    cudaEventRecord(e); cudaEventSynchronize(e);
    double ms = to_ms(s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    
    if(workspace) cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c);
    cublasLtMatmulDescDestroy(op);
    return ms;
}