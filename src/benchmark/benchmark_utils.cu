#include "../../include/benchmark.h"
#include "../../include/kernels.h"

double calculateRMSError(const std::vector<float>& ref,
                        const std::vector<float>& test)
{
    double diff2 = 0, ref2 = 0;
    for(size_t i = 0; i < ref.size(); ++i) {
        const double d = ref[i] - test[i];
        diff2 += d * d; 
        ref2 += ref[i] * ref[i];
    }
    return std::sqrt(diff2 / ref2);
}

BenchmarkResult runCompleteBenchmark(
    const float* dW, const float* dA, const float* dB, float* dP,
    int M, int K, int N, int tile)
{
    BenchmarkResult result = {};
    
    // Create cuBLAS handles
    cublasHandle_t handle; cublasCreate(&handle);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    
    size_t szB = (size_t)M * N;
    std::vector<float> hPd(szB), hPs_basic(szB), hPs_warp_gather(szB);
    
    // Warmup runs (not profiled)
    NVTX_RANGE_PUSH("Warmup");
    denseSGEMM(handle, dW, dA, dB, dP, M, K, N);
    denseLtTF32(lt, dW, dA, dB, dP, M, K, N);
    denseLtOptimal(lt, dW, dA, dB, dP, M, K, N);
    NVTX_RANGE_POP();
    
    // Dense baselines
    NVTX_RANGE_PUSH("Dense_cuBLAS_SGEMM");
    result.ms_sgemm = denseSGEMM(handle, dW, dA, dB, dP, M, K, N);
    NVTX_RANGE_POP();
    CHECK_CUDA(cudaMemcpy(hPd.data(), dP, szB * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    NVTX_RANGE_PUSH("Dense_cuBLASLt_TF32");
    result.ms_lt_tf32 = denseLtTF32(lt, dW, dA, dB, dP, M, K, N);
    NVTX_RANGE_POP();
    
    NVTX_RANGE_PUSH("Dense_cuBLASLt_Optimal");
    result.ms_lt_optimal = denseLtOptimal(lt, dW, dA, dB, dP, M, K, N);
    NVTX_RANGE_POP();
    
    result.best_dense_ms = std::min({result.ms_sgemm, result.ms_lt_tf32, result.ms_lt_optimal});
    
    // Sparse implementations
    NVTX_RANGE_PUSH("Sparse_Basic");
    result.ms_sparse_basic = runBasicSparse(dW, dA, dB, dP, M, K, N, tile);
    NVTX_RANGE_POP();
    CHECK_CUDA(cudaMemcpy(hPs_basic.data(), dP, szB * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    NVTX_RANGE_PUSH("Sparse_WarpGather");
    result.ms_sparse_warp_gather = runWarpGatherSparse(dW, dA, dB, dP, M, K, N, tile);
    NVTX_RANGE_POP();
    CHECK_CUDA(cudaMemcpy(hPs_warp_gather.data(), dP, szB * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    // Calculate errors
    result.rms_error_basic = calculateRMSError(hPd, hPs_basic);
    result.rms_error_warp_gather = calculateRMSError(hPd, hPs_warp_gather);
    
    // Cleanup
    cublasDestroy(handle);
    cublasLtDestroy(lt);
    
    return result;
}