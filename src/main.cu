#include "../include/common.h"
#include "../include/kernels.h"
#include "../include/benchmark.h"

int main(int argc, char* argv[])
{
    const int M   = argc > 1 ? atoi(argv[1]) : 1024;
    const int K   = argc > 2 ? atoi(argv[2]) : 1024;
    const int N   = argc > 3 ? atoi(argv[3]) : 1024;
    const int tile= argc > 4 ? atoi(argv[4]) : 32;
    const float sp= argc > 5 ? atof(argv[5]) : 0.9f;
    assert(tile > 0 && tile <= SH_TILE_MAX);

    const size_t szW = size_t(M) * K, szA = size_t(K) * N, szB = size_t(M) * N;
    std::vector<float> hW(szW), hA(szA), hB(szB);
    gen_matrices(hW, hA, hB, M, K, N, tile, sp);

    float *dW, *dA, *dB, *dP;
    CHECK_CUDA(cudaMalloc(&dW, szW * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dA, szA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, szB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dP, szB * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dW, hW.data(), szW * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), szA * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), szB * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Run complete benchmark
    BenchmarkResult result = runCompleteBenchmark(dW, dA, dB, dP, M, K, N, tile);

    // Report results
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "M=" << M << " K=" << K << " N=" << N
              << " tile=" << tile << " sparsity=" << sp << std::endl << std::endl;
    
    std::cout << "=== Dense Baseline Comparison ===" << std::endl;
    std::cout << "cuBLAS SGEMM      : " << result.ms_sgemm << " ms" << std::endl;
    std::cout << "cuBLASLt TF32     : " << result.ms_lt_tf32 << " ms" << std::endl;
    std::cout << "cuBLASLt Optimal  : " << result.ms_lt_optimal << " ms" << std::endl;
    std::cout << "Best Dense        : " << result.best_dense_ms << " ms" << std::endl << std::endl;
    
    std::cout << "=== Sparse Performance ===" << std::endl;
    std::cout << "Sparse Basic         : " << result.ms_sparse_basic << " ms" << std::endl;
    std::cout << "Sparse Warp Gather   : " << result.ms_sparse_warp_gather << " ms" << std::endl;
    // std::cout << "Sparse Warp Gather TC: " << result.ms_sparse_warp_gather_tc << " ms" << std::endl;
    
    std::cout << "=== Speedup Analysis ===" << std::endl;
    std::cout << "Basic vs Best Dense       : " << result.best_dense_ms / result.ms_sparse_basic << "x" << std::endl;
    // std::cout << "Pipeline vs Best Dense    : " << result.best_dense_ms / result.ms_sparse_pipeline << "x" << std::endl;
    std::cout << "Warp Gather vs Best Dense : " << result.best_dense_ms / result.ms_sparse_warp_gather << "x" << std::endl;
    // std::cout << "Warp Gather TC vs Best Dense: " << result.best_dense_ms / result.ms_sparse_warp_gather_tc << "x" << std::endl;
    // std::cout << "Fused vs Best Dense       : " << result.best_dense_ms / result.ms_sparse_fused << "x" << std::endl;
    // std::cout << "Gather Scatter vs Best Dense: " << result.best_dense_ms / result.ms_sparse_gather_scatter << "x" << std::endl;

    std::cout << "=== Accuracy ===" << std::endl;
    std::cout << "Basic RMS Error       : " << result.rms_error_basic << std::endl;
    // std::cout << "Pipeline RMS Error    : " << result.rms_error_pipeline << std::endl;
    std::cout << "Warp Gather RMS Error : " << result.rms_error_warp_gather << std::endl;
    // std::cout << "Warp Gather TC RMS Error: " << result.rms_error_warp_gather_tc << std::endl;
    // std::cout << "Fused RMS Error       : " << result.rms_error_fused << std::endl;
    // std::cout << "Gather Scatter RMS Error: " << result.rms_error_gather_scatter << std::endl;

    // Cleanup
    cudaFree(dW); cudaFree(dA); cudaFree(dB); cudaFree(dP);
    return 0;
}