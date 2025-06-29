/***********************************************************************
 *  binary_sparse_benchmark.cu  —  Binary sparse matrix benchmark
 *  ----------------------------------------------------------------
 *  Compares two approaches for P = A * B + C where B ∈ {0, 1}:
 *    1. Standard matrix multiplication: acc += a * b + c
 *    2. Conditional addition: if(b == 1) acc += a
 *
 *  Tests different workload sizes and B sparsity levels.
 **********************************************************************/

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <tuple>
#include <cmath>
#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePush(name)
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_PUSH(name) 
#define NVTX_POP()
#endif

#define CHECK_CUDA(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d : %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// Timing utility
static inline double to_ms(cudaEvent_t s, cudaEvent_t e) {
    float ms = 0.f;
    cudaEventElapsedTime(&ms, s, e);
    return static_cast<double>(ms);
}

// ------------------------------------------------------------
// 公共宏
// ------------------------------------------------------------
#ifndef TILE
#define TILE 16          // 线程块尺寸
#endif

// ------------------------------------------------------------
// 1) 标准 GEMM：每步做一次 FMA
// ------------------------------------------------------------
__global__ void matmul_standard(const float* __restrict__ A,
                                const float* __restrict__ B,
                                const float* __restrict__ C,
                                float*       __restrict__ P,
                                int M, int K, int N)
{
    __shared__ float sA[TILE][TILE];   // A 的 tile : [row, k]
    __shared__ float sB[TILE][TILE];   // B 的 tile : [k, col]

    // 本线程计算的全局坐标
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.f;

    // 每次取 TILE 个 k 值
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // 边界检查后装入共享内存
        int k  = t * TILE + threadIdx.x;              // 为 A 读
        int kk = t * TILE + threadIdx.y;              // 为 B 读

        sA[threadIdx.y][threadIdx.x] =
            (row < M && k < K) ? A[row * K + k] : 0.f;

        sB[threadIdx.y][threadIdx.x] =
            (kk < K && col < N) ? B[kk * N + col] : 0.f;

        __syncthreads();

        // 逐 k 做 FMA
        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc = fmaf(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);
        }
        __syncthreads();
    }

    if (row < M && col < N)
        P[row * N + col] = acc + C[row * N + col];
}

// ------------------------------------------------------------
// 2) Binary‑Sparse GEMM：b∈{0,1} 时，仅做条件加法
// ------------------------------------------------------------
__global__ void matmul_binary_sparse(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     const float* __restrict__ C,
                                     float*       __restrict__ P,
                                     int M, int K, int N)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int k  = t * TILE + threadIdx.x;
        int kk = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] =
            (row < M && k < K) ? A[row * K + k] : 0.f;

        sB[threadIdx.y][threadIdx.x] =
            (kk < K && col < N) ? B[kk * N + col] : 0.f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            float b = sB[i][threadIdx.x];  // b ∈ {0,1}
            if (b != 0.0f)                 // 避免乘法
                acc += sA[threadIdx.y][i];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        P[row * N + col] = acc + C[row * N + col];
}


/**
 * Generate binary sparse matrix B with specified sparsity
 * sparsity: fraction of zeros (0.0 = dense, 0.9 = 90% zeros)
 */
void generateBinaryMatrix(std::vector<float>& B, int K, int N, float sparsity) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < K * N; ++i) {
        B[i] = (dis(gen) > sparsity) ? 1.0f : 0.0f;
    }
}

/**
 * Generate random matrix with normal distribution
 */
void generateRandomMatrix(std::vector<float>& mat, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < size; ++i) {
        mat[i] = dis(gen);
    }
}

/**
 * Verify correctness by comparing two result matrices
 */
bool verifyResults(const std::vector<float>& P1, const std::vector<float>& P2, 
                  int size, float tolerance = 1e-4f) {
    for (int i = 0; i < size; ++i) {
        if (std::abs(P1[i] - P2[i]) > tolerance) {
            printf("Mismatch at index %d: %f vs %f (diff: %f)\n", 
                   i, P1[i], P2[i], std::abs(P1[i] - P2[i]));
            return false;
        }
    }
    return true;
}

/**
 * Run benchmark for given matrix size and sparsity
 */
void runBenchmark(int M, int K, int N, float sparsity, int iterations = 100) {
    // Host memory allocation
    std::vector<float> hA(M * K), hB(K * N), hC(M * N);
    std::vector<float> hP1(M * N), hP2(M * N);  // Results from two methods
    
    // Generate test data
    generateRandomMatrix(hA, M * K);
    generateBinaryMatrix(hB, K, N, sparsity);
    generateRandomMatrix(hC, M * N);
    
    // Device memory allocation
    float *dA, *dB, *dC, *dP1, *dP2;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dP1, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dP2, M * N * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Setup grid and block dimensions (using TILE size)
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warmup (5 iterations)
    NVTX_PUSH("Warmup");
    for (int i = 0; i < 5; ++i) {
        matmul_standard<<<gridDim, blockDim>>>(dA, dB, dC, dP1, M, K, N);
        matmul_binary_sparse<<<gridDim, blockDim>>>(dA, dB, dC, dP2, M, K, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    NVTX_POP();
    
    // Benchmark standard method
    NVTX_PUSH("Benchmark_Standard_GEMM");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        char iteration_name[64];
        snprintf(iteration_name, sizeof(iteration_name), "Standard_Iteration_%d", i);
        NVTX_PUSH(iteration_name);
        matmul_standard<<<gridDim, blockDim>>>(dA, dB, dC, dP1, M, K, N);
        NVTX_POP();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    double time_standard = to_ms(start, stop) / iterations;
    NVTX_POP();
    
    // Benchmark binary sparse method
    NVTX_PUSH("Benchmark_Binary_Sparse");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        char iteration_name[64];
        snprintf(iteration_name, sizeof(iteration_name), "Sparse_Iteration_%d", i);
        NVTX_PUSH(iteration_name);
        matmul_binary_sparse<<<gridDim, blockDim>>>(dA, dB, dC, dP2, M, K, N);
        NVTX_POP();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    double time_binary = to_ms(start, stop) / iterations;
    NVTX_POP();
    
    // Copy results back for verification (only once)
    CHECK_CUDA(cudaMemcpy(hP1.data(), dP1, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hP2.data(), dP2, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify correctness
    bool correct = verifyResults(hP1, hP2, M * N, 1e-3f);  // More relaxed tolerance
    
    // Calculate actual sparsity in generated matrix
    int zeros = 0;
    for (const float& val : hB) {
        if (val == 0.0f) zeros++;
    }
    float actual_sparsity = static_cast<float>(zeros) / (K * N);
    
    // Calculate performance metrics
    double gflops_standard = (2.0 * M * K * N * 1e-9) / (time_standard * 1e-3);
    double gflops_binary = (2.0 * M * K * N * (1.0 - actual_sparsity) * 1e-9) / (time_binary * 1e-3);
    double speedup = time_standard / time_binary;
    
    // Output CSV format for Python parsing
    // Format: M,K,N,target_sparsity,actual_sparsity,time_standard,time_binary,gflops_standard,gflops_binary,speedup,correct
    printf("%d,%d,%d,%.3f,%.3f,%.6f,%.6f,%.2f,%.2f,%.3f,%s\n",
           M, K, N, sparsity, actual_sparsity, 
           time_standard, time_binary,
           gflops_standard, gflops_binary,
           speedup, correct ? "PASS" : "FAIL");
    
    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(dP1));
    CHECK_CUDA(cudaFree(dP2));
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    if (argc != 5) {
        printf("Usage: %s <M> <K> <N> <sparsity>\n", argv[0]);
        printf("  M, K, N: Matrix dimensions for P = A*B + C where A(M,K), B(K,N)\n");
        printf("  sparsity: Fraction of zeros in B (0.0-1.0)\n");
        printf("Example: %s 1024 1024 1024 0.8\n", argv[0]);
        return 1;
    }
    
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    float sparsity = atof(argv[4]);
    
    if (M <= 0 || K <= 0 || N <= 0) {
        printf("Error: Matrix dimensions must be positive\n");
        return 1;
    }
    
    if (sparsity < 0.0f || sparsity > 1.0f) {
        printf("Error: Sparsity must be between 0.0 and 1.0\n");
        return 1;
    }
    
    // Run single benchmark with multiple iterations for averaging
    char benchmark_name[256];
    snprintf(benchmark_name, sizeof(benchmark_name), 
             "Binary_Sparse_Benchmark_M%d_K%d_N%d_Sparsity%.1f", 
             M, K, N, sparsity * 100);
    NVTX_PUSH(benchmark_name);
    
    runBenchmark(M, K, N, sparsity, 20);  // 20 iterations for better averaging
    
    NVTX_POP();
    return 0;
}