// /***********************************************************************
// *  sparse_mm_gather_scatter.cu - Gather‑GEMM‑Scatter SPMM demo        *
// *                                                                     *
// *  1. Build per‑(kTile,nTile) 稀疏掩码 (共用原文件中的 buildMask).     *
// *  2. Reduce 得到列‑tile 向量 colMask[nTiles]，判断哪些 16×K 块非零。 *
// *  3. Host 端对 colMask 做 prefix‑sum → 获得 activeTiles 与映射表。   *
// *  4. <<<gatherTiles>>>  将非零列块拷贝到紧凑矩阵 A_packed。         *
// *  5. cublas(S/T/F)gemm(Ex)          完整密集矩阵乘：                 *
// *           P_packed[M × Npacked] = W[M×K] · A_packed[K×Npacked]      *
// *  6. <<<scatterTilesAdd>>>          把结果加上偏置 B 并写回 P        *
// ***********************************************************************/

// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <cstdint>
// #include <cstdio>
// #include <cstdlib>
// #include <vector>
// #include <numeric>
// #include <algorithm>

// /* ---------------- 同 basic_spmm_tc.cu 中的工具宏 ----------------- */
// #ifndef BLK_X
// #   define BLK_X 16      // 软件 SPMM 路径线程块宽度（仍保留）
// #endif
// #ifndef BLK_Y
// #   define BLK_Y 16      // 软件 SPMM 路径线程块高度
// #endif
// #ifndef PAD
// #   define PAD   1       // shared‑memory padding
// #endif

// #define CHECK_CUDA(call)                                                   \
// do {                                                                       \
//     cudaError_t _e = (call);                                               \
//     if (_e != cudaSuccess) {                                               \
//         fprintf(stderr,"CUDA error %s:%d : %s\n",                          \
//                 __FILE__,__LINE__, cudaGetErrorString(_e));                \
//         exit(EXIT_FAILURE);                                                \
//     }                                                                      \
// } while (0)

// #define CHECK_CUBLAS(call)                                                 \
// do {                                                                       \
//     cublasStatus_t _s = (call);                                            \
//     if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
//         fprintf(stderr,"cuBLAS error %s:%d : %d\n", __FILE__,__LINE__, _s);\
//         exit(EXIT_FAILURE);                                                \
//     }                                                                      \
// } while (0)

// static inline double to_ms(cudaEvent_t beg, cudaEvent_t end)
// {
//     float ms = 0.f;
//     cudaEventElapsedTime(&ms, beg, end);
//     return static_cast<double>(ms);
// }

// /* ===================================================================== *
//  *           0. buildMask 复用原实现（贴一份，以便单文件可编译）          *
//  * ===================================================================== */
// __global__ void buildMask(const float *__restrict__ A,
//                           uint8_t     *__restrict__ mask,
//                           int K, int N, int tile)
// {
//     const int nt = blockIdx.x;
//     const int kt = blockIdx.y;
//     const int k0   = kt * tile;
//     const int n0   = nt * tile;
//     const int kEnd = min(k0 + tile, K);
//     const int nEnd = min(n0 + tile, N);

//     bool hasNZ = false;
//     for (int k = k0 + threadIdx.x; k < kEnd && !hasNZ; k += BLK_X)
//         for (int n = n0 + threadIdx.y; n < nEnd && !hasNZ; n += BLK_Y)
//             if (__ldg(&A[k + (size_t)n * K]) != 0.f) { hasNZ = true; break; }

//     __syncthreads();
//     hasNZ = __syncthreads_or(hasNZ);
//     if (threadIdx.x == 0 && threadIdx.y == 0)
//         mask[kt * gridDim.x + nt] = static_cast<uint8_t>(hasNZ);
// }

// /* ===================================================================== *
//  *               1. reduceColumnMask  (kTiles × nTiles) → nTiles         *
//  * ===================================================================== */
// __global__ void reduceColumnMask(const uint8_t *__restrict__ fullMask,
//                                        uint8_t *__restrict__ colMask,
//                                  int kTiles, int nTiles)
// {
//     const int nt = blockIdx.x * blockDim.x + threadIdx.x;
//     if (nt >= nTiles) return;

//     uint8_t nz = 0;
//     for (int kt = 0; kt < kTiles && nz == 0; ++kt)
//         nz |= fullMask[kt * nTiles + nt];

//     colMask[nt] = nz;           // 1 = 该列 tile 至少含一非零
// }

// /* ===================================================================== *
//  *               2. gatherTiles – 拷贝非零列‑tile 进 A_packed            *
//  * ===================================================================== *
//  *  ‣ 输入  A      : [K × N]  (col‑major)                                 *
//  *  ‣ 输出  A_pk   : [K × Npacked] (col‑major，按 activeTile 顺序)        *
//  *  ‣ activeIdx[j] = 原始 nTileId                                         *
//  *  ‣ prefix[nTile] = 该 nTile 在 packed 中的 tile 序号                   *
//  *    （host 端通过前缀和得到）                                           *
//  * --------------------------------------------------------------------- */
// #ifndef GATHER_TX
// #   define GATHER_TX 32
// #endif
// #ifndef GATHER_TY
// #   define GATHER_TY 8
// #endif
// __launch_bounds__(GATHER_TX * GATHER_TY, 2)
// __global__ void gatherTiles_kernel(const float *__restrict__ A,
//                                    const uint8_t *__restrict__ colMask,
//                                    const int     *__restrict__ prefix,
//                                          int     *__restrict__ activeIdx,
//                                          float   *__restrict__ A_packed,
//                                    int K, int N, int tile)
// {
//     const int nTile = blockIdx.x;            // 1个 block 负责 1 个列‑tile
//     if (!colMask[nTile]) return;             // 整块全零 → 无需 gather

//     const int pkId  = prefix[nTile];         // packed 中的 tile 序号
//     const int n0    = nTile * tile;          // 源矩阵列起点
//     const int cols  = min(tile, N - n0);

//     /* 线程块二维布局：(tx,ty) copy K×cols */
//     for (int k = threadIdx.y; k < K; k += blockDim.y)
//         for (int c = threadIdx.x; c < cols; c += blockDim.x)
//         {
//             size_t src = k + (size_t)(n0 + c) * K;
//             size_t dst = k + (size_t)(pkId * tile + c) * K;
//             A_packed[dst] = __ldg(&A[src]);
//         }

//     /* 记录映射：packed tile → 原始 nTileId */
//     if (threadIdx.x == 0 && threadIdx.y == 0)
//         activeIdx[pkId] = nTile;
// }

// /* ===================================================================== *
//  *               3. scatterTilesAdd – 写回结果并加上偏置 B                *
//  * ===================================================================== */
// #ifndef SCAT_TX
// #   define SCAT_TX 32
// #endif
// #ifndef SCAT_TY
// #   define SCAT_TY 8
// #endif
// __launch_bounds__(SCAT_TX * SCAT_TY, 2)
// __global__ void scatterTilesAdd_kernel(const float *__restrict__ P_packed,
//                                        const int   *__restrict__ activeIdx,
//                                        const float *__restrict__ B,
//                                              float *__restrict__ P,
//                                        int M, int N, int tile, int numActive)
// {
//     const int pkId = blockIdx.x;             // packed tile 序号
//     const int m    = blockIdx.y * blockDim.y + threadIdx.y;   // 行 index

//     if (pkId >= numActive || m >= M) return;

//     const int nTile = activeIdx[pkId];
//     const int n0    = nTile * tile;
//     const int threadsX = blockDim.x;

//     /* 每个线程 y 负责一行，x 方向展开 tile 列 */
//     for (int c = threadIdx.x; c < tile && n0 + c < N; c += threadsX)
//     {
//         size_t src = m + (size_t)(pkId * tile + c) * M;
//         size_t dst = m + (size_t)(n0   + c) * M;
//         P[dst] = P_packed[src] + __ldg(&B[dst]);   // += B
//     }
// }

// /* ===================================================================== *
//  *                     4. Host helper: prefix‑sum (CPU)                  *
//  * ===================================================================== */
// static void prefixSumCPU(const std::vector<uint8_t>& mask,
//                          std::vector<int>& prefix, int& total)
// {
//     const int n = static_cast<int>(mask.size());
//     prefix.resize(n);
//     int running = 0;
//     for (int i = 0; i < n; ++i)
//     {
//         prefix[i] = running;
//         running  += mask[i] ? 1 : 0;
//     }
//     total = running;   // active tiles
// }

// /* ===================================================================== *
//  *            5. 外部调用接口：runBasicSparse (Gather‑Scatter)            *
//  * ===================================================================== *
//  *   参数/含义 100% 兼容 basic_spmm_tc.cu 中的版本：                       *
//  *   ‣ W[M×K], A[K×N], B[M×N] ‑> P[M×N]                                   *
//  *   ‣ 只假设  A 稀疏，W/B 稠密                                            *
//  * --------------------------------------------------------------------- */
// double runGSSparse(const float *dW, const float *dA,
//                       const float *dB,       float *dP,
//                       int M, int K, int N, int tile)
// {
//     /* ---------------- 0. buildMask : 完全复用 ---------------- */
//     const int nTiles = (N + tile - 1) / tile;
//     const int kTiles = (K + tile - 1) / tile;

//     uint8_t *dMaskFull = nullptr;      // [kTiles × nTiles]
//     CHECK_CUDA(cudaMalloc(&dMaskFull, kTiles * nTiles));

//     dim3 thrMask(BLK_X, BLK_Y);
//     dim3 gridMask(nTiles, kTiles);
//     buildMask<<<gridMask, thrMask>>>(dA, dMaskFull, K, N, tile);
//     CHECK_CUDA(cudaGetLastError());

//     /* ---------------- 1. reduceColumnMask ------------------- */
//     uint8_t *dColMask = nullptr;
//     CHECK_CUDA(cudaMalloc(&dColMask, nTiles));

//     const int REDUCE_BLOCK = 256;
//     int reduceGrid = (nTiles + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
//     reduceColumnMask<<<reduceGrid, REDUCE_BLOCK>>>(dMaskFull, dColMask,
//                                                    kTiles, nTiles);
//     CHECK_CUDA(cudaGetLastError());
//     cudaFree(dMaskFull);

//     /* 拷回 host，做 prefix‑sum 得到 activeTiles 个数与 mapping */
//     std::vector<uint8_t>  hColMask(nTiles);
//     CHECK_CUDA(cudaMemcpy(hColMask.data(), dColMask, nTiles,
//                           cudaMemcpyDeviceToHost));

//     std::vector<int> hPrefix;      // prefix[nTiles]
//     int numActive = 0;
//     prefixSumCPU(hColMask, hPrefix, numActive);

//     if (numActive == 0) {          // 极端情况：全零
//         CHECK_CUDA(cudaMemcpy(dP, dB, (size_t)M * N * sizeof(float),
//                               cudaMemcpyDeviceToDevice));
//         cudaFree(dColMask);
//         return 0.0;    // 耗时几乎可忽略
//     }

//     /* Device 上也需要 prefix & activeIdx */
//     int *dPrefix = nullptr, *dActiveIdx = nullptr;
//     CHECK_CUDA(cudaMalloc(&dPrefix, nTiles * sizeof(int)));
//     CHECK_CUDA(cudaMemcpy(dPrefix, hPrefix.data(),
//                           nTiles * sizeof(int), cudaMemcpyHostToDevice));

//     CHECK_CUDA(cudaMalloc(&dActiveIdx, numActive * sizeof(int)));

//     /* ---------------- 2. 为 packed 矩阵 / 结果 分配显存 -------- */
//     const int Npacked = numActive * tile;
//     float *dA_packed = nullptr;   // [K × Npacked] col‑major
//     float *dP_packed = nullptr;   // [M × Npacked]
//     CHECK_CUDA(cudaMalloc(&dA_packed, (size_t)K * Npacked * sizeof(float)));
//     CHECK_CUDA(cudaMalloc(&dP_packed, (size_t)M * Npacked * sizeof(float)));

//     /* ---------------- 3. GatherTiles ------------------------ */
//     dim3 thrGather(GATHER_TX, GATHER_TY);
//     dim3 gridGather(nTiles);     // 每个 nTile 一个 block
//     gatherTiles_kernel<<<gridGather, thrGather>>>(dA, dColMask,
//                                                   dPrefix, dActiveIdx,
//                                                   dA_packed,
//                                                   K, N, tile);
//     CHECK_CUDA(cudaGetLastError());
//     cudaFree(dColMask);
//     cudaFree(dPrefix);           // prefix 只在 gather 用

//     /* ---------------- 4. cuBLAS Dense GEMM ------------------ */
//     cublasHandle_t handle;
//     CHECK_CUBLAS(cublasCreate(&handle));

//     /* 使用 TF32 Tensor Core（如果 -DUSE_TENSOR_CORE 且 GPU >= Ampere） */
//     float alpha = 1.f, beta = 0.f;
// #if defined(USE_TENSOR_CORE)
//     CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
// #endif
//     /*  A_packed: K×Npacked, W: M×K, P_packed: M×Npacked (列主序) */
//     CHECK_CUBLAS(
//         cublasSgemm(handle,
//                     CUBLAS_OP_N, CUBLAS_OP_N,
//                     M,            /* m  */
//                     Npacked,      /* n  */
//                     K,            /* k  */
//                     &alpha,
//                     dW,  M,       /* lda = M */
//                     dA_packed, K, /* ldb = K */
//                     &beta,
//                     dP_packed, M) /* ldc = M */
//     );

//     CHECK_CUBLAS(cublasDestroy(handle));
//     cudaFree(dA_packed);         /* 不再需要 */

//     /* ---------------- 5. Scatter + Add B -------------------- */
//     dim3 thrScat(SCAT_TX, SCAT_TY);
//     dim3 gridScat(numActive,
//                  (M + SCAT_TY - 1) / SCAT_TY);
//     scatterTilesAdd_kernel<<<gridScat, thrScat>>>(dP_packed, dActiveIdx,
//                                                   dB, dP,
//                                                   M, N, tile, numActive);
//     CHECK_CUDA(cudaGetLastError());
//     cudaFree(dP_packed);
//     cudaFree(dActiveIdx);

//     /* --------------- 6. 计时 + 返回 -------------------------- */
//     /* ⚠ 若需要精确计时，可把 cudaEvent 置于关键阶段前后。下面给出示例 */
//     static cudaEvent_t beg = nullptr, end = nullptr;
//     if (!beg) { cudaEventCreate(&beg); cudaEventCreate(&end); }

//     cudaEventRecord(beg);
//     /* （这里只记录空事件差——示例中已完成计算） */
//     cudaEventRecord(end);
//     cudaEventSynchronize(end);
//     return to_ms(beg, end);      // 仅示范：真实耗时请用外部计时
// }