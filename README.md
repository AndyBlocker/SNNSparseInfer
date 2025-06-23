# Sparse Matrix Multiplication Benchmark

A modular CUDA implementation comparing sparse matrix multiplication kernels against dense baselines.

## Project Structure

```
SNNSparseInfer/
├── include/
│   ├── common.h           # Common definitions, macros, and inline functions
│   ├── kernels.h          # Kernel interface declarations
│   └── benchmark.h        # Benchmark utilities interface
├── src/
│   ├── kernels/
│   │   ├── kernel_utils.cu        # Common host functions (frand, etc.)
│   │   ├── sparse_mm_basic.cu     # Basic sparse implementation (mask + compute)
│   │   └── sparse_mm_pipeline.cu  # Optimized pipeline implementation
│   ├── benchmark/
│   │   ├── matrix_generator.cu    # Sparse matrix generation
│   │   ├── dense_baselines.cu     # cuBLAS/cuBLASLt baselines
│   │   └── benchmark_utils.cu     # Timing and error calculation
│   └── main.cu                    # Main benchmark program
├── scripts/
│   └── profile_sparse_mm.sh       # Nsight Compute profiling script
├── Makefile                       # Build system
├── sparse_mm.cu                   # Legacy monolithic version
└── new.cu                         # Original pipeline implementation
```

## Building

### Prerequisites
- CUDA 12.6+ (for TF32 Tensor Core support)
- GCC 11+ 
- cuBLAS and cuBLASLt libraries

### Quick Start
```bash
# Set CUDA paths (if not already done)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Build
make

# Test
make test
```

### Build Targets
- `make` or `make all` - Build main modular benchmark
- `make sparse_mm_legacy` - Build legacy monolithic version 
- `make pipeline` - Build pipeline-only version
- `make test` - Run basic test (1024×1024, 90% sparsity)
- `make test-variants` - Test different sparsity levels
- `make debug` - Build with debug symbols
- `make clean` - Remove build artifacts

## Usage

```bash
./sparse_mm_benchmark [M] [K] [N] [tile] [sparsity]
```

**Parameters:**
- `M, K, N`: Matrix dimensions (W: M×K, A: K×N, result: M×N)
- `tile`: Tile size for sparse computation (default: 32)
- `sparsity`: Fraction of zero tiles (0.0-1.0, default: 0.9)

**Example:**
```bash
./sparse_mm_benchmark 2048 2048 2048 32 0.8
```

## Kernel Implementations

### 1. Basic Sparse (`sparse_mm_basic.cu`)
- Two-phase approach: mask generation + computation
- Based on the original implementation
- Simpler but less efficient

### 2. Pipeline Sparse (`sparse_mm_pipeline.cu`) 
- Triple-stream pipeline: mask generation, async loading, computation
- Double buffering for overlapped execution
- cp.async support for sm_80+
- 4-way vectorized output

### 3. Dense Baselines
- **cuBLAS SGEMM**: Traditional FP32 GEMM
- **cuBLASLt TF32**: TensorFloat-32 with Tensor Cores
- **cuBLASLt Optimal**: Algorithm selection with workspace

## Performance Analysis

The benchmark provides comprehensive performance comparison:

```
=== Dense Baseline Comparison ===
cuBLAS SGEMM      : 0.051 ms
cuBLASLt TF32     : 0.036 ms  ← Best dense baseline
cuBLASLt Optimal  : 0.036 ms

=== Sparse Performance ===
Sparse Basic      : 0.293 ms
Sparse Pipeline   : 0.899 ms

=== Speedup Analysis ===
Basic vs Best Dense     : 0.12x  (sparse slower than dense)
Pipeline vs Best Dense  : 0.04x  (pipeline needs optimization)
Pipeline vs Basic       : 0.33x  (basic faster than pipeline)
```

## Key Insights

1. **TF32 Tensor Cores are extremely efficient** for dense GEMM on RTX 4090
2. **High sparsity (90%) creates overhead** that exceeds computation savings
3. **Pipeline version needs tuning** - currently slower than basic version
4. **Matrix size matters** - larger matrices may favor sparse approaches

## Profiling

Use Nsight Compute for detailed analysis:
```bash
./scripts/profile_sparse_mm.sh [M] [K] [N] [tile] [sparsity]
```

Results saved in `profile_results/`:
- `sparse_mm_detailed.ncu-rep` - Complete profiling data
- `sparse_mm_metrics.csv` - Key performance metrics
- `analysis_summary.txt` - Profiling guide

## Python Benchmark Suite

For comprehensive automated benchmarking with visualization:

```bash
cd python_benchmark
pip install -r requirements.txt
./run_benchmark.py
```

**Features:**
- 🎯 **Automated testing** across multiple matrix sizes and sparsity levels
- 📊 **Professional visualizations**: Sparsity curves, workload performance bars, algorithm comparisons
- 🔧 **YAML configuration** for easy customization
- 📈 **Statistical analysis** with error bars and efficiency metrics
- 💾 **Export results** in CSV/JSON formats

**Generated Plots:**
- `sparsity_curves.png` - Performance vs sparsity analysis with efficiency metrics
- `workload_size_performance.png` - Bar charts comparing matrix sizes with GFLOPS/bandwidth
- `algorithm_comparison_summary.png` - Overall algorithm comparison with recommendations

See `python_benchmark/README.md` for detailed usage.

## Future Work

1. **Optimize pipeline kernel** - reduce overhead, improve memory access
2. **Add more sparsity patterns** - structured sparsity, block sparsity
3. **Multi-GPU support** - distributed sparse computation
4. **FP16 variants** - half-precision implementations
5. **Adaptive tile sizing** - dynamic tile selection based on sparsity
6. **CI/CD integration** - automated performance regression testing

## Architecture Support

- **sm_89** (RTX 4090): Primary target, TF32 + cp.async
- **sm_80** (A100): TF32 + cp.async support  
- **sm_75** (RTX 2080): TF32 fallback to manual load
- **sm_70** (V100): Manual loading only

## Legacy Files

- `sparse_mm.cu` - Original monolithic implementation with multiple baselines
- `new.cu` - Pipeline implementation before modularization

These are kept for reference and comparison purposes.