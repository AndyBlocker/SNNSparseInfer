#!/bin/bash

# Profile script for sparse_mm CUDA kernels
# Usage: ./profile_sparse_mm.sh [M K N tile sparsity]

set -e

# Set up temp directory for NCU without sudo
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
export NCU_CACHE_PATH=$TMPDIR/ncu-cache
mkdir -p $NCU_CACHE_PATH

# Default parameters
M=${1:-1024}
K=${2:-1024}
N=${3:-1024}
TILE=${4:-32}
SPARSITY=${5:-0.9}

# Compile the CUDA program with debug info for source mapping
echo "Compiling sparse_mm.cu..."
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo -src-in-ptx sparse_mm.cu \
     -lcublasLt -lcublas -o sparse_mm

# Create output directory
mkdir -p profile_results

# Run comprehensive profiling with NCU (profile all kernels first)
echo "Running NCU profiling with detailed metrics..."
TMPDIR=$TMPDIR ncu --target-processes all \
    --set full \
    --import-source on \
    --source-folders . \
    --export profile_results/sparse_mm_detailed \
    --force-overwrite \
    ./sparse_mm $M $K $N $TILE $SPARSITY

# Run focused kernel analysis (profile all kernels with key metrics)
echo "Running focused kernel analysis..."
TMPDIR=$TMPDIR ncu --target-processes all \
    --metrics \
    smsp__cycles_elapsed.avg,\
    sm__cycles_elapsed.avg,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
    smsp__inst_executed.sum,\
    smsp__warps_active.avg.pct_of_peak_sustained_active,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_shared_op_st.sum \
    --csv \
    --log-file profile_results/sparse_mm_metrics.log \
    ./sparse_mm $M $K $N $TILE $SPARSITY > profile_results/sparse_mm_metrics.csv

# Run memory access pattern analysis
echo "Running memory access pattern analysis..."
TMPDIR=$TMPDIR ncu --target-processes all \
    --section MemoryWorkloadAnalysis \
    --export profile_results/sparse_mm_memory \
    --force-overwrite \
    ./sparse_mm $M $K $N $TILE $SPARSITY

# Generate summary report
echo "Generating summary report..."
cat > profile_results/analysis_summary.txt << EOF
=== Sparse Matrix Multiplication Profiling Summary ===
Configuration: M=$M, K=$K, N=$N, Tile=$TILE, Sparsity=$SPARSITY

Files generated:
1. sparse_mm_detailed.ncu-rep - Complete detailed profiling data
2. sparse_mm_memory.ncu-rep - Memory access pattern analysis  
3. sparse_mm_metrics.csv - Key performance metrics in CSV format
4. sparse_mm_metrics.log - Detailed logging information

Key areas to analyze:
1. Compute utilization (SpeedOfLight section)
2. Memory throughput and coalescing efficiency
3. Shared memory bank conflicts
4. Warp execution efficiency
5. Occupancy limitations
6. Instruction mix and pipeline utilization

Open .ncu-rep files with:
  ncu-ui sparse_mm_detailed.ncu-rep

View CSV metrics with:
  cat sparse_mm_metrics.csv
EOF

echo "Profiling complete! Results saved in profile_results/"
echo "Key files:"
echo "  - profile_results/sparse_mm_detailed.ncu-rep (open with ncu-ui)"
echo "  - profile_results/sparse_mm_metrics.csv (performance metrics)"
echo "  - profile_results/analysis_summary.txt (this summary)"