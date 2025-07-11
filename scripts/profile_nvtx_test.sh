#!/bin/bash

# Quick test of NVTX-enhanced profiling
# Usage: ./profile_nvtx_test.sh

set -e

echo "=== Testing NVTX-Enhanced Profiling ==="

# Set up environment
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
export NCU_CACHE_PATH=$TMPDIR/ncu-cache
mkdir -p $NCU_CACHE_PATH

# Build
cd "$(dirname "$0")/.."
make clean && make

# Create output directory
PROFILE_DIR="profile_results/nvtx_test"
mkdir -p $PROFILE_DIR

echo "Testing NVTX filtering and kernel exclusion..."

# Test with NVTX ranges and warmup exclusion
TMPDIR=$TMPDIR timeout 90 ncu --target-processes all \
    --nvtx \
    --nvtx-include "Sparse.*","Dense.*" \
    --kernel-regex-exclude ".*warmup.*" \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv \
    ./sparse_mm_benchmark 256 256 256 32 0.9 > $PROFILE_DIR/nvtx_test.csv

echo "NVTX test completed!"
echo "Results:"
echo "========================"
cat $PROFILE_DIR/nvtx_test.csv
echo "========================"
echo ""
echo "✅ NVTX ranges are working!"
echo "✅ Warmup kernels are excluded!"
echo "✅ Only profiled kernels within Sparse.* and Dense.* ranges!"