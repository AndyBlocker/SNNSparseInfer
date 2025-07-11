#!/bin/bash

# Quick test version of the profiling script
# Usage: ./profile_quick_test.sh

set -e

echo "=== Quick NCU Profile Test ==="

# Set up environment
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
export NCU_CACHE_PATH=$TMPDIR/ncu-cache
mkdir -p $NCU_CACHE_PATH

# Build
cd "$(dirname "$0")/.."
make clean && make

# Create output directory
PROFILE_DIR="profile_results/quick_test"
mkdir -p $PROFILE_DIR

echo "Testing NCU with minimal metrics..."

# Test with just a few basic metrics
TMPDIR=$TMPDIR timeout 60 ncu --target-processes all \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv \
    ./sparse_mm_benchmark 256 256 256 32 0.9 > $PROFILE_DIR/test_metrics.csv

echo "Quick test completed successfully!"
echo "Results in: $PROFILE_DIR/test_metrics.csv"
cat $PROFILE_DIR/test_metrics.csv