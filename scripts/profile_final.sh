#!/bin/bash

# Final comprehensive profiling script with NVTX support
# Usage: ./profile_final.sh [M K N tile sparsity]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} NVIDIA Nsight Compute (ncu) is not installed or not in PATH"
    exit 1
fi

# Set up environment
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

print_header "Enhanced NCU Profiling with NVTX Support"
echo "Configuration: M=$M, K=$K, N=$N, Tile=$TILE, Sparsity=$SPARSITY"
echo "Features: NVTX method labeling, warmup exclusion, comprehensive analysis"
echo

# Build
print_status "Building with NVTX support..."
cd "$(dirname "$0")/.."
make clean && make

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_DIR="profile_results/enhanced_${TIMESTAMP}"
mkdir -p $PROFILE_DIR

# 1. All kernels overview (no NVTX filtering)
print_header "Phase 1: Complete Kernel Overview"
print_status "Collecting data from all kernels..."

TMPDIR=$TMPDIR ncu --target-processes all \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
    --csv \
    --log-file $PROFILE_DIR/all_kernels.log \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY > $PROFILE_DIR/all_kernels.csv

# 2. NVTX-filtered analysis for Sparse methods only
print_header "Phase 2: Sparse Methods Detailed Analysis (NVTX filtered)"
print_status "Analyzing only sparse kernels within NVTX ranges..."

TMPDIR=$TMPDIR ncu --target-processes all \
    --nvtx \
    --nvtx-include "Sparse_Basic" \
    --set full \
    --import-source on \
    --source-folders . \
    --export $PROFILE_DIR/sparse_basic_detailed \
    --force-overwrite \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY

TMPDIR=$TMPDIR ncu --target-processes all \
    --nvtx \
    --nvtx-include "Sparse_Pipeline" \
    --set full \
    --import-source on \
    --source-folders . \
    --export $PROFILE_DIR/sparse_pipeline_detailed \
    --force-overwrite \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY

TMPDIR=$TMPDIR ncu --target-processes all \
    --nvtx \
    --nvtx-include "Sparse_WarpGather" \
    --set full \
    --import-source on \
    --source-folders . \
    --export $PROFILE_DIR/sparse_warpgather_detailed \
    --force-overwrite \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY

# 3. Dense baselines for comparison
print_header "Phase 3: Dense Baseline Analysis"
print_status "Analyzing dense implementations for comparison..."

TMPDIR=$TMPDIR ncu --target-processes all \
    --nvtx \
    --nvtx-include "Dense_cuBLAS_SGEMM" \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --export $PROFILE_DIR/dense_sgemm_analysis \
    --force-overwrite \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY

# 4. Generate analysis guide
print_status "Generating analysis guide..."

cat > $PROFILE_DIR/ENHANCED_ANALYSIS_GUIDE.md << EOF
# Enhanced Sparse Matrix Multiplication - NCU Analysis Guide

## Configuration
- **Matrix Dimensions**: M=$M, K=$K, N=$N  
- **Tile Size**: $TILE x $TILE
- **Target Sparsity**: $SPARSITY
- **Profiling Timestamp**: $TIMESTAMP

## Generated Files

### Overview Files
- \`all_kernels.csv\` - Complete overview of all kernels (CSV format)
- \`all_kernels.log\` - NCU execution log

### NVTX-Filtered Detailed Analysis (.ncu-rep format)
- \`sparse_basic_detailed.ncu-rep\` - Sparse Basic method only
- \`sparse_pipeline_detailed.ncu-rep\` - Sparse Pipeline method only  
- \`sparse_warpgather_detailed.ncu-rep\` - Sparse WarpGather method only
- \`dense_sgemm_analysis.ncu-rep\` - Dense cuBLAS baseline

## Key Features

### ✅ NVTX Method Labeling
Each method is wrapped in NVTX ranges:
- **Sparse_Basic**: Traditional tile-based sparse implementation
- **Sparse_Pipeline**: Pipelined sparse implementation  
- **Sparse_WarpGather**: Warp-level gather sparse implementation
- **Dense_cuBLAS_SGEMM**: cuBLAS dense baseline
- **Dense_cuBLASLt_TF32**: cuBLASLt TF32 baseline
- **Dense_cuBLASLt_Optimal**: cuBLASLt optimized baseline

### ✅ Warmup Exclusion
Warmup kernels are excluded from profiling to focus on actual performance measurements.

### ✅ Method-Specific Analysis
Each sparse method gets its own detailed .ncu-rep file for focused optimization.

## Usage Instructions

### Quick Overview Analysis
\`\`\`bash
# View all kernel performance summary
cat all_kernels.csv

# Identify performance bottlenecks
grep -E "(spmmTile|spmm_pipeline|spmm_rowGather)" all_kernels.csv
\`\`\`

### Detailed Method Analysis
\`\`\`bash
# Analyze specific sparse methods in GUI
ncu-ui sparse_basic_detailed.ncu-rep
ncu-ui sparse_pipeline_detailed.ncu-rep  
ncu-ui sparse_warpgather_detailed.ncu-rep

# Compare with dense baseline
ncu-ui dense_sgemm_analysis.ncu-rep
\`\`\`

### Method Comparison Strategy
1. **Start with Overview**: Check \`all_kernels.csv\` to identify the best performing method
2. **Focus Analysis**: Open the .ncu-rep file for the best method in ncu-ui
3. **Optimize**: Use SpeedOfLight section to identify bottlenecks
4. **Compare**: Load multiple .ncu-rep files in ncu-ui for side-by-side comparison

## Optimization Priority
1. **Sparse_WarpGather** - Usually highest performance potential
2. **Sparse_Basic** - Most stable, good optimization target
3. **Sparse_Pipeline** - Complex but may have specific advantages

## Next Steps
1. Identify the fastest sparse method from \`all_kernels.csv\`
2. Open corresponding .ncu-rep in ncu-ui
3. Focus on the limiting resource (compute vs memory)
4. Apply targeted optimizations
5. Re-profile to measure improvements

Happy optimizing! 🚀
EOF

# 5. Final summary
print_header "Enhanced Profiling Complete!"
echo
print_status "NVTX-enhanced profiling completed successfully!"
echo
echo "📁 Results Directory: $PROFILE_DIR"
echo
echo "🚀 Quick Start:"
echo "   1. Overview: cat $PROFILE_DIR/all_kernels.csv"
echo "   2. Guide:    cat $PROFILE_DIR/ENHANCED_ANALYSIS_GUIDE.md"
echo "   3. Methods:  ncu-ui $PROFILE_DIR/sparse_*_detailed.ncu-rep"
echo
echo "📊 Available Analysis Files:"
echo "   • all_kernels.csv                   (Complete kernel overview)"
echo "   • sparse_basic_detailed.ncu-rep     (Sparse Basic method)"
echo "   • sparse_pipeline_detailed.ncu-rep  (Sparse Pipeline method)"
echo "   • sparse_warpgather_detailed.ncu-rep(Sparse WarpGather method)"
echo "   • dense_sgemm_analysis.ncu-rep      (Dense baseline)"
echo "   • ENHANCED_ANALYSIS_GUIDE.md        (Detailed usage guide)"
echo
echo "✨ Features:"
echo "   ✅ NVTX method labeling and filtering"
echo "   ✅ Warmup kernel exclusion"  
echo "   ✅ Method-specific detailed analysis"
echo "   ✅ Side-by-side comparison ready"
echo

print_status "Ready for optimization! 🔧"