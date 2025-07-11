#!/bin/bash

# Comprehensive NCU Profiling Script for Sparse Matrix Multiplication
# Author: SparseInfer Team
# Usage: ./profile_comprehensive.sh [M K N tile sparsity]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    print_error "NVIDIA Nsight Compute (ncu) is not installed or not in PATH"
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

print_header "Comprehensive NCU Profiling for Sparse Matrix Multiplication"
echo "Configuration: M=$M, K=$K, N=$N, Tile=$TILE, Sparsity=$SPARSITY"
echo

# Build the benchmark
print_status "Building sparse matrix multiplication benchmark..."
cd "$(dirname "$0")/.."
make clean && make

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_DIR="profile_results/profile_${TIMESTAMP}"
mkdir -p $PROFILE_DIR

# 1. Quick Preview - Essential metrics only
print_header "Phase 1: Quick Performance Preview"
print_status "Collecting essential metrics for quick analysis..."

TMPDIR=$TMPDIR ncu --target-processes all \
    --nvtx \
    --nvtx-include "Sparse_Basic" --nvtx-include "Sparse_Pipeline" --nvtx-include "Sparse_WarpGather" --nvtx-include "Dense_cuBLAS_SGEMM" --nvtx-include "Dense_cuBLASLt_TF32" --nvtx-include "Dense_cuBLASLt_Optimal" \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum \
    --csv \
    --log-file $PROFILE_DIR/quick_preview.log \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY > $PROFILE_DIR/quick_preview.csv

# 2. Full Detailed Analysis
print_header "Phase 2: Comprehensive Detailed Analysis"
print_status "Collecting full kernel analysis data (this may take several minutes)..."

TMPDIR=$TMPDIR ncu --target-processes all \
    --nvtx \
    --nvtx-include "Sparse_Basic" --nvtx-include "Sparse_Pipeline" --nvtx-include "Sparse_WarpGather" --nvtx-include "Dense_cuBLAS_SGEMM" --nvtx-include "Dense_cuBLASLt_TF32" --nvtx-include "Dense_cuBLASLt_Optimal" \
    --set full \
    --import-source on \
    --source-folders . \
    --export $PROFILE_DIR/sparse_mm_full_analysis \
    --force-overwrite \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY

# 3. Memory-focused Analysis
print_header "Phase 3: Memory Access Pattern Analysis"
print_status "Analyzing memory access patterns and cache behavior..."

TMPDIR=$TMPDIR ncu --target-processes all \
    --nvtx \
    --nvtx-include "Sparse_Basic" --nvtx-include "Sparse_Pipeline" --nvtx-include "Sparse_WarpGather" --nvtx-include "Dense_cuBLAS_SGEMM" --nvtx-include "Dense_cuBLASLt_TF32" --nvtx-include "Dense_cuBLASLt_Optimal" \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    --section SpeedOfLight \
    --export $PROFILE_DIR/sparse_mm_memory_analysis \
    --force-overwrite \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY

# 4. Compute-focused Analysis
print_header "Phase 4: Compute and Instruction Analysis"
print_status "Analyzing compute utilization and instruction efficiency..."

TMPDIR=$TMPDIR ncu --target-processes all \
    --nvtx \
    --nvtx-include "Sparse_Basic" --nvtx-include "Sparse_Pipeline" --nvtx-include "Sparse_WarpGather" --nvtx-include "Dense_cuBLAS_SGEMM" --nvtx-include "Dense_cuBLASLt_TF32" --nvtx-include "Dense_cuBLASLt_Optimal" \
    --section ComputeWorkloadAnalysis \
    --section InstructionStats \
    --section WarpStateStats \
    --section SchedulerStats \
    --export $PROFILE_DIR/sparse_mm_compute_analysis \
    --force-overwrite \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY

# 5. Generate Quick Preview Report
print_header "Phase 5: Generating Analysis Reports"
print_status "Processing quick preview data..."

# Parse the CSV for quick insights
python3 - << EOF
import csv
import sys
import os

csv_file = '$PROFILE_DIR/quick_preview.csv'
if not os.path.exists(csv_file):
    print("CSV file not found, skipping quick analysis")
    sys.exit(0)

print("=== QUICK PERFORMANCE PREVIEW ===\\n")

kernels = {}
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        kernel_name = row.get('Kernel Name', 'Unknown')
        if 'sparse' in kernel_name.lower() or 'kernel' in kernel_name.lower():
            duration = float(row.get('gpu__time_duration.sum', 0))
            compute_util = float(row.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 0))
            memory_util = float(row.get('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 0))
            warp_efficiency = float(row.get('smsp__warps_active.avg.pct_of_peak_sustained_active', 0))
            
            kernels[kernel_name] = {
                'duration': duration,
                'compute_util': compute_util,
                'memory_util': memory_util,
                'warp_efficiency': warp_efficiency
            }

if kernels:
    print(f"{'Kernel':<30} {'Duration(ms)':<12} {'Compute%':<10} {'Memory%':<10} {'Warp%':<8}")
    print("-" * 70)
    for name, metrics in kernels.items():
        short_name = name.split('(')[0][-25:] if len(name) > 25 else name
        print(f"{short_name:<30} {metrics['duration']:<12.4f} {metrics['compute_util']:<10.1f} {metrics['memory_util']:<10.1f} {metrics['warp_efficiency']:<8.1f}")
    
    print("\\n=== OPTIMIZATION RECOMMENDATIONS ===")
    for name, metrics in kernels.items():
        short_name = name.split('(')[0]
        if metrics['compute_util'] < 50:
            print(f"⚠️  {short_name}: LOW COMPUTE UTILIZATION ({metrics['compute_util']:.1f}%)")
        if metrics['memory_util'] < 50:
            print(f"⚠️  {short_name}: LOW MEMORY UTILIZATION ({metrics['memory_util']:.1f}%)")
        if metrics['warp_efficiency'] < 70:
            print(f"⚠️  {short_name}: LOW WARP EFFICIENCY ({metrics['warp_efficiency']:.1f}%)")
else:
    print("No sparse kernels found in profiling data")
EOF

# 6. Generate comprehensive summary
print_status "Generating comprehensive analysis summary..."

cat > $PROFILE_DIR/ANALYSIS_GUIDE.md << EOF
# Sparse Matrix Multiplication - NCU Profiling Analysis Guide

## Configuration
- **Matrix Dimensions**: M=$M, K=$K, N=$N
- **Tile Size**: $TILE x $TILE
- **Target Sparsity**: $SPARSITY
- **Profiling Timestamp**: $TIMESTAMP

## Generated Files

### 1. Quick Analysis Files
- \`quick_preview.csv\` - Essential performance metrics in CSV format
- \`quick_preview.log\` - NCU execution log for quick run

### 2. Detailed Analysis Files (.ncu-rep format - open with ncu-ui)
- \`sparse_mm_full_analysis.ncu-rep\` - **MAIN FILE** - Complete analysis with all sections
- \`sparse_mm_memory_analysis.ncu-rep\` - Memory-focused analysis
- \`sparse_mm_compute_analysis.ncu-rep\` - Compute-focused analysis

## How to Use These Files

### Quick Command Line Analysis
\`\`\`bash
# View quick metrics
cat quick_preview.csv

# Search for specific kernels
grep -i "sparse" quick_preview.csv
\`\`\`

### GUI Analysis (Recommended)
\`\`\`bash
# Open main analysis file in Nsight Compute GUI
ncu-ui sparse_mm_full_analysis.ncu-rep

# For memory-specific analysis
ncu-ui sparse_mm_memory_analysis.ncu-rep

# For compute-specific analysis  
ncu-ui sparse_mm_compute_analysis.ncu-rep
\`\`\`

## Key Sections to Analyze in NCU GUI

### 1. Speed of Light (SOL)
- **Purpose**: Overall performance bottleneck identification
- **Focus**: Compute vs Memory bound analysis
- **Look for**: Which resource is limiting performance

### 2. Memory Workload Analysis
- **Purpose**: Memory access pattern efficiency
- **Focus**: 
  - Global memory coalescing efficiency
  - Shared memory bank conflicts
  - Cache hit rates
- **Key Metrics**: 
  - \`l1tex__t_sectors_pipe_lsu_mem_global_op_ld_efficiency.pct\`
  - \`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum\`

### 3. Compute Workload Analysis  
- **Purpose**: Instruction mix and compute efficiency
- **Focus**:
  - FFMA/FMUL instruction utilization
  - Warp execution efficiency
  - Register usage
- **Key Metrics**:
  - \`smsp__sass_thread_inst_executed_op_ffma_pred_on.sum\`
  - \`smsp__warps_active.avg.pct_of_peak_sustained_active\`

### 4. Occupancy Analysis
- **Purpose**: Resource utilization and launch configuration
- **Focus**:
  - Theoretical vs achieved occupancy
  - Limiting factors (registers, shared memory, blocks)
- **Key Metrics**:
  - \`sm__warps_active.avg.pct_of_peak_sustained_active\`

## Optimization Strategy

### If Memory Bound:
1. Improve global memory coalescing
2. Reduce shared memory bank conflicts  
3. Optimize data layout and access patterns
4. Consider prefetching strategies

### If Compute Bound:
1. Increase arithmetic intensity
2. Optimize instruction mix
3. Reduce divergent branches
4. Improve warp utilization

### If Launch Configuration Issues:
1. Adjust block size for better occupancy
2. Optimize shared memory usage
3. Consider register pressure reduction

## Next Steps
1. Open \`sparse_mm_full_analysis.ncu-rep\` in ncu-ui
2. Start with "Speed of Light" section
3. Dive deeper into the limiting resource section
4. Compare different sparse kernel implementations
5. Focus optimization efforts on the most impactful areas

EOF

# 7. Final summary
print_header "Profiling Complete!"
echo
print_status "All profiling data collected successfully!"
echo
echo "📁 Results Directory: $PROFILE_DIR"
echo
echo "🚀 Quick Start:"
echo "   1. Read: cat $PROFILE_DIR/ANALYSIS_GUIDE.md"
echo "   2. GUI:  ncu-ui $PROFILE_DIR/sparse_mm_full_analysis.ncu-rep"
echo "   3. CSV:  cat $PROFILE_DIR/quick_preview.csv"
echo
echo "📊 Key Files:"
echo "   • sparse_mm_full_analysis.ncu-rep    (Main analysis - open with ncu-ui)"
echo "   • sparse_mm_memory_analysis.ncu-rep  (Memory focus)"
echo "   • sparse_mm_compute_analysis.ncu-rep (Compute focus)"
echo "   • quick_preview.csv                  (Quick metrics)"
echo "   • ANALYSIS_GUIDE.md                  (This guide)"
echo

print_status "Happy optimizing! 🔧"