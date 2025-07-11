#!/bin/bash

# Unified NCU Profiling Script - All methods in one comprehensive .ncu-rep file
# Usage: ./profile_unified.sh [M K N tile sparsity]

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

print_header "Unified NCU Profiling - All Methods in One File"
echo "Configuration: M=$M, K=$K, N=$N, Tile=$TILE, Sparsity=$SPARSITY"
echo "Features: NVTX method labeling, warmup exclusion, unified analysis"
echo

# Build
print_status "Building with NVTX support..."
cd "$(dirname "$0")/.."
make clean && make

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_DIR="profile_results/unified_${TIMESTAMP}"
mkdir -p $PROFILE_DIR

# 1. Quick preview without NVTX filtering - all kernels
print_header "Phase 1: Quick Performance Preview"
print_status "Collecting essential metrics from all kernels..."

TMPDIR=$TMPDIR ncu --target-processes all \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active \
    --csv \
    --log-file $PROFILE_DIR/quick_preview.log \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY > $PROFILE_DIR/quick_preview.csv

# 2. Comprehensive analysis with NVTX filtering - ALL methods in ONE file
print_header "Phase 2: Unified Comprehensive Analysis"
print_status "Analyzing ALL methods (Dense + Sparse) in a single .ncu-rep file..."
print_status "NVTX ranges: All baselines and sparse implementations"

TMPDIR=$TMPDIR ncu --target-processes all \
    --nvtx \
    --nvtx-include "Dense_cuBLAS_SGEMM" \
    --nvtx-include "Dense_cuBLASLt_TF32" \
    --nvtx-include "Dense_cuBLASLt_Optimal" \
    --nvtx-include "Sparse_Basic" \
    --nvtx-include "Sparse_Pipeline" \
    --nvtx-include "Sparse_WarpGather" \
    --set full \
    --import-source on \
    --source-folders . \
    --export $PROFILE_DIR/unified_all_methods_analysis \
    --force-overwrite \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY

# 3. Memory-focused unified analysis
print_header "Phase 3: Unified Memory Analysis"
print_status "Memory-focused analysis of all methods in one file..."

TMPDIR=$TMPDIR ncu --target-processes all \
    --nvtx \
    --nvtx-include "Dense_cuBLAS_SGEMM" \
    --nvtx-include "Dense_cuBLASLt_TF32" \
    --nvtx-include "Dense_cuBLASLt_Optimal" \
    --nvtx-include "Sparse_Basic" \
    --nvtx-include "Sparse_Pipeline" \
    --nvtx-include "Sparse_WarpGather" \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    --section SpeedOfLight \
    --export $PROFILE_DIR/unified_memory_analysis \
    --force-overwrite \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY

# 4. Compute-focused unified analysis  
print_header "Phase 4: Unified Compute Analysis"
print_status "Compute-focused analysis of all methods in one file..."

TMPDIR=$TMPDIR ncu --target-processes all \
    --nvtx \
    --nvtx-include "Dense_cuBLAS_SGEMM" \
    --nvtx-include "Dense_cuBLASLt_TF32" \
    --nvtx-include "Dense_cuBLASLt_Optimal" \
    --nvtx-include "Sparse_Basic" \
    --nvtx-include "Sparse_Pipeline" \
    --nvtx-include "Sparse_WarpGather" \
    --section ComputeWorkloadAnalysis \
    --section InstructionStats \
    --section WarpStateStats \
    --section SchedulerStats \
    --export $PROFILE_DIR/unified_compute_analysis \
    --force-overwrite \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY

# 5. Generate quick analysis
print_header "Phase 5: Generating Analysis Reports"
print_status "Processing unified profiling data..."

# Parse the CSV for quick insights
python3 - << EOF
import csv
import sys
import os

csv_file = '$PROFILE_DIR/quick_preview.csv'
if not os.path.exists(csv_file):
    print("CSV file not found, skipping quick analysis")
    sys.exit(0)

print("=== UNIFIED PERFORMANCE ANALYSIS ===\\n")

# Find kernel data in CSV
kernels = {}
found_data = False
with open(csv_file, 'r') as f:
    for line in f:
        if '"ID"' in line:  # Found CSV header
            found_data = True
            reader = csv.DictReader(f)
            for row in reader:
                kernel_name = row.get('Kernel Name', 'Unknown')
                if any(x in kernel_name.lower() for x in ['gemm', 'spmm', 'buildmask', 'gather']):
                    duration = float(row.get('gpu__time_duration.sum', 0)) / 1000000  # Convert to ms
                    compute_util = float(row.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 0))
                    memory_util = float(row.get('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 0))
                    warp_efficiency = float(row.get('smsp__warps_active.avg.pct_of_peak_sustained_active', 0))
                    
                    kernels[kernel_name] = {
                        'duration': duration,
                        'compute_util': compute_util,
                        'memory_util': memory_util,
                        'warp_efficiency': warp_efficiency
                    }
            break

if kernels and found_data:
    print(f"{'Kernel':<35} {'Duration(ms)':<12} {'Compute%':<10} {'Memory%':<10} {'Warp%':<8}")
    print("-" * 75)
    
    # Sort by duration to show performance ranking
    sorted_kernels = sorted(kernels.items(), key=lambda x: x[1]['duration'])
    
    for name, metrics in sorted_kernels:
        short_name = name.split('(')[0][-30:] if len(name) > 30 else name
        print(f"{short_name:<35} {metrics['duration']:<12.4f} {metrics['compute_util']:<10.1f} {metrics['memory_util']:<10.1f} {metrics['warp_efficiency']:<8.1f}")
    
    print("\\n=== PERFORMANCE INSIGHTS ===")
    # Find sparse kernels
    sparse_kernels = [k for k in kernels.keys() if any(x in k.lower() for x in ['spmm', 'sparse'])]
    dense_kernels = [k for k in kernels.keys() if any(x in k.lower() for x in ['gemm', 'cutlass', 'ampere'])]
    
    if sparse_kernels:
        best_sparse = min(sparse_kernels, key=lambda k: kernels[k]['duration'])
        print(f"🚀 Best Sparse Method: {best_sparse.split('(')[0]} ({kernels[best_sparse]['duration']:.4f} ms)")
    
    if dense_kernels:
        best_dense = min(dense_kernels, key=lambda k: kernels[k]['duration'])
        print(f"⚡ Best Dense Method: {best_dense.split('(')[0]} ({kernels[best_dense]['duration']:.4f} ms)")
    
    print("\\n=== OPTIMIZATION RECOMMENDATIONS ===")
    for name, metrics in sorted_kernels:
        short_name = name.split('(')[0]
        if any(x in name.lower() for x in ['spmm', 'sparse']):
            if metrics['compute_util'] < 50:
                print(f"⚠️  {short_name}: LOW COMPUTE UTILIZATION ({metrics['compute_util']:.1f}%) - Focus on arithmetic intensity")
            if metrics['memory_util'] < 50:
                print(f"⚠️  {short_name}: LOW MEMORY UTILIZATION ({metrics['memory_util']:.1f}%) - Optimize memory access patterns")
            if metrics['warp_efficiency'] < 70:
                print(f"⚠️  {short_name}: LOW WARP EFFICIENCY ({metrics['warp_efficiency']:.1f}%) - Reduce divergence")
else:
    print("No kernel performance data found in CSV")
EOF

# 6. Generate comprehensive guide
print_status "Generating unified analysis guide..."

cat > $PROFILE_DIR/UNIFIED_ANALYSIS_GUIDE.md << EOF
# Unified Sparse Matrix Multiplication - NCU Analysis Guide

## Configuration
- **Matrix Dimensions**: M=$M, K=$K, N=$N  
- **Tile Size**: $TILE x $TILE
- **Target Sparsity**: $SPARSITY
- **Profiling Timestamp**: $TIMESTAMP

## Generated Files

### Quick Analysis Files
- \`quick_preview.csv\` - All kernels performance overview (CSV format)
- \`quick_preview.log\` - NCU execution log

### Unified Analysis Files (.ncu-rep format - open with ncu-ui)
- 🎯 \`unified_all_methods_analysis.ncu-rep\` - **MAIN FILE** - ALL methods complete analysis
- 📊 \`unified_memory_analysis.ncu-rep\` - ALL methods memory-focused analysis
- 🔧 \`unified_compute_analysis.ncu-rep\` - ALL methods compute-focused analysis

## Key Features

### ✅ Unified View
- **Single .ncu-rep file** contains all Dense + Sparse methods
- **Side-by-side comparison** in NCU GUI
- **NVTX method labeling** for easy identification

### ✅ Method Coverage
Each method is clearly labeled with NVTX ranges:
- **Dense_cuBLAS_SGEMM**: Standard cuBLAS baseline
- **Dense_cuBLASLt_TF32**: cuBLASLt with TF32 precision
- **Dense_cuBLASLt_Optimal**: cuBLASLt with optimal algorithm
- **Sparse_Basic**: Traditional tile-based sparse implementation
- **Sparse_Pipeline**: Pipelined sparse implementation  
- **Sparse_WarpGather**: Warp-level gather sparse implementation

### ✅ Warmup Exclusion
Warmup kernels are wrapped in separate NVTX ranges and excluded from analysis.

## Usage Instructions

### 🚀 Quick Start
\`\`\`bash
# Open the main unified analysis in NCU GUI
ncu-ui unified_all_methods_analysis.ncu-rep
\`\`\`

### 📊 Performance Overview
\`\`\`bash
# View all kernel performance summary
cat quick_preview.csv

# Get quick performance insights
grep -A 20 "UNIFIED PERFORMANCE ANALYSIS" quick_preview.csv
\`\`\`

### 🔍 Detailed Analysis Strategy

#### Step 1: Open Unified Analysis
\`\`\`bash
ncu-ui unified_all_methods_analysis.ncu-rep
\`\`\`

#### Step 2: Method Comparison in NCU GUI
1. **Select Multiple Kernels**: Use Ctrl+Click to select kernels from different methods
2. **View Side-by-Side**: NCU GUI will show comparison views
3. **NVTX Identification**: Each kernel is labeled with its method name
4. **Focus on Sparse**: Compare \`Sparse_*\` kernels against \`Dense_*\` baselines

#### Step 3: Specialized Analysis
- **Memory Issues**: Open \`unified_memory_analysis.ncu-rep\`
- **Compute Issues**: Open \`unified_compute_analysis.ncu-rep\`

### 🎯 Optimization Workflow

#### 1. Identify Best Method
- Check quick_preview.csv for performance ranking
- Look for fastest sparse method vs dense baseline

#### 2. Deep Dive Analysis
- Open unified_all_methods_analysis.ncu-rep in NCU GUI
- Select the best sparse method kernel
- Analyze SpeedOfLight section for bottlenecks

#### 3. Method Comparison
- Select both best sparse and best dense kernels
- Use NCU's comparison view to understand differences
- Focus on limiting resources

#### 4. Targeted Optimization
- **If Memory Bound**: Check memory coalescing, bank conflicts
- **If Compute Bound**: Analyze instruction mix, warp utilization  
- **If Launch Bound**: Optimize occupancy, block size

## NCU GUI Tips

### Finding Your Methods
- **NVTX Timeline**: Look for labeled ranges in timeline view
- **Kernel List**: Kernels are grouped by NVTX ranges
- **Search**: Use search functionality to find specific methods

### Comparison Views
- **Multiple Selection**: Select kernels from different methods
- **Metrics Comparison**: Side-by-side metric values
- **Roofline Analysis**: Compare against theoretical limits

## Next Steps
1. 📈 **Quick Check**: Review quick_preview.csv for performance overview
2. 🔍 **Deep Analysis**: Open unified_all_methods_analysis.ncu-rep in NCU GUI
3. ⚖️ **Compare Methods**: Select multiple kernels for side-by-side analysis
4. 🎯 **Optimize**: Focus on the most promising sparse method
5. 🔄 **Iterate**: Re-profile after optimizations

## File Priority
1. **START HERE**: \`unified_all_methods_analysis.ncu-rep\` (comprehensive analysis)
2. **Memory Focus**: \`unified_memory_analysis.ncu-rep\` (if memory bound)
3. **Compute Focus**: \`unified_compute_analysis.ncu-rep\` (if compute bound)

Happy optimizing! 🚀
EOF

# 7. Final summary
print_header "Unified Profiling Complete!"
echo
print_status "Unified NVTX profiling completed successfully!"
echo
echo "📁 Results Directory: $PROFILE_DIR"
echo
echo "🎯 MAIN FILE (All methods in one): unified_all_methods_analysis.ncu-rep"
echo
echo "🚀 Quick Start:"
echo "   1. Overview: cat $PROFILE_DIR/quick_preview.csv"
echo "   2. Main GUI: ncu-ui $PROFILE_DIR/unified_all_methods_analysis.ncu-rep"
echo "   3. Guide:    cat $PROFILE_DIR/UNIFIED_ANALYSIS_GUIDE.md"
echo
echo "📊 Available Files:"
echo "   • unified_all_methods_analysis.ncu-rep  🎯 MAIN - All methods complete analysis"
echo "   • unified_memory_analysis.ncu-rep       📊 All methods memory-focused"
echo "   • unified_compute_analysis.ncu-rep      🔧 All methods compute-focused"
echo "   • quick_preview.csv                     📈 Performance overview"
echo "   • UNIFIED_ANALYSIS_GUIDE.md             📖 Detailed usage guide"
echo
echo "✨ Key Benefits:"
echo "   ✅ Single file contains ALL baselines + sparse methods"
echo "   ✅ NVTX labeling for easy method identification"  
echo "   ✅ Side-by-side comparison ready in NCU GUI"
echo "   ✅ Warmup kernels excluded from analysis"
echo

print_status "Ready for unified analysis! 🔧"