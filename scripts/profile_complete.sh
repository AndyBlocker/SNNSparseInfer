#!/bin/bash

# Complete NCU Profiling Script - All kernels in one comprehensive .ncu-rep file
# Usage: ./profile_complete.sh [M K N tile sparsity]

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
M=${1:-2048}
K=${2:-2048}
N=${3:-2048}
TILE=${4:-32}
SPARSITY=${5:-0.9}

print_header "Complete NCU Profiling - All Methods & Kernels in One File"
echo "Configuration: M=$M, K=$K, N=$N, Tile=$TILE, Sparsity=$SPARSITY"
echo "Features: NVTX method labeling, comprehensive analysis, warmup included"
echo

# Build
print_status "Building with NVTX support..."
cd "$(dirname "$0")/.."
make clean && make

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_DIR="profile_results/complete_${TIMESTAMP}"
mkdir -p $PROFILE_DIR

# 1. Quick preview with essential metrics - all kernels
print_header "Phase 1: Quick Performance Overview"
print_status "Collecting essential metrics from ALL kernels..."

TMPDIR=$TMPDIR ncu --target-processes all \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active \
    --csv \
    --log-file $PROFILE_DIR/quick_overview.log \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY > $PROFILE_DIR/quick_overview.csv

# 2. Complete comprehensive analysis - ALL kernels, no filtering
print_header "Phase 2: Complete Comprehensive Analysis"
print_status "Collecting FULL analysis of ALL kernels in a single .ncu-rep file..."
print_status "This includes Dense baselines, Sparse methods, and support kernels"

TMPDIR=$TMPDIR ncu --target-processes all \
    --set full \
    --import-source on \
    --source-folders . \
    --export $PROFILE_DIR/complete_all_kernels_analysis \
    --force-overwrite \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY

# 3. Memory-focused analysis - ALL kernels
print_header "Phase 3: Complete Memory Analysis"
print_status "Memory-focused analysis of ALL kernels in one file..."

TMPDIR=$TMPDIR ncu --target-processes all \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    --section SpeedOfLight \
    --export $PROFILE_DIR/complete_memory_analysis \
    --force-overwrite \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY

# 4. Compute-focused analysis - ALL kernels
print_header "Phase 4: Complete Compute Analysis"
print_status "Compute-focused analysis of ALL kernels in one file..."

TMPDIR=$TMPDIR ncu --target-processes all \
    --section ComputeWorkloadAnalysis \
    --section InstructionStats \
    --section WarpStateStats \
    --section SchedulerStats \
    --export $PROFILE_DIR/complete_compute_analysis \
    --force-overwrite \
    ./sparse_mm_benchmark $M $K $N $TILE $SPARSITY

# 5. Generate quick analysis
print_header "Phase 5: Generating Analysis Reports"
print_status "Processing complete profiling data..."

# Parse the CSV for quick insights
python3 - << EOF
import csv
import sys
import os

csv_file = '$PROFILE_DIR/quick_overview.csv'
if not os.path.exists(csv_file):
    print("CSV file not found, skipping quick analysis")
    sys.exit(0)

print("=== COMPLETE KERNEL PERFORMANCE ANALYSIS ===\\n")

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
                if kernel_name and kernel_name != 'Unknown':
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
    print(f"{'Kernel':<45} {'Duration(ms)':<12} {'Compute%':<10} {'Memory%':<10} {'Warp%':<8}")
    print("-" * 85)
    
    # Sort by duration to show performance ranking
    sorted_kernels = sorted(kernels.items(), key=lambda x: x[1]['duration'])
    
    # Categorize kernels
    dense_kernels = []
    sparse_kernels = []
    support_kernels = []
    
    for name, metrics in sorted_kernels:
        if any(x in name.lower() for x in ['gemm', 'cutlass', 'ampere']):
            dense_kernels.append((name, metrics))
        elif any(x in name.lower() for x in ['spmm', 'sparse']):
            sparse_kernels.append((name, metrics))
        else:
            support_kernels.append((name, metrics))
    
    # Display by category
    if dense_kernels:
        print("\\n🏭 DENSE BASELINES:")
        for name, metrics in dense_kernels:
            short_name = name.split('(')[0][-40:] if len(name) > 40 else name
            print(f"  {short_name:<43} {metrics['duration']:<12.4f} {metrics['compute_util']:<10.1f} {metrics['memory_util']:<10.1f} {metrics['warp_efficiency']:<8.1f}")
    
    if sparse_kernels:
        print("\\n🔗 SPARSE METHODS:")
        for name, metrics in sparse_kernels:
            short_name = name.split('(')[0][-40:] if len(name) > 40 else name
            print(f"  {short_name:<43} {metrics['duration']:<12.4f} {metrics['compute_util']:<10.1f} {metrics['memory_util']:<10.1f} {metrics['warp_efficiency']:<8.1f}")
    
    if support_kernels:
        print("\\n🔧 SUPPORT KERNELS:")
        for name, metrics in support_kernels:
            short_name = name.split('(')[0][-40:] if len(name) > 40 else name
            print(f"  {short_name:<43} {metrics['duration']:<12.4f} {metrics['compute_util']:<10.1f} {metrics['memory_util']:<10.1f} {metrics['warp_efficiency']:<8.1f}")
    
    print("\\n=== PERFORMANCE INSIGHTS ===")
    
    if sparse_kernels:
        best_sparse = min(sparse_kernels, key=lambda x: x[1]['duration'])
        print(f"🚀 Best Sparse Method: {best_sparse[0].split('(')[0]} ({best_sparse[1]['duration']:.4f} ms)")
    
    if dense_kernels:
        best_dense = min(dense_kernels, key=lambda x: x[1]['duration'])
        print(f"⚡ Best Dense Method: {best_dense[0].split('(')[0]} ({best_dense[1]['duration']:.4f} ms)")
    
    print("\\n=== OPTIMIZATION RECOMMENDATIONS ===")
    for name, metrics in sparse_kernels:
        short_name = name.split('(')[0]
        if metrics['compute_util'] < 50:
            print(f"⚠️  {short_name}: LOW COMPUTE UTILIZATION ({metrics['compute_util']:.1f}%) - Focus on arithmetic intensity")
        if metrics['memory_util'] < 50:
            print(f"⚠️  {short_name}: LOW MEMORY UTILIZATION ({metrics['memory_util']:.1f}%) - Optimize memory access patterns")
        if metrics['warp_efficiency'] < 70:
            print(f"⚠️  {short_name}: LOW WARP EFFICIENCY ({metrics['warp_efficiency']:.1f}%) - Reduce divergence")
else:
    print("No kernel performance data found in CSV")
    print("Check the CSV file directly for any issues.")
EOF

# 6. Generate comprehensive guide
print_status "Generating complete analysis guide..."

cat > $PROFILE_DIR/COMPLETE_ANALYSIS_GUIDE.md << EOF
# Complete Sparse Matrix Multiplication - NCU Analysis Guide

## Configuration
- **Matrix Dimensions**: M=$M, K=$K, N=$N  
- **Tile Size**: $TILE x $TILE
- **Target Sparsity**: $SPARSITY
- **Profiling Timestamp**: $TIMESTAMP

## Generated Files

### Quick Analysis Files
- \`quick_overview.csv\` - All kernels performance overview (CSV format)
- \`quick_overview.log\` - NCU execution log

### Complete Analysis Files (.ncu-rep format - open with ncu-ui)
- 🎯 \`complete_all_kernels_analysis.ncu-rep\` - **MAIN FILE** - ALL kernels complete analysis
- 📊 \`complete_memory_analysis.ncu-rep\` - ALL kernels memory-focused analysis
- 🔧 \`complete_compute_analysis.ncu-rep\` - ALL kernels compute-focused analysis

## Key Features

### ✅ Complete Coverage
- **ALL kernels** in single .ncu-rep file
- **Dense baselines**: cuBLAS SGEMM, cuBLASLt variants
- **Sparse methods**: Basic, Pipeline, WarpGather implementations
- **Support kernels**: buildMask and other utility kernels
- **NVTX method labeling** for easy identification

### ✅ Method Identification
The code uses NVTX ranges to label each method:
- **Dense_cuBLAS_SGEMM**: Standard cuBLAS baseline
- **Dense_cuBLASLt_TF32**: cuBLASLt with TF32 precision
- **Dense_cuBLASLt_Optimal**: cuBLASLt with optimal algorithm
- **Sparse_Basic**: Traditional tile-based sparse implementation
- **Sparse_Pipeline**: Pipelined sparse implementation  
- **Sparse_WarpGather**: Warp-level gather sparse implementation
- **Warmup**: Warmup kernels (can be filtered out in analysis)

## Usage Instructions

### 🚀 Quick Start
\`\`\`bash
# Open the main complete analysis in NCU GUI
ncu-ui complete_all_kernels_analysis.ncu-rep
\`\`\`

### 📊 Performance Overview
\`\`\`bash
# View categorized kernel performance
cat quick_overview.csv

# Look for the performance analysis section
grep -A 30 "COMPLETE KERNEL PERFORMANCE ANALYSIS" quick_overview.csv
\`\`\`

### 🔍 Detailed Analysis Strategy

#### Step 1: Overview Analysis
\`\`\`bash
# Check performance rankings
cat quick_overview.csv
\`\`\`

#### Step 2: NCU GUI Analysis
\`\`\`bash
# Open complete analysis
ncu-ui complete_all_kernels_analysis.ncu-rep
\`\`\`

#### Step 3: Method Comparison
1. **Kernel Selection**: Look for kernels with NVTX range labels
2. **Category Focus**: 
   - Dense baselines: Look for \`ampere_sgemm\` and \`cutlass\` kernels
   - Sparse methods: Look for \`spmm\` kernels
   - Support: Look for \`buildMask\` and other utility kernels
3. **Side-by-side**: Select multiple kernels for comparison

#### Step 4: Specialized Analysis
- **Memory Issues**: Open \`complete_memory_analysis.ncu-rep\`
- **Compute Issues**: Open \`complete_compute_analysis.ncu-rep\`

### 🎯 Optimization Workflow

#### 1. Identify Target Method
- Check \`quick_overview.csv\` for performance ranking
- Focus on the best performing sparse method
- Compare against best dense baseline

#### 2. Kernel-Level Analysis
- Open \`complete_all_kernels_analysis.ncu-rep\` in NCU GUI
- Find your target sparse kernel (e.g., \`spmmTile_sw\`, \`spmm_pipeline\`, \`spmm_rowGather_v2\`)
- Analyze SpeedOfLight section

#### 3. Method Comparison
- Select best sparse kernel + best dense kernel
- Use NCU's comparison features
- Identify performance gaps and opportunities

#### 4. Deep Dive Optimization
- **Memory Bound**: Focus on \`complete_memory_analysis.ncu-rep\`
- **Compute Bound**: Focus on \`complete_compute_analysis.ncu-rep\`
- **Launch Config**: Check occupancy and block sizing

## Kernel Identification Guide

### Dense Baselines
- \`ampere_sgemm_*\`: cuBLAS SGEMM implementations
- \`cutlass::Kernel2<*>\`: cuBLASLt implementations

### Sparse Methods
- \`spmmTile_sw\`: Sparse Basic method kernel
- \`spmm_pipeline\`: Sparse Pipeline method kernel
- \`spmm_rowGather_v2\`: Sparse WarpGather method kernel

### Support Kernels
- \`buildMask\`: Sparsity mask generation
- Other utility kernels as needed

## NCU GUI Tips

### Navigation
- **Timeline View**: Look for NVTX ranges to identify methods
- **Kernel List**: Kernels are listed chronologically
- **Search**: Use kernel name patterns to find specific implementations

### Analysis Focus
- **SpeedOfLight**: Start here for bottleneck identification
- **MemoryWorkloadAnalysis**: For memory-bound kernels
- **ComputeWorkloadAnalysis**: For compute-bound kernels
- **Occupancy**: For launch configuration optimization

### Comparison Strategy
1. **Baseline Comparison**: Compare sparse vs best dense baseline
2. **Method Comparison**: Compare different sparse implementations
3. **Metric Focus**: Focus on limiting factors identified in SpeedOfLight

## Next Steps
1. 📈 **Quick Assessment**: Review \`quick_overview.csv\` for kernel performance ranking
2. 🔍 **Main Analysis**: Open \`complete_all_kernels_analysis.ncu-rep\` in NCU GUI
3. 🎯 **Focus Method**: Identify and analyze the best performing sparse method
4. ⚖️ **Compare**: Compare best sparse vs best dense baseline
5. 🔧 **Optimize**: Apply targeted optimizations based on bottleneck analysis
6. 🔄 **Iterate**: Re-profile after changes

## File Priority
1. **START HERE**: \`complete_all_kernels_analysis.ncu-rep\` (comprehensive analysis)
2. **Memory Deep Dive**: \`complete_memory_analysis.ncu-rep\` (if memory bound)
3. **Compute Deep Dive**: \`complete_compute_analysis.ncu-rep\` (if compute bound)

Happy optimizing! 🚀
EOF

# 7. Final summary
print_header "Complete Profiling Finished!"
echo
print_status "Complete NCU profiling completed successfully!"
echo
echo "📁 Results Directory: $PROFILE_DIR"
echo
echo "🎯 MAIN FILE (All kernels): complete_all_kernels_analysis.ncu-rep"
echo
echo "🚀 Quick Start:"
echo "   1. Overview: cat $PROFILE_DIR/quick_overview.csv"
echo "   2. Main GUI: ncu-ui $PROFILE_DIR/complete_all_kernels_analysis.ncu-rep"
echo "   3. Guide:    cat $PROFILE_DIR/COMPLETE_ANALYSIS_GUIDE.md"
echo
echo "📊 Available Files:"
echo "   • complete_all_kernels_analysis.ncu-rep  🎯 MAIN - All kernels complete analysis"
echo "   • complete_memory_analysis.ncu-rep       📊 All kernels memory-focused"
echo "   • complete_compute_analysis.ncu-rep      🔧 All kernels compute-focused"  
echo "   • quick_overview.csv                     📈 Kernel performance overview"
echo "   • COMPLETE_ANALYSIS_GUIDE.md             📖 Detailed usage guide"
echo
echo "✨ Key Benefits:"
echo "   ✅ Single file with ALL kernels (dense + sparse + support)"
echo "   ✅ NVTX method labeling for identification"
echo "   ✅ Complete kernel coverage including warmup"
echo "   ✅ Side-by-side comparison ready in NCU GUI"
echo "   ✅ Categorized performance analysis"
echo

print_status "Ready for complete analysis! 🔧"