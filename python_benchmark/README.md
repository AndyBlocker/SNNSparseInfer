# Python Benchmark Suite for Sparse Matrix Multiplication

This comprehensive Python benchmark suite runs multiple configurations of sparse matrix multiplication and generates detailed performance visualizations.

## Features

🎯 **Automated Benchmarking**
- Multiple matrix sizes (512x512 to 8192x8192)
- Various sparsity levels (50% to 99%)
- Different tile sizes (16, 32, 64)
- Multiple measurement runs for accuracy

📊 **Professional Visualizations**
- **Sparsity Curves**: Performance vs sparsity level analysis
- **Workload Size Performance**: Bar charts comparing different matrix sizes  
- **Algorithm Comparison**: Summary of all algorithms with recommendations

🔧 **Configurable & Extensible**
- YAML configuration files
- Easy to add new test cases
- Robust error handling and progress tracking

## Quick Start

### 1. Install Dependencies
```bash
cd python_benchmark
pip install -r requirements.txt
```

### 2. Run Complete Benchmark
```bash
./run_benchmark.py
```

### 3. View Results
Results are saved in:
- `results/` - Raw CSV and JSON data
- `reports/plots/` - Generated PNG visualizations

## Generated Visualizations

### 1. Sparsity Curves (`sparsity_curves.png`)
- **Speedup vs Sparsity**: Shows how performance changes with sparsity level
- **Absolute Performance**: Raw execution times for all algorithms
- **Implementation Efficiency**: Actual vs theoretical maximum speedup
- **Numerical Accuracy**: RMS error analysis

### 2. Workload Size Performance (`workload_size_performance.png`)
- **Performance Comparison**: Bar chart of execution times by matrix size
- **Speedup Comparison**: Speedup factors with value labels
- **Computational Performance**: GFLOPS scaling analysis
- **Memory Bandwidth**: Effective bandwidth utilization

### 3. Algorithm Summary (`algorithm_comparison_summary.png`)
- **Performance Distribution**: Box plots for all algorithms
- **Speedup Distribution**: Statistical summary of speedup factors
- **Performance vs Accuracy**: Trade-off analysis with sparsity coloring
- **Best Algorithm**: Pie chart showing optimal choice distribution

## Configuration

Edit `config/benchmark_config.yaml` to customize:

```yaml
# Matrix sizes to test
matrix_sizes:
  small: [512, 1024, 1536]
  medium: [2048, 3072, 4096] 
  large: [6144, 8192]

# Sparsity levels (0.0-1.0)
sparsity_levels: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# Tile sizes for sparse computation
tile_sizes: [16, 32, 64]

# Number of runs for averaging
measurement_runs: 5
```

## Advanced Usage

### Custom Configuration
```bash
# Edit config file
vim config/benchmark_config.yaml

# Run with custom settings
python benchmark_runner.py
```

### Analyze Existing Results
```bash
# Generate plots from existing CSV data
python -c "
from visualization import BenchmarkVisualizer
import pandas as pd
from pathlib import Path
import yaml

# Load config and data
with open('config/benchmark_config.yaml') as f:
    config = yaml.safe_load(f)
df = pd.read_csv('results/benchmark_results_YYYYMMDD_HHMMSS.csv')

# Generate plots
viz = BenchmarkVisualizer(config)
viz.generate_all_plots(df, Path('reports'))
"
```

### Test Visualization Only
```bash
python visualization.py
```

## Output Files

### Results Directory (`results/`)
- `benchmark_results_TIMESTAMP.csv` - Complete results in tabular format
- `benchmark_results_TIMESTAMP.json` - Raw results with full metadata

### Reports Directory (`reports/plots/`)
- `sparsity_curves.png` - Sparsity analysis plots
- `workload_size_performance.png` - Matrix size performance analysis
- `algorithm_comparison_summary.png` - Algorithm comparison summary

## Performance Metrics

The benchmark measures and compares:

**Dense Baselines:**
- cuBLAS SGEMM (traditional FP32)
- cuBLASLt TF32 (Tensor Core accelerated)
- cuBLASLt Optimal (algorithm selection)

**Sparse Implementations:**
- Basic Sparse (mask generation + computation)
- Pipeline Sparse (triple-stream pipeline)

**Calculated Metrics:**
- Speedup factors (sparse vs dense)
- GFLOPS performance
- Memory bandwidth utilization
- Implementation efficiency vs theoretical maximum
- Numerical accuracy (RMS error)

## System Requirements

- **CUDA**: 12.6+ with TF32 support
- **GPU**: RTX 4090 or equivalent (sm_89)
- **Python**: 3.8+
- **Dependencies**: Listed in `requirements.txt`

## Troubleshooting

**Import Errors:**
```bash
pip install -r requirements.txt
```

**Build Failures:**
```bash
cd .. && make clean && make
```

**CUDA Path Issues:**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Permission Errors:**
```bash
chmod +x run_benchmark.py
```

## Example Results

Typical benchmark output:
```
🎯 Sparse Matrix Multiplication Benchmark Suite
============================================================
🖥️  System: 32 CPUs, 128GB RAM
📁 Working directory: /home/user/SNNSparseInfer/python_benchmark
🚀 Starting benchmark suite with 147 configurations
📊 Matrix sizes: [512, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
🎯 Sparsity levels: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
🔲 Tile sizes: [16, 32, 64]

🔨 Building benchmark executable...
✅ Build successful
🔥 Running 3 warmup iterations...
📈 Running main benchmark suite...
Benchmarking: 100%|██████████| 147/147 [15:30<00:00, 6.33s/it]
✅ Benchmark completed: 147/147 successful

🎨 Generating visualization plots...
📊 Plots saved to reports/plots

🎉 Benchmark completed successfully!
📈 147 configurations tested
📊 Results saved to: results
📋 Reports saved to: reports
```

This comprehensive benchmark suite provides deep insights into sparse matrix multiplication performance across different configurations and generates publication-ready visualizations.