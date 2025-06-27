#!/usr/bin/env python3
"""
Sparse Matrix Multiplication Benchmark Runner
Comprehensive benchmarking with multiple configurations and visualization
"""

import os
import sys
import subprocess
import json
import yaml
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import psutil

@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run"""
    M: int
    K: int  
    N: int
    tile: int
    sparsity: float
    
    def __str__(self):
        return f"M{self.M}_K{self.K}_N{self.N}_tile{self.tile}_sp{self.sparsity:.2f}"

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    config: BenchmarkConfig
    ms_sgemm: float
    ms_lt_tf32: float
    ms_lt_optimal: float
    best_dense_ms: float
    ms_sparse_basic: float
    ms_sparse_warp_gather: float
    rms_error_basic: float
    rms_error_warp_gather: float
    timestamp: str
    
    @property
    def speedup_basic_vs_dense(self) -> float:
        return self.best_dense_ms / self.ms_sparse_basic if self.ms_sparse_basic > 0 else 0
    
    @property 
    def speedup_warp_gather_vs_dense(self) -> float:
        return self.best_dense_ms / self.ms_sparse_warp_gather if self.ms_sparse_warp_gather > 0 else 0
    
    @property
    def speedup_warp_gather_vs_basic(self) -> float:
        return self.ms_sparse_basic / self.ms_sparse_warp_gather if self.ms_sparse_warp_gather > 0 else 0

class BenchmarkRunner:
    """Main benchmark runner class"""
    
    def __init__(self, config_path: str = "config/benchmark_config.yaml"):
        self.config_path = Path(config_path)
        self.load_config()
        self.results: List[BenchmarkResult] = []
        self.start_time = datetime.now()
        
    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Create output directories
        self.results_dir = Path(self.config['output']['results_dir'])
        self.reports_dir = Path(self.config['output']['reports_dir'])
        self.results_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
    def generate_test_configs(self) -> List[BenchmarkConfig]:
        """Generate all test configurations"""
        configs = []
        
        # Combine all matrix sizes
        all_sizes = []
        for size_group in self.config['matrix_sizes'].values():
            all_sizes.extend(size_group)
        
        for size in all_sizes:
            for sparsity in self.config['sparsity_levels']:
                for tile in self.config['tile_sizes']:
                    # Square matrices for simplicity, can be extended
                    configs.append(BenchmarkConfig(
                        M=size, K=size, N=size, 
                        tile=tile, sparsity=sparsity
                    ))
        
        return configs
    
    def build_benchmark(self) -> bool:
        """Build the benchmark executable"""
        try:
            print("🔨 Building benchmark executable...")
            result = subprocess.run(
                self.config['executable']['build_command'], 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=Path(__file__).parent
            )
            
            if result.returncode != 0:
                print(f"❌ Build failed:\n{result.stderr}")
                return False
                
            print("✅ Build successful")
            return True
            
        except Exception as e:
            print(f"❌ Build error: {e}")
            return False
    
    def run_single_benchmark(self, config: BenchmarkConfig) -> Optional[BenchmarkResult]:
        """Run a single benchmark configuration"""
        executable = Path(self.config['executable']['path'])
        
        if not executable.exists():
            print(f"❌ Executable not found: {executable}")
            return None
            
        cmd = [
            str(executable.resolve()),
            str(config.M), str(config.K), str(config.N),
            str(config.tile), str(config.sparsity)
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, 
                cwd=Path(__file__).parent.parent,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                print(f"❌ Benchmark failed for {config}: {result.stderr}")
                return None
                
            return self.parse_benchmark_output(config, result.stdout)
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Benchmark timeout for {config}")
            return None
        except Exception as e:
            print(f"❌ Error running {config}: {e}")
            return None
    
    def parse_benchmark_output(self, config: BenchmarkConfig, output: str) -> Optional[BenchmarkResult]:
        """Parse benchmark output into structured result"""
        try:
            # Regular expressions to extract timing data
            patterns = {
                'ms_sgemm': r'cuBLAS SGEMM\s*:\s*([\d.]+)\s*ms',
                'ms_lt_tf32': r'cuBLASLt TF32\s*:\s*([\d.]+)\s*ms', 
                'ms_lt_optimal': r'cuBLASLt Optimal\s*:\s*([\d.]+)\s*ms',
                'best_dense_ms': r'Best Dense\s*:\s*([\d.]+)\s*ms',
                'ms_sparse_basic': r'Sparse Basic\s*:\s*([\d.]+)\s*ms',
                'ms_sparse_warp_gather': r'Sparse Warp Gather\s*:\s*([\d.]+)\s*ms',
                'rms_error_basic': r'Basic RMS Error\s*:\s*([\d.e+-]+)',
                'rms_error_warp_gather': r'Warp Gather RMS Error\s*:\s*([\d.e+-]+)'
            }
            
            extracted = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, output)
                if match:
                    extracted[key] = float(match.group(1))
                else:
                    print(f"⚠️  Could not extract {key} from output")
                    return None
            
            return BenchmarkResult(
                config=config,
                timestamp=datetime.now().isoformat(),
                **extracted
            )
            
        except Exception as e:
            print(f"❌ Error parsing output for {config}: {e}")
            return None
    
    def run_benchmark_suite(self):
        """Run complete benchmark suite"""
        configs = self.generate_test_configs()
        print(f"🚀 Starting benchmark suite with {len(configs)} configurations")
        print(f"📊 Matrix sizes: {[c.M for c in configs if c.sparsity == 0.9 and c.tile == 32]}")
        print(f"🎯 Sparsity levels: {sorted(set(c.sparsity for c in configs))}")
        print(f"🔲 Tile sizes: {sorted(set(c.tile for c in configs))}")
        
        # Build benchmark first
        if not self.build_benchmark():
            return False
        
        # Run warmup
        print(f"🔥 Running {self.config['warmup_runs']} warmup iterations...")
        warmup_config = configs[0]  # Use first config for warmup
        for _ in range(self.config['warmup_runs']):
            self.run_single_benchmark(warmup_config)
        
        # Main benchmark loop
        print("📈 Running main benchmark suite...")
        
        with tqdm(total=len(configs), desc="Benchmarking") as pbar:
            for config in configs:
                # Run multiple times and average (for more stable results)
                run_results = []
                for run_idx in range(self.config['measurement_runs']):
                    result = self.run_single_benchmark(config)
                    if result:
                        run_results.append(result)
                
                if run_results:
                    # Average the results
                    avg_result = self.average_results(run_results)
                    self.results.append(avg_result)
                    pbar.set_postfix({
                        'Config': str(config)[:30],
                        'Results': len(self.results)
                    })
                else:
                    print(f"❌ All runs failed for {config}")
                
                pbar.update(1)
        
        print(f"✅ Benchmark completed: {len(self.results)}/{len(configs)} successful")
        return True
    
    def average_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """Average multiple benchmark results"""
        if not results:
            raise ValueError("No results to average")
        
        if len(results) == 1:
            return results[0]
        
        # Average all numeric fields
        avg_result = BenchmarkResult(
            config=results[0].config,
            timestamp=results[0].timestamp,
            ms_sgemm=np.mean([r.ms_sgemm for r in results]),
            ms_lt_tf32=np.mean([r.ms_lt_tf32 for r in results]),
            ms_lt_optimal=np.mean([r.ms_lt_optimal for r in results]),
            best_dense_ms=np.mean([r.best_dense_ms for r in results]),
            ms_sparse_basic=np.mean([r.ms_sparse_basic for r in results]),
            ms_sparse_warp_gather=np.mean([r.ms_sparse_warp_gather for r in results]),
            rms_error_basic=np.mean([r.rms_error_basic for r in results]),
            rms_error_warp_gather=np.mean([r.rms_error_warp_gather for r in results])
        )
        
        return avg_result
    
    def save_results(self):
        """Save results to files"""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Save raw results as JSON
        if self.config['output']['save_raw_data']:
            json_path = self.results_dir / f"benchmark_results_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump([asdict(r) for r in self.results], f, indent=2)
            print(f"💾 Raw results saved to {json_path}")
        
        # Save as CSV for analysis
        if self.results:
            df = self.results_to_dataframe()
            csv_path = self.results_dir / f"benchmark_results_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"📊 CSV results saved to {csv_path}")
            return df
        
        return None
    
    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        data = []
        for result in self.results:
            row = {
                'M': result.config.M,
                'K': result.config.K, 
                'N': result.config.N,
                'tile': result.config.tile,
                'sparsity': result.config.sparsity,
                'ms_sgemm': result.ms_sgemm,
                'ms_lt_tf32': result.ms_lt_tf32,
                'ms_lt_optimal': result.ms_lt_optimal,
                'best_dense_ms': result.best_dense_ms,
                'ms_sparse_basic': result.ms_sparse_basic,
                'ms_sparse_warp_gather': result.ms_sparse_warp_gather,
                'speedup_basic_vs_dense': result.speedup_basic_vs_dense,
                'speedup_warp_gather_vs_dense': result.speedup_warp_gather_vs_dense,
                'speedup_warp_gather_vs_basic': result.speedup_warp_gather_vs_basic,
                'rms_error_basic': result.rms_error_basic,
                'rms_error_warp_gather': result.rms_error_warp_gather,
                'timestamp': result.timestamp
            }
            data.append(row)
        
        return pd.DataFrame(data)

def main():
    """Main entry point"""
    print("🎯 Sparse Matrix Multiplication Benchmark Suite")
    print("=" * 60)
    
    runner = BenchmarkRunner()
    
    # Check system info
    print(f"🖥️  System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total // (1024**3)}GB RAM")
    print(f"📁 Working directory: {Path.cwd()}")
    
    success = runner.run_benchmark_suite()
    
    if success and runner.results:
        df = runner.save_results()
        
        # Generate visualizations
        if runner.config['output']['generate_plots'] and df is not None:
            from visualization import BenchmarkVisualizer
            visualizer = BenchmarkVisualizer(runner.config)
            visualizer.generate_all_plots(df, runner.reports_dir)
        
        print(f"🎉 Benchmark completed successfully!")
        print(f"📈 {len(runner.results)} configurations tested")
        print(f"📊 Results saved to: {runner.results_dir}")
        print(f"📋 Reports saved to: {runner.reports_dir}")
        
    else:
        print("❌ Benchmark failed or no results generated")
        sys.exit(1)

if __name__ == "__main__":
    main()