#!/usr/bin/env python3
"""
Binary Sparse Matrix Multiplication Benchmark Runner
====================================================

This script runs the binary sparse benchmark across different matrix sizes
and sparsity levels, then generates visualization plots.

Usage:
    python3 benchmark_runner.py [--quick] [--output results.csv]
"""

import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

def run_benchmark(executable, M, K, N, sparsity, runs=5):
    """
    Run the benchmark multiple times and return averaged results
    """
    results = []
    
    for run in range(runs):
        try:
            cmd = [executable, str(M), str(K), str(N), f"{sparsity:.3f}"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"Error running benchmark: {result.stderr}", file=sys.stderr)
                continue
                
            # Parse CSV output
            line = result.stdout.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 11:  # Expected number of CSV fields
                    results.append({
                        'M': int(parts[0]),
                        'K': int(parts[1]), 
                        'N': int(parts[2]),
                        'target_sparsity': float(parts[3]),
                        'actual_sparsity': float(parts[4]),
                        'time_standard': float(parts[5]),
                        'time_binary': float(parts[6]),
                        'gflops_standard': float(parts[7]),
                        'gflops_binary': float(parts[8]),
                        'speedup': float(parts[9]),
                        'correct': parts[10] == 'PASS'
                    })
        except subprocess.TimeoutExpired:
            print(f"Benchmark timed out for M={M}, K={K}, N={N}, sparsity={sparsity}")
        except Exception as e:
            print(f"Error running benchmark: {e}")
    
    if not results:
        return None
    
    # Average the results
    avg_result = {}
    for key in results[0].keys():
        if key in ['M', 'K', 'N', 'correct']:
            avg_result[key] = results[0][key]  # These should be the same
        else:
            avg_result[key] = np.mean([r[key] for r in results])
    
    avg_result['runs'] = len(results)
    return avg_result

def main():
    parser = argparse.ArgumentParser(description='Run binary sparse benchmark suite')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick test with smaller matrix sizes')
    parser.add_argument('--output', default='benchmark_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--executable', default='./binary_sparse_benchmark',
                       help='Path to benchmark executable')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of runs per configuration for averaging')
    args = parser.parse_args()
    
    # Check if executable exists
    if not os.path.exists(args.executable):
        print(f"Error: Benchmark executable '{args.executable}' not found")
        print("Please compile it first with: make binary_sparse_benchmark")
        return 1
    
    # Test configurations
    if args.quick:
        matrix_sizes = [
            (512, 512, 512),
            (1024, 1024, 1024),
        ]
        sparsity_levels = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    else:
        matrix_sizes = [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (1024, 2048, 1024),  # Rectangular
            (2048, 1024, 2048),  # Rectangular
        ]
        sparsity_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
    
    print(f"Running benchmark suite with {len(matrix_sizes)} matrix sizes and {len(sparsity_levels)} sparsity levels")
    print(f"Total configurations: {len(matrix_sizes) * len(sparsity_levels)}")
    print(f"Runs per configuration: {args.runs}")
    print()
    
    all_results = []
    total_configs = len(matrix_sizes) * len(sparsity_levels)
    current_config = 0
    
    for M, K, N in matrix_sizes:
        print(f"Testing matrix size: {M}x{K}x{N}")
        
        for sparsity in sparsity_levels:
            current_config += 1
            print(f"  [{current_config}/{total_configs}] Sparsity: {sparsity:.2f}...", end=' ', flush=True)
            
            result = run_benchmark(args.executable, M, K, N, sparsity, args.runs)
            
            if result:
                all_results.append(result)
                print(f"✓ Standard: {result['time_standard']:.3f}ms, Binary: {result['time_binary']:.3f}ms, Speedup: {result['speedup']:.2f}x")
            else:
                print("✗ Failed")
    
    if not all_results:
        print("No successful benchmark runs!")
        return 1
    
    # Save results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # Generate plots
    generate_plots(df, args.output.replace('.csv', ''))
    
    # Print summary statistics
    print_summary(df)
    
    return 0

def generate_plots(df, output_prefix):
    """Generate visualization plots"""
    
    # Set up matplotlib for better plots
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Get unique matrix sizes
    matrix_sizes = df[['M', 'K', 'N']].drop_duplicates()
    
    # 1. Latency vs Sparsity for different matrix sizes
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Binary Sparse Matrix Multiplication Performance Analysis', fontsize=16)
    
    # Plot 1: Execution time vs sparsity
    ax1 = axes[0, 0]
    for _, size in matrix_sizes.iterrows():
        subset = df[(df['M'] == size['M']) & (df['K'] == size['K']) & (df['N'] == size['N'])]
        if len(subset) > 0:
            label = f"{size['M']}x{size['K']}x{size['N']}"
            ax1.plot(subset['actual_sparsity'], subset['time_standard'], 'o-', label=f'Standard {label}', alpha=0.7)
            ax1.plot(subset['actual_sparsity'], subset['time_binary'], 's--', label=f'Binary {label}', alpha=0.7)
    
    ax1.set_xlabel('Sparsity (fraction of zeros)')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Execution Time vs Sparsity')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup vs sparsity
    ax2 = axes[0, 1]
    for _, size in matrix_sizes.iterrows():
        subset = df[(df['M'] == size['M']) & (df['K'] == size['K']) & (df['N'] == size['N'])]
        if len(subset) > 0:
            label = f"{size['M']}x{size['K']}x{size['N']}"
            ax2.plot(subset['actual_sparsity'], subset['speedup'], 'o-', label=label)
    
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No speedup')
    ax2.set_xlabel('Sparsity (fraction of zeros)')
    ax2.set_ylabel('Speedup (Standard/Binary)')
    ax2.set_title('Speedup vs Sparsity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: GFLOPS comparison
    ax3 = axes[1, 0]
    for _, size in matrix_sizes.iterrows():
        subset = df[(df['M'] == size['M']) & (df['K'] == size['K']) & (df['N'] == size['N'])]
        if len(subset) > 0:
            label = f"{size['M']}x{size['K']}x{size['N']}"
            ax3.plot(subset['actual_sparsity'], subset['gflops_standard'], 'o-', label=f'Standard {label}', alpha=0.7)
            ax3.plot(subset['actual_sparsity'], subset['gflops_binary'], 's--', label=f'Binary {label}', alpha=0.7)
    
    ax3.set_xlabel('Sparsity (fraction of zeros)')
    ax3.set_ylabel('Performance (GFLOPS)')
    ax3.set_title('Performance vs Sparsity')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency analysis (speedup vs theoretical maximum)
    ax4 = axes[1, 1]
    for _, size in matrix_sizes.iterrows():
        subset = df[(df['M'] == size['M']) & (df['K'] == size['K']) & (df['N'] == size['N'])]
        if len(subset) > 0:
            label = f"{size['M']}x{size['K']}x{size['N']}"
            # Theoretical maximum speedup is 1/(1-sparsity) for compute-bound operations
            theoretical_speedup = 1.0 / (1.0 - subset['actual_sparsity'] + 1e-6)
            efficiency = subset['speedup'] / theoretical_speedup
            ax4.plot(subset['actual_sparsity'], efficiency, 'o-', label=label)
    
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect efficiency')
    ax4.set_xlabel('Sparsity (fraction of zeros)')
    ax4.set_ylabel('Efficiency (Actual/Theoretical Speedup)')
    ax4.set_title('Sparse Optimization Efficiency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_analysis.pdf', bbox_inches='tight')
    print(f"Analysis plots saved to {output_prefix}_analysis.png/pdf")
    
    # Create a focused plot for best results
    plt.figure(figsize=(10, 6))
    
    # Find the configuration with best speedup at high sparsity
    high_sparsity_df = df[df['actual_sparsity'] >= 0.8]
    if len(high_sparsity_df) > 0:
        best_config = high_sparsity_df.loc[high_sparsity_df['speedup'].idxmax()]
        best_subset = df[(df['M'] == best_config['M']) & 
                        (df['K'] == best_config['K']) & 
                        (df['N'] == best_config['N'])]
        
        plt.plot(best_subset['actual_sparsity'], best_subset['time_standard'], 
                'o-', linewidth=2, markersize=8, label='Standard Method')
        plt.plot(best_subset['actual_sparsity'], best_subset['time_binary'], 
                's-', linewidth=2, markersize=8, label='Binary Sparse Method')
        
        plt.xlabel('Sparsity (fraction of zeros)', fontsize=12)
        plt.ylabel('Execution Time (ms)', fontsize=12)
        plt.title(f'Performance Comparison: {best_config["M"]}×{best_config["K"]}×{best_config["N"]} Matrix', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add speedup annotations
        for _, row in best_subset.iterrows():
            if row['actual_sparsity'] >= 0.5:  # Only annotate high sparsity points
                plt.annotate(f'{row["speedup"]:.1f}x', 
                           xy=(row['actual_sparsity'], row['time_binary']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_best.png', dpi=300, bbox_inches='tight')
    print(f"Best results plot saved to {output_prefix}_best.png")
    
    plt.show()

def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"Total configurations tested: {len(df)}")
    print(f"Correctness check pass rate: {df['correct'].mean()*100:.1f}%")
    
    # Find best speedups
    max_speedup = df['speedup'].max()
    best_row = df.loc[df['speedup'].idxmax()]
    print(f"\nBest speedup: {max_speedup:.2f}x")
    print(f"  Configuration: {best_row['M']}×{best_row['K']}×{best_row['N']}")
    print(f"  Sparsity: {best_row['actual_sparsity']:.1%}")
    print(f"  Times: {best_row['time_standard']:.3f}ms → {best_row['time_binary']:.3f}ms")
    
    # Analyze sparsity threshold for speedup
    speedup_configs = df[df['speedup'] > 1.0]
    if len(speedup_configs) > 0:
        min_beneficial_sparsity = speedup_configs['actual_sparsity'].min()
        print(f"\nSpeedup threshold: {min_beneficial_sparsity:.1%} sparsity")
        print(f"Configurations with speedup: {len(speedup_configs)}/{len(df)} ({len(speedup_configs)/len(df)*100:.1f}%)")
    
    # Performance analysis by matrix size
    print(f"\nPerformance by matrix size:")
    for _, size in df[['M', 'K', 'N']].drop_duplicates().iterrows():
        subset = df[(df['M'] == size['M']) & (df['K'] == size['K']) & (df['N'] == size['N'])]
        if len(subset) > 0:
            avg_speedup = subset['speedup'].mean()
            max_speedup = subset['speedup'].max()
            print(f"  {size['M']}×{size['K']}×{size['N']}: avg={avg_speedup:.2f}x, max={max_speedup:.2f}x")

if __name__ == '__main__':
    sys.exit(main())