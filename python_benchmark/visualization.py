#!/usr/bin/env python3
"""
Simplified visualization module for sparse matrix multiplication benchmarks
Focuses on key performance plots: sparsity curves and workload size comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class BenchmarkVisualizer:
    """Simplified visualization class for benchmark results"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.plot_config = config.get('plots', {})
        self.setup_matplotlib_style()
        
    def setup_matplotlib_style(self):
        """Setup matplotlib styling"""
        plt.style.use(self.plot_config.get('style', 'seaborn-v0_8'))
        plt.rcParams['figure.figsize'] = (
            self.plot_config.get('width', 1200) / 100,
            self.plot_config.get('height', 800) / 100
        )
        plt.rcParams['figure.dpi'] = self.plot_config.get('dpi', 300)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        
    def generate_all_plots(self, df: pd.DataFrame, output_dir: Path):
        """Generate key visualization plots"""
        print("🎨 Generating visualization plots...")
        
        # Create plots subdirectory
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate key plots
        self.plot_sparsity_curves(df, plots_dir)
        self.plot_workload_size_performance(df, plots_dir)
        self.plot_algorithm_comparison_summary(df, plots_dir)
        
        print(f"📊 Plots saved to {plots_dir}")
    
    def plot_sparsity_curves(self, df: pd.DataFrame, output_dir: Path):
        """Generate sparsity performance curves"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance vs Sparsity Level Analysis', fontsize=18, y=0.95)
        
        # Filter for standard configuration (tile=32, medium matrix size)
        df_filtered = df[df['tile'] == 32].copy()
        
        # If we have multiple matrix sizes, pick a representative one
        if len(df_filtered['M'].unique()) > 1:
            medium_size = sorted(df_filtered['M'].unique())[len(df_filtered['M'].unique())//2]
            df_main = df_filtered[df_filtered['M'] == medium_size].copy()
        else:
            df_main = df_filtered.copy()
        
        # If no data, use all available
        if df_main.empty:
            df_main = df.copy()
        
        # Aggregate by sparsity level
        sparsity_data = df_main.groupby('sparsity').agg({
            'speedup_basic_vs_dense': ['mean', 'std'],
            'speedup_warp_gather_vs_dense': ['mean', 'std'],
            'best_dense_ms': 'mean',
            'ms_sparse_basic': 'mean',
            'ms_sparse_warp_gather': 'mean',
            'rms_error_basic': 'mean',
            'rms_error_warp_gather': 'mean'
        }).reset_index()
        
        # Plot 1: Speedup curves
        ax1 = axes[0, 0]
        x = sparsity_data['sparsity']
        y1_mean = sparsity_data[('speedup_basic_vs_dense', 'mean')]
        y1_std = sparsity_data[('speedup_basic_vs_dense', 'std')]
        y2_mean = sparsity_data[('speedup_warp_gather_vs_dense', 'mean')]
        y2_std = sparsity_data[('speedup_warp_gather_vs_dense', 'std')]
        
        ax1.errorbar(x, y1_mean, yerr=y1_std, marker='o', linewidth=3, markersize=8,
                    label='Basic Sparse vs Dense', capsize=5, capthick=2)
        ax1.errorbar(x, y2_mean, yerr=y2_std, marker='s', linewidth=3, markersize=8,
                    label='Warp Gather Sparse vs Dense', capsize=5, capthick=2)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Break-even')
        ax1.set_xlabel('Sparsity Level', fontweight='bold')
        ax1.set_ylabel('Speedup vs Dense', fontweight='bold')
        ax1.set_title('Speedup vs Sparsity Level', fontweight='bold')
        ax1.legend(frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # Plot 2: Absolute performance
        ax2 = axes[0, 1]
        ax2.plot(x, sparsity_data[('best_dense_ms', 'mean')], '^-', linewidth=3, markersize=10,
                label='Best Dense', color='green')
        ax2.plot(x, sparsity_data[('ms_sparse_basic', 'mean')], 'o-', linewidth=3, markersize=8,
                label='Basic Sparse', color='blue')
        ax2.plot(x, sparsity_data[('ms_sparse_warp_gather', 'mean')], 's-', linewidth=3, markersize=8,
                label='Warp Gather Sparse', color='orange')
        ax2.set_xlabel('Sparsity Level', fontweight='bold')
        ax2.set_ylabel('Execution Time (ms)', fontweight='bold')
        ax2.set_title('Absolute Performance vs Sparsity', fontweight='bold')
        ax2.legend(frameon=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Efficiency analysis
        ax3 = axes[1, 0]
        # Calculate theoretical maximum speedup (1 / (1 - sparsity))
        theoretical_speedup = 1 / (1 - x.clip(upper=0.999))  # Avoid division by zero
        actual_speedup = y1_mean
        efficiency = actual_speedup / theoretical_speedup * 100
        
        ax3.plot(x, efficiency, 'o-', linewidth=3, markersize=8, color='purple', label='Basic Sparse Efficiency')
        ax3.axhline(y=100, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Perfect Efficiency (100%)')
        ax3.set_xlabel('Sparsity Level', fontweight='bold')
        ax3.set_ylabel('Efficiency (%)', fontweight='bold')
        ax3.set_title('Implementation Efficiency vs Theoretical Maximum', fontweight='bold')
        ax3.legend(frameon=True, shadow=True)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, max(efficiency.max() * 1.1, 110))
        
        # Plot 4: Error vs sparsity
        ax4 = axes[1, 1]
        ax4.semilogy(x, sparsity_data[('rms_error_basic', 'mean')], 'o-', linewidth=3, markersize=8,
                    label='Basic Sparse', color='blue')
        ax4.semilogy(x, sparsity_data[('rms_error_warp_gather', 'mean')], 's-', linewidth=3, markersize=8,
                    label='Warp Gather Sparse', color='orange')
        ax4.set_xlabel('Sparsity Level', fontweight='bold')
        ax4.set_ylabel('RMS Error', fontweight='bold')
        ax4.set_title('Numerical Accuracy vs Sparsity', fontweight='bold')
        ax4.legend(frameon=True, shadow=True)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sparsity_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_workload_size_performance(self, df: pd.DataFrame, output_dir: Path):
        """Generate workload size performance bar plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance vs Matrix Size Analysis', fontsize=18, y=0.95)
        
        # Filter for high sparsity (worst case) and standard tile
        df_filtered = df[(df['sparsity'] >= 0.8) & (df['tile'] == 32)].copy()
        if df_filtered.empty:
            df_filtered = df[df['tile'] == 32].copy()
        if df_filtered.empty:
            df_filtered = df.copy()
        
        # Aggregate by matrix size
        size_data = df_filtered.groupby('M').agg({
            'best_dense_ms': 'mean',
            'ms_sparse_basic': 'mean',
            'ms_sparse_warp_gather': 'mean',
            'speedup_basic_vs_dense': 'mean',
            'speedup_warp_gather_vs_dense': 'mean'
        }).reset_index()
        
        # Plot 1: Performance comparison bar chart
        ax1 = axes[0, 0]
        x_pos = np.arange(len(size_data))
        width = 0.25
        
        bars1 = ax1.bar(x_pos - width, size_data['best_dense_ms'], width, 
                       label='Best Dense', color='green', alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x_pos, size_data['ms_sparse_basic'], width,
                       label='Basic Sparse', color='blue', alpha=0.8, edgecolor='black')
        bars3 = ax1.bar(x_pos + width, size_data['ms_sparse_warp_gather'], width,
                       label='Warp Gather Sparse', color='orange', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        ax1.set_xlabel('Matrix Size', fontweight='bold')
        ax1.set_ylabel('Execution Time (ms)', fontweight='bold')
        ax1.set_title('Performance Comparison by Matrix Size', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'{int(m)}x{int(m)}' for m in size_data['M']])
        ax1.legend(frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_yscale('log')
        
        # Plot 2: Speedup comparison
        ax2 = axes[0, 1]
        bars1 = ax2.bar(x_pos - width/2, size_data['speedup_basic_vs_dense'], width,
                       label='Basic vs Dense', color='blue', alpha=0.8, edgecolor='black')
        bars2 = ax2.bar(x_pos + width/2, size_data['speedup_warp_gather_vs_dense'], width,
                       label='Warp Gather vs Dense', color='orange', alpha=0.8, edgecolor='black')
        
        # Add speedup value labels
        for bar in bars1:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}x',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}x',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Break-even')
        ax2.set_xlabel('Matrix Size', fontweight='bold')
        ax2.set_ylabel('Speedup vs Dense', fontweight='bold')
        ax2.set_title('Speedup Comparison by Matrix Size', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{int(m)}x{int(m)}' for m in size_data['M']])
        ax2.legend(frameon=True, shadow=True)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Scaling efficiency
        ax3 = axes[1, 0]
        # Calculate GFLOPS for each implementation
        flops = 2 * size_data['M'] * size_data['M'] * size_data['M']  # 2*M*K*N for square matrices
        gflops_dense = flops / (size_data['best_dense_ms'] / 1000) / 1e9
        gflops_basic = flops / (size_data['ms_sparse_basic'] / 1000) / 1e9
        gflops_warp_gather = flops / (size_data['ms_sparse_warp_gather'] / 1000) / 1e9
        
        ax3.plot(size_data['M'], gflops_dense, '^-', linewidth=3, markersize=10,
                label='Dense GFLOPS', color='green')
        ax3.plot(size_data['M'], gflops_basic, 'o-', linewidth=3, markersize=8,
                label='Basic Sparse GFLOPS', color='blue')
        ax3.plot(size_data['M'], gflops_warp_gather, 's-', linewidth=3, markersize=8,
                label='Warp Gather Sparse GFLOPS', color='orange')
        
        ax3.set_xlabel('Matrix Size', fontweight='bold')
        ax3.set_ylabel('Performance (GFLOPS)', fontweight='bold')
        ax3.set_title('Computational Performance Scaling', fontweight='bold')
        ax3.legend(frameon=True, shadow=True)
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # Plot 4: Memory bandwidth utilization
        ax4 = axes[1, 1]
        # Estimate memory bandwidth (assuming 3 matrices: W, A, B in FP32)
        memory_gb = 3 * size_data['M'] * size_data['M'] * 4 / (1024**3)
        bandwidth_dense = memory_gb / (size_data['best_dense_ms'] / 1000)
        bandwidth_basic = memory_gb / (size_data['ms_sparse_basic'] / 1000)
        
        ax4.plot(size_data['M'], bandwidth_dense, '^-', linewidth=3, markersize=10,
                label='Dense Bandwidth', color='green')
        ax4.plot(size_data['M'], bandwidth_basic, 'o-', linewidth=3, markersize=8,
                label='Basic Sparse Bandwidth', color='blue')
        
        ax4.set_xlabel('Matrix Size', fontweight='bold')
        ax4.set_ylabel('Effective Bandwidth (GB/s)', fontweight='bold')
        ax4.set_title('Memory Bandwidth Utilization', fontweight='bold')
        ax4.legend(frameon=True, shadow=True)
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'workload_size_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_algorithm_comparison_summary(self, df: pd.DataFrame, output_dir: Path):
        """Generate summary comparison of all algorithms"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithm Performance Summary', fontsize=18, y=0.95)
        
        # Plot 1: Overall performance distribution
        ax1 = axes[0, 0]
        algorithms = ['cuBLAS\nSGEMM', 'cuBLASLt\nTF32', 'cuBLASLt\nOptimal', 'Basic\nSparse', 'Warp Gather\nSparse']
        performance_data = [
            df['ms_sgemm'].values,
            df['ms_lt_tf32'].values,
            df['ms_lt_optimal'].values,
            df['ms_sparse_basic'].values,
            df['ms_sparse_warp_gather'].values
        ]
        
        bp = ax1.boxplot(performance_data, labels=algorithms, patch_artist=True, showfliers=False)
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'khaki', 'plum']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax1.set_ylabel('Execution Time (ms)', fontweight='bold')
        ax1.set_title('Performance Distribution by Algorithm', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_yscale('log')
        
        # Plot 2: Speedup summary
        ax2 = axes[0, 1]
        speedup_categories = ['Basic vs\nDense', 'Warp Gather vs\nDense', 'Warp Gather vs\nBasic']
        speedup_data = [
            df['speedup_basic_vs_dense'].values,
            df['speedup_warp_gather_vs_dense'].values,
            df['speedup_warp_gather_vs_basic'].values
        ]
        
        bp2 = ax2.boxplot(speedup_data, labels=speedup_categories, patch_artist=True)
        colors2 = ['lightblue', 'lightcoral', 'lightgreen']
        for patch, color in zip(bp2['boxes'], colors2):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Break-even')
        ax2.set_ylabel('Speedup Factor', fontweight='bold')
        ax2.set_title('Speedup Distribution Summary', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Performance vs accuracy scatter
        ax3 = axes[1, 0]
        scatter1 = ax3.scatter(df['rms_error_basic'], df['speedup_basic_vs_dense'], 
                             alpha=0.6, s=60, c=df['sparsity'], cmap='viridis', 
                             label='Basic Sparse', edgecolors='black', linewidth=0.5)
        scatter2 = ax3.scatter(df['rms_error_warp_gather'], df['speedup_warp_gather_vs_dense'],
                             alpha=0.6, s=60, c=df['sparsity'], cmap='plasma', marker='^',
                             label='Warp Gather Sparse', edgecolors='black', linewidth=0.5)
        
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax3.set_xlabel('RMS Error', fontweight='bold')
        ax3.set_ylabel('Speedup vs Dense', fontweight='bold')
        ax3.set_title('Performance vs Accuracy Trade-off', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Add colorbar
        cbar = plt.colorbar(scatter1, ax=ax3)
        cbar.set_label('Sparsity Level', fontweight='bold')
        
        # Plot 4: Best algorithm recommendation
        ax4 = axes[1, 1]
        
        # Determine best algorithm for each configuration
        def get_best_algorithm(row):
            times = {
                'Dense (TF32)': row['ms_lt_tf32'],
                'Basic Sparse': row['ms_sparse_basic'],
                'Warp Gather Sparse': row['ms_sparse_warp_gather']
            }
            return min(times, key=times.get)
        
        df['best_algorithm'] = df.apply(get_best_algorithm, axis=1)
        best_counts = df['best_algorithm'].value_counts()
        
        colors_pie = ['lightgreen', 'lightblue', 'lightcoral']
        wedges, texts, autotexts = ax4.pie(best_counts.values, labels=best_counts.index, 
                                          autopct='%1.1f%%', colors=colors_pie,
                                          explode=[0.05] * len(best_counts),
                                          shadow=True, startangle=90)
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        ax4.set_title('Best Algorithm Distribution\n(Across All Configurations)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'algorithm_comparison_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

if __name__ == "__main__":
    # Test visualization with sample data
    print("Testing simplified visualization module...")
    
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'M': [1024, 2048, 4096] * 7,
        'sparsity': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99] * 3,
        'tile': [32] * 21,
        'ms_sgemm': np.random.uniform(0.1, 2.0, 21),
        'ms_lt_tf32': np.random.uniform(0.05, 1.0, 21),
        'ms_lt_optimal': np.random.uniform(0.05, 1.0, 21),
        'best_dense_ms': np.random.uniform(0.05, 1.0, 21),
        'ms_sparse_basic': np.random.uniform(0.1, 5.0, 21),
        'ms_sparse_warp_gather': np.random.uniform(0.2, 6.0, 21),
        'speedup_basic_vs_dense': np.random.uniform(0.1, 2.0, 21),
        'speedup_warp_gather_vs_dense': np.random.uniform(0.05, 1.5, 21),
        'speedup_warp_gather_vs_basic': np.random.uniform(0.5, 1.2, 21),
        'rms_error_basic': np.random.uniform(1e-8, 1e-3, 21),
        'rms_error_warp_gather': np.random.uniform(1e-7, 1e-2, 21)
    }
    
    df = pd.DataFrame(sample_data)
    
    config = {
        'plots': {
            'width': 1200,
            'height': 800,
            'dpi': 300,
            'style': 'seaborn-v0_8'
        }
    }
    
    viz = BenchmarkVisualizer(config)
    output_dir = Path('test_plots')
    output_dir.mkdir(exist_ok=True)
    
    viz.generate_all_plots(df, output_dir)
    print("✅ Test visualization completed")