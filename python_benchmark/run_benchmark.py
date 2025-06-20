#!/usr/bin/env python3
"""
Simple script to run the complete benchmark suite
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install Python requirements"""
    try:
        print("📦 Installing Python requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def main():
    """Main entry point"""
    print("🚀 Sparse Matrix Multiplication Benchmark Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("config/benchmark_config.yaml").exists():
        print("❌ Config file not found. Please run from python_benchmark directory.")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Run benchmark
    try:
        print("\n🎯 Starting benchmark runner...")
        from benchmark_runner import main as run_benchmark
        run_benchmark()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()