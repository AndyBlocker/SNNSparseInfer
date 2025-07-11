#!/bin/bash

# Debug NVTX functionality
set -e

cd "$(dirname "$0")/.."

echo "=== Debug NVTX Setup ==="

echo "1. Testing with nvtx-dump to see available NVTX ranges..."
timeout 10 nvtx-dump ./sparse_mm_benchmark 256 256 256 32 0.9 || echo "nvtx-dump not available"

echo ""
echo "2. Testing with ncu --print-nvtx-rename..."
timeout 30 ncu --nvtx --print-nvtx-rename kernel --metrics gpu__time_duration.sum --csv ./sparse_mm_benchmark 256 256 256 32 0.9

echo ""
echo "3. Testing without NVTX filters to see all kernels..."
timeout 30 ncu --metrics gpu__time_duration.sum --csv ./sparse_mm_benchmark 256 256 256 32 0.9