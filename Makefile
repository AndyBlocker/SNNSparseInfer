# CUDA Sparse Matrix Multiplication Benchmark
# Author: SNNSparseInfer
# Date: 2025-06-20

# Configuration
NVCC = nvcc
CUDA_ARCH = sm_86,sm_89
NVCC_FLAGS = -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -lineinfo -src-in-ptx
CUDA_LIBS = -lcublasLt -lcublas

# Optional optimizations
USE_CP_ASYNC ?= 1
ifeq ($(USE_CP_ASYNC), 1)
    NVCC_FLAGS += -DUSE_CP_ASYNC
endif

# Directories
SRC_DIR = src
KERNELS_DIR = $(SRC_DIR)/kernels
BENCHMARK_DIR = $(SRC_DIR)/benchmark
INCLUDE_DIR = include

# Source files
KERNEL_SOURCES = $(KERNELS_DIR)/kernel_utils.cu \
                 $(KERNELS_DIR)/sparse_mm_basic.cu \
                 $(KERNELS_DIR)/sparse_mm_pipeline.cu \
                 $(KERNELS_DIR)/sparse_mm_warp_gather.cu \
                 $(KERNELS_DIR)/sparse_mm_fused.cu \
                 $(KERNELS_DIR)/sparse_mm_gather_scatter.cu

BENCHMARK_SOURCES = $(BENCHMARK_DIR)/matrix_generator.cu \
                    $(BENCHMARK_DIR)/dense_baselines.cu \
                    $(BENCHMARK_DIR)/benchmark_utils.cu

MAIN_SOURCE = $(SRC_DIR)/main.cu

ALL_SOURCES = $(KERNEL_SOURCES) $(BENCHMARK_SOURCES) $(MAIN_SOURCE)

# Targets
TARGET = sparse_mm_benchmark
LEGACY_TARGET = sparse_mm_legacy

# Default target
all: $(TARGET)

# Main benchmark target
$(TARGET): $(ALL_SOURCES)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) $(ALL_SOURCES) $(CUDA_LIBS) -o $@

# Legacy target (for comparison with old version)
$(LEGACY_TARGET): sparse_mm.cu
	$(NVCC) $(NVCC_FLAGS) sparse_mm.cu $(CUDA_LIBS) -o $@

# Pipeline-only target (from new.cu)
pipeline: new.cu
	$(NVCC) $(NVCC_FLAGS) new.cu -lcublasLt -o sparse_mm_pipeline_only

# Clean
clean:
	rm -f $(TARGET) $(LEGACY_TARGET) sparse_mm_pipeline_only sparse_mm

# Test targets
test: $(TARGET)
	./$(TARGET) 1024 1024 1024 32 0.9

test-large: $(TARGET)
	./$(TARGET) 2048 2048 2048 32 0.9

test-variants: $(TARGET)
	@echo "Testing different sparsity levels..."
	./$(TARGET) 1024 1024 1024 32 0.5
	./$(TARGET) 1024 1024 1024 32 0.7
	./$(TARGET) 1024 1024 1024 32 0.9
	./$(TARGET) 1024 1024 1024 32 0.95

# CP.ASYNC comparison targets
test-cp-async: clean
	@echo "=== Testing WITH cp.async ==="
	$(MAKE) USE_CP_ASYNC=1
	./$(TARGET) 1024 1024 1024 32 0.9
	@echo ""
	@echo "=== Testing WITHOUT cp.async ==="
	$(MAKE) clean
	$(MAKE) USE_CP_ASYNC=0
	./$(TARGET) 1024 1024 1024 32 0.9

# Debugging targets
debug: NVCC_FLAGS += -g -G
debug: $(TARGET)

# Profile target
profile: $(TARGET)
	./scripts/profile_sparse_mm.sh

# Install CUDA paths (for convenience)
setup-cuda:
	@echo "export PATH=/usr/local/cuda/bin:\$$PATH" >> ~/.bashrc
	@echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$$LD_LIBRARY_PATH" >> ~/.bashrc
	@echo "CUDA paths added to ~/.bashrc. Please run 'source ~/.bashrc' or restart terminal."

# Help
help:
	@echo "Available targets:"
	@echo "  all         - Build main benchmark (default, with cp.async)"
	@echo "  $(TARGET)   - Build modular benchmark"
	@echo "  $(LEGACY_TARGET) - Build legacy version"
	@echo "  pipeline    - Build pipeline-only version"
	@echo "  test        - Run basic test"
	@echo "  test-large  - Run large matrix test"
	@echo "  test-variants - Test different sparsity levels"
	@echo "  test-cp-async - Compare with/without cp.async optimization"
	@echo "  debug       - Build with debug symbols"
	@echo "  profile     - Run profiling"
	@echo "  clean       - Remove build artifacts"
	@echo "  setup-cuda  - Add CUDA paths to bashrc"
	@echo "  help        - Show this help"
	@echo ""
	@echo "Build options:"
	@echo "  USE_CP_ASYNC=1  - Enable cp.async optimization (default)"
	@echo "  USE_CP_ASYNC=0  - Disable cp.async optimization"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build with cp.async"
	@echo "  make USE_CP_ASYNC=0     # Build without cp.async"
	@echo "  make test-cp-async      # Compare both versions"

.PHONY: all clean test test-large test-variants debug profile setup-cuda help