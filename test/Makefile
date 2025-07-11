# Makefile for SNNSparseInfer test programs
# Compiles binary sparse benchmark and MMA sparse capability checker

# Compiler and flags
NVCC = nvcc
CXXFLAGS = -O3 -std=c++17
NVCCFLAGS = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
LIBS = -lcublas

# Targets
TARGETS = binary_sparse_benchmark mma_sparse_check

# Default target
all: $(TARGETS)

# Binary sparse benchmark
binary_sparse_benchmark: binary_sparse_benchmark.cu
	$(NVCC) $(CXXFLAGS) $(NVCCFLAGS) $< -o $@ $(LIBS)

# MMA sparse capability checker
mma_sparse_check: mma_sparse_check.cu
	$(NVCC) $(CXXFLAGS) $(NVCCFLAGS) $< -o $@

# Clean
clean:
	rm -f $(TARGETS)

# Run tests
test-binary: binary_sparse_benchmark
	./binary_sparse_benchmark

test-sparse: mma_sparse_check
	./mma_sparse_check

test: test-sparse test-binary

# Help
help:
	@echo "Available targets:"
	@echo "  all                - Build all test programs"
	@echo "  binary_sparse_benchmark - Build binary sparse matrix benchmark"
	@echo "  mma_sparse_check   - Build MMA sparse capability checker"
	@echo "  test-binary        - Run binary sparse benchmark"
	@echo "  test-sparse        - Run MMA sparse capability check"
	@echo "  test               - Run all tests"
	@echo "  clean              - Remove built programs"
	@echo "  help               - Show this help message"

.PHONY: all clean test test-binary test-sparse help