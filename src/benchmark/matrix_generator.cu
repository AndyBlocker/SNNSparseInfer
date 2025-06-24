#include "../../include/benchmark.h"

void gen_matrices(std::vector<float>& W,
                 std::vector<float>& A,
                 std::vector<float>& B,
                 int M, int K, int N, int tile, float sparsity)
{
    // Generate dense weight matrix W (always full)
    std::generate(W.begin(), W.end(), frand);
    
    // Generate dense bias matrix B (always full)  
    std::generate(B.begin(), B.end(), frand);

    // Generate activation matrix A with structured sparsity (Group Lasso pattern)
    // Each tile is either completely full or completely empty
    const int nTiles = (N + tile - 1) / tile;
    const int kTiles = (K + tile - 1) / tile;
    const int totalTiles = nTiles * kTiles;
    
    // Use tile-level sparsity: each tile has probability (1-sparsity) of being active
    std::bernoulli_distribution tileActive(1.f - sparsity);
    std::mt19937 rng{std::random_device{}()};
    
    // Initialize all of A to zero first
    std::fill(A.begin(), A.end(), 0.f);
    
    int activeTiles = 0;
    int totalElements = 0;
    int activeElements = 0;
    
    // Generate structured sparse pattern
    for(int kt = 0; kt < kTiles; ++kt) {
        for(int nt = 0; nt < nTiles; ++nt) {
            const bool isActive = tileActive(rng);
            if (isActive) activeTiles++;
            
            const int kStart = kt * tile;
            const int kEnd = std::min((kt + 1) * tile, K);
            const int nStart = nt * tile;
            const int nEnd = std::min((nt + 1) * tile, N);
            
            for(int k = kStart; k < kEnd; ++k) {
                for(int n = nStart; n < nEnd; ++n) {
                    totalElements++;
                    if (isActive) {
                        // Fill entire tile with non-zero values when active
                        A[k + size_t(n) * K] = frand();
                        activeElements++;
                    }
                    // else: remains 0.f (tile is completely empty)
                }
            }
        }
    }
    
    // Print structured sparsity statistics
    const float actualSparsity = 1.f - (float(activeElements) / float(totalElements));
    const float tileSparsity = 1.f - (float(activeTiles) / float(totalTiles));
    
    std::cout << "=== Structured Sparsity Statistics ===" << std::endl;
    std::cout << "Total tiles: " << totalTiles << " (K=" << kTiles << " x N=" << nTiles << ")" << std::endl;
    std::cout << "Active tiles: " << activeTiles << " (" << (100.f * activeTiles / totalTiles) << "%)" << std::endl;
    std::cout << "Tile-level sparsity: " << tileSparsity << std::endl;
    std::cout << "Element-level sparsity: " << actualSparsity << std::endl;
    std::cout << "Target sparsity: " << sparsity << std::endl;
    std::cout << "Tile size: " << tile << "x" << tile << std::endl << std::endl;
}