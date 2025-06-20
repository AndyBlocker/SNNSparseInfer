#include "../../include/benchmark.h"

void gen_matrices(std::vector<float>& W,
                 std::vector<float>& A,
                 std::vector<float>& B,
                 int M, int K, int N, int tile, float sparsity)
{
    std::generate(W.begin(), W.end(), frand);
    std::generate(B.begin(), B.end(), frand);

    const int nTiles = (N + tile - 1) / tile, kTiles = (K + tile - 1) / tile;
    std::bernoulli_distribution bern(1.f - sparsity);
    std::mt19937 rng{std::random_device{}()};
    
    for(int kt = 0; kt < kTiles; ++kt) {
        for(int nt = 0; nt < nTiles; ++nt) {
            const bool active = bern(rng);
            for(int k = kt * tile; k < std::min((kt + 1) * tile, K); ++k) {
                for(int n = nt * tile; n < std::min((nt + 1) * tile, N); ++n) {
                    A[k + size_t(n) * K] = active ? frand() : 0.f;
                }
            }
        }
    }
}