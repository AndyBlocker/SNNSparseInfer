#include "../../include/common.h"

// Host random number generator
__host__ float frand() {
    static thread_local std::mt19937 g{std::random_device{}()};
    static thread_local std::uniform_real_distribution<float> d(-1.f, 1.f);
    return d(g);
}

// Note: ld4 device function is now defined inline in common.h