#pragma once
// Counter-based random numbers (FORGE_HD): the same numbers on the CPU and the GPU.
// A stream is fully set by (seed, a, b), for example (run seed, env id, episode
// number). Env e's numbers do not depend on how many envs exist or on the order
// envs run in, so a 16-env run and a 4096-env run agree on env e.
#include <cstdint>

#include "forge/math/hd.h"

namespace forge {

FORGE_HD uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

struct Rng {
    uint64_t state;

    FORGE_HD Rng(uint64_t seed, uint64_t a, uint64_t b)
        : state(splitmix64(splitmix64(splitmix64(seed) ^ a) ^ (b * 0xD1B54A32D192ED03ull))) {}

    FORGE_HD uint64_t next() {
        state = splitmix64(state);
        return state;
    }
    // [0, 1), 24 random bits: exact in float.
    FORGE_HD float uniform() { return float(next() >> 40) * (1.0f / 16777216.0f); }
    // [lo, hi)
    FORGE_HD float uniform(float lo, float hi) { return lo + (hi - lo) * uniform(); }
};

}  // namespace forge
