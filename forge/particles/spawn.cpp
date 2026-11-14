#include "forge/particles/spawn.h"

#include <random>

namespace forge {

namespace {

// Uniform float in [-1, 1) from the top 24 bits of one mt19937 output.
float uniform_pm1(std::mt19937& g) {
    const uint32_t bits = g() >> 8;                   // 24 random bits
    return float(bits) * (2.0f / 16777216.0f) - 1.0f;  // 2^24 = 16777216
}

}  // namespace

void spawn_box(ParticlesHost& out, int num_envs, int particles_per_env, const SpawnBox& box,
               uint32_t seed) {
    out.resize(num_envs, particles_per_env);
    for (int e = 0; e < num_envs; ++e) {
        // Seed per env, so env 3 looks the same whether we spawn 4 envs or 400.
        std::mt19937 g(seed ^ (0x9E3779B9u * static_cast<uint32_t>(e + 1)));
        for (int b = 0; b < particles_per_env; ++b) {
            const int i = env_index(e, b, particles_per_env);
            out.px[i] = box.center.x + box.half_extent.x * uniform_pm1(g);
            out.py[i] = box.center.y + box.half_extent.y * uniform_pm1(g);
            out.pz[i] = box.center.z + box.half_extent.z * uniform_pm1(g);
            out.vx[i] = box.max_speed * uniform_pm1(g);
            out.vy[i] = box.max_speed * uniform_pm1(g);
            out.vz[i] = box.max_speed * uniform_pm1(g);
            out.radius[i] = box.radius;
            out.inv_mass[i] = 1.0f;
        }
    }
}

}  // namespace forge
