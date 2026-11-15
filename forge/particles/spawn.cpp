#include "forge/particles/spawn.h"

#include <random>
#include <stdexcept>

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

void spawn_lattice(ParticlesHost& out, int num_envs, int particles_per_env, const SpawnLattice& lat,
                   uint32_t seed) {
    if (lat.spacing <= 2.0f * lat.radius) throw std::invalid_argument("spawn_lattice: spacing must be > 2 * radius");
    if (lat.nx <= 0 || lat.ny <= 0) throw std::invalid_argument("spawn_lattice: nx and ny must be > 0");
    out.resize(num_envs, particles_per_env);
    const float gap = lat.spacing - 2.0f * lat.radius;
    const float j = 0.5f * lat.jitter * gap;  // max offset per axis keeps neighbors apart
    for (int e = 0; e < num_envs; ++e) {
        std::mt19937 g(seed ^ (0x9E3779B9u * static_cast<uint32_t>(e + 1)));
        for (int b = 0; b < particles_per_env; ++b) {
            const int ix = b % lat.nx;
            const int iy = (b / lat.nx) % lat.ny;
            const int iz = b / (lat.nx * lat.ny);
            const int i = env_index(e, b, particles_per_env);
            out.px[i] = lat.origin.x + lat.radius + ix * lat.spacing + j * uniform_pm1(g);
            out.py[i] = lat.origin.y + lat.radius + iy * lat.spacing + j * uniform_pm1(g);
            out.pz[i] = lat.origin.z + lat.radius + iz * lat.spacing + j * uniform_pm1(g);
            out.vx[i] = out.vy[i] = out.vz[i] = 0.0f;
            out.radius[i] = lat.radius;
            out.inv_mass[i] = 1.0f;
        }
    }
}

}  // namespace forge
