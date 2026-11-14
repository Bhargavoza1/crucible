#pragma once
// Fills particles with random start positions and velocities inside a box.
// Same seed gives the same particles on every machine and compiler: the random
// numbers come straight from std::mt19937 bits, not from std::uniform_real_distribution
// (its output is not the same across standard libraries).
#include <cstdint>

#include "forge/math/vec3.h"
#include "forge/particles/particles.h"

namespace forge {

struct SpawnBox {
    Vec3 center{0.0f, 0.0f, 3.0f};       // m, env-local, Z-up
    Vec3 half_extent{1.0f, 1.0f, 1.0f};  // m
    float max_speed = 1.0f;              // m/s, each velocity component in [-max, max]
    float radius = 0.05f;                // m
};

// Resizes 'out' to num_envs * particles_per_env and fills every env.
// Each env gets different particles (the env index is mixed into the seed).
void spawn_box(ParticlesHost& out, int num_envs, int particles_per_env, const SpawnBox& box,
               uint32_t seed);

}  // namespace forge
