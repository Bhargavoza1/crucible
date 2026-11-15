#pragma once
// CPU answer key for particles WITH collisions (Phase 4).
// Sphere-sphere uses brute force: every pair inside each env. That is O(n^2)
// and slow, but it does not use the grid at all, so it really tests the GPU grid.
// Use small scenes (a few thousand particles per env at most).
#include <vector>

#include "forge/particles/particles.h"

namespace forge {

// Runs 'steps' steps of params.dt. Each step: compute every acceleration
// (gravity + planes + spheres) from the current state, then integrate all.
void step_particles_cpu(const ParticleView& state, const ParticleParams& params, int steps);

// Number of other particles each particle touches (same env). For tests.
std::vector<int> contact_counts_cpu(const ParticleView& state);

}  // namespace forge
