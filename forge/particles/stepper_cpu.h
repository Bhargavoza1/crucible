#pragma once
// CPU particle stepper using the same uniform grid as the GPU (same keys, same
// stable sort order, same for_each_neighbor). Much faster than the brute-force
// answer key (step_particles_cpu), so the sandbox CPU backend can run big piles.
// It is checked against brute force in the tests, like the GPU stepper.
#include <cstdint>
#include <vector>

#include "forge/math/vec3.h"
#include "forge/particles/particles.h"

namespace forge {

class ParticleStepperCpu {
public:
    void step(const ParticleView& state, const ParticleParams& params, int steps);

private:
    void build_grid(const ParticleView& s, const ParticleParams& p);

    std::vector<uint32_t> keys_, cell_start_, cell_end_;
    std::vector<int> sorted_;
    std::vector<Vec3> accel_;
    uint32_t table_size_ = 0;
};

}  // namespace forge
