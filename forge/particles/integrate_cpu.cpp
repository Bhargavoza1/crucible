#include "forge/particles/integrate_cpu.h"

#include "forge/particles/particle_step.h"

namespace forge {

void integrate_cpu(const ParticleView& state, const ParticleParams& params, int steps) {
    const int n = state.count();
    for (int s = 0; s < steps; ++s)
        for (int i = 0; i < n; ++i) integrate_particle(state, i, params);
}

}  // namespace forge
