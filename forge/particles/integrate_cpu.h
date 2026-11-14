#pragma once
// CPU reference integrator: the answer key for integrate_gpu (doc 9, rule:
// every physics kernel gets a CPU version first). Slow, simple, single thread.
#include "forge/particles/particles.h"

namespace forge {

// Runs 'steps' physics steps on host memory. steps <= 0 does nothing.
void integrate_cpu(const ParticleView& state, const ParticleParams& params, int steps);

}  // namespace forge
