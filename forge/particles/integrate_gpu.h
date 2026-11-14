#pragma once
// GPU integrator. Same math as integrate_cpu (both call integrate_particle).
#include <cuda_runtime.h>

#include "forge/particles/particles.h"

namespace forge {

// Launches 'steps' kernels on 'stream', one per physics step. Returns at once
// (asynchronous). 'state' must point to device memory (ParticlesDevice::view()).
// Later phases put collision and solver kernels between the steps, so we keep
// one launch per step instead of looping inside the kernel.
void integrate_gpu(const ParticleView& state, const ParticleParams& params, int steps,
                   cudaStream_t stream = nullptr);

}  // namespace forge
