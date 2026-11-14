#include "forge/particles/integrate_gpu.h"

#include "forge/particles/particle_step.h"
#include "forge/util/cuda_check.h"

namespace forge {

namespace {

// One thread per particle. Neighbor threads read neighbor floats in each SoA
// array, so every warp reads one contiguous block of memory (coalesced).
__global__ void integrate_kernel(ParticleView s, ParticleParams p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.count()) return;
    integrate_particle(s, i, p);
}

constexpr int kBlock = 256;

}  // namespace

void integrate_gpu(const ParticleView& state, const ParticleParams& params, int steps,
                   cudaStream_t stream) {
    const int n = state.count();
    if (n == 0 || steps <= 0) return;  // a launch with 0 blocks is an error
    const int grid = (n + kBlock - 1) / kBlock;
    for (int s = 0; s < steps; ++s) {
        integrate_kernel<<<grid, kBlock, 0, stream>>>(state, params);
        KERNEL_CHECK();
    }
}

}  // namespace forge
