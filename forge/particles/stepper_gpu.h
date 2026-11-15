#pragma once
// GPU particle stepper WITH collisions (Phase 4). Same math as
// step_particles_cpu, but sphere-sphere pairs come from a uniform grid instead
// of brute force.
//
// Per step (all on one stream, no CPU wait):
//   1. cell key per particle            (hash of env + cell)
//   2. sort particles by key            (CUB radix sort, stable -> same order every run)
//   3. start/end of each key            (in the sorted list)
//   4. copy particles in sorted order   (sorted layout only, see below)
//   5. acceleration per particle        (gravity + planes + the 27 cells around it)
//   6. integrate                        (semi-implicit Euler, original order)
// Each particle gathers its own forces, so there are no atomics.
//
// Sorted layout (default, from profiling with Nsight Compute): step 4 copies
// the particles in cell order into packed float4 arrays, and thread k works on
// the k-th particle in that order. Neighbor threads are then neighbors in
// space: they read the same cells (coalesced, cached) and loop about the same
// number of times (less idle threads). Forces are added in exactly the same
// order as the original layout, so both give bit-for-bit the same result.
//
// Host code only; owns its scratch buffers. Reuse one object for the whole run.
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "forge/particles/particles.h"
#include "forge/util/device_buffer.h"

namespace forge {

class ParticleStepperGpu {
public:
    // 'state' must point to device memory. Asynchronous.
    void step(const ParticleView& state, const ParticleParams& params, int steps, cudaStream_t stream = nullptr);

    // Number of other particles each particle touches, found through the grid.
    // For tests: compare with contact_counts_cpu. Waits for the GPU.
    std::vector<int> contact_counts(const ParticleView& state, const ParticleParams& params,
                                    cudaStream_t stream = nullptr);

    // true (default): sorted packed layout. false: original layout (one thread
    // per particle index, 9 separate arrays). Kept to compare speed and results.
    void set_sorted_layout(bool on) { sorted_layout_ = on; }
    bool sorted_layout() const { return sorted_layout_; }

    // Tests only: force a tiny hash table so many cells share a hash.
    // 0 = automatic (next power of two >= 2 * particles, at least 1024).
    void force_table_size(uint32_t size) { forced_table_size_ = size; }
    uint32_t table_size() const { return table_size_; }

private:
    void ensure_capacity(int n);
    void build_grid(const ParticleView& s, const ParticleParams& p, cudaStream_t stream);

    DeviceBuffer<uint32_t> keys_, keys_sorted_;
    DeviceBuffer<int> idx_, idx_sorted_;
    DeviceBuffer<uint32_t> cell_start_, cell_end_;
    DeviceBuffer<float> ax_, ay_, az_;
    DeviceBuffer<float4> pos_r_sorted_;  // x, y, z, radius     (sorted layout)
    DeviceBuffer<float4> vel_m_sorted_;  // vx, vy, vz, inv_mass (sorted layout)
    DeviceBuffer<int> env_sorted_;
    bool sorted_layout_ = true;
    DeviceBuffer<unsigned char> sort_temp_;
    int capacity_ = 0;
    uint32_t table_size_ = 0;
    uint32_t forced_table_size_ = 0;
    int key_bits_ = 0;
};

}  // namespace forge
