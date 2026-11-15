#include "forge/particles/stepper_gpu.h"

#include <algorithm>

#include <cub/device/device_radix_sort.cuh>

#include "forge/collision/particle_grid.h"
#include "forge/particles/grid_traversal.h"
#include "forge/particles/particle_step.h"
#include "forge/util/cuda_check.h"

namespace forge {

namespace {

constexpr int kBlock = 256;

int grid_for(int n) { return (n + kBlock - 1) / kBlock; }

__global__ void compute_keys_kernel(ParticleView s, float inv_cell, uint32_t mask, uint32_t* keys, int* idx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.count()) return;
    keys[i] = cell_hash(s.env_id[i], cell_of(particle_position(s, i), inv_cell), mask);
    idx[i] = i;
}

// After sorting, equal keys sit together. Mark where each run starts and ends.
// Empty keys keep start = end = 0 (from the memset), so their loop does nothing.
__global__ void cell_bounds_kernel(const uint32_t* keys_sorted, int n, uint32_t* start, uint32_t* end) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t k = keys_sorted[i];
    if (i == 0 || keys_sorted[i - 1] != k) start[k] = static_cast<uint32_t>(i);
    if (i == n - 1 || keys_sorted[i + 1] != k) end[k] = static_cast<uint32_t>(i + 1);
}

__global__ void accel_kernel(ParticleView s, ParticleParams p, ContactCoeffs cc, GridView g, bool use_grid,
                             float* ax, float* ay, float* az) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.count()) return;
    Vec3 a = accel_gravity_and_planes(s, i, p, cc);
    if (use_grid && s.inv_mass[i] != 0.0f)
        for_each_neighbor(s, i, g, [&](int j) { a += accel_from_particle(s, i, j, cc); });
    ax[i] = a.x;
    ay[i] = a.y;
    az[i] = a.z;
}

__global__ void integrate_accel_kernel(ParticleView s, const float* ax, const float* ay, const float* az,
                                       float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.count()) return;
    integrate_particle_accel(s, i, Vec3{ax[i], ay[i], az[i]}, dt);
}

__global__ void count_kernel(ParticleView s, GridView g, int* counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.count()) return;
    int c = 0;
    for_each_neighbor(s, i, g, [&](int j) {
        if (particles_touch(s, i, j)) ++c;
    });
    counts[i] = c;
}

uint32_t next_pow2(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

int log2_pow2(uint32_t v) {
    int b = 0;
    while ((1u << b) < v) ++b;
    return b;
}

}  // namespace

void ParticleStepperGpu::ensure_capacity(int n) {
    const uint32_t want_table =
        forced_table_size_ ? next_pow2(forced_table_size_) : next_pow2(std::max<uint32_t>(1024u, 2u * uint32_t(n)));
    if (n <= capacity_ && want_table == table_size_) return;

    if (n > capacity_) {
        for (auto* b : {&keys_, &keys_sorted_}) b->resize(n);
        for (auto* b : {&idx_, &idx_sorted_}) b->resize(n);
        for (auto* b : {&ax_, &ay_, &az_}) b->resize(n);
        capacity_ = n;
    }
    table_size_ = want_table;
    key_bits_ = std::max(1, log2_pow2(table_size_));
    cell_start_.resize(table_size_);
    cell_end_.resize(table_size_);

    // Ask CUB how much scratch memory the sort needs for this size (no work done).
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys_.data(), keys_sorted_.data(), idx_.data(),
                                               idx_sorted_.data(), capacity_, 0, key_bits_));
    sort_temp_.resize(std::max<size_t>(temp_bytes, 1));
}

void ParticleStepperGpu::build_grid(const ParticleView& s, const ParticleParams& p, cudaStream_t stream) {
    const int n = s.count();
    const float inv_cell = 1.0f / p.cell_size;
    const uint32_t mask = table_size_ - 1;
    compute_keys_kernel<<<grid_for(n), kBlock, 0, stream>>>(s, inv_cell, mask, keys_.data(), idx_.data());
    KERNEL_CHECK();
    size_t temp_bytes = sort_temp_.size();
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(sort_temp_.data(), temp_bytes, keys_.data(), keys_sorted_.data(),
                                               idx_.data(), idx_sorted_.data(), n, 0, key_bits_, stream));
    CUDA_CHECK(cudaMemsetAsync(cell_start_.data(), 0, table_size_ * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(cell_end_.data(), 0, table_size_ * sizeof(uint32_t), stream));
    cell_bounds_kernel<<<grid_for(n), kBlock, 0, stream>>>(keys_sorted_.data(), n, cell_start_.data(),
                                                            cell_end_.data());
    KERNEL_CHECK();
}

void ParticleStepperGpu::step(const ParticleView& s, const ParticleParams& p, int steps, cudaStream_t stream) {
    const int n = s.count();
    if (n == 0 || steps <= 0) return;
    ensure_capacity(n);
    const ContactCoeffs cc = contact_coeffs(p.contact);
    const GridView g{cell_start_.data(), cell_end_.data(), idx_sorted_.data(), 1.0f / p.cell_size,
                     table_size_ - 1};
    for (int k = 0; k < steps; ++k) {
        if (p.particle_collisions) build_grid(s, p, stream);
        accel_kernel<<<grid_for(n), kBlock, 0, stream>>>(s, p, cc, g, p.particle_collisions, ax_.data(),
                                                         ay_.data(), az_.data());
        KERNEL_CHECK();
        integrate_accel_kernel<<<grid_for(n), kBlock, 0, stream>>>(s, ax_.data(), ay_.data(), az_.data(), p.dt);
        KERNEL_CHECK();
    }
}

std::vector<int> ParticleStepperGpu::contact_counts(const ParticleView& s, const ParticleParams& p,
                                                    cudaStream_t stream) {
    const int n = s.count();
    if (n == 0) return {};
    ensure_capacity(n);
    build_grid(s, p, stream);
    DeviceBuffer<int> counts(n);
    const GridView g{cell_start_.data(), cell_end_.data(), idx_sorted_.data(), 1.0f / p.cell_size,
                     table_size_ - 1};
    count_kernel<<<grid_for(n), kBlock, 0, stream>>>(s, g, counts.data());
    KERNEL_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return counts.download();
}

}  // namespace forge
