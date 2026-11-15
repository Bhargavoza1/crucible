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
// Smaller blocks for the force kernel: more blocks, so small scenes still
// spread over all 60 SMs (ncu: "grid too small" with 40 blocks of 256).
constexpr int kAccelBlock = 128;

int grid_for(int n, int block = kBlock) { return (n + block - 1) / block; }

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

// ---- Sorted layout ------------------------------------------------------------

// Copy particle data in sorted (cell) order into packed arrays.
__global__ void gather_sorted_kernel(ParticleView s, const int* idx_sorted, float4* pos_r, float4* vel_m,
                                     int* env) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= s.count()) return;
    const int i = idx_sorted[k];
    pos_r[k] = make_float4(s.px[i], s.py[i], s.pz[i], s.radius[i]);
    vel_m[k] = make_float4(s.vx[i], s.vy[i], s.vz[i], s.inv_mass[i]);
    env[k] = s.env_id[i];
}

// Thread k = k-th particle in sorted order. Same math and the same order of
// additions as accel_kernel (gravity, planes in order, then neighbors cell by
// cell, sorted order inside a cell), so the result is bit for bit the same.
__global__ void accel_sorted_kernel(int n, const float4* pos_r, const float4* vel_m, const int* env_s,
                                    const int* idx_sorted, ParticleParams p, ContactCoeffs cc, GridView g,
                                    float* ax, float* ay, float* az) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const float4 pr = pos_r[k];
    const float4 vm = vel_m[k];
    const Vec3 pi{pr.x, pr.y, pr.z};
    const Vec3 vi{vm.x, vm.y, vm.z};
    const float ri = pr.w, inv_mi = vm.w;

    Vec3 a = p.gravity;
    if (inv_mi != 0.0f) {
        const float m = 1.0f / inv_mi;
        for (int q = 0; q < p.num_planes; ++q) a += plane_contact_force(p.planes[q], pi, ri, vi, m, cc) * inv_mi;

        const int env = env_s[k];
        const CellCoord c = cell_of(pi, g.inv_cell);
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const CellCoord nc{c.x + dx, c.y + dy, c.z + dz};
                    const uint32_t key = cell_hash(env, nc, g.mask);
                    const uint32_t end = g.cell_end[key];
                    for (uint32_t kk = g.cell_start[key]; kk < end; ++kk) {
                        if (int(kk) == k || env_s[kk] != env) continue;
                        const float4 pj4 = pos_r[kk];  // one 16-byte load: position + radius
                        const Vec3 pj{pj4.x, pj4.y, pj4.z};
                        if (!(cell_of(pj, g.inv_cell) == nc)) continue;
                        float d2;
                        if (!spheres_overlap(pi, ri, pj, pj4.w, d2)) continue;  // far: skip the velocity load
                        const float4 vj4 = vel_m[kk];
                        a += sphere_contact_force(pi, ri, vi, inv_mi, pj, pj4.w, Vec3{vj4.x, vj4.y, vj4.z}, vj4.w,
                                                  cc) *
                             inv_mi;
                    }
                }
    }
    const int i = idx_sorted[k];  // write back to the particle's own index
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
        pos_r_sorted_.resize(n);
        vel_m_sorted_.resize(n);
        env_sorted_.resize(n);
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
    const bool sorted = sorted_layout_ && p.particle_collisions;
    for (int k = 0; k < steps; ++k) {
        if (p.particle_collisions) build_grid(s, p, stream);
        if (sorted) {
            gather_sorted_kernel<<<grid_for(n), kBlock, 0, stream>>>(s, idx_sorted_.data(), pos_r_sorted_.data(),
                                                                     vel_m_sorted_.data(), env_sorted_.data());
            KERNEL_CHECK();
            accel_sorted_kernel<<<grid_for(n, kAccelBlock), kAccelBlock, 0, stream>>>(
                n, pos_r_sorted_.data(), vel_m_sorted_.data(), env_sorted_.data(), idx_sorted_.data(), p, cc, g,
                ax_.data(), ay_.data(), az_.data());
        } else {
            accel_kernel<<<grid_for(n, kAccelBlock), kAccelBlock, 0, stream>>>(
                s, p, cc, g, p.particle_collisions, ax_.data(), ay_.data(), az_.data());
        }
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
