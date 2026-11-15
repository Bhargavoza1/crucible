#include "forge/particles/stepper_cpu.h"

#include <algorithm>
#include <numeric>

#include "forge/collision/particle_grid.h"
#include "forge/particles/grid_traversal.h"
#include "forge/particles/particle_step.h"

namespace forge {

void ParticleStepperCpu::build_grid(const ParticleView& s, const ParticleParams& p) {
    const int n = s.count();
    uint32_t table = 1024;
    while (table < 2u * uint32_t(n)) table <<= 1;  // same rule as the GPU
    if (table != table_size_) {
        table_size_ = table;
        cell_start_.assign(table, 0);
        cell_end_.assign(table, 0);
    } else {
        std::fill(cell_start_.begin(), cell_start_.end(), 0u);
        std::fill(cell_end_.begin(), cell_end_.end(), 0u);
    }
    const float inv = 1.0f / p.cell_size;
    const uint32_t mask = table_size_ - 1;
    keys_.resize(n);
    sorted_.resize(n);
    for (int i = 0; i < n; ++i) keys_[i] = cell_hash(s.env_id[i], cell_of(particle_position(s, i), inv), mask);
    std::iota(sorted_.begin(), sorted_.end(), 0);
    std::stable_sort(sorted_.begin(), sorted_.end(), [&](int a, int b) { return keys_[a] < keys_[b]; });
    for (int k = 0; k < n; ++k) {
        const uint32_t key = keys_[sorted_[k]];
        if (k == 0 || keys_[sorted_[k - 1]] != key) cell_start_[key] = uint32_t(k);
        if (k == n - 1 || keys_[sorted_[k + 1]] != key) cell_end_[key] = uint32_t(k + 1);
    }
}

void ParticleStepperCpu::step(const ParticleView& s, const ParticleParams& p, int steps) {
    const int n = s.count();
    if (n == 0 || steps <= 0) return;
    const ContactCoeffs cc = contact_coeffs(p.contact);
    accel_.resize(n);
    for (int k = 0; k < steps; ++k) {
        if (p.particle_collisions) build_grid(s, p);
        const GridView g{cell_start_.data(), cell_end_.data(), sorted_.data(), 1.0f / p.cell_size, table_size_ - 1};
        for (int i = 0; i < n; ++i) {
            Vec3 a = accel_gravity_and_planes(s, i, p, cc);
            if (p.particle_collisions && s.inv_mass[i] != 0.0f)
                for_each_neighbor(s, i, g, [&](int j) { a += accel_from_particle(s, i, j, cc); });
            accel_[i] = a;
        }
        for (int i = 0; i < n; ++i) integrate_particle_accel(s, i, accel_[i], p.dt);
    }
}

}  // namespace forge
