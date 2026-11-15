// GPU stepper with collisions vs the CPU brute-force answer key.
#include <algorithm>
#include <cmath>
#include <cstdio>

#include <gtest/gtest.h>

#include "forge/particles/integrate_gpu.h"
#include "forge/particles/particles_device.h"
#include "forge/particles/spawn.h"
#include "forge/particles/step_cpu.h"
#include "forge/particles/stepper_gpu.h"

using namespace forge;

namespace {

constexpr float kDt = 0.0005f;

// Overlapping random particles: many contacts, several envs, negative cells.
ParticlesHost dense_scene(int envs, int per_env) {
    SpawnBox box;
    box.center = {0, 0, 0.5f};
    box.half_extent = {0.4f, 0.4f, 0.4f};
    box.max_speed = 0.5f;
    ParticlesHost h;
    spawn_box(h, envs, per_env, box, 99);
    for (int i = 0; i < h.count(); i += 53) h.inv_mass[i] = 0.0f;  // a few static ones
    return h;
}

ParticleParams bin_params(float half) {
    ParticleParams p;
    p.dt = kDt;
    add_bin_planes(p, half);
    return p;
}

float max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]) / std::max(1.0f, std::fabs(b[i])));
    return m;
}

ParticlesHost download(const ParticlesDevice& dev, const ParticlesHost& like) {
    ParticlesHost out = like;
    dev.download(out);
    return out;
}

}  // namespace

// The grid must find exactly the contacts brute force finds (integers: exact).
TEST(StepperGpu, ContactCountsMatchBruteForce) {
    ParticlesHost h = dense_scene(16, 500);
    ParticlesDevice dev;
    dev.upload(h);
    ParticleStepperGpu stepper;
    const auto gpu = stepper.contact_counts(dev.view(), bin_params(1.0f));
    const auto cpu = contact_counts_cpu(h.view());
    ASSERT_EQ(gpu.size(), cpu.size());
    long long total = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        ASSERT_EQ(gpu[i], cpu[i]) << "i=" << i;
        total += cpu[i];
    }
    EXPECT_GT(total, 10000);  // really dense
}

// Same with a tiny hash table: many cells share a hash, pairs still found once.
TEST(StepperGpu, ContactCountsWithTinyHashTable) {
    ParticlesHost h = dense_scene(4, 500);
    ParticlesDevice dev;
    dev.upload(h);
    ParticleStepperGpu stepper;
    stepper.force_table_size(16);
    const auto gpu = stepper.contact_counts(dev.view(), bin_params(1.0f));
    EXPECT_EQ(stepper.table_size(), 16u);
    EXPECT_EQ(gpu, contact_counts_cpu(h.view()));
}

// Forces are summed in a different order (grid order vs index order), so the
// result is not bit for bit. A few steps must still agree to FP32 rounding.
TEST(StepperGpu, StepsMatchCpu) {
    ParticlesHost cpu = dense_scene(8, 400);
    const ParticleParams p = bin_params(0.5f);
    ParticlesDevice dev;
    dev.upload(cpu);
    ParticleStepperGpu stepper;
    stepper.step(dev.view(), p, 5);
    CUDA_CHECK(cudaDeviceSynchronize());
    const ParticlesHost gpu = download(dev, cpu);
    step_particles_cpu(cpu.view(), p, 5);

    const float dv = std::max({max_rel_diff(gpu.vx, cpu.vx), max_rel_diff(gpu.vy, cpu.vy), max_rel_diff(gpu.vz, cpu.vz)});
    const float dp = std::max({max_rel_diff(gpu.px, cpu.px), max_rel_diff(gpu.py, cpu.py), max_rel_diff(gpu.pz, cpu.pz)});
    std::printf("after 5 steps: max rel diff  velocity %.3g  position %.3g\n", dv, dp);
    EXPECT_LT(dv, 1e-4f);
    EXPECT_LT(dp, 1e-6f);
}

// Same input, same output, every run (stable sort, no atomics).
TEST(StepperGpu, Deterministic) {
    ParticlesHost h = dense_scene(8, 400);
    const ParticleParams p = bin_params(0.5f);
    ParticlesHost out[2];
    for (int run = 0; run < 2; ++run) {
        ParticlesDevice dev;
        dev.upload(h);
        ParticleStepperGpu stepper;
        stepper.step(dev.view(), p, 200);
        CUDA_CHECK(cudaDeviceSynchronize());
        out[run] = download(dev, h);
    }
    EXPECT_EQ(out[0].px, out[1].px);
    EXPECT_EQ(out[0].vz, out[1].vz);
}

// The sorted packed layout (default) adds forces in the same order as the
// original layout, so 300 steps must give bit-for-bit the same state.
TEST(StepperGpu, SortedLayoutEqualsOriginal) {
    ParticlesHost h = dense_scene(8, 600);
    const ParticleParams p = bin_params(0.5f);
    ParticlesHost out[2];
    for (int layout = 0; layout < 2; ++layout) {
        ParticlesDevice dev;
        dev.upload(h);
        ParticleStepperGpu stepper;
        stepper.set_sorted_layout(layout == 1);
        stepper.step(dev.view(), p, 300);
        CUDA_CHECK(cudaDeviceSynchronize());
        out[layout] = download(dev, h);
    }
    EXPECT_EQ(out[0].px, out[1].px);
    EXPECT_EQ(out[0].py, out[1].py);
    EXPECT_EQ(out[0].pz, out[1].pz);
    EXPECT_EQ(out[0].vx, out[1].vx);
    EXPECT_EQ(out[0].vz, out[1].vz);
}

// No planes, no sphere collisions: exactly the Phase 2 free-flight integrator.
TEST(StepperGpu, WithoutCollisionsEqualsFreeFlight) {
    ParticlesHost h = dense_scene(4, 300);
    ParticleParams p;
    p.dt = 0.005f;
    p.particle_collisions = false;
    ParticlesDevice a, b;
    a.upload(h);
    b.upload(h);
    ParticleStepperGpu stepper;
    stepper.step(a.view(), p, 50);
    integrate_gpu(b.view(), p, 50);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(download(a, h).pz, download(b, h).pz);
}

// The CPU test scene (200 particles in a 0.6 m bin): the pile on the GPU must end
// as low as the pile on the CPU. Collisions are chaotic, so individual particles
// differ, but the pile height (potential energy) must agree closely.
TEST(StepperGpu, SmallPileMatchesCpuStatistically) {
    const ParticleParams p = bin_params(0.3f);
    SpawnLattice lat;
    lat.origin = {-0.28f, -0.28f, 0.1f};
    lat.nx = lat.ny = 5;
    ParticlesHost cpu;
    spawn_lattice(cpu, 1, 200, lat, 3);
    ParticlesDevice dev;
    dev.upload(cpu);
    ParticleStepperGpu stepper;
    stepper.step(dev.view(), p, 4000);  // 2 s
    CUDA_CHECK(cudaDeviceSynchronize());
    const ParticlesHost gpu = download(dev, cpu);
    step_particles_cpu(cpu.view(), p, 4000);
    float pe_cpu = 0, pe_gpu = 0;
    for (int i = 0; i < cpu.count(); ++i) {
        pe_cpu += 9.81f * cpu.pz[i];
        pe_gpu += 9.81f * gpu.pz[i];
    }
    std::printf("pile potential energy: cpu %.2f  gpu %.2f\n", pe_cpu, pe_gpu);
    EXPECT_NEAR(pe_gpu, pe_cpu, 0.02f * pe_cpu);
}

// The Phase 4 done check scene: 10,000 particles dropped into a 2 m bin.
// After 4 s: all inside, none sunk into the floor, no NaN, and settled.
TEST(StepperGpu, TenThousandParticlePile) {
    const float half = 1.0f;
    const ParticleParams p = bin_params(half);
    SpawnLattice lat;  // 16 x 16 per layer, 40 layers, 0.3 m above the floor
    ParticlesHost h;
    spawn_lattice(h, 1, 10000, lat, 1);
    ParticlesDevice dev;
    dev.upload(h);
    ParticleStepperGpu stepper;
    float peak_ke = 0, ke = 0;
    for (int chunk = 0; chunk < 80; ++chunk) {  // 80 x 0.05 s = 4 s
        stepper.step(dev.view(), p, 100);
        CUDA_CHECK(cudaDeviceSynchronize());
        dev.download(h);
        ke = 0;
        for (int i = 0; i < h.count(); ++i) ke += 0.5f * (h.vx[i] * h.vx[i] + h.vy[i] * h.vy[i] + h.vz[i] * h.vz[i]);
        peak_ke = std::max(peak_ke, ke);
    }
    float top = 0;
    for (int i = 0; i < h.count(); ++i) {
        ASSERT_TRUE(std::isfinite(h.px[i]) && std::isfinite(h.pz[i])) << "i=" << i;
        ASSERT_LT(std::fabs(h.px[i]), half) << "i=" << i;
        ASSERT_LT(std::fabs(h.py[i]), half) << "i=" << i;
        ASSERT_GT(h.pz[i], 0.025f) << "i=" << i;
        top = std::max(top, h.pz[i]);
    }
    std::printf("10k pile: top %.2f m, kinetic energy %.3g (peak %.3g)\n", top, ke, peak_ke);
    EXPECT_LT(ke, 0.01f * peak_ke);
    EXPECT_LT(top, 3.5f);  // the 4.9 m column has collapsed into a pile
}

TEST(StepperGpu, EmptyAndZeroStepsAreSafe) {
    ParticlesHost h;
    h.resize(0, 0);
    ParticlesDevice dev;
    dev.upload(h);
    ParticleStepperGpu stepper;
    stepper.step(dev.view(), bin_params(1.0f), 5);
    EXPECT_TRUE(stepper.contact_counts(dev.view(), bin_params(1.0f)).empty());
    ParticlesHost one = dense_scene(1, 10);
    dev.upload(one);
    stepper.step(dev.view(), bin_params(1.0f), 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    SUCCEED();
}
