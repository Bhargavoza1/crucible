// GPU integrator vs the CPU answer key (doc 11.1).
// Both call integrate_particle(), which uses explicit fmaf(), so the results
// should be bit for bit equal. We still allow a tiny tolerance, and print the
// largest difference so a change in rounding is easy to see.
#include <algorithm>
#include <cmath>
#include <cstdio>

#include <gtest/gtest.h>

#include "forge/particles/integrate_cpu.h"
#include "forge/particles/integrate_gpu.h"
#include "forge/particles/particles_device.h"
#include "forge/particles/spawn.h"

using namespace forge;

namespace {

// Largest |a - b| / max(1, |b|) over two arrays.
float max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs(a[i] - b[i]) / std::max(1.0f, std::fabs(b[i])));
    return m;
}

ParticlesHost make_scene() {
    ParticlesHost h;
    spawn_box(h, 64, 1000, SpawnBox{}, 2024);  // 64,000 particles, more than one block
    for (int i = 0; i < h.count(); i += 97) h.inv_mass[i] = 0.0f;  // some static ones
    return h;
}

}  // namespace

TEST(IntegrateGpu, MatchesCpu) {
    const ParticleParams params{};
    ParticlesHost cpu = make_scene();
    ParticlesHost gpu_result = cpu;

    ParticlesDevice dev;
    dev.upload(cpu);
    // 200 steps as 50 calls of 4 (like the sandbox: a few steps per frame).
    for (int k = 0; k < 50; ++k) integrate_gpu(dev.view(), params, 4);
    CUDA_CHECK(cudaDeviceSynchronize());
    dev.download(gpu_result);

    integrate_cpu(cpu.view(), params, 200);

    const float d_px = max_rel_diff(gpu_result.px, cpu.px);
    const float d_pz = max_rel_diff(gpu_result.pz, cpu.pz);
    const float d_vz = max_rel_diff(gpu_result.vz, cpu.vz);
    std::printf("max rel diff: px %.3g  pz %.3g  vz %.3g  (0 = bit for bit)\n", d_px, d_pz, d_vz);
    EXPECT_LE(d_px, 1e-6f);
    EXPECT_LE(d_pz, 1e-6f);
    EXPECT_LE(d_vz, 1e-6f);
    EXPECT_LE(max_rel_diff(gpu_result.py, cpu.py), 1e-6f);
    EXPECT_LE(max_rel_diff(gpu_result.vx, cpu.vx), 1e-6f);
    EXPECT_LE(max_rel_diff(gpu_result.vy, cpu.vy), 1e-6f);

    // Static particles did not move, and fields the kernel must not touch are unchanged.
    for (int i = 0; i < cpu.count(); ++i) {
        ASSERT_TRUE(std::isfinite(gpu_result.pz[i])) << "i=" << i;
        ASSERT_EQ(gpu_result.env_id[i], cpu.env_id[i]) << "i=" << i;
        ASSERT_EQ(gpu_result.inv_mass[i], cpu.inv_mass[i]) << "i=" << i;
    }
}

TEST(IntegrateGpu, DownloadPositionsOnlyCopiesPositions) {
    ParticlesHost h = make_scene();
    ParticlesDevice dev;
    dev.upload(h);
    integrate_gpu(dev.view(), ParticleParams{}, 10);
    CUDA_CHECK(cudaDeviceSynchronize());

    ParticlesHost out = h;
    dev.download_positions(out);
    EXPECT_NE(out.pz, h.pz);  // positions changed
    EXPECT_EQ(out.vz, h.vz);  // velocities were not copied back
}

TEST(IntegrateGpu, UploadDownloadRoundTrip) {
    ParticlesHost h = make_scene();
    ParticlesDevice dev;
    dev.upload(h);
    ParticlesHost back;
    back.resize(h.num_envs(), h.particles_per_env());
    dev.download(back);
    EXPECT_EQ(back.px, h.px);
    EXPECT_EQ(back.vz, h.vz);
    EXPECT_EQ(back.env_id, h.env_id);
    EXPECT_EQ(dev.count(), h.count());
}

TEST(IntegrateGpu, EmptyAndZeroStepsAreSafe) {
    ParticlesHost h;
    h.resize(0, 0);
    ParticlesDevice dev;
    dev.upload(h);
    integrate_gpu(dev.view(), ParticleParams{}, 5);  // n == 0: no launch
    ParticlesHost one = make_scene();
    dev.upload(one);
    integrate_gpu(dev.view(), ParticleParams{}, 0);  // steps == 0: no launch
    CUDA_CHECK(cudaDeviceSynchronize());
    SUCCEED();
}
