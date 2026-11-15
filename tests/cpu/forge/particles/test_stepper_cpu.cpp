// The CPU grid stepper vs the brute-force answer key.
#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

#include "forge/particles/integrate_cpu.h"
#include "forge/particles/spawn.h"
#include "forge/particles/step_cpu.h"
#include "forge/particles/stepper_cpu.h"

using namespace forge;

namespace {

ParticlesHost dense_scene() {
    SpawnBox box;
    box.center = {0, 0, 0.5f};
    box.half_extent = {0.4f, 0.4f, 0.4f};
    box.max_speed = 0.5f;
    ParticlesHost h;
    spawn_box(h, 4, 300, box, 99);
    for (int i = 0; i < h.count(); i += 53) h.inv_mass[i] = 0.0f;
    return h;
}

float max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]) / std::max(1.0f, std::fabs(b[i])));
    return m;
}

}  // namespace

// Forces are summed in grid order instead of index order: not bit for bit, but
// a few steps must agree to FP32 rounding.
TEST(StepperCpu, StepsMatchBruteForce) {
    ParticleParams p;
    p.dt = 0.0005f;
    add_bin_planes(p, 0.5f);
    ParticlesHost a = dense_scene();
    ParticlesHost b = a;
    ParticleStepperCpu stepper;
    stepper.step(a.view(), p, 5);
    step_particles_cpu(b.view(), p, 5);
    EXPECT_LT(std::max({max_rel_diff(a.vx, b.vx), max_rel_diff(a.vy, b.vy), max_rel_diff(a.vz, b.vz)}), 1e-4f);
    EXPECT_LT(std::max({max_rel_diff(a.px, b.px), max_rel_diff(a.py, b.py), max_rel_diff(a.pz, b.pz)}), 1e-6f);
}

TEST(StepperCpu, WithoutCollisionsEqualsFreeFlight) {
    ParticleParams p;
    p.particle_collisions = false;
    ParticlesHost a = dense_scene();
    ParticlesHost b = a;
    ParticleStepperCpu stepper;
    stepper.step(a.view(), p, 50);
    integrate_cpu(b.view(), p, 50);
    EXPECT_EQ(a.pz, b.pz);  // bit for bit: same math, no contacts
}
