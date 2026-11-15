// Tests for the CPU answer key with collisions. Each test checks a physical
// fact (rest depth, momentum, friction, isolation), not other code.
#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

#include "forge/particles/particle_step.h"
#include "forge/particles/spawn.h"
#include "forge/particles/step_cpu.h"

using namespace forge;

namespace {

constexpr float kDt = 0.0005f;  // 2 kHz, like the sandbox substep
constexpr float kR = 0.05f;

ParticleParams params_with_floor() {
    ParticleParams p;
    p.dt = kDt;
    p.add_plane({0, 0, 1}, 0.0f);
    return p;
}

ParticlesHost particles(int envs, int per_env) {
    ParticlesHost h;
    h.resize(envs, per_env);
    for (auto& r : h.radius) r = kR;
    return h;
}

float speed(const ParticlesHost& h, int i) { return std::sqrt(h.vx[i] * h.vx[i] + h.vy[i] * h.vy[i] + h.vz[i] * h.vz[i]); }

}  // namespace

// At rest the spring holds the weight: k * depth = g, so depth = g * timeconst^2.
TEST(StepCpu, RestsOnFloorAtExpectedDepth) {
    ParticleParams p = params_with_floor();
    ParticlesHost h = particles(1, 1);
    h.pz[0] = 0.1f;
    step_particles_cpu(h.view(), p, 4000);  // 2 s
    const float depth = 9.81f * p.contact.timeconst * p.contact.timeconst;  // 0.157 mm
    EXPECT_NEAR(h.pz[0], kR - depth, 1e-5f);
    EXPECT_LT(speed(h, 0), 1e-3f);
}

// Dropped from 3 m (7.7 m/s at impact): the spring stops it before the center
// reaches the floor. Max depth is about v / omega = 7.7 / 250 = 3 cm < radius.
TEST(StepCpu, HighDropDoesNotFallThrough) {
    ParticleParams p = params_with_floor();
    ParticlesHost h = particles(1, 1);
    h.pz[0] = 3.0f;
    float lowest = h.pz[0];
    for (int s = 0; s < 3000; ++s) {
        step_particles_cpu(h.view(), p, 1);
        lowest = std::min(lowest, h.pz[0]);
    }
    EXPECT_GT(lowest, 0.0f);
    EXPECT_LT(lowest, kR);  // it did touch and compress
}

// No gravity, no planes: two spheres hit head on. Total momentum stays the same
// (exact antisymmetric forces) and they bounce apart with some energy lost.
// Restitution e = (separation speed) / (approach speed). For a spring-damper the
// theory value is exp(-pi * z / sqrt(1 - z^2)) = 0.163 for z = 0.5. Ours is higher,
// because a contact lets go as soon as it would start to pull (it never pulls).
TEST(StepCpu, HeadOnCollisionConservesMomentum) {
    ParticleParams p;
    p.dt = kDt;
    p.gravity = {0, 0, 0};
    ParticlesHost h = particles(1, 2);
    h.inv_mass[1] = 0.5f;  // second one is twice as heavy
    h.px[0] = -0.2f;
    h.vx[0] = 2.0f;
    h.px[1] = 0.2f;
    h.vx[1] = -1.0f;
    auto momentum = [&] { return h.vx[0] / h.inv_mass[0] + h.vx[1] / h.inv_mass[1]; };
    const float p0 = momentum();
    const float approach = h.vx[0] - h.vx[1];  // 3 m/s
    step_particles_cpu(h.view(), p, 1000);  // 0.5 s: hit and separate
    EXPECT_NEAR(momentum(), p0, 1e-5f);
    EXPECT_GT(h.px[1] - h.px[0], 2 * kR);  // apart again
    const float e = (h.vx[1] - h.vx[0]) / approach;
    const float z = p.contact.dampratio;
    const float e_theory = std::exp(-3.14159265f * z / std::sqrt(1 - z * z));
    EXPECT_GT(e, e_theory);
    EXPECT_LT(e, 0.5f);
}

TEST(StepCpu, FrictionStopsSliding) {
    ParticleParams p = params_with_floor();
    p.contact.friction = 0.5f;  // deceleration about mu * g = 4.9 m/s^2 -> stops in ~0.4 s
    ParticlesHost h = particles(1, 1);
    h.pz[0] = kR - 9.81f * p.contact.timeconst * p.contact.timeconst;  // resting depth
    h.vx[0] = 2.0f;
    step_particles_cpu(h.view(), p, 2000);  // 1 s
    EXPECT_LT(std::fabs(h.vx[0]), 0.01f);
    EXPECT_NEAR(h.px[0], 2.0f * 2.0f / (2.0f * 0.5f * 9.81f), 0.03f);  // v^2 / (2 mu g) = 0.41 m
}

TEST(StepCpu, NoFrictionKeepsSliding) {
    ParticleParams p = params_with_floor();
    p.contact.friction = 0.0f;
    ParticlesHost h = particles(1, 1);
    h.pz[0] = kR - 9.81f * p.contact.timeconst * p.contact.timeconst;
    h.vx[0] = 2.0f;
    step_particles_cpu(h.view(), p, 2000);
    EXPECT_FLOAT_EQ(h.vx[0], 2.0f);  // the floor force has no x part at all
}

// Two particles at the same place but in different envs never touch (doc 3.4).
TEST(StepCpu, EnvsDoNotInteract) {
    ParticleParams p;
    p.dt = kDt;
    p.gravity = {0, 0, 0};
    ParticlesHost h = particles(2, 1);  // env 0 and env 1, one particle each
    h.px[0] = h.px[1] = 0.0f;
    step_particles_cpu(h.view(), p, 100);
    EXPECT_FLOAT_EQ(h.vx[0], 0.0f);
    EXPECT_FLOAT_EQ(h.vx[1], 0.0f);
    EXPECT_EQ(contact_counts_cpu(h.view())[0], 0);
}

// A static particle is a fixed obstacle: it does not move, the other one rests on it.
TEST(StepCpu, StaticParticleIsAnObstacle) {
    ParticleParams p;
    p.dt = kDt;
    ParticlesHost h = particles(1, 2);
    h.inv_mass[0] = 0.0f;  // static, at the origin
    h.pz[1] = 0.2f;        // falls on top of it
    step_particles_cpu(h.view(), p, 4000);
    EXPECT_FLOAT_EQ(h.pz[0], 0.0f);
    const float depth = 9.81f * p.contact.timeconst * p.contact.timeconst;
    EXPECT_NEAR(h.pz[1], 2 * kR - depth, 1e-4f);
}

TEST(StepCpu, ContactCountsInALine) {
    ParticlesHost h = particles(1, 3);
    h.px[0] = 0.0f;
    h.px[1] = 0.09f;  // touches 0 and 2
    h.px[2] = 0.18f;
    const auto c = contact_counts_cpu(h.view());
    EXPECT_EQ(c[0], 1);
    EXPECT_EQ(c[1], 2);
    EXPECT_EQ(c[2], 1);
}

namespace {

struct PileResult {
    ParticlesHost h;
    float peak_ke = 0.0f;
    float start_pe = 0.0f;
};

// 200 particles dropped into a 0.6 m bin, run for 'seconds'.
PileResult run_small_pile(float dt, float seconds) {
    ParticleParams p = params_with_floor();
    p.dt = dt;
    const float half = 0.3f;
    p.add_plane({1, 0, 0}, -half);   // wall at x = -half, facing +x
    p.add_plane({-1, 0, 0}, -half);  // wall at x = +half, facing -x
    p.add_plane({0, 1, 0}, -half);
    p.add_plane({0, -1, 0}, -half);
    SpawnLattice lat;
    lat.origin = {-0.28f, -0.28f, 0.1f};
    lat.nx = lat.ny = 5;
    PileResult r;
    spawn_lattice(r.h, 1, 200, lat, 3);
    for (int i = 0; i < r.h.count(); ++i) r.start_pe += 9.81f * r.h.pz[i];
    const int chunk = int(0.05f / dt);
    const int chunks = int(seconds / 0.05f);
    for (int c = 0; c < chunks; ++c) {
        step_particles_cpu(r.h.view(), p, chunk);
        float ke = 0;
        for (int i = 0; i < r.h.count(); ++i) ke += 0.5f * speed(r.h, i) * speed(r.h, i);
        r.peak_ke = std::max(r.peak_ke, ke);
    }
    return r;
}

float potential_energy(const ParticlesHost& h) {
    float pe = 0;
    for (int i = 0; i < h.count(); ++i) pe += 9.81f * h.pz[i];
    return pe;
}

}  // namespace

// A small pile in a bin. Smooth spheres have no rolling friction, so small
// avalanches keep happening for a while (a few particles roll down the top).
// So we do NOT ask "everything at rest". We ask what physics promises:
// energy drained, pile got lower, everything inside, no NaN (doc 11.1).
TEST(StepCpu, SmallPileSettlesInBin) {
    PileResult r = run_small_pile(kDt, 3.0f);
    ParticlesHost& h = r.h;
    float ke = 0;
    for (int i = 0; i < h.count(); ++i) {
        ASSERT_TRUE(std::isfinite(h.px[i]) && std::isfinite(h.pz[i])) << "i=" << i;
        EXPECT_LT(std::fabs(h.px[i]), 0.3f) << "i=" << i;
        EXPECT_LT(std::fabs(h.py[i]), 0.3f) << "i=" << i;
        EXPECT_GT(h.pz[i], 0.5f * kR) << "i=" << i;  // not sunk into the floor
        ke += 0.5f * speed(h, i) * speed(h, i);
    }
    EXPECT_LT(ke, 0.01f * r.peak_ke);                // >100x less motion than at impact
    EXPECT_LT(potential_energy(h), 0.8f * r.start_pe);  // the pile is lower than the column
    const auto counts = contact_counts_cpu(h.view());
    EXPECT_GT(*std::max_element(counts.begin(), counts.end()), 2);  // really piled up
}

// Halving the step must not change the result much. If it did, the step would
// be too big and we would be looking at numerical error, not physics.
TEST(StepCpu, PileConvergesWithSmallerStep) {
    const float pe_full = potential_energy(run_small_pile(kDt, 2.0f).h);
    const float pe_half = potential_energy(run_small_pile(0.5f * kDt, 2.0f).h);
    EXPECT_NEAR(pe_full, pe_half, 0.01f * pe_half);
}
