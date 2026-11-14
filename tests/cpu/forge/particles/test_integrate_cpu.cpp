// Tests for the CPU reference integrator. This is the answer key for the GPU,
// so it is checked against exact formulas, not against other code.
#include <cmath>

#include <gtest/gtest.h>

#include "forge/particles/integrate_cpu.h"
#include "forge/particles/particles.h"

using namespace forge;

namespace {

ParticlesHost one_particle(float pz, float vx, float vz) {
    ParticlesHost h;
    h.resize(1, 1);
    h.pz[0] = pz;
    h.vx[0] = vx;
    h.vz[0] = vz;
    return h;
}

}  // namespace

// Semi-implicit Euler with constant gravity has an exact closed form after n steps:
//   v_n = v_0 + n g dt
//   p_n = p_0 + n v_0 dt + g dt^2 n (n + 1) / 2
// (each step adds the NEW velocity, so the sum is 1 + 2 + ... + n).
TEST(IntegrateCpu, FreeFallMatchesClosedForm) {
    ParticleParams p;  // g = (0, 0, -9.81), dt = 0.005
    const float pz0 = 10.0f, vx0 = 1.5f, vz0 = 2.0f;
    ParticlesHost h = one_particle(pz0, vx0, vz0);
    const int n = 200;  // 1 second
    integrate_cpu(h.view(), p, n);

    const double g = p.gravity.z, dt = p.dt;
    const double vz_n = vz0 + n * g * dt;
    const double pz_n = pz0 + n * vz0 * dt + g * dt * dt * n * (n + 1) / 2.0;
    // FP32 rounding over 200 steps: about 1e-5 relative is the honest tolerance.
    EXPECT_NEAR(h.vz[0], vz_n, 1e-4);
    EXPECT_NEAR(h.pz[0], pz_n, 1e-4);
    // No force in x: constant velocity, linear position.
    EXPECT_FLOAT_EQ(h.vx[0], vx0);
    EXPECT_NEAR(h.px[0], n * vx0 * dt, 1e-4);
    EXPECT_FLOAT_EQ(h.py[0], 0.0f);
}

// Semi-implicit Euler is first order: compared with the exact physics
// p(t) = p0 + v0 t + g t^2 / 2, it is ahead by g dt t / 2. This test documents
// that known error, so nobody "fixes" it by mistake.
TEST(IntegrateCpu, ErrorVsExactPhysicsIsFirstOrder) {
    ParticleParams p;
    ParticlesHost h = one_particle(0.0f, 0.0f, 0.0f);
    const int n = 200;
    integrate_cpu(h.view(), p, n);
    const double t = n * double(p.dt);
    const double exact = 0.5 * p.gravity.z * t * t;           // -4.905 m
    const double expected_error = 0.5 * p.gravity.z * p.dt * t;  // -0.0245 m
    EXPECT_NEAR(h.pz[0] - exact, expected_error, 1e-4);
}

TEST(IntegrateCpu, StaticParticleDoesNotMove) {
    ParticlesHost h = one_particle(5.0f, 1.0f, 1.0f);
    h.inv_mass[0] = 0.0f;
    integrate_cpu(h.view(), ParticleParams{}, 100);
    EXPECT_FLOAT_EQ(h.pz[0], 5.0f);
    EXPECT_FLOAT_EQ(h.px[0], 0.0f);
    EXPECT_FLOAT_EQ(h.vz[0], 1.0f);  // velocity untouched too
}

TEST(IntegrateCpu, ZeroStepsDoesNothing) {
    ParticlesHost h = one_particle(5.0f, 1.0f, 1.0f);
    integrate_cpu(h.view(), ParticleParams{}, 0);
    integrate_cpu(h.view(), ParticleParams{}, -3);
    EXPECT_FLOAT_EQ(h.pz[0], 5.0f);
}

TEST(IntegrateCpu, EmptyStateIsSafe) {
    ParticlesHost h;
    h.resize(0, 0);
    integrate_cpu(h.view(), ParticleParams{}, 10);  // must not crash
    SUCCEED();
}

// Steps add up: 2 calls of 50 steps == 1 call of 100 steps (bit for bit).
TEST(IntegrateCpu, StepsCompose) {
    ParticlesHost a = one_particle(3.0f, 0.5f, 1.0f);
    ParticlesHost b = a;
    integrate_cpu(a.view(), ParticleParams{}, 100);
    integrate_cpu(b.view(), ParticleParams{}, 50);
    integrate_cpu(b.view(), ParticleParams{}, 50);
    EXPECT_EQ(a.pz[0], b.pz[0]);
    EXPECT_EQ(a.vz[0], b.vz[0]);
}

// Invariant (doc 11.1): no NaN or Inf after a long run.
TEST(IntegrateCpu, NoNanAfterLongRun) {
    ParticlesHost h = one_particle(0.0f, 3.0f, 3.0f);
    integrate_cpu(h.view(), ParticleParams{}, 20000);  // 100 s
    EXPECT_TRUE(std::isfinite(h.pz[0]));
    EXPECT_TRUE(std::isfinite(h.vz[0]));
}
