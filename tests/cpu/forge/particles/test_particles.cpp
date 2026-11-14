#include <gtest/gtest.h>

#include "forge/particles/particles.h"

using namespace forge;

TEST(Particles, ResizeSetsSizesAndDefaults) {
    ParticlesHost h;
    h.resize(3, 4);
    EXPECT_EQ(h.count(), 12);
    EXPECT_EQ(h.px.size(), 12u);
    EXPECT_EQ(h.env_id.size(), 12u);
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(h.inv_mass[i], 1.0f);
        EXPECT_GT(h.radius[i], 0.0f);
        EXPECT_FLOAT_EQ(h.vz[i], 0.0f);
    }
}

// Layout rule (doc 3.1): i = e * particles_per_env + b, and env_id[i] == e.
TEST(Particles, EnvIdFollowsLayout) {
    ParticlesHost h;
    h.resize(5, 7);
    for (int e = 0; e < 5; ++e)
        for (int b = 0; b < 7; ++b) {
            const int i = env_index(e, b, 7);
            EXPECT_EQ(i, e * 7 + b);
            EXPECT_EQ(h.env_id[i], e);
        }
}

TEST(Particles, ViewPointsIntoVectors) {
    ParticlesHost h;
    h.resize(2, 3);
    ParticleView v = h.view();
    EXPECT_EQ(v.count(), 6);
    EXPECT_EQ(v.num_envs, 2);
    EXPECT_EQ(v.particles_per_env, 3);
    v.pz[4] = 1.25f;
    EXPECT_FLOAT_EQ(h.pz[4], 1.25f);
    EXPECT_EQ(v.env_id, h.env_id.data());
}

TEST(Particles, NegativeSizeThrows) {
    ParticlesHost h;
    EXPECT_THROW(h.resize(-1, 4), std::invalid_argument);
}
