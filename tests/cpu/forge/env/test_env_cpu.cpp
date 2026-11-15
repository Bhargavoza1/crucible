// Phase 7a: Env API semantics on the CPU backend (doc 3.6), with cartpole.
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/env/env.h"
#include "forge/util/rng.h"

using namespace forge;

namespace {

EnvOptions cartpole_opts(int num_envs, int max_steps = 0) {
    EnvOptions o;
    o.task = "cartpole";
    o.num_envs = num_envs;
    o.seed = 7;
    o.backend = EnvBackend::kCpu;
    o.max_episode_steps = max_steps;
    o.asset_dir = std::string(CRUCIBLE_SOURCE_DIR) + "/assets";
    return o;
}

TEST(EnvCpu, CartpoleShapesAndResetState) {
    auto env = make_env_cpu(cartpole_opts(64));
    EXPECT_EQ(env->obs_dim(), 4);
    EXPECT_EQ(env->act_dim(), 1);
    EXPECT_EQ(env->max_episode_steps(), 500);
    Env::HostCopy h;
    env->download(h);
    ASSERT_EQ(h.obs.size(), 64u * 4);
    bool all_same = true;
    for (int e = 0; e < 64; ++e)
        for (int k = 0; k < 4; ++k) {
            EXPECT_LE(std::fabs(h.obs[e * 4 + k]), 0.05f) << "reset noise is +-0.05";
            all_same = all_same && h.obs[e * 4 + k] == h.obs[k];
        }
    EXPECT_FALSE(all_same) << "every env should start from a different random state";
}

// No push: the pole falls (cartpole is unstable), the episode ends as terminated,
// final_obs holds the failed state, and obs is already the new episode.
TEST(EnvCpu, FallingPoleTerminatesAndResets) {
    auto env = make_env_cpu(cartpole_opts(8));
    std::vector<float> zero(8, 0.0f);
    Env::HostCopy h;
    int ended = 0;
    for (int t = 0; t < 400 && ended == 0; ++t) {
        env->step(zero.data());
        env->download(h);
        for (int e = 0; e < 8; ++e) {
            EXPECT_EQ(h.truncated[e], 0) << "no time limit reached this early";
            if (!h.terminated[e]) continue;
            ++ended;
            const float* fo = &h.final_obs[e * 4];
            EXPECT_TRUE(std::fabs(fo[2]) > 0.5f || std::fabs(fo[0]) > 2.4f) << "final_obs is the failed state";
            for (int k = 0; k < 4; ++k) EXPECT_LE(std::fabs(h.obs[e * 4 + k]), 0.05f) << "obs is the new episode";
        }
    }
    EXPECT_GT(ended, 0) << "the pole should fall within 8 s";
    int lengths = 0;
    for (int e = 0; e < 8; ++e) lengths += env->last_length()[e];
    EXPECT_GT(lengths, 0) << "the ended episode's length is kept";
}

// With a short time limit and a pole that cannot fall in time, every env is
// truncated (not terminated) at exactly that step.
TEST(EnvCpu, TimeLimitTruncates) {
    auto env = make_env_cpu(cartpole_opts(16, 5));
    std::vector<float> zero(16, 0.0f);
    Env::HostCopy h;
    for (int t = 1; t <= 5; ++t) {
        env->step(zero.data());
        env->download(h);
        for (int e = 0; e < 16; ++e) {
            EXPECT_EQ(h.terminated[e], 0);
            EXPECT_EQ(h.truncated[e], t == 5 ? 1 : 0) << "step " << t << " env " << e;
        }
    }
    for (int e = 0; e < 16; ++e) {
        EXPECT_EQ(env->last_length()[e], 5);
        EXPECT_NEAR(env->last_return()[e], 5.0f, 1e-5f) << "+1 per step, no action cost with zero action";
    }
}

// Env e does the same thing in a run with 8 envs and in a run with 64 envs (doc
// scene 9): resets and actions use per-env random streams.
TEST(EnvCpu, EnvsDoNotDependOnEnvCount) {
    auto small = make_env_cpu(cartpole_opts(8));
    auto big = make_env_cpu(cartpole_opts(64));
    std::vector<float> a_small(8), a_big(64);
    Env::HostCopy hs, hb;
    for (int t = 0; t < 300; ++t) {
        small->random_actions(a_small.data(), t);
        big->random_actions(a_big.data(), t);
        for (int e = 0; e < 8; ++e) ASSERT_EQ(a_small[e], a_big[e]);
        small->step(a_small.data());
        big->step(a_big.data());
    }
    small->download(hs);
    big->download(hb);
    for (int i = 0; i < 8 * 4; ++i) EXPECT_EQ(hs.obs[i], hb.obs[i]) << "index " << i;
}

TEST(EnvCpu, NanActionIsTreatedAsZero) {
    auto a = make_env_cpu(cartpole_opts(4));
    auto b = make_env_cpu(cartpole_opts(4));
    std::vector<float> nan(4, std::nanf("")), zero(4, 0.0f);
    a->step(nan.data());
    b->step(zero.data());
    Env::HostCopy ha, hb;
    a->download(ha);
    b->download(hb);
    EXPECT_EQ(ha.obs, hb.obs);
}

TEST(EnvCpu, BadOptionsThrow) {
    EnvOptions o = cartpole_opts(4);
    o.task = "nope";
    EXPECT_THROW(make_env_cpu(o), std::runtime_error);
    o = cartpole_opts(4);
    o.solver = "pgs";
    EXPECT_THROW(make_env_cpu(o), std::runtime_error);
    o = cartpole_opts(0);
    EXPECT_THROW(make_env_cpu(o), std::runtime_error);
}

TEST(Rng, SameStreamSameNumbersAndRange) {
    Rng a(1, 2, 3), b(1, 2, 3), c(1, 2, 4);
    bool differ = false;
    for (int i = 0; i < 1000; ++i) {
        const float x = a.uniform(), y = b.uniform(), z = c.uniform();
        EXPECT_EQ(x, y);
        EXPECT_GE(x, 0.0f);
        EXPECT_LT(x, 1.0f);
        differ = differ || x != z;
    }
    EXPECT_TRUE(differ);
}

}  // namespace
