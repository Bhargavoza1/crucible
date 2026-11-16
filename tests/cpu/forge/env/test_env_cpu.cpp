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

namespace {

EnvOptions ant_opts(int n) {
    EnvOptions o = cartpole_opts(n);
    o.task = "ant";
    return o;
}

// Action 0 = every joint target in the middle of its range: the ant lands and
// stands; no episode ends; the torso height settles near 0.5 m. Upright, the
// gravity direction in the base frame is (0, 0, -1).
TEST(AntTask, StandsWithZeroAction) {
    auto env = make_env_cpu(ant_opts(8));
    EXPECT_EQ(env->obs_dim(), 36);
    EXPECT_EQ(env->act_dim(), 8);
    EXPECT_EQ(env->max_episode_steps(), 1000);
    std::vector<float> zero(8 * 8, 0.0f);
    Env::HostCopy h;
    for (int t = 0; t < 150; ++t) {  // 3 s
        env->step(zero.data());
        env->download(h);
        for (int e = 0; e < 8; ++e) ASSERT_EQ(h.terminated[e], 0) << "step " << t << " env " << e;
    }
    for (int e = 0; e < 8; ++e) {
        const float z = h.qpos[e * env->model().nq + 2];
        EXPECT_NEAR(z, 0.50f, 0.03f) << "env " << e;
        EXPECT_NEAR(h.obs[e * 36 + 2], -1.0f, 0.02f) << "gravity z in the base frame";
        EXPECT_NEAR(h.obs[e * 36 + 3], 1.0f, 0.02f) << "facing +x: world +x is the base's own +x";
        EXPECT_NEAR(h.obs[e * 36 + 6], 0.0f, 0.05f) << "standing still: forward velocity ~0";
    }
}

// 20 s of random actions (full joint ranges, a new target every step): no
// dynamics error, all obs finite. Falling over is allowed (that ends an episode).
TEST(AntTask, RandomActionsStayFinite) {
    auto env = make_env_cpu(ant_opts(16));
    std::vector<float> act(16 * 8);
    Env::HostCopy h;
    int ended = 0;
    for (int t = 0; t < 1000; ++t) {
        env->random_actions(act.data(), t);
        env->step(act.data());
        env->download(h);
        for (float x : h.obs) ASSERT_TRUE(std::isfinite(x)) << "step " << t;
        for (int e = 0; e < 16; ++e) ended += h.terminated[e];
    }
    std::printf("ant, 16 envs x 20 s random actions: %d episodes ended by falling\n", ended);
}

}  // namespace

// ---- Phase 8b-1: G1 stand ------------------------------------------------------
#include "forge/env/tasks/g1_stand.h"

namespace {

EnvOptions g1_opts(int n) {
    EnvOptions o = cartpole_opts(n);
    o.task = "g1_stand";
    o.physics_dt = 0.005f;  // configs/g1_stand.txt
    o.decimation = 4;
    return o;
}

// Shapes, and the start state: upright, feet on the floor (pelvis at its standing
// height + 1 cm), gravity in the base frame about (0, 0, -1) (obs noise <= 0.05).
TEST(G1StandTask, ShapesAndStartStanding) {
    auto env = make_env_cpu(g1_opts(8));
    EXPECT_EQ(env->obs_dim(), 42);
    EXPECT_EQ(env->act_dim(), 12);
    EXPECT_EQ(env->max_episode_steps(), 1000);
    EXPECT_EQ(env->model().nv, 18);
    EXPECT_FLOAT_EQ(env->model().dof_armature.at(6), 0.01f) << "the task loads Unitree's armature";
    Env::HostCopy h;
    env->download(h);
    const auto p = G1StandTask::make_params(env->model());
    EXPECT_NEAR(p.stand_height, 0.78f, 0.03f) << "pelvis height standing, from the foot spheres";
    for (int e = 0; e < 8; ++e) {
        EXPECT_NEAR(h.qpos[e * env->model().nq + 2], p.stand_height + 0.01f, 1e-5f);
        EXPECT_NEAR(h.obs[e * 42 + 5], -1.0f, 0.06f) << "gravity z in the base frame, env " << e;
    }
}

// The push schedule: first push after 5 to 10 s, then every 5 to 10 s, at most
// 4 pushes in a 20 s episode; different envs and episodes get different times.
TEST(G1StandTask, PushScheduleIsFiveToTenSeconds) {
    G1StandTask::Params p;
    std::vector<uint32_t> episode(4, 0);
    EnvView v;
    v.seed = 3;
    v.episode_index = episode.data();
    std::vector<std::vector<int>> times(4);
    for (int e = 0; e < 4; ++e)
        for (int s = 0; s < 1000; ++s)
            if (G1StandTask::push_now(p, v, e, s)) times[e].push_back(s);
    for (int e = 0; e < 4; ++e) {
        ASSERT_FALSE(times[e].empty());
        EXPECT_GE(times[e][0], 250);
        EXPECT_LT(times[e][0], 500);
        for (size_t i = 1; i < times[e].size(); ++i) {
            EXPECT_GE(times[e][i] - times[e][i - 1], 250);
            EXPECT_LT(times[e][i] - times[e][i - 1], 500);
        }
        EXPECT_LE(times[e].size(), 4u);
    }
    EXPECT_NE(times[0], times[1]);
    episode[0] = 1;
    std::vector<int> next;
    for (int s = 0; s < 1000; ++s)
        if (G1StandTask::push_now(p, v, 0, s)) next.push_back(s);
    EXPECT_NE(next, times[0]) << "a new episode gets new push times";
}

// With action 0 the PD only holds the standing pose: no balance, so the G1 tips
// over within a few seconds (MuJoCo: about 1-2 s). Falling must end the episode
// and the next episode must start standing again.
TEST(G1StandTask, FallingTerminatesAndResetsStanding) {
    auto env = make_env_cpu(g1_opts(8));
    std::vector<float> zero(8 * 12, 0.0f);
    Env::HostCopy h;
    int fell = 0, first = -1;
    for (int t = 0; t < 250 && fell < 8; ++t) {  // 5 s
        env->step(zero.data());
        env->download(h);
        for (int e = 0; e < 8; ++e)
            if (h.terminated[e]) {
                ++fell;
                if (first < 0) first = t + 1;
                EXPECT_LT(h.reward[e], -5.0f) << "fall penalty";
                EXPECT_NEAR(h.qpos[e * env->model().nq + 2], 0.79f, 0.03f) << "reset to standing";
            }
    }
    std::printf("g1_stand, 8 envs, action 0: %d falls in 5 s, first after %.2f s\n", fell, first * 0.02);
    EXPECT_GT(fell, 0);
}

// 20 s of random actions: no dynamics error, all obs finite.
TEST(G1StandTask, RandomActionsStayFinite) {
    auto env = make_env_cpu(g1_opts(16));
    std::vector<float> act(16 * 12);
    Env::HostCopy h;
    int ended = 0;
    for (int t = 0; t < 1000; ++t) {
        env->random_actions(act.data(), t);
        env->step(act.data());
        env->download(h);
        for (float x : h.obs) ASSERT_TRUE(std::isfinite(x)) << "step " << t;
        for (int e = 0; e < 16; ++e) ended += h.terminated[e];
    }
    std::printf("g1_stand, 16 envs x 20 s random actions: %d episodes ended by falling\n", ended);
}

}  // namespace
