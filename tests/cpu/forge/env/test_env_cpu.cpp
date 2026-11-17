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
    EXPECT_EQ(env->obs_dim(), 45);
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
        EXPECT_NEAR(h.obs[e * 45 + 5], -1.0f, 0.06f) << "gravity z in the base frame, env " << e;
        EXPECT_NEAR(h.obs[e * 45 + 6], 0.0f, 0.16f) << "base x velocity ~0 at the start (noise 0.1 + reset 0.05)";
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

// ---- Phase 8b-2: domain randomization and foot slip -------------------------------
#include "forge/dynamics/robot_work.h"
#include "forge/robots/urdf.h"
#include "forge/solver/constrained_dynamics.h"
#include "forge/solver/constraint_work.h"
#include "forge/solver/constraint_model.h"

namespace {

struct G1Physics {
    RobotModel m;
    explicit G1Physics() {
        UrdfOptions u;
        u.floating_base = true;
        G1StandTask::urdf_options(u);
        m = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/g1/g1_12dof.urdf", u);
    }
    // qacc of one standing state with the given model / constraints / actuators / randomization.
    std::vector<float> qacc(const RobotModel& model, const ConstraintModel& cm, const Actuators& act,
                            const EnvRandom& r = EnvRandom{}) const {
        const auto p = G1StandTask::make_params(model);
        std::vector<float> q = model.qpos0(), qd(model.nv, 0.0f), ctrl(act.nu());
        q[p.base_q + 2] = p.stand_height - 0.002f;  // feet 2 mm inside the floor: contacts on
        for (int k = 0; k < 12; ++k) {
            q[p.q[k]] = p.q0[k] + 0.05f * float(k % 3 - 1);
            ctrl[k] = p.q0[k];
            qd[p.d[k]] = 0.3f * float(k % 5 - 2);
        }
        qd[p.base_d] = 0.2f;  // sliding: friction rows matter
        RobotWork work(model);
        DynamicsWork w = work.view();
        ConstraintWorkHost cwh(cm);
        ConstraintWork cw = cwh.view();
        std::vector<float> out(model.nv);
        EXPECT_EQ(forward_dynamics_constrained(model.view(), cm.view(), act.view(), q.data(), qd.data(), nullptr,
                                               ctrl.data(), w, cw, out.data(), r),
                  kDynamicsOk);
        EXPECT_GT(cw.nefc, 0);
        return out;
    }
};

// Each randomization must equal building the model with that value directly.
TEST(DomainRandomization, SameAsChangingTheModel) {
    G1Physics g;
    ConstraintModel cm(g.m);
    cm.add_ground();
    const Actuators act = G1StandTask::make_actuators(g.m);

    // Mass: factor s per body == a model with mass and inertia times s.
    std::vector<float> scale(g.m.nbody());
    RobotModel heavy = g.m;
    for (int b = 0; b < g.m.nbody(); ++b) {
        scale[b] = 0.9f + 0.3f * float(b % 4) / 3.0f;
        heavy.mass[b] *= scale[b];
        for (int k = 0; k < 9; ++k) heavy.inertia[9 * b + k] *= scale[b];
    }
    EnvRandom r;
    r.body_scale = scale.data();
    // Both use the base model's constraint model (its invweight0), as the env does.
    const auto a1 = g.qacc(g.m, cm, act, r), a2 = g.qacc(heavy, cm, act);
    for (int i = 0; i < g.m.nv; ++i) EXPECT_NEAR(a1[i], a2[i], 1e-3f * std::max(1.0f, std::fabs(a2[i]))) << "mass, dof " << i;

    // Friction: per-env value == a model with that friction.
    ConstraintOptions o;
    o.friction = 0.6f;
    ConstraintModel cm_f(g.m, o);
    cm_f.add_ground();
    EnvRandom rf;
    rf.friction = 0.6f;
    const auto f1 = g.qacc(g.m, cm, act, rf), f2 = g.qacc(g.m, cm_f, act);
    for (int i = 0; i < g.m.nv; ++i) EXPECT_EQ(f1[i], f2[i]) << "friction, dof " << i;

    // PD gains: factors == actuators with scaled kp, kv.
    Actuators act2 = act;
    for (int k = 0; k < act.nu(); ++k) {
        act2.kp[k] *= 1.1f;
        act2.kv[k] *= 0.9f;
    }
    EnvRandom rk;
    rk.kp = 1.1f;
    rk.kv = 0.9f;
    const auto k1 = g.qacc(g.m, cm, act, rk), k2 = g.qacc(g.m, cm, act2);
    for (int i = 0; i < g.m.nv; ++i) EXPECT_NEAR(k1[i], k2[i], 1e-4f * std::max(1.0f, std::fabs(k2[i]))) << "gains, dof " << i;

    // Neutral randomization changes nothing at all.
    EnvRandom n;
    n.friction = cm.opt.friction;
    std::vector<float> ones(g.m.nbody(), 1.0f);
    n.body_scale = ones.data();
    const auto n1 = g.qacc(g.m, cm, act, n), n2 = g.qacc(g.m, cm, act);
    for (int i = 0; i < g.m.nv; ++i) EXPECT_EQ(n1[i], n2[i]) << "neutral, dof " << i;
}

// Foot slip: feet resting still -> ~0; the whole robot sliding at 1 m/s with all
// 8 foot spheres on the floor -> 8 x 1^2; feet in the air -> 0.
TEST(G1StandTask, FootSlipPenaltyMeasuresSlidingFeet) {
    G1Physics g;
    const auto p = G1StandTask::make_params(g.m);
    std::vector<float> q = g.m.qpos0(), qd(g.m.nv, 0.0f);
    for (int k = 0; k < 12; ++k) q[p.q[k]] = p.q0[k];
    q[p.base_q + 2] = p.stand_height;
    EnvView v;
    v.model = g.m.view();
    v.state.qpos = q.data();
    v.state.qvel = qd.data();
    EXPECT_NEAR(G1StandTask::foot_slip2(p, v, 0), 0.0f, 1e-6f);
    qd[p.base_d] = 1.0f;  // free joint qvel: world linear velocity
    EXPECT_NEAR(G1StandTask::foot_slip2(p, v, 0), 8.0f, 1e-3f);
    qd[p.base_d] = 0.0f;
    qd[p.base_d + 5] = 1.0f;  // spinning about the vertical: feet move sideways, each by |r| = distance to the axis
    EXPECT_GT(G1StandTask::foot_slip2(p, v, 0), 0.01f);
    q[p.base_q + 2] = p.stand_height + 0.05f;  // lifted 5 cm: no foot touches
    EXPECT_EQ(G1StandTask::foot_slip2(p, v, 0), 0.0f);
}

}  // namespace

// ---- Phase 8c: G1 walk ------------------------------------------------------------
#include "forge/env/tasks/g1_walk.h"

namespace {

EnvOptions g1_walk_opts(int n) {
    EnvOptions o = g1_opts(n);
    o.task = "g1_walk";
    return o;
}

TEST(G1WalkTask, ShapesClockAndCommandInObs) {
    auto env = make_env_cpu(g1_walk_opts(8));
    EXPECT_EQ(env->obs_dim(), 50);
    EXPECT_EQ(env->act_dim(), 12);
    Env::HostCopy h;
    env->download(h);
    for (int e = 0; e < 8; ++e) {
        EXPECT_NEAR(h.obs[e * 50 + 48], 0.0f, 1e-6f) << "sin(2 pi phase) at step 0";
        EXPECT_NEAR(h.obs[e * 50 + 49], 1.0f, 1e-6f) << "cos(2 pi phase) at step 0";
    }
    const auto p = G1WalkTask::make_params(env->model());
    EXPECT_FLOAT_EQ(G1WalkTask::phase(p, 20), 0.5f) << "0.4 s into the 0.8 s period";
    EXPECT_FLOAT_EQ(G1WalkTask::phase(p, 40), 0.0f);
}

// Commands: inside their ranges, small ones become "stand" (vx = vy = 0), constant
// for 10 s (500 steps), new after that, different per env.
TEST(G1WalkTask, CommandsRangesAndSchedule) {
    G1WalkTask::Params p;
    std::vector<uint32_t> episode(64, 0);
    EnvView v;
    v.seed = 5;
    v.episode_index = episode.data();
    int stand = 0, changed = 0;
    for (int e = 0; e < 64; ++e) {
        const Vec3 c0 = G1WalkTask::command(p, v, e, 0), c1 = G1WalkTask::command(p, v, e, 499);
        const Vec3 c2 = G1WalkTask::command(p, v, e, 500);
        EXPECT_EQ(c0.x, c1.x);
        EXPECT_EQ(c0.z, c1.z);
        changed += c2.z != c0.z;
        EXPECT_GE(c0.x, -0.5f);
        EXPECT_LE(c0.x, 1.0f);
        EXPECT_LE(std::fabs(c0.y), 0.5f);
        EXPECT_LE(std::fabs(c0.z), 1.0f);
        if (c0.x == 0.0f && c0.y == 0.0f) ++stand;
        else EXPECT_GE(c0.x * c0.x + c0.y * c0.y, 0.04f - 1e-6f);
    }
    EXPECT_EQ(changed, 64) << "a new command after 500 steps";
    EXPECT_GT(stand, 0) << "some envs get a stand command";
    EXPECT_LT(stand, 32);
}

// Random actions for 20 s: finite obs, the reward never negative (clipped at 0).
TEST(G1WalkTask, RandomActionsFiniteRewardNotNegative) {
    auto env = make_env_cpu(g1_walk_opts(16));
    std::vector<float> act(16 * 12);
    Env::HostCopy h;
    double sum = 0;
    for (int t = 0; t < 1000; ++t) {
        env->random_actions(act.data(), t);
        env->step(act.data());
        env->download(h);
        for (float x : h.obs) ASSERT_TRUE(std::isfinite(x)) << "step " << t;
        for (int e = 0; e < 16; ++e) {
            ASSERT_GE(h.reward[e], 0.0f);
            sum += h.reward[e];
        }
    }
    std::printf("g1_walk, 16 envs x 20 s random actions: mean reward per step %.3f\n", sum / (16 * 1000));
}

}  // namespace

namespace {

// Task input (keyboard driving): g1_walk takes 3 numbers that replace its random
// command in the obs; nullptr goes back to the random command. Stand has none.
TEST(G1WalkTask, TaskInputReplacesTheCommand) {
    auto stand = make_env_cpu(g1_opts(2));
    EXPECT_EQ(stand->task_input_dim(), 0);
    auto env = make_env_cpu(g1_walk_opts(4));
    ASSERT_EQ(env->task_input_dim(), 3);
    std::vector<float> zero(4 * 12, 0.0f);
    Env::HostCopy h;
    env->step(zero.data());
    env->download(h);
    const float random_vx = h.obs[9];
    const float cmd[3] = {0.7f, -0.2f, 0.3f};
    env->set_task_input(cmd);
    env->step(zero.data());
    env->download(h);
    for (int e = 0; e < 4; ++e)
        for (int k = 0; k < 3; ++k) EXPECT_EQ(h.obs[e * 50 + 9 + k], cmd[k]) << "env " << e;
    env->set_task_input(nullptr);
    env->step(zero.data());
    env->download(h);
    EXPECT_EQ(h.obs[9], random_vx) << "same episode, same 10 s window: the random command is back";
}

}  // namespace

namespace {

// Reward terms (doc 7.4): g1_walk names 15; their means per env-step add up to
// the mean reward (when no step was clipped at 0); reading restarts the sums.
TEST(G1WalkTask, RewardTermsAddUpToTheReward) {
    EXPECT_TRUE(make_env_cpu(g1_opts(2))->reward_term_names().empty());
    auto env = make_env_cpu(g1_walk_opts(8));
    ASSERT_EQ(env->reward_term_names().size(), 15u);
    EXPECT_EQ(env->reward_term_names()[0], "track_lin_vel");
    std::vector<float> zero(8 * 12, 0.0f);
    Env::HostCopy h;
    double reward = 0;
    bool clipped = false;
    const int T = 20;
    for (int t = 0; t < T; ++t) {
        env->step(zero.data());
        env->download(h);
        for (int e = 0; e < 8; ++e) {
            reward += h.reward[e];
            clipped = clipped || h.reward[e] == 0.0f;
        }
    }
    const auto terms = env->read_reward_terms();
    double sum = 0;
    for (double x : terms) sum += x;
    EXPECT_NEAR(terms[10], 0.15, 1e-6) << "alive";
    if (!clipped) EXPECT_NEAR(sum, reward / (8 * T), 1e-4);
    for (double x : env->read_reward_terms()) EXPECT_EQ(x, 0.0) << "sums restart after a read";
}

}  // namespace

// ---- Phase 8d: terrain ------------------------------------------------------------
#include "forge/env/terrain.h"

namespace {

// 32 m field of 4 x 4 tiles, 10 cm cells, centered on 0; columns: smooth, rough,
// hill, valley; the last row is the hardest.
TEST(Terrain, LayoutAndKinds) {
    const Terrain t = make_terrain();
    EXPECT_EQ(t.nx, 321);
    EXPECT_EQ(t.ny, 321);
    EXPECT_FLOAT_EQ(t.x0, -16.0f);
    auto tile_range = [&](int tx, int ty, float& lo, float& hi) {
        lo = 1e9f;
        hi = -1e9f;
        for (int j = ty * 80 + 2; j <= ty * 80 + 78; ++j)
            for (int i = tx * 80 + 2; i <= tx * 80 + 78; ++i) {
                lo = std::min(lo, t.h[size_t(j) * t.nx + i]);
                hi = std::max(hi, t.h[size_t(j) * t.nx + i]);
            }
    };
    float lo, hi;
    tile_range(0, 0, lo, hi);  // smooth, easy
    EXPECT_LE(std::max(-lo, hi), 0.0051f);
    tile_range(1, 3, lo, hi);  // rough, hard: up to 7 cm
    EXPECT_GT(hi, 0.04f);
    EXPECT_LE(std::max(-lo, hi), 0.0701f);
    tile_range(2, 3, lo, hi);  // hill, hard: slope 0.29 over 3.5 m -> ~1 m
    EXPECT_NEAR(hi, 0.29f * 3.5f, 0.02f);
    tile_range(3, 0, lo, hi);  // valley, easy: slope 0.05 -> ~ -0.18 m
    EXPECT_NEAR(lo, -0.05f * 3.5f, 0.02f);
}

EnvOptions g1_rough_opts(int n) {
    EnvOptions o = g1_walk_opts(n);
    o.task = "g1_walk_rough";
    return o;
}

// Episodes start at different spots of the field, standing just above the ground there.
TEST(G1WalkRoughTask, StartsOnTheTerrain) {
    auto env = make_env_cpu(g1_rough_opts(16));
    ASSERT_FALSE(env->terrain().h.empty());
    EXPECT_EQ(env->obs_dim(), 50);
    const auto p = G1WalkTask::make_params(env->model());
    HeightfieldView hv{env->terrain().nx, env->terrain().ny, env->terrain().cell, env->terrain().x0,
                       env->terrain().y0, env->terrain().h.data()};
    Env::HostCopy h;
    env->download(h);
    const int nq = env->model().nq;
    for (int e = 0; e < 16; ++e) {
        const float x = h.qpos[e * nq], y = h.qpos[e * nq + 1], z = h.qpos[e * nq + 2];
        EXPECT_LE(std::fabs(x), 14.0f);
        EXPECT_LE(std::fabs(y), 14.0f);
        const float above = z - hfield_height(hv, x, y);
        EXPECT_GE(above, p.stand_height + 0.05f - 1e-4f) << "env " << e;
        EXPECT_LE(above, p.stand_height + 0.35f) << "env " << e << " (slopes lift it a little more)";
    }
    EXPECT_NE(h.qpos[0], h.qpos[nq]) << "different spots";
}

// 20 s of random actions on the terrain: finite, rewards >= 0.
TEST(G1WalkRoughTask, RandomActionsStayFinite) {
    auto env = make_env_cpu(g1_rough_opts(16));
    std::vector<float> act(16 * 12);
    Env::HostCopy h;
    for (int t = 0; t < 1000; ++t) {
        env->random_actions(act.data(), t);
        env->step(act.data());
        env->download(h);
        for (float x : h.obs) ASSERT_TRUE(std::isfinite(x)) << "step " << t;
        for (int e = 0; e < 16; ++e) ASSERT_GE(h.reward[e], 0.0f);
    }
}

}  // namespace

// ---- Phase 8e: Go2 walk -----------------------------------------------------------
#include "forge/env/tasks/go2_walk.h"

namespace {

EnvOptions go2_opts(int n, const char* task = "go2_walk") {
    EnvOptions o = cartpole_opts(n);
    o.task = task;
    o.physics_dt = 0.005f;  // configs/go2_walk.txt
    o.decimation = 4;
    return o;
}

// Joints, standing pose and gains from unitree_rl_gym's Go2; 4 feet in trot pairs.
TEST(Go2WalkTask, SetupFromTheModel) {
    auto env = make_env_cpu(go2_opts(4));
    EXPECT_EQ(env->obs_dim(), 50);
    EXPECT_EQ(env->act_dim(), 12);
    EXPECT_EQ(env->task_input_dim(), 3);
    ASSERT_EQ(env->reward_term_names().size(), 11u);
    const RobotModel& m = env->model();
    const auto p = Go2WalkTask::make_params(m);
    // body order: FL, FR, RL, RR legs; hip, thigh, calf
    const float q0[12] = {0.1f, 0.8f, -1.5f, -0.1f, 0.8f, -1.5f, 0.1f, 1.0f, -1.5f, -0.1f, 1.0f, -1.5f};
    for (int k = 0; k < 12; ++k) {
        EXPECT_FLOAT_EQ(p.q0[k], q0[k]) << "joint " << k;
        EXPECT_FLOAT_EQ(p.kp[k], 20.0f);
        EXPECT_FLOAT_EQ(p.kv[k], 0.5f);
        EXPECT_GT(p.fmax[k], 20.0f);
    }
    for (int f = 0; f < 4; ++f) {
        const std::string& n = m.body_name[p.foot_body[f]];
        const bool pair_a = n.rfind("FL", 0) == 0 || n.rfind("RR", 0) == 0;
        EXPECT_FLOAT_EQ(p.foot_offset[f], pair_a ? 0.0f : 0.5f) << n;
    }
    EXPECT_NEAR(p.stand_height, 0.324f, 0.01f);
    // start: upright, lowest foot 2 cm above the floor
    Env::HostCopy h;
    env->download(h);
    for (int e = 0; e < 4; ++e) EXPECT_NEAR(h.qpos[e * m.nq + 2], p.stand_height + 0.02f, 1e-5f);
}

// Random actions for 20 s, flat and rough: finite obs, reward never negative.
TEST(Go2WalkTask, RandomActionsFiniteRewardNotNegative) {
    for (const char* task : {"go2_walk", "go2_walk_rough"}) {
        auto env = make_env_cpu(go2_opts(16, task));
        std::vector<float> act(16 * 12);
        Env::HostCopy h;
        double sum = 0;
        for (int t = 0; t < 1000; ++t) {
            env->random_actions(act.data(), t);
            env->step(act.data());
            env->download(h);
            for (float x : h.obs) ASSERT_TRUE(std::isfinite(x)) << task << " step " << t;
            for (int e = 0; e < 16; ++e) {
                ASSERT_GE(h.reward[e], 0.0f);
                sum += h.reward[e];
            }
        }
        std::printf("%s, 16 envs x 20 s random actions: mean reward per step %.3f\n", task, sum / (16 * 1000));
    }
}

// The 11 reward terms add up to the reward (when no step was clipped at 0).
TEST(Go2WalkTask, RewardTermsAddUpToTheReward) {
    auto env = make_env_cpu(go2_opts(8));
    std::vector<float> zero(8 * 12, 0.0f);
    Env::HostCopy h;
    double reward = 0;
    bool clipped = false;
    const int T = 20;
    for (int t = 0; t < T; ++t) {
        env->step(zero.data());
        env->download(h);
        for (int e = 0; e < 8; ++e) {
            reward += h.reward[e];
            clipped = clipped || h.reward[e] == 0.0f;
        }
    }
    double sum = 0;
    for (double x : env->read_reward_terms()) sum += x;
    if (!clipped) {
        EXPECT_NEAR(sum, reward / (8 * T), 1e-4);
    }
}

// Rough: episodes start at different spots of the field, above the ground there.
TEST(Go2WalkRoughTask, StartsOnTheTerrain) {
    auto env = make_env_cpu(go2_opts(16, "go2_walk_rough"));
    ASSERT_FALSE(env->terrain().h.empty());
    const auto p = Go2WalkTask::make_params(env->model());
    const Terrain& t = env->terrain();
    HeightfieldView hv{t.nx, t.ny, t.cell, t.x0, t.y0, t.h.data()};
    Env::HostCopy h;
    env->download(h);
    const int nq = env->model().nq;
    for (int e = 0; e < 16; ++e) {
        const float x = h.qpos[e * nq], y = h.qpos[e * nq + 1], z = h.qpos[e * nq + 2];
        EXPECT_LE(std::fabs(x), 14.0f);
        EXPECT_LE(std::fabs(y), 14.0f);
        const float above = z - hfield_height(hv, x, y);
        EXPECT_GE(above, p.stand_height + 0.05f - 1e-4f) << "env " << e;
        EXPECT_LE(above, p.stand_height + 0.35f) << "env " << e;
    }
    EXPECT_NE(h.qpos[0], h.qpos[nq]) << "different spots";
}

}  // namespace

// ---- Phase 9b: Panda pick -----------------------------------------------------------
#include "forge/env/tasks/panda_pick.h"

namespace {

EnvOptions pick_opts(int n) {
    EnvOptions o = cartpole_opts(n);
    o.task = "panda_pick";
    o.physics_dt = 0.005f;  // configs/panda_pick.txt
    o.decimation = 4;
    return o;
}

// Hand position from the obs (actual pose), env e.
Vec3 hand_of(const Env::HostCopy& h, int e) { return {h.obs[e * 39 + 18], h.obs[e * 39 + 19], h.obs[e * 39 + 20]}; }

TEST(PandaPickTask, ShapesAndStart) {
    auto env = make_env_cpu(pick_opts(8));
    EXPECT_EQ(env->obs_dim(), 39);
    EXPECT_EQ(env->act_dim(), 5);
    EXPECT_TRUE(env->has_expert());
    EXPECT_EQ(env->model().nq, 16) << "9 Panda joints + the cube's free joint";
    ASSERT_EQ(env->reward_term_names().size(), 4u);
    Env::HostCopy h;
    env->download(h);
    float cube_spread = 0.0f;
    for (int e = 0; e < 8; ++e) {
        const Vec3 hp = hand_of(h, e);  // random start spot (IK at reset): x 0.35-0.60, y +-0.25, z 0.20-0.35
        EXPECT_GT(hp.x, 0.35f - 2e-3f) << "env " << e;
        EXPECT_LT(hp.x, 0.60f + 2e-3f) << "env " << e;
        EXPECT_LT(std::fabs(hp.y), 0.25f + 2e-3f) << "env " << e;
        EXPECT_GT(hp.z, 0.20f - 2e-3f) << "env " << e;
        EXPECT_LT(hp.z, 0.35f + 2e-3f) << "env " << e;
        EXPECT_GT(h.obs[e * 39 + 22], std::cos(0.8f + 0.01f)) << "turned at most 0.8 rad, pointing down";
        EXPECT_NEAR(h.obs[e * 39 + 23], 0.08f, 1e-5f) << "fingers open";
        EXPECT_NEAR(h.obs[e * 39 + 26], 0.025f, 1e-5f) << "cube on the floor";
        cube_spread = std::fmax(cube_spread, std::fabs(h.obs[e * 39 + 25]));
    }
    EXPECT_GT(cube_spread, 0.1f) << "cubes spread over the table";
    EXPECT_FALSE(make_env_cpu(g1_opts(1))->has_expert());
}

// End-effector actions: zero holds the hand (gravity compensated), dz = +0.5 lifts the
// hand target 1 cm per step, the gripper closes.
TEST(PandaPickTask, EndEffectorActions) {
    auto env = make_env_cpu(pick_opts(1));
    std::vector<float> a(5, 0.0f);
    a[4] = -1.0f;
    Env::HostCopy h;
    env->download(h);
    const Vec3 h0 = hand_of(h, 0);
    for (int t = 0; t < 50; ++t) env->step(a.data());
    env->download(h);
    const Vec3 h1 = hand_of(h, 0);
    EXPECT_LT(length(h1 - h0), 0.005f) << "zero action: the hand stays (1 s)";
    a[2] = 0.5f;  // the start is at most 0.35 m high: 4 cm up stays under the box top (0.40 m)
    for (int t = 0; t < 4; ++t) env->step(a.data());
    a[2] = 0.0f;
    for (int t = 0; t < 25; ++t) env->step(a.data());
    env->download(h);
    EXPECT_NEAR(hand_of(h, 0).z - h1.z, 0.04f, 0.005f) << "4 steps of dz = +0.5: 4 cm up";
    a[4] = 1.0f;
    for (int t = 0; t < 25; ++t) env->step(a.data());
    env->download(h);
    EXPECT_LT(h.obs[23], 0.005f) << "closed (nothing between the fingers)";
}

// The scripted expert lifts the cube in (almost) every env: friction and cube mass
// randomized, 64 envs, one episode.
TEST(PandaPickTask, ExpertLiftsTheCube) {
    const int N = 64;
    auto env = make_env_cpu(pick_opts(N));
    std::vector<float> a(N * 5);
    Env::HostCopy h;
    int lifted = 0;
    for (int t = 0; t < 250; ++t) {
        env->expert_actions(a.data());
        env->step(a.data());
        env->download(h);
        if (t == 249)
            for (int e = 0; e < N; ++e) {
                ASSERT_TRUE(h.truncated[e]) << "the episode ends after 250 steps";
                lifted += h.final_obs[e * 39 + 26] - 0.025f > 0.08f;
            }
    }
    std::printf("panda_pick expert: %d / %d envs lifted the cube more than 8 cm\n", lifted, N);
    EXPECT_GE(lifted, N * 95 / 100);
    const auto terms = env->read_reward_terms();
    EXPECT_GT(terms[2], 0.5) << "lift reward over the episode";
}

// panda_pick_color (Phase 9e): three cubes with different colors, new each episode,
// 12 cm apart; the expert lifts the named one (body 'cube', obs 24-26) and leaves the
// others on the floor.
TEST(PandaPickColorTask, ColorsAndExpert) {
    const int N = 64, D = 50;
    EnvOptions o = pick_opts(N);
    o.task = "panda_pick_color";
    auto env = make_env_cpu(o);
    ASSERT_EQ(env->obs_dim(), D);
    EXPECT_EQ(env->model().nq, 9 + 3 * 7);
    Env::HostCopy h;
    env->download(h);
    int count[3] = {0, 0, 0};
    for (int e = 0; e < N; ++e) {
        const float* ob = h.obs.data() + e * D;
        int target = -1, ones = 0;
        for (int k = 0; k < 3; ++k)
            if (ob[39 + k] == 1.0f) target = k, ++ones;
        ASSERT_EQ(ones, 1) << "one-hot target color, env " << e;
        ++count[target];
        const int c1 = int(ob[48]), c2 = int(ob[49]);
        EXPECT_TRUE(target != c1 && target != c2 && c1 != c2) << "three different colors, env " << e;
        const Vec3 c[3] = {{ob[24], ob[25], ob[26]}, {ob[42], ob[43], ob[44]}, {ob[45], ob[46], ob[47]}};
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(c[i].z, 0.025f, 1e-5f) << "on the floor";
            for (int j = i + 1; j < 3; ++j)
                EXPECT_GE(std::hypot(c[i].x - c[j].x, c[i].y - c[j].y), 0.12f - 1e-5f) << "12 cm apart, env " << e;
        }
    }
    for (int k = 0; k < 3; ++k) EXPECT_GT(count[k], N / 6) << "each color is the target sometimes";
    std::vector<float> a(N * 5);
    int lifted = 0, wrong = 0;
    for (int t = 0; t < 250; ++t) {
        env->expert_actions(a.data());
        env->step(a.data());
        env->download(h);
        if (t == 249)
            for (int e = 0; e < N; ++e) {
                const float* f = h.final_obs.data() + e * D;
                lifted += f[26] - 0.025f > 0.08f;
                wrong += f[44] - 0.025f > 0.02f || f[47] - 0.025f > 0.02f;
            }
    }
    std::printf("panda_pick_color expert: %d / %d envs lifted the named cube, %d lifted another\n", lifted, N, wrong);
    EXPECT_GE(lifted, N * 95 / 100);
    EXPECT_EQ(wrong, 0);
}

// Random end-effector actions for 4 episodes: finite, the elbow stays natural
// (posture term), the hand stays inside its box.
TEST(PandaPickTask, RandomActionsStayNatural) {
    const int N = 16;
    auto env = make_env_cpu(pick_opts(N));
    const auto p = PandaPickTask::make_params(env->model());
    std::vector<float> a(N * 5);
    Env::HostCopy h;
    float worst_elbow = 0.0f;
    for (int t = 0; t < 1000; ++t) {
        env->random_actions(a.data(), t);
        env->step(a.data());
        env->download(h);
        for (float x : h.obs) ASSERT_TRUE(std::isfinite(x)) << "step " << t;
        for (int e = 0; e < N; ++e) {
            // the box (0.30-0.65, +-0.30, 0.113-0.40) + 3 cm: PD lag at full speed (1 m/s)
            const Vec3 hp = hand_of(h, e);
            ASSERT_GT(hp.x, 0.27f) << "env " << e << " step " << t;
            ASSERT_LT(hp.x, 0.68f) << "env " << e << " step " << t;
            ASSERT_LT(std::fabs(hp.y), 0.33f) << "env " << e << " step " << t;
            ASSERT_GT(hp.z, 0.083f) << "env " << e << " step " << t;
            ASSERT_LT(hp.z, 0.43f) << "env " << e << " step " << t;
            ASSERT_GT(h.obs[e * 39 + 22], std::cos(1.3f)) << "hand turn within +-1.2 rad + 0.1 lag (pointing down kept)";
            worst_elbow = std::fmax(worst_elbow, std::fabs(h.obs[e * 39 + 2] - p.start[2]));  // joint 3: the elbow turn
        }
    }
    std::printf("panda_pick random actions: joint 3 at most %.2f rad from its start value\n", worst_elbow);
    EXPECT_LT(worst_elbow, 1.0f);
}


}  // namespace

#include <algorithm>

#include "forge/env/tasks/openarm_pick_place.h"

namespace {

// ---- openarm_pick_place (Phase 10) ----------------------------------------------------
EnvOptions openarm_opts(int n) {
    EnvOptions o = cartpole_opts(n);
    o.task = "openarm_pick_place";
    o.physics_dt = 0.002f;  // configs/openarm_pick_place.txt (0.005 s: the wrist PD shakes)
    o.decimation = 10;
    return o;
}
constexpr int kOaObs = 43;
Vec3 tcp_of(const Env::HostCopy& h, int e) {
    return {h.obs[e * kOaObs + 18], h.obs[e * kOaObs + 19], h.obs[e * kOaObs + 20]};
}
// placed: cube within the tray's inner square minus its half size (+5 mm), resting
// (on the tray's floor, 6 mm up)
bool oa_placed(const float* o) {
    return std::fabs(o[36]) < 0.03f && std::fabs(o[37]) < 0.03f && o[26] < 0.031f;
}

TEST(OpenArmPickPlace, ShapesAndStart) {
    const int N = 64;
    auto env = make_env_cpu(openarm_opts(N));
    EXPECT_EQ(env->obs_dim(), kOaObs);
    EXPECT_EQ(env->act_dim(), 5);
    EXPECT_TRUE(env->has_expert());
    EXPECT_EQ(env->model().nq, 18)
        << "9 joints of the left arm (the right arm is frozen) + the cube's free joint + the tray's 2 slide joints";
    EXPECT_FALSE(env->model().visuals.empty()) << "tray and room floor to draw";
    ASSERT_EQ(env->reward_term_names().size(), 8u);
    const auto p = OpenArmPickPlaceTask::make_params(env->model());
    Env::HostCopy h;
    env->download(h);
    float lo[4] = {1e9f, 1e9f, 1e9f, 1e9f}, hi[4] = {-1e9f, -1e9f, -1e9f, -1e9f};
    for (int e = 0; e < N; ++e) {
        const float* o = h.obs.data() + e * kOaObs;
        const float v4[4] = {o[24], o[25], o[34], o[35]};  // cube x y, tray x y
        for (int k = 0; k < 4; ++k) lo[k] = std::fmin(lo[k], v4[k]), hi[k] = std::fmax(hi[k], v4[k]);
        EXPECT_TRUE(OpenArmPickPlaceTask::in_area(p, o[24], o[25])) << "env " << e << ": cube in the work area";
        EXPECT_TRUE(OpenArmPickPlaceTask::in_area(p, o[34], o[35])) << "env " << e << ": tray in the work area";
        EXPECT_GE(std::fmax(std::fabs(o[36]), std::fabs(o[37])), p.cube_tray_gap) << "env " << e << ": cube away from the tray";
        const Vec3 t = tcp_of(h, e);
        EXPECT_LT(length(t - Vec3{p.home_tcp[0], p.home_tcp[1], p.home_tcp[2]}), 0.03f) << "env " << e << ": starts at home";
        EXPECT_NEAR(o[23], 0.5f, 1e-4f) << "fingers open";
        EXPECT_NEAR(o[26], 0.02f, 1e-5f) << "cube on the table";
        EXPECT_FALSE(oa_placed(o)) << "the cube starts outside the tray";
    }
    std::printf("openarm_pick_place starts: cube x %.2f-%.2f y %.2f-%.2f, tray x %.2f-%.2f y %.2f-%.2f, home TCP %.3f %.3f %.3f\n",
                lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], lo[3], hi[3], p.home_tcp[0], p.home_tcp[1], p.home_tcp[2]);
    EXPECT_LT(lo[1], 0.0f) << "cubes in the middle of the table, toward the right arm";
    EXPECT_GT(hi[1], 0.30f) << "cubes on the left arm's side";
    EXPECT_LT(lo[3], 0.0f) << "trays in the middle too";
    EXPECT_GT(hi[3], 0.30f);
    EXPECT_GT(hi[0] - lo[0], 0.12f);
    EXPECT_GT(hi[2] - lo[2], 0.12f);
}

// Zero action holds the TCP (2 s); dz = -0.5 for 4 steps lowers it 4 cm; closing
// on nothing closes the fingers.
TEST(OpenArmPickPlace, EndEffectorActions) {
    auto env = make_env_cpu(openarm_opts(1));
    std::vector<float> a(5, 0.0f);
    a[4] = -1.0f;
    Env::HostCopy h;
    for (int t = 0; t < 25; ++t) env->step(a.data());  // settle
    env->download(h);
    const Vec3 t0 = tcp_of(h, 0);
    for (int t = 0; t < 100; ++t) env->step(a.data());
    env->download(h);
    const Vec3 t1 = tcp_of(h, 0);
    EXPECT_LT(length(t1 - t0), 0.003f) << "zero action: the TCP stays (2 s)";
    a[2] = -0.5f;
    for (int t = 0; t < 4; ++t) env->step(a.data());
    a[2] = 0.0f;
    for (int t = 0; t < 25; ++t) env->step(a.data());
    env->download(h);
    EXPECT_NEAR(tcp_of(h, 0).z - t1.z, -0.04f, 0.005f) << "4 steps of dz = -0.5: 4 cm down (home is 0.11 m up)";
    a[4] = 1.0f;
    for (int t = 0; t < 25; ++t) env->step(a.data());
    env->download(h);
    EXPECT_LT(h.obs[23], 0.02f) << "closed (nothing between the fingers)";
}

// The work area is reachable from home by the axis IK walked there in small steps
// (as the arm moves): every 2.5 cm of it, at the grasp, hover and return heights,
// with the closing axis at turn_pref - 0.8 .. + 0.8 (any cube turn needs a 90-degree
// window; at the return height +- 0.4, as the expert holds it), to 3 mm and 0.02 rad.
// And home again from there.
TEST(OpenArmPickPlace, WorkspaceReachable) {
    auto env = make_env_cpu(openarm_opts(1));
    const RobotModel& m = env->model();
    const auto p = OpenArmPickPlaceTask::make_params(m);
    using T = OpenArmPickPlaceTask;
    const Vec3 home{p.home_tcp[0], p.home_tcp[1], p.home_tcp[2]};
    int bad = 0, total = 0, bad_home = 0;
    for (float x = p.area_x[0]; x <= p.area_x[1] + 1e-4f; x += 0.025f)
        for (float y = p.area_y[0]; y <= p.area_y[1] + 1e-4f; y += 0.025f) {
            if (!T::in_area(p, x, y)) continue;
            for (float z : {0.02f, 0.08f, 0.14f})
                for (float dt : {-0.8f, -0.4f, 0.0f, 0.4f, 0.8f}) {
                    if (z > 0.1f && std::fabs(dt) > 0.5f) continue;  // up high the expert keeps pref +- 0.4
                    std::vector<float> q(size_t(m.nq), 0.0f);
                    for (int j = 0; j < 9; ++j) q[size_t(p.q[j])] = p.start[j];
                    T::Fk f;
                    const Vec3 t{x, y, z};
                    const float turn = T::turn_pref(p, x, y) + dt;
                    auto walk = [&](Vec3 s, Vec3 g, float u0, float u1) {
                        const int n = 1 + int(length(g - s) / 0.008f + std::fabs(u1 - u0) / 0.04f);
                        for (int i = 1; i <= n + 20; ++i) {
                            const float al = std::fmin(1.0f, float(i) / float(n));
                            T::ik_to(p, m.view(), s + (g - s) * al, u0 + (u1 - u0) * al, 3, q.data(), f);
                        }
                        forward_kinematics(m.view(), q.data(), f.w);
                    };
                    walk(home, t, p.turn0, turn);
                    const Vec3 got = T::tcp_pos(p, f);
                    const float gt = T::hand_turn(p, f);
                    const bool ok = length(got - t) < 3e-3f && std::fabs(gt - turn) < 0.02f;
                    walk(got, home, gt, p.turn0);
                    const bool back = length(T::tcp_pos(p, f) - home) < 3e-3f;
                    if ((!ok || !back) && bad + bad_home < 10)
                        std::printf("  (%.3f %.3f %.3f) turn %+.2f: error %.4f m, turn error %.3f, home error %.4f m\n", x, y,
                                    z, turn, length(got - t), gt - turn, length(T::tcp_pos(p, f) - home));
                    bad += !ok;
                    bad_home += !back;
                    ++total;
                }
        }
    std::printf("openarm_pick_place work area: %d / %d (spot, height, turn) reached, %d back home\n", total - bad, total,
                total - bad_home);
    EXPECT_EQ(bad, 0);
    EXPECT_EQ(bad_home, 0);
}

// Run the expert (with optional random actions at the start of each env's first
// episode); count each env's first episode: done (the cube placed, the arm home)
// or not within the episode limit.
struct OaRun {
    int done = 0, slowest = 0, total_steps = 0;
};
OaRun oa_run_expert(int N, int random_steps, uint64_t seed) {
    auto env = make_env_cpu(openarm_opts(N));
    std::vector<float> a(N * 5);
    Rng rng(seed, 0, 0);
    Env::HostCopy h;
    OaRun r;
    std::vector<int> ended(N, 0);
    int fails = 0;
    for (int t = 0; t < OpenArmPickPlaceTask::kMaxEpisodeSteps; ++t) {
        env->expert_actions(a.data());
        if (t < random_steps)
            for (float& x : a) x = 0.5f * rng.uniform(-1.0f, 1.0f);
        env->step(a.data());
        env->download(h);
        for (int e = 0; e < N; ++e) {
            if (ended[e] || !(h.terminated[e] || h.truncated[e])) continue;
            ended[e] = 1;
            const float* o = h.final_obs.data() + e * kOaObs;
            if (h.terminated[e] && oa_placed(o)) {
                ++r.done;
                r.slowest = std::max(r.slowest, t + 1);
                r.total_steps += t + 1;
            } else if (++fails <= 3) {
                std::printf("  first failure env %d: cube %.3f %.3f %.3f, tray %.3f %.3f, TCP %.3f %.3f %.3f, opening %.2f\n",
                            e, o[24], o[25], o[26], o[34], o[35], o[18], o[19], o[20], o[23]);
            }
        }
    }
    return r;
}

// Random actions for 1 s at half size (the arm off its path, the cube maybe bumped),
// then the expert from that state: it still finishes in most envs (DAgger asks the
// expert in states like these). Not all: a bumped cube can end up out of reach.
TEST(OpenArmPickPlace, ExpertRecoversAfterRandomActions) {
    const int N = 64;
    const OaRun r = oa_run_expert(N, 50, 7);
    std::printf("openarm_pick_place: expert after 50 random steps finished %d / %d\n", r.done, N);
    EXPECT_GE(r.done, N * 85 / 100);
}

// The scripted expert puts the cube into the tray and brings the arm home, then the
// episode ends, in (almost) every env: tray and cube anywhere, friction and cube
// mass randomized, 128 envs.
TEST(OpenArmPickPlace, ExpertPlacesTheCube) {
    const int N = 128;
    const OaRun r = oa_run_expert(N, 0, 1);
    std::printf("openarm_pick_place expert: %d / %d envs placed the cube and came home (mean %d steps, slowest %d)\n",
                r.done, N, r.done ? r.total_steps / r.done : 0, r.slowest);
    EXPECT_GE(r.done, N * 97 / 100);
}

}  // namespace
