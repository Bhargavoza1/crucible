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
