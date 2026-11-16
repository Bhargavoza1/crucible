// Phase 7e: SAC, TQC, the replay buffer and the Algorithm interface on the GPU.
//   - replay buffer: keeps final_obs where an episode ended, done = terminated only
//   - SAC learns cartpole (16 envs). The PyTorch baseline (tools/sac_baseline.py,
//     same algorithm and settings) went from 36 to ~200 in 45k steps on a CPU.
//   - same seed, same numbers; checkpoint round trip gives the same actions
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/env/env.h"
#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"
#include "temper/algos/algorithm.h"
#include "temper/algos/sac/sac.h"
#include "temper/buffers/replay/replay_buffer.h"
#include "temper/common/train_loop.h"

using namespace temper;

namespace {

std::unique_ptr<forge::Env> cartpole(int n, uint64_t seed) {
    forge::EnvOptions o;
    o.task = "cartpole";
    o.num_envs = n;
    o.seed = seed;
    o.backend = forge::EnvBackend::kGpu;
    o.asset_dir = std::string(CRUCIBLE_SOURCE_DIR) + "/assets";
    return forge::make_env(o);
}

}  // namespace

TEST(ReplayBuffer, KeepsFinalObsAndTerminatedOnly) {
    // 3 envs, obs 2, act 1. Env 1 terminated, env 2 truncated.
    ReplayBuffer rb(8, 2, 1);
    Tensor obs(3, 2), act(3, 1), rew(1, 3), nxt(3, 2), fin(3, 2);
    forge::DeviceBuffer<int> term(3), trunc(3);
    obs.upload({1, 1, 2, 2, 3, 3});
    act.upload({0.1f, 0.2f, 0.3f});
    rew.upload({10, 20, 30});
    nxt.upload({4, 4, 5, 5, 6, 6});  // obs after the step (new episodes for envs 1, 2)
    fin.upload({0, 0, 50, 50, 60, 60});
    term.upload({0, 1, 0});
    trunc.upload({0, 0, 1});
    rb.add(obs.data(), act.data(), rew.data(), nxt.data(), fin.data(), term.data(), trunc.data(), 3, nullptr);
    EXPECT_EQ(rb.size(), 3);
    Tensor bo(64, 2), ba(64, 1), br(1, 64), bn(64, 2), bd(1, 64);
    rb.sample(64, 1, 0, bo, ba, br, bn, bd, nullptr);
    const auto o = bo.download(), n = bn.download(), r = br.download(), d = bd.download();
    int seen[3] = {0, 0, 0};
    for (int i = 0; i < 64; ++i) {
        const int e = int(o[2 * i]) - 1;  // which env this row came from
        ASSERT_GE(e, 0);
        ASSERT_LT(e, 3);
        ++seen[e];
        EXPECT_EQ(r[i], 10.0f * (e + 1));
        EXPECT_EQ(n[2 * i], e == 0 ? 4.0f : (e == 1 ? 50.0f : 60.0f)) << "ended envs keep final_obs";
        EXPECT_EQ(d[i], e == 1 ? 1.0f : 0.0f) << "done = terminated only (truncated still bootstraps)";
    }
    EXPECT_GT(seen[0] * seen[1] * seen[2], 0) << "uniform sampling should hit all three";
}

TEST(Sac, LearnsCartpole) {
    auto env = cartpole(16, 1);
    auto algo = make_algorithm("sac", 4, 1, 16, nullptr, static_cast<cudaStream_t>(env->stream()));
    TrainOptions opt;
    opt.max_iterations = 120;  // 120 x 24 steps x 16 envs = 46k steps
    opt.quiet = true;
    const TrainResult r = train(*env, *algo, opt);
    std::printf("SAC cartpole, 16 envs x 120 iterations (46k steps): last mean return %.1f, best %.1f, "
                "%.3g env-steps/s\n", r.last_mean_return, r.best_mean_return, r.env_steps_per_s);
    EXPECT_FALSE(r.stopped_on_nan);
    EXPECT_GT(r.best_mean_return, 120.0) << "the baseline passes 130 by about 18k steps";
}

TEST(Sac, SameSeedSameNumbers) {
    auto run = [] {
        auto env = cartpole(16, 2);
        SacConfig c;
        c.learning_starts = 400;
        auto algo = std::make_unique<Sac>(4, 1, 16, c, static_cast<cudaStream_t>(env->stream()));
        TrainOptions opt;
        opt.max_iterations = 3;
        opt.quiet = true;
        train(*env, *algo, opt);
        std::vector<float> w;
        const Checkpoint ck = algo->checkpoint("cartpole");
        for (const char* name : {"actor.0.W", "q1.0.W", "q2t.2.b", "log_alpha"}) {
            const auto& a = ck.array(name);
            w.insert(w.end(), a.begin(), a.end());
        }
        return w;
    };
    EXPECT_EQ(run(), run());
}

TEST(Sac, CheckpointRoundTripGivesSameActions) {
    auto env = cartpole(16, 3);
    cudaStream_t s = static_cast<cudaStream_t>(env->stream());
    SacConfig c;
    c.learning_starts = 400;
    Sac a(4, 1, 16, c, s);
    TrainOptions opt;
    opt.max_iterations = 2;
    opt.quiet = true;
    train(*env, a, opt);
    const std::string path = "/tmp/crucible_test_sac_ckpt.bin";
    a.checkpoint("cartpole").save(path);
    SacConfig other = c;
    other.seed = 77;
    Sac b(4, 1, 16, other, s);
    b.load(Checkpoint::load(path), true);
    std::remove(path.c_str());
    forge::DeviceBuffer<float> out_a(16), out_b(16);
    a.act_deterministic(env->obs(), out_a.data(), 16);
    b.act_deterministic(env->obs(), out_b.data(), 16);
    EXPECT_EQ(out_a.download(), out_b.download());
    EXPECT_EQ(b.iteration(), a.iteration());
    auto ppo = make_algorithm("ppo", 4, 1, 16, nullptr, s);
    EXPECT_THROW(ppo->load(a.checkpoint("cartpole"), false), std::runtime_error) << "a sac checkpoint is not ppo";
}

// Writes one SAC gradient step (weights before and after, batch, noise,
// gradients) for tools/sac_check_step.py. Runs only with CRUCIBLE_SAC_STEP_DIR set.
TEST(Sac, DumpOneStepForPytorchCheck) {
    const char* dir = std::getenv("CRUCIBLE_SAC_STEP_DIR");
    if (!dir) GTEST_SKIP() << "set CRUCIBLE_SAC_STEP_DIR to write the files";
    const char* algo = std::getenv("CRUCIBLE_SAC_STEP_ALGO");  // sac (default) or tqc
    const bool tqc = algo && std::string(algo) == "tqc";
    auto env = cartpole(16, 5);
    cudaStream_t s = static_cast<cudaStream_t>(env->stream());
    SacConfig c = tqc ? SacConfig::tqc_defaults() : SacConfig{};
    c.learning_starts = 2000;
    c.updates_per_step = 1;
    c.seed = 5;
    Sac sac(4, 1, 16, c, s);
    forge::DeviceBuffer<float> actions(16);
    const TransitionView tv{env->reward(), env->terminated(), env->truncated(), env->final_obs(), env->obs()};
    // Warmup, then 300 normal gradient steps so the networks and Adam are not at their start.
    int updates = 0;
    while (updates < 300) {
        sac.act(env->obs(), actions.data());
        env->step(actions.data());
        sac.observe(tv);
        if (sac.ready_to_update()) {
            sac.update(env->obs());
            ++updates;
        }
    }
    const std::string d(dir);
    sac.checkpoint("cartpole").save(d + "/before.bin");
    SacStepCapture cap;
    sac.capture_next_step(&cap);
    sac.update(env->obs());
    sac.checkpoint("cartpole").save(d + "/after.bin");
    Checkpoint step;
    step.algo = "sac_step";
    step.task = "cartpole";
    step.obs_dim = 4;
    step.act_dim = 1;
    step.arrays = cap.arrays;
    step.arrays["gamma"] = {c.gamma};
    step.arrays["tau"] = {c.tau};
    step.arrays["lr"] = {c.q_lr, c.policy_lr, c.alpha_lr};
    step.save(d + "/step.bin");
    std::printf("wrote %s before.bin, step.bin, after.bin to %s\n", tqc ? "tqc" : "sac", dir);
}

TEST(Tqc, LearnsCartpole) {
    auto env = cartpole(16, 1);
    auto algo = make_algorithm("tqc", 4, 1, 16, nullptr, static_cast<cudaStream_t>(env->stream()));
    EXPECT_STREQ(algo->name(), "tqc");
    TrainOptions opt;
    opt.max_iterations = 120;  // 46k steps
    opt.quiet = true;
    const TrainResult r = train(*env, *algo, opt);
    std::printf("TQC cartpole, 16 envs x 120 iterations (46k steps): last mean return %.1f, best %.1f, "
                "%.3g env-steps/s\n", r.last_mean_return, r.best_mean_return, r.env_steps_per_s);
    EXPECT_FALSE(r.stopped_on_nan);
    EXPECT_GT(r.best_mean_return, 100.0) << "starts near 36; the PyTorch TQC was at 82 after 11k steps";
}

TEST(Tqc, SameSeedSameNumbersAndCheckpoint) {
    auto run = [](Checkpoint* out) {
        auto env = cartpole(16, 2);
        SacConfig c = SacConfig::tqc_defaults();
        c.learning_starts = 400;
        auto algo = std::make_unique<Sac>(4, 1, 16, c, static_cast<cudaStream_t>(env->stream()));
        TrainOptions opt;
        opt.max_iterations = 3;
        opt.quiet = true;
        train(*env, *algo, opt);
        *out = algo->checkpoint("cartpole");
    };
    Checkpoint a, b;
    run(&a);
    run(&b);
    for (const char* name : {"actor.0.W", "q1.0.W", "q5t.2.b", "log_alpha"}) EXPECT_EQ(a.array(name), b.array(name)) << name;
    EXPECT_EQ(a.algo, "tqc");
    EXPECT_EQ(a.layer_sizes.at("critic").back(), 25);

    auto env = cartpole(16, 3);
    cudaStream_t s = static_cast<cudaStream_t>(env->stream());
    Sac full(4, 1, 16, SacConfig::tqc_defaults(), s);
    full.load(a, true);  // full resume
    SacConfig other = SacConfig::tqc_defaults();
    other.n_critics = 3;
    Sac small(4, 1, 16, other, s);
    EXPECT_THROW(small.load(a, true), std::runtime_error) << "critic count differs";
    small.load(a, false);  // play mode: actor only
    forge::DeviceBuffer<float> o1(16), o2(16);
    full.act_deterministic(env->obs(), o1.data(), 16);
    small.act_deterministic(env->obs(), o2.data(), 16);
    EXPECT_EQ(o1.download(), o2.download());
    auto sac = make_algorithm("sac", 4, 1, 16, nullptr, s);
    EXPECT_THROW(sac->load(a, false), std::runtime_error) << "a tqc checkpoint is not sac";
}
