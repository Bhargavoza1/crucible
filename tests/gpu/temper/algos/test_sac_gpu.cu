// Phase 7e: SAC, the replay buffer and the Algorithm interface on the GPU.
//   - replay buffer: keeps final_obs where an episode ended, done = terminated only
//   - SAC learns cartpole (16 envs). The PyTorch baseline (tools/sac_baseline.py,
//     same algorithm and settings) went from 36 to ~200 in 45k steps on a CPU.
//   - same seed, same numbers; checkpoint round trip gives the same actions
#include <cmath>
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
