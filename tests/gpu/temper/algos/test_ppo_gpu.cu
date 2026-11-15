// Phase 7c: PPO on the GPU with the cartpole Env.
//   - learns: 1024 envs, 60 iterations (1.5M steps); mean episode return > 400
//     (the max is ~500). A numpy version of the same algorithm and settings
//     on MuJoCo reaches ~497 after ~190k steps (256 envs), so 400 has room.
//   - deterministic: the same seed gives the same numbers twice
//   - checkpoint: save + load gives the same deterministic actions
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/env/env.h"
#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"
#include "temper/algos/ppo/ppo.h"
#include "temper/algos/ppo/train_ppo.h"

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

TEST(Ppo, LearnsCartpole) {
    auto env = cartpole(1024, 1);
    PpoConfig cfg;
    TrainOptions opt;
    opt.max_iterations = 60;
    opt.quiet = true;
    const TrainResult r = train_ppo(*env, cfg, opt);
    std::printf("cartpole, 1024 envs x 60 iterations: last mean return %.1f (episode length %.1f), best %.1f, "
                "%.3g env-steps/s overall\n", r.last_mean_return, r.last_mean_length, r.best_mean_return,
                r.env_steps_per_s);
    EXPECT_FALSE(r.stopped_on_nan);
    EXPECT_GT(r.last_mean_return, 400.0);
}

TEST(Ppo, SameSeedSameNumbers) {
    auto run = [] {
        auto env = cartpole(256, 3);
        cudaStream_t s = static_cast<cudaStream_t>(env->stream());
        Ppo ppo(4, 1, 256, PpoConfig{}, s);
        forge::DeviceBuffer<float> act(256);
        std::vector<float> out;
        for (int it = 0; it < 3; ++it) {
            for (int t = 0; t < 24; ++t) {
                ppo.act(env->obs(), act.data());
                env->step(act.data());
                ppo.observe(env->reward(), env->terminated(), env->truncated(), env->final_obs());
            }
            const PpoStats st = ppo.update(env->obs());
            out.insert(out.end(), {st.policy_loss, st.value_loss, st.approx_kl, st.grad_norm});
        }
        return out;
    };
    EXPECT_EQ(run(), run());
}

TEST(Ppo, CheckpointRoundTripGivesSameActions) {
    auto env = cartpole(64, 5);
    cudaStream_t s = static_cast<cudaStream_t>(env->stream());
    Ppo a(4, 1, 64, PpoConfig{}, s);
    forge::DeviceBuffer<float> act(64), out_a(64), out_b(64);
    for (int t = 0; t < 24; ++t) {
        a.act(env->obs(), act.data());
        env->step(act.data());
        a.observe(env->reward(), env->terminated(), env->truncated(), env->final_obs());
    }
    a.update(env->obs());
    const std::string path = "/tmp/crucible_test_ppo_ckpt.bin";
    a.checkpoint("cartpole").save(path);
    PpoConfig other;
    other.seed = 99;  // different init: only the loaded weights can make b equal a
    Ppo b(4, 1, 64, other, s);
    b.load(Checkpoint::load(path), true);
    std::remove(path.c_str());
    EXPECT_EQ(b.iteration(), a.iteration());
    a.act_deterministic(env->obs(), out_a.data(), 64);
    b.act_deterministic(env->obs(), out_b.data(), 64);
    EXPECT_EQ(out_a.download(), out_b.download());
    // A checkpoint for other sizes is refused.
    Ppo c(5, 1, 64, PpoConfig{}, s);
    EXPECT_THROW(c.load(a.checkpoint("cartpole"), true), std::runtime_error);
}
