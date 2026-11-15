// Phase 7a: the GPU Env backend vs the CPU answer key, cartpole, random actions,
// many episodes (resets inside step). Done flags must agree; obs, reward and
// final_obs within float rounding (FMA on the GPU, none on the CPU).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/env/env.h"
#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"

using namespace forge;

namespace {

EnvOptions opts(EnvBackend b, int n) {
    EnvOptions o;
    o.task = "cartpole";
    o.num_envs = n;
    o.seed = 3;
    o.backend = b;
    o.asset_dir = std::string(CRUCIBLE_SOURCE_DIR) + "/assets";
    return o;
}

}  // namespace

TEST(EnvGpu, CartpoleMatchesCpu) {
    const int N = 1024, steps = 300;
    auto cpu = make_env(opts(EnvBackend::kCpu, N));
    auto gpu = make_env(opts(EnvBackend::kGpu, N));
    std::vector<float> act(N);
    DeviceBuffer<float> dact(N);
    Env::HostCopy hc, hg;
    // An env whose done flag differs once (a pole exactly at the limit, rounding)
    // follows a different episode from then on: stop comparing it.
    std::vector<char> diverged(N, 0);
    double worst_obs = 0, worst_rew = 0;
    long long dones = 0;
    for (int t = 0; t < steps; ++t) {
        cpu->random_actions(act.data(), t);
        gpu->random_actions(dact.data(), t);
        cpu->step(act.data());
        gpu->step(dact.data());
        cpu->download(hc);
        gpu->download(hg);
        for (int e = 0; e < N; ++e) {
            if (diverged[e]) continue;
            if (hc.terminated[e] != hg.terminated[e] || hc.truncated[e] != hg.truncated[e]) {
                diverged[e] = 1;
                continue;
            }
            dones += hc.terminated[e] + hc.truncated[e];
            worst_rew = std::max(worst_rew, double(std::fabs(hc.reward[e] - hg.reward[e])));
            for (int k = 0; k < 4; ++k) {
                worst_obs = std::max(worst_obs, double(std::fabs(hc.obs[e * 4 + k] - hg.obs[e * 4 + k])));
                if (hc.terminated[e] || hc.truncated[e])
                    worst_obs = std::max(worst_obs, double(std::fabs(hc.final_obs[e * 4 + k] - hg.final_obs[e * 4 + k])));
            }
        }
    }
    const int ndiv = int(std::count(diverged.begin(), diverged.end(), 1));
    std::printf("%d envs x %d steps: %lld episodes ended, worst |obs| diff %.2e, |reward| diff %.2e, "
                "%d envs with a different done flag\n", N, steps, dones, worst_obs, worst_rew, ndiv);
    EXPECT_GT(dones, N) << "random actions should end many episodes";
    EXPECT_LE(ndiv, N / 100);
    EXPECT_LT(worst_obs, 1e-3);
    EXPECT_LT(worst_rew, 1e-5);
}

// The GPU backend does not sync inside step(): 100 steps queue up without error,
// and the result is the same as stepping with a sync after each step.
TEST(EnvGpu, StepIsAsyncAndDeterministic) {
    auto a = make_env(opts(EnvBackend::kGpu, 256));
    auto b = make_env(opts(EnvBackend::kGpu, 256));
    // One action buffer per env: a and b run on different streams, so a shared
    // buffer would be a race (b writing actions while a still reads them).
    DeviceBuffer<float> act_a(256), act_b(256);
    for (int t = 0; t < 100; ++t) {
        a->random_actions(act_a.data(), t);
        a->step(act_a.data());
    }
    for (int t = 0; t < 100; ++t) {
        b->random_actions(act_b.data(), t);
        b->step(act_b.data());
        CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(b->stream())));
    }
    Env::HostCopy ha, hb;
    a->download(ha);
    b->download(hb);
    EXPECT_EQ(ha.obs, hb.obs);
}
