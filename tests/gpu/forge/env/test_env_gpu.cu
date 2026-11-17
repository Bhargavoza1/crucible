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

// The ant: contacts, limits, PD and the free base inside the Env, random
// full-range targets (joints hit their limits every step).
// A limit or contact row starts the step its distance crosses 0, with a sudden
// push. If GPU math (FMA, sinf, powf) moves a joint across that line one step
// earlier than the CPU, that env's path splits by several rad/s, and that is
// physics, not a bug. A bug moves most envs. So: report how many envs differ
// and from which step, for both GPU kernels; most envs must still agree.
// Differences are relative (|gpu - cpu| / max(1, |cpu|)): a joint at 25 rad/s that
// differs by 0.02 rad/s has not split.
// Terrain tasks: only envs that spawn within 6 m of the origin are checked. Far out,
// float32 holds a position only to ~1 um (14 m: 1 ulp = 0.95 um); on a slope that
// enters the contact distance, so FMA (GPU) vs no FMA (CPU) rounding grows apart on
// rough ground within a few steps. The same split shows between two CPU builds,
// with and without -mfma (256 envs, 10 steps: 0 of 22 split within 4 m, 100 of 109
// beyond 12 m). The far envs are still stepped and must stay finite.
static void task_gpu_vs_cpu(const char* task, const char* solver) {
    const int N = 256, T = 10;
    EnvOptions oc = opts(EnvBackend::kCpu, N), og = opts(EnvBackend::kGpu, N);
    oc.task = og.task = task;
    og.solver = solver;
    auto cpu = make_env(oc);
    auto gpu = make_env(og);
    const int D = cpu->obs_dim(), A = cpu->act_dim();
    std::vector<float> act(N * A);
    DeviceBuffer<float> dact(N * A);
    Env::HostCopy hc, hg;
    std::vector<int> split_at(N, -1);
    std::vector<char> checked(N, 1);
    int n_checked = N;
    double worst_agree = 0;
    {  // the start states must be the same (reset on both backends)
        cpu->download(hc);
        gpu->download(hg);
        double d0 = 0;
        for (size_t i = 0; i < hc.qpos.size(); ++i) d0 = std::max(d0, double(std::fabs(hc.qpos[i] - hg.qpos[i])));
        std::printf("%s: start qpos GPU vs CPU, worst difference %.2e\n", task, d0);
        EXPECT_LT(d0, 1e-4) << "the reset gives different start states";
        if (!cpu->terrain().h.empty()) {  // free base: qpos[0], qpos[1] are x, y
            const int nq = cpu->model().nq;
            n_checked = 0;
            for (int e = 0; e < N; ++e) {
                checked[e] = std::hypot(hc.qpos[e * nq], hc.qpos[e * nq + 1]) < 6.0f;
                n_checked += checked[e];
            }
            std::printf("%s: terrain, checking the %d envs that spawn within 6 m of the origin\n", task, n_checked);
            ASSERT_GE(n_checked, 20);
        }
    }
    bool told = false;
    for (int t = 0; t < 200; ++t) {
        cpu->random_actions(act.data(), t);
        gpu->random_actions(dact.data(), t);
        cpu->step(act.data());
        gpu->step(dact.data());
        gpu->download(hg);
        for (float x : hg.obs) ASSERT_TRUE(std::isfinite(x)) << "step " << t;
        if (t >= T) continue;
        cpu->download(hc);
        for (int e = 0; e < N; ++e) {
            if (!checked[e]) continue;
            double d = 0;
            for (int k = 0; k < D; ++k)
                d = std::max(d, std::fabs(double(hc.obs[e * D + k]) - hg.obs[e * D + k]) /
                                    std::max(1.0, std::fabs(double(hc.obs[e * D + k]))));
            if (d > 1e-2 && split_at[e] < 0) {
                split_at[e] = t + 1;
                if (!told) {  // details of the first env that splits
                    told = true;
                    const int nq = cpu->model().nq;
                    double dq = 0;
                    int iq = 0;
                    for (int i = 0; i < nq; ++i) {
                        const double x = std::fabs(hc.qpos[e * nq + i] - hg.qpos[e * nq + i]);
                        if (x > dq) { dq = x; iq = i; }
                    }
                    std::printf("  first split: env %d at step %d; worst qpos difference %.3e at index %d "
                                "(cpu %.5f gpu %.5f); obs:", e, t + 1, dq, iq, hc.qpos[e * nq + iq], hg.qpos[e * nq + iq]);
                    for (int k = 0; k < D; ++k) {
                        const float x = std::fabs(hc.obs[e * D + k] - hg.obs[e * D + k]);
                        if (x > 1e-2f) std::printf(" [%d] %.3f/%.3f", k, hc.obs[e * D + k], hg.obs[e * D + k]);
                    }
                    std::printf("\n");
                }
            }
            if (split_at[e] < 0) worst_agree = std::max(worst_agree, d);
        }
    }
    int split = 0, first = T + 1;
    for (int e = 0; e < N; ++e)
        if (split_at[e] > 0) {
            ++split;
            first = std::min(first, split_at[e]);
        }
    std::printf("%s, %d envs checked, solver %s: %d envs split from the CPU in the first %d steps (first at step %d); "
                "worst relative obs difference in the others %.2e\n", task, n_checked, solver, split, T,
                split ? first : 0, worst_agree);
    EXPECT_LE(split, n_checked / 20) << "more than 5% of envs split: a real difference, not one-step contact timing";
    EXPECT_LT(worst_agree, 1e-2);
}

TEST(EnvGpu, AntMatchesCpuWarpKernel) { task_gpu_vs_cpu("ant", "newton"); }
TEST(EnvGpu, AntMatchesCpuThreadKernel) { task_gpu_vs_cpu("ant", "newton_thread"); }
// Phase 8b-1: same check for the G1 (noise, pushes and delay use the same counter
// RNG streams on both backends, so they must agree too).
TEST(EnvGpu, G1StandMatchesCpuWarpKernel) { task_gpu_vs_cpu("g1_stand", "newton"); }
TEST(EnvGpu, G1StandMatchesCpuThreadKernel) { task_gpu_vs_cpu("g1_stand", "newton_thread"); }
TEST(EnvGpu, G1WalkMatchesCpuWarpKernel) { task_gpu_vs_cpu("g1_walk", "newton"); }  // Phase 8c
TEST(EnvGpu, G1WalkRoughMatchesCpuWarpKernel) { task_gpu_vs_cpu("g1_walk_rough", "newton"); }  // Phase 8d
TEST(EnvGpu, G1WalkRoughMatchesCpuThreadKernel) { task_gpu_vs_cpu("g1_walk_rough", "newton_thread"); }
TEST(EnvGpu, Go2WalkMatchesCpuWarpKernel) { task_gpu_vs_cpu("go2_walk", "newton"); }  // Phase 8e
TEST(EnvGpu, Go2WalkMatchesCpuThreadKernel) { task_gpu_vs_cpu("go2_walk", "newton_thread"); }
TEST(EnvGpu, Go2WalkRoughMatchesCpuWarpKernel) { task_gpu_vs_cpu("go2_walk_rough", "newton"); }
TEST(EnvGpu, PandaPickMatchesCpuWarpKernel) { task_gpu_vs_cpu("panda_pick", "newton"); }  // Phase 9b
TEST(EnvGpu, PandaPickMatchesCpuThreadKernel) { task_gpu_vs_cpu("panda_pick", "newton_thread"); }

// Phase 9b: the scripted expert runs on the GPU (IK and gravity compensation per env
// in the env kernels): its first actions equal the CPU expert's, and it lifts the
// cube in as many envs.
// The scripted expert on both backends: same first actions, both lift the cube.
void expert_gpu_vs_cpu(const char* task) {
    const int N = 256;
    EnvOptions oc = opts(EnvBackend::kCpu, N), og = opts(EnvBackend::kGpu, N);
    oc.task = og.task = task;
    oc.physics_dt = og.physics_dt = 0.005f;  // configs/panda_pick.txt
    oc.decimation = og.decimation = 4;
    auto cpu = make_env(oc);
    auto gpu = make_env(og);
    const int A = cpu->act_dim(), D = cpu->obs_dim();
    std::vector<float> ac(N * A), ag(N * A);
    DeviceBuffer<float> dact(N * A);
    Env::HostCopy hc, hg;
    int lifted_cpu = 0, lifted_gpu = 0;
    double first_diff = 0;
    for (int t = 0; t < 250; ++t) {
        cpu->expert_actions(ac.data());
        gpu->expert_actions(dact.data());
        CUDA_CHECK(cudaMemcpy(ag.data(), dact.data(), ag.size() * 4, cudaMemcpyDeviceToHost));
        if (t == 0)
            for (size_t i = 0; i < ac.size(); ++i) first_diff = std::max(first_diff, double(std::fabs(ac[i] - ag[i])));
        cpu->step(ac.data());
        gpu->step(dact.data());
        if (t == 249) {
            cpu->download(hc);
            gpu->download(hg);
            for (int e = 0; e < N; ++e) {
                lifted_cpu += hc.final_obs[e * D + 26] - 0.025f > 0.08f;
                lifted_gpu += hg.final_obs[e * D + 26] - 0.025f > 0.08f;
            }
        }
    }
    std::printf("%s expert, %d envs: first actions GPU vs CPU within %.2e; lifted %d (CPU) and %d (GPU)\n", task, N,
                first_diff, lifted_cpu, lifted_gpu);
    EXPECT_LT(first_diff, 1e-4);
    EXPECT_GE(lifted_gpu, N * 95 / 100);
    EXPECT_GE(lifted_cpu, N * 95 / 100);
}

TEST(EnvGpu, PandaPickExpertMatchesCpu) { expert_gpu_vs_cpu("panda_pick"); }
// Phase 10: OpenArm pick and place (physics 0.002 s x 10): the expert's first actions
// agree, and on both backends (almost) every env's first episode ends done (the
// cube in the tray, the arm back home; tray and cube anywhere in the work area).
TEST(EnvGpu, OpenArmPickPlaceExpertMatchesCpu) {
    const int N = 256;
    EnvOptions oc = opts(EnvBackend::kCpu, N), og = opts(EnvBackend::kGpu, N);
    oc.task = og.task = "openarm_pick_place";
    oc.physics_dt = og.physics_dt = 0.002f;  // configs/openarm_pick_place.txt
    oc.decimation = og.decimation = 10;
    auto cpu = make_env(oc);
    auto gpu = make_env(og);
    const int A = cpu->act_dim(), D = cpu->obs_dim();
    std::vector<float> ac(N * A), ag(N * A);
    DeviceBuffer<float> dact(N * A);
    Env::HostCopy hc, hg;
    int done_cpu = 0, done_gpu = 0;
    std::vector<int> ended_cpu(N, 0), ended_gpu(N, 0);
    double first_diff = 0;
    auto placed = [](const float* o) { return std::fabs(o[36]) < 0.03f && std::fabs(o[37]) < 0.03f && o[26] < 0.025f; };
    auto count = [&](const Env::HostCopy& h, std::vector<int>& ended, int& done) {
        for (int e = 0; e < N; ++e) {
            if (ended[e] || !(h.terminated[e] || h.truncated[e])) continue;
            ended[e] = 1;
            done += h.terminated[e] && placed(h.final_obs.data() + e * D);
        }
    };
    for (int t = 0; t < 400; ++t) {
        cpu->expert_actions(ac.data());
        gpu->expert_actions(dact.data());
        CUDA_CHECK(cudaMemcpy(ag.data(), dact.data(), ag.size() * 4, cudaMemcpyDeviceToHost));
        if (t == 0)
            for (size_t i = 0; i < ac.size(); ++i) first_diff = std::max(first_diff, double(std::fabs(ac[i] - ag[i])));
        cpu->step(ac.data());
        gpu->step(dact.data());
        cpu->download(hc);
        gpu->download(hg);
        count(hc, ended_cpu, done_cpu);
        count(hg, ended_gpu, done_gpu);
    }
    std::printf("openarm_pick_place expert, %d envs: first actions GPU vs CPU within %.2e; done %d (CPU) and %d (GPU)\n",
                N, first_diff, done_cpu, done_gpu);
    EXPECT_LT(first_diff, 1e-4);
    EXPECT_GE(done_gpu, N * 95 / 100);
    EXPECT_GE(done_cpu, N * 95 / 100);
}

// Phase 9e: three colored cubes (the GPU solver's general kernel: too many contact
// rows for the warp kernel)
TEST(EnvGpu, PandaPickColorExpertMatchesCpu) { expert_gpu_vs_cpu("panda_pick_color"); }
