// Phase 11: openarm_stack (both arms build a tower on the spot: red, then green, then blue).
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/env/env.h"
#include "forge/env/tasks/openarm_stack.h"

using namespace forge;

namespace {

using T = OpenArmStackTask;

EnvOptions stack_opts(int n, uint64_t seed = 1) {
    EnvOptions o;
    o.task = "openarm_stack";
    o.num_envs = n;
    o.seed = seed;
    o.backend = EnvBackend::kCpu;
    o.asset_dir = std::string(CRUCIBLE_SOURCE_DIR) + "/assets";
    o.physics_dt = 0.002f;  // configs/openarm_stack.txt
    o.decimation = 10;
    return o;
}
Vec3 at(const float* o, int i) { return {o[i], o[i + 1], o[i + 2]}; }
constexpr int kTcp[2] = {36, 42}, kCube[3] = {48, 55, 62};

// Sizes; both hands start at home (mirror images), fingers open; the cubes on the
// table anywhere the arms reach, 12 cm or more from the spot, 10 cm or more apart.
TEST(OpenArmStack, ShapesAndStart) {
    const int N = 256;
    auto env = make_env_cpu(stack_opts(N));
    EXPECT_EQ(env->obs_dim(), 79);
    EXPECT_EQ(env->act_dim(), 10);
    EXPECT_EQ(env->model().nq, 18 + 21) << "2 x (7 arm + 2 fingers) + 3 free cubes";
    const T::Params p = T::make_params(env->model());
    Env::HostCopy h;
    env->download(h);
    int left[3] = {0, 0, 0};
    float lo_x = 1, hi_x = 0, lo_y = 1, hi_y = -1;
    for (int e = 0; e < N; ++e) {
        const float* o = h.obs.data() + size_t(e) * 79;
        EXPECT_LT(length(at(o, kTcp[0]) - T::home_of(p, 0)), 0.03f) << "env " << e;
        EXPECT_LT(length(at(o, kTcp[1]) - T::home_of(p, 1)), 0.03f) << "env " << e;
        EXPECT_NEAR(o[41], 0.5f, 0.02f);
        EXPECT_NEAR(o[47], 0.5f, 0.02f);
        for (int c = 0; c < 3; ++c) {
            const Vec3 x = at(o, kCube[c]);
            EXPECT_NEAR(x.z, 0.02f, 1e-4f);
            EXPECT_GE(std::fmax(std::fabs(x.x - p.spot[0]), std::fabs(x.y - p.spot[1])), 0.12f - 1e-4f) << "env " << e;
            for (int d = 0; d < c; ++d) {
                const Vec3 y = at(o, kCube[d]);
                EXPECT_GE(std::fmax(std::fabs(x.x - y.x), std::fabs(x.y - y.y)), 0.10f - 1e-4f) << "env " << e;
            }
            left[c] += x.y >= 0.0f;
            lo_x = std::fmin(lo_x, x.x), hi_x = std::fmax(hi_x, x.x), lo_y = std::fmin(lo_y, x.y), hi_y = std::fmax(hi_y, x.y);
        }
    }
    for (int c = 0; c < 3; ++c) EXPECT_TRUE(left[c] > N / 4 && left[c] < 3 * N / 4) << "cube " << c << " on both halves";
    const Vec3 hl = T::home_of(p, 0), hr = T::home_of(p, 1);
    std::printf("openarm_stack: cubes x %.2f-%.2f y %.2f-%.2f (left half: red %d, green %d, blue %d of %d); home TCPs "
                "left %.3f %.3f %.3f, right %.3f %.3f %.3f\n", lo_x, hi_x, lo_y, hi_y, left[0], left[1], left[2], N,
                hl.x, hl.y, hl.z, hr.x, hr.y, hr.z);
}

// The right arm is the left's mirror: the same moves with y and the turn flipped move
// the two TCPs (and turn the hands) as mirror images (within 1.5 cm over 15 cm: the
// start postures differ by the reset's joint noise).
TEST(OpenArmStack, MirroredActions) {
    EnvOptions o = stack_opts(1);
    auto env = make_env_cpu(o);
    Env::HostCopy h0, h;
    env->download(h0);
    const float a[10] = {-0.3f, 0.25f, -0.3f, 0.2f, -1.0f, -0.3f, -0.25f, -0.3f, -0.2f, -1.0f};
    for (int t = 0; t < 25; ++t) env->step(a);
    env->download(h);
    const Vec3 dl = at(h.obs.data(), kTcp[0]) - at(h0.obs.data(), kTcp[0]);
    const Vec3 dr = at(h.obs.data(), kTcp[1]) - at(h0.obs.data(), kTcp[1]);
    std::printf("openarm_stack mirrored moves: left %.3f %.3f %.3f, right %.3f %.3f %.3f\n", dl.x, dl.y, dl.z, dr.x,
                dr.y, dr.z);
    EXPECT_GT(length(dl), 0.08f) << "the hands moved";
    EXPECT_NEAR(dl.x, dr.x, 0.015f);
    EXPECT_NEAR(dl.y, -dr.y, 0.015f);
    EXPECT_NEAR(dl.z, dr.z, 0.015f);
    // turns: world yaw of the closing axis, mirrored (sin flips, cos stays)
    EXPECT_NEAR(h.obs[39], -h.obs[45], 0.05f);
    EXPECT_NEAR(h.obs[40], h.obs[46], 0.05f);
}

// The scripted expert builds the tower and brings both hands home in 90% or more of
// the envs, under the sparse reward: success once per done episode.
TEST(OpenArmStack, ExpertStacks) {
    const int N = 128;
    auto env = make_env_cpu(stack_opts(N));
    std::vector<float> a(N * 10);
    std::vector<int> ended(N, 0);
    Env::HostCopy h;
    env->read_reward_terms();
    int done = 0, steps_done = 0, slowest = 0, done_all = 0;  // first episodes; every done episode
    double ret = 0;
    for (int t = 0; t < T::kMaxEpisodeSteps; ++t) {
        env->expert_actions(a.data());
        env->step(a.data());
        env->download(h);
        for (int e = 0; e < N; ++e) {
            done_all += h.terminated[e];
            if (ended[e] || !(h.terminated[e] || h.truncated[e])) continue;
            ended[e] = 1;
            if (h.terminated[e]) {
                ++done, steps_done += t + 1, slowest = std::max(slowest, t + 1);
                ret += env->last_return()[e];
            }
        }
    }
    const auto m = env->read_reward_terms();
    const auto& names = env->reward_term_names();
    ASSERT_EQ(names.size(), 11u);
    ASSERT_EQ(names[10], "progress");
    std::printf("openarm_stack expert: %d / %d towers built, hands home (mean %d steps, slowest %d); per done episode:",
                done, N, steps_done / std::max(1, done), slowest);
    std::vector<double> per(m.size());
    for (size_t k = 0; k < m.size(); ++k) {
        per[k] = m[k] * T::kMaxEpisodeSteps * N / std::max(1, done_all);
        std::printf(" %s %.2f", names[k].c_str(), per[k]);
    }
    std::printf("; return %.2f\n", ret / std::max(1, done));
    EXPECT_GE(done, N * 90 / 100);  // the misses: a cube that slid off the one below and landed out of reach
    EXPECT_NEAR(per[0], 20.0, 1e-3) << "success: +20 once per done episode";
    EXPECT_GT(per[1], 3.0) << "the expert stacks near the centers";
    EXPECT_GT(per[3], -0.1) << "the arms (almost) never overlap";
    double pen = 0;
    for (size_t k = 2; k < 10; ++k) EXPECT_LE(per[k], 0.0), pen += per[k];
    EXPECT_GT(pen, -12.0) << "penalties well below the success";
    // progress: phi(done) - phi(start) is 3 cubes x 10 plus both hands home (2 each,
    // within 2 cm: > 1.2 each) minus the start's (~0.3) per done episode, > 31.5. This
    // sum also has the restarted episodes' partial progress (they run on after a done:
    // per done episode it is more); ProgressFollowsStages checks single episodes.
    EXPECT_GT(per[10], 31.5);
}

// The progress term (potential-based): an episode's sum is phi(end) - phi(start).
// Holding still it is (about) 0 every step; the expert's climbs the stages; in a
// random policy's episodes it stays small (nothing to farm by moving around).
TEST(OpenArmStack, ProgressHoldStill) {
    const int N = 16;
    auto env = make_env_cpu(stack_opts(N, 3));
    std::vector<float> a(size_t(N) * 10, 0.0f);
    for (int e = 0; e < N; ++e) a[size_t(e) * 10 + 4] = a[size_t(e) * 10 + 9] = -1.0f;  // fingers open
    env->read_reward_terms();
    Env::HostCopy h;
    double worst = 0;
    for (int t = 0; t < 100; ++t) {
        env->step(a.data());
        const auto m = env->read_reward_terms();  // mean over the envs this step
        worst = std::fmax(worst, std::fabs(m[10]));
    }
    std::printf("openarm_stack progress, holding still: largest step mean %.2e\n", worst);
    EXPECT_LT(worst, 2e-3) << "no progress without moving";
}

// The expert's progress, one env at a time (the term's mean is then that env's): the
// running sum is phi(now) - phi(start). Before the k-th cube is placed (and let go) it
// stays at most 10 (k - 1) + 7 (the hold's most); the step it first reaches 10 k - 0.5
// (the start's phi is ~0.3), the obs shows that cube in its place; at the done step,
// all three and both hands home.
TEST(OpenArmStack, ProgressFollowsStages) {
    const T::Params p = T::make_params(make_env_cpu(stack_opts(1))->model());
    int full = 0;
    const int kRuns = 6;
    for (int run = 0; run < kRuns; ++run) {
        auto env = make_env_cpu(stack_opts(1, uint64_t(11 + run)));
        float a[10];
        Env::HostCopy h;
        env->read_reward_terms();
        double cum = 0, top = 0;
        int gained = 0;  // stages the sum has reached
        for (int t = 0; t < T::kMaxEpisodeSteps; ++t) {
            env->expert_actions(a);
            env->step(a);
            env->download(h);
            cum += env->read_reward_terms()[10];
            const bool end = h.terminated[0] || h.truncated[0];
            const float* o = end ? h.final_obs.data() : h.obs.data();
            const Vec3 r = at(o, kCube[0]), g = at(o, kCube[1]), b = at(o, kCube[2]);
            auto on = [&](Vec3 x, Vec3 u) {
                return std::hypot(x.x - u.x, x.y - u.y) < p.stack_xy && std::fabs(x.z - u.z - 0.04f) < p.stack_dz;
            };
            const bool rp = std::hypot(r.x - p.spot[0], r.y - p.spot[1]) < p.spot_xy && r.z < 0.025f;
            const int now = rp ? (on(g, r) ? (on(b, g) ? 3 : 2) : 1) : 0;
            while (gained < 3 && cum >= 10.0 * (gained + 1) - 0.5) {  // stage gained + 1 just reached
                ++gained;
                EXPECT_GE(now, gained) << "run " << run << " step " << t << ": the obs shows the cube placed";
                EXPECT_LE(top, 10.0 * (gained - 1) + 7.01) << "run " << run << ": before it, at most the hold's 7";
            }
            top = std::fmax(top, cum);
            if (end) {
                if (h.terminated[0]) {
                    ++full;
                    EXPECT_GT(cum, 31.5) << "run " << run << ": done, phi(done) - phi(start)";
                }
                std::printf("openarm_stack progress, expert run %d: %s at step %d, sum %.2f\n", run,
                            h.terminated[0] ? "done" : "time out", t + 1, cum);
                break;
            }
        }
    }
    EXPECT_GE(full, kRuns - 1);
}

// Random actions: per episode the progress stays small (it pays only for real stages:
// moving the hands about near the cubes earns at most the reach's 1, and loses it again).
TEST(OpenArmStack, ProgressRandomIsSmall) {
    const int N = 64;
    auto env = make_env_cpu(stack_opts(N, 7));
    std::vector<float> a(size_t(N) * 10);
    env->read_reward_terms();
    for (int t = 0; t < 200; ++t) {
        env->random_actions(a.data(), uint64_t(t));
        env->step(a.data());
    }
    const auto m = env->read_reward_terms();
    const double per_env = m[10] * 200.0;  // the sum over 200 steps, mean per env
    std::printf("openarm_stack progress, random actions: %.3f per env over 200 steps\n", per_env);
    EXPECT_LT(std::fabs(per_env), 1.0);
}

}  // namespace
