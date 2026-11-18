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
    ASSERT_EQ(names.size(), 10u);
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
    for (size_t k = 2; k < m.size(); ++k) EXPECT_LE(per[k], 0.0), pen += per[k];
    EXPECT_GT(pen, -12.0) << "penalties well below the success";
}

}  // namespace
