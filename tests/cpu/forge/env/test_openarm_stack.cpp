// Phase 11: openarm_stack (both arms stack 3 cubes: green on red, blue on green).
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

// Sizes; both hands start at home (mirror images), fingers open; red in the middle,
// green on the left arm's side, blue on the right arm's, apart, on the table.
TEST(OpenArmStack, ShapesAndStart) {
    const int N = 64;
    auto env = make_env_cpu(stack_opts(N));
    EXPECT_EQ(env->obs_dim(), 79);
    EXPECT_EQ(env->act_dim(), 10);
    EXPECT_EQ(env->model().nq, 18 + 21) << "2 x (7 arm + 2 fingers) + 3 free cubes";
    const T::Params p = T::make_params(env->model());
    Env::HostCopy h;
    env->download(h);
    for (int e = 0; e < N; ++e) {
        const float* o = h.obs.data() + size_t(e) * 79;
        const Vec3 l = at(o, kTcp[0]), r = at(o, kTcp[1]);
        EXPECT_LT(length(l - T::home_of(p, 0)), 0.03f) << "env " << e;
        EXPECT_LT(length(r - T::home_of(p, 1)), 0.03f) << "env " << e;
        EXPECT_NEAR(o[41], 0.5f, 0.02f);
        EXPECT_NEAR(o[47], 0.5f, 0.02f);
        const Vec3 red = at(o, kCube[0]), g = at(o, kCube[1]), b = at(o, kCube[2]);
        EXPECT_TRUE(red.x >= 0.37f && red.x <= 0.44f && std::fabs(red.y) <= 0.025f) << "env " << e;
        EXPECT_GE(g.y, 0.10f);
        EXPECT_LE(b.y, -0.10f);
        for (const Vec3& c : {red, g, b}) EXPECT_NEAR(c.z, 0.02f, 1e-4f);
        EXPECT_GE(std::fmax(std::fabs(g.x - red.x), std::fabs(g.y - red.y)), 0.12f - 1e-4f);
        EXPECT_GE(std::fmax(std::fabs(b.x - red.x), std::fabs(b.y - red.y)), 0.12f - 1e-4f);
    }
    const Vec3 hl = T::home_of(p, 0), hr = T::home_of(p, 1);
    std::printf("openarm_stack home TCPs: left %.3f %.3f %.3f, right %.3f %.3f %.3f\n", hl.x, hl.y, hl.z, hr.x, hr.y,
                hr.z);
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

// The scripted expert builds the tower and brings both hands home in (almost) every
// env, under the sparse reward: success once per done episode, small penalties.
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
    EXPECT_GE(done, N * 95 / 100);
    EXPECT_NEAR(per[0], 20.0, 1e-3) << "success: +20 once per done episode";
    EXPECT_GT(per[1], 3.0) << "the expert stacks near the centers";
    EXPECT_EQ(per[3], 0.0) << "the arms never overlap";
    double pen = 0;
    for (size_t k = 2; k < m.size(); ++k) EXPECT_LE(per[k], 0.0), pen += per[k];
    EXPECT_GT(pen, -8.0) << "small penalties next to the success";
}

}  // namespace
