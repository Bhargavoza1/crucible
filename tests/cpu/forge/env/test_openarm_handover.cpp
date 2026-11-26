// Phase 12: openarm_handover (the left hand picks a block, hands it to the right hand,
// the right hand puts it on the place spot).
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/env/env.h"
#include "forge/env/tasks/openarm_handover.h"

using namespace forge;

namespace {

using T = OpenArmHandoverTask;

EnvOptions handover_opts(int n, uint64_t seed = 1) {
    EnvOptions o;
    o.task = "openarm_handover";
    o.num_envs = n;
    o.seed = seed;
    o.backend = EnvBackend::kCpu;
    o.asset_dir = std::string(CRUCIBLE_SOURCE_DIR) + "/assets";
    o.physics_dt = 0.002f;  // configs/openarm_handover.txt
    o.decimation = 10;
    return o;
}
Vec3 at(const float* o, int i) { return {o[i], o[i + 1], o[i + 2]}; }
constexpr int kTcp[2] = {36, 42}, kBlock = 48;

// The stage of every env from a host copy of the state (T::stage reads qpos only).
std::vector<int> stages(const Env& env, const T::Params& p, const Env::HostCopy& h, int n) {
    std::vector<float> ctrl(size_t(n) * 18, 0.0f);  // not read (measured hands only)
    EnvView v;
    v.model = env.model().view();
    v.state.qpos = const_cast<float*>(h.qpos.data());
    v.ctrl = ctrl.data(), v.nu = 18;
    std::vector<int> s(static_cast<size_t>(n));
    for (int e = 0; e < n; ++e) {
        T::Fk f;
        forward_kinematics(v.model, v.qpos(e), f.w);
        s[size_t(e)] = T::stage(p, v, e, f);
    }
    return s;
}

// Sizes; both hands at home, fingers open; the block flat on the left half (y >= 0.17,
// out of the right arm's reach), spread over the left arm's area.
TEST(OpenArmHandover, ShapesAndStart) {
    const int N = 256;
    auto env = make_env_cpu(handover_opts(N));
    EXPECT_EQ(env->obs_dim(), 65);
    EXPECT_EQ(env->act_dim(), 10);
    EXPECT_EQ(env->model().nq, 18 + 7) << "2 x (7 arm + 2 fingers) + the free block";
    const T::Params p = T::make_params(env->model());
    Env::HostCopy h;
    env->download(h);
    float lo_x = 1, hi_x = 0, lo_y = 1, hi_y = -1;
    for (int e = 0; e < N; ++e) {
        const float* o = h.obs.data() + size_t(e) * 65;
        EXPECT_LT(length(at(o, kTcp[0]) - T::home_of(p, 0)), 0.03f) << "env " << e;
        EXPECT_LT(length(at(o, kTcp[1]) - T::home_of(p, 1)), 0.03f) << "env " << e;
        EXPECT_NEAR(o[41], 0.5f, 0.02f);
        EXPECT_NEAR(o[47], 0.5f, 0.02f);
        const Vec3 x = at(o, kBlock);
        EXPECT_NEAR(x.z, T::kHalfYZ, 1e-4f);
        EXPECT_GE(x.y, p.spawn_y - 1e-4f) << "env " << e;
        lo_x = std::fmin(lo_x, x.x), hi_x = std::fmax(hi_x, x.x), lo_y = std::fmin(lo_y, x.y), hi_y = std::fmax(hi_y, x.y);
    }
    std::printf("openarm_handover: block x %.2f-%.2f y %.2f-%.2f\n", lo_x, hi_x, lo_y, hi_y);
    EXPECT_GT(hi_x - lo_x, 0.08f) << "spread in x";
    EXPECT_GT(hi_y - lo_y, 0.12f) << "spread in y";
    const auto s = stages(*env, p, h, N);
    for (int e = 0; e < N; ++e) EXPECT_EQ(s[size_t(e)], int(T::kStart));
}

// The scripted expert hands the block over and places it, both hands home, in 90% or
// more of the envs; every stage is counted (the highest stage of each first episode).
TEST(OpenArmHandover, ExpertHandsOver) {
    const int N = 128;
    auto env = make_env_cpu(handover_opts(N));
    const T::Params p = T::make_params(env->model());
    std::vector<float> a(N * 10);
    std::vector<int> ended(N, 0), best(N, 0);
    Env::HostCopy h;
    env->read_reward_terms();
    int done = 0, steps_done = 0, slowest = 0, done_all = 0;
    double ret = 0;
    for (int t = 0; t < T::kMaxEpisodeSteps; ++t) {
        env->expert_actions(a.data());
        env->step(a.data());
        env->download(h);
        const auto s = stages(*env, p, h, N);
        for (int e = 0; e < N; ++e) {
            done_all += h.terminated[e];
            if (ended[e]) continue;
            if (!(h.terminated[e] || h.truncated[e])) {
                best[size_t(e)] = std::max(best[size_t(e)], s[size_t(e)]);
                continue;
            }
            ended[e] = 1;
            if (h.terminated[e]) {
                best[size_t(e)] = T::kDone;
                ++done, steps_done += t + 1, slowest = std::max(slowest, t + 1);
                ret += env->last_return()[e];
            }
        }
    }
    const auto m = env->read_reward_terms();
    const auto& names = env->reward_term_names();
    ASSERT_EQ(names.size(), 10u);
    ASSERT_EQ(names[9], "progress");
    const char* stage_name[T::kStages] = {"start", "left holds", "at M", "both hold", "handed", "placed", "done"};
    std::printf("openarm_handover expert: %d / %d done (mean %d steps, slowest %d); stages reached:", done, N,
                steps_done / std::max(1, done), slowest);
    for (int k = 1; k < T::kStages; ++k) {
        int c = 0;
        for (int e = 0; e < N; ++e) c += best[size_t(e)] >= k;
        std::printf(" %s %d", stage_name[k], c);
    }
    std::printf("\nper done episode:");
    for (size_t k = 0; k < m.size(); ++k)
        std::printf(" %s %.2f", names[k].c_str(), m[k] * T::kMaxEpisodeSteps * N / std::max(1, done_all));
    std::printf("; return %.2f\n", ret / std::max(1, done));
    EXPECT_GE(done, N * 9 / 10);
}

// Holding still earns no progress (the potential only moves with the state).
TEST(OpenArmHandover, ProgressHoldingStill) {
    const int N = 16;
    auto env = make_env_cpu(handover_opts(N));
    std::vector<float> a(N * 10, 0.0f);
    for (int e = 0; e < N; ++e) a[size_t(e) * 10 + 4] = a[size_t(e) * 10 + 9] = -1.0f;
    env->read_reward_terms();
    for (int t = 0; t < 100; ++t) env->step(a.data());
    const auto m = env->read_reward_terms();
    std::printf("openarm_handover progress, holding still: mean per step %.2e\n", m[9]);
    EXPECT_LT(std::fabs(m[9]), 1e-3);
}

}  // namespace
