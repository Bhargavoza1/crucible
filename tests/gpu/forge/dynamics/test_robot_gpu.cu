// GPU robot dynamics: vs real MuJoCo (the Phase 5 done check, on the GPU) and
// vs the CPU answer key on many envs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/mujoco_setup.h"
#include "common/robot_states.h"
#include "forge/dynamics/robot_batch_host.h"
#include "forge/dynamics/robot_gpu.h"
#include "forge/robots/urdf.h"

using namespace forge;

namespace {

RobotModel load(const std::string& name, bool floating) {
    return load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/" + name + ".urdf", {floating});
}

double max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs(double(a[i]) - b[i]) / std::max(1.0, std::fabs(double(b[i]))));
    return m;
}

// Each MuJoCo reference state becomes one env; one GPU launch does all of them.
void check_gpu_vs_mujoco(const std::string& robot) {
    testutil::Setup s = testutil::load_mujoco_setup(robot);
    const RobotModel& model = s.model;
    const int n = static_cast<int>(s.ref.cases.size());
    RobotBatchHost h(model, n);
    for (int e = 0; e < n; ++e) {
        const auto q = testutil::to_ours(s.ref.cases[e].qpos, s.q_mj);
        const auto v = testutil::to_ours(s.ref.cases[e].qvel, s.d_mj);
        const auto t = testutil::to_ours(s.ref.cases[e].tau, s.d_mj);
        std::copy(q.begin(), q.end(), h.env_qpos(e));
        std::copy(v.begin(), v.end(), h.env_qvel(e));
        std::copy(t.begin(), t.end(), h.tau.begin() + size_t(e) * model.nv);
    }
    RobotModelDevice dm(model);
    RobotBatchDevice d(model, n);
    d.upload_state(h);
    robot_dynamics_gpu(dm.view(), d.view(), d.work());
    CUDA_CHECK(cudaDeviceSynchronize());
    d.download_all(h);

    const int nv = model.nv, nb = model.nbody();
    double worst_M = 0, worst_bias = 0, worst_pos = 0;
    for (int e = 0; e < n; ++e) {
        const auto& rc = s.ref.cases[e];
        ASSERT_EQ(h.status[e], 0);
        for (int b = 0; b < nb; ++b)
            for (int a = 0; a < 3; ++a) {
                const double ours = h.xpos[size_t(e) * 3 * nb + 3 * b + a], mj = rc.xpos[3 * s.b_mj[b] + a];
                worst_pos = std::max(worst_pos, std::fabs(ours - mj));
                EXPECT_TRUE(testutil::close(ours, mj, testutil::kMujocoTol)) << "env " << e << " body " << b;
            }
        for (int i = 0; i < nv; ++i) {
            for (int j = 0; j < nv; ++j) {
                const double ours = h.M[size_t(e) * nv * nv + i * nv + j], mj = rc.M[s.d_mj[i] * nv + s.d_mj[j]];
                worst_M = std::max(worst_M, std::fabs(ours - mj) / std::max(1.0, std::fabs(mj)));
                EXPECT_TRUE(testutil::close(ours, mj, testutil::kMujocoTol)) << "env " << e << " M " << i << "," << j;
            }
            const double b_ours = h.bias[size_t(e) * nv + i], b_mj = rc.bias[s.d_mj[i]];
            worst_bias = std::max(worst_bias, std::fabs(b_ours - b_mj) / std::max(1.0, std::fabs(b_mj)));
            EXPECT_TRUE(testutil::close(b_ours, b_mj, testutil::kMujocoTol)) << "env " << e << " bias " << i;
            EXPECT_TRUE(testutil::close(h.qacc[size_t(e) * nv + i], rc.qacc[s.d_mj[i]], 1e-3)) << "env " << e << " qacc " << i;
        }
    }
    std::printf("GPU %s vs MuJoCo, %d envs: worst xpos %.2g m, M %.2g, bias %.2g\n", robot.c_str(), n, worst_pos,
                worst_M, worst_bias);
}

}  // namespace

TEST(RobotGpu, PendulumMatchesMujoco) { check_gpu_vs_mujoco("pendulum"); }
TEST(RobotGpu, Arm3MatchesMujoco) { check_gpu_vs_mujoco("arm3"); }
TEST(RobotGpu, QuadrupedMatchesMujoco) { check_gpu_vs_mujoco("quadruped"); }

// Largest per-env normwise difference: ||a_e - b_e|| / max(1, ||b_e||), where
// each env's block has 'len' numbers.
static double max_normwise(const std::vector<float>& a, const std::vector<float>& b, int len) {
    double worst = 0;
    for (size_t e = 0; e * len < b.size(); ++e) {
        double d = 0, s = 0;
        for (int i = 0; i < len; ++i) {
            const double x = a[e * len + i], y = b[e * len + i];
            d += (x - y) * (x - y);
            s += y * y;
        }
        worst = std::max(worst, std::sqrt(d) / std::max(1.0, std::sqrt(s)));
    }
    return worst;
}

// 1024 envs, all different. GPU and CPU are both FP32 but round differently:
// nvcc fuses a*b+c into one FMA instruction, g++ does not (the particle code
// avoids this with explicit fmaf; the dynamics code does not). The difference
// is measured normwise per env. Elementwise it looks much bigger, because the
// bias is a sum of large terms that cancel, and qacc comes out of a solve that
// multiplies rounding by the condition number of M.
// Limits: 10x the difference between an FMA and a no-FMA CPU build of the same
// code on these same states (measured: M 9e-7, bias 7.5e-7, qacc 2.1e-5).
TEST(RobotGpu, DynamicsMatchCpuManyEnvs) {
    const RobotModel m = load("quadruped", true);
    RobotBatchHost cpu(m, 1024);
    testutil::randomize(m, cpu, 11);
    RobotBatchHost gpu = cpu;
    RobotModelDevice dm(m);
    RobotBatchDevice d(m, 1024);
    d.upload_state(gpu);
    robot_dynamics_gpu(dm.view(), d.view(), d.work());
    CUDA_CHECK(cudaDeviceSynchronize());
    d.download_all(gpu);
    robot_dynamics_cpu(m.view(), cpu.view(), cpu.work());
    const int nv = m.nv;
    const double dM = max_normwise(gpu.M, cpu.M, nv * nv), db = max_normwise(gpu.bias, cpu.bias, nv);
    const double da = max_normwise(gpu.qacc, cpu.qacc, nv);
    std::printf("GPU vs CPU, 1024 envs, normwise per env: M %.2g, bias %.2g, qacc %.2g\n", dM, db, da);
    std::printf("  (elementwise, for reference: M %.2g, bias %.2g, qacc %.2g)\n", max_rel(gpu.M, cpu.M),
                max_rel(gpu.bias, cpu.bias), max_rel(gpu.qacc, cpu.qacc));
    EXPECT_LT(dM, 1e-5);
    EXPECT_LT(db, 1e-5);
    EXPECT_LT(da, 2e-4);
    EXPECT_LT(max_rel(gpu.xpos, cpu.xpos), 1e-6);
}

// 0.5 s of motion for 256 envs of each robot. Rounding differences grow a little
// over time, so the limit is looser than for one state.
TEST(RobotGpu, TrajectoriesMatchCpu) {
    const std::vector<std::pair<const char*, bool>> robots = {{"arm3", false}, {"quadruped", true}};
    for (const auto& [name, floating] : robots) {
        const RobotModel m = load(name, floating);
        RobotBatchHost cpu(m, 256);
        testutil::randomize(m, cpu, 3);
        for (auto& t : cpu.tau) t = 0.0f;
        RobotBatchHost gpu = cpu;
        RobotModelDevice dm(m);
        RobotBatchDevice d(m, 256);
        d.upload_state(gpu);
        robot_step_gpu(dm.view(), d.view(), d.work(), 0.002f, 250);
        CUDA_CHECK(cudaDeviceSynchronize());
        d.download_state(gpu);
        robot_step_cpu(m.view(), cpu.view(), cpu.work(), 0.002f, 250);
        double worst = 0;
        for (size_t i = 0; i < cpu.qpos.size(); ++i) worst = std::max(worst, double(std::fabs(gpu.qpos[i] - cpu.qpos[i])));
        std::printf("%s: 256 envs x 250 steps, worst qpos GPU vs CPU %.2g\n", name, worst);
        EXPECT_LT(worst, 1e-4) << name;
        for (int s : gpu.status) EXPECT_EQ(s, 0);
    }
}

// One env with a NaN velocity must not change any other env.
TEST(RobotGpu, EnvsAreIndependent) {
    const RobotModel m = load("arm3", false);
    RobotBatchHost a(m, 64);
    testutil::randomize(m, a, 5);
    RobotBatchHost b = a;
    b.env_qvel(17)[1] = std::nanf("");
    RobotModelDevice dm(m);
    RobotBatchDevice da(m, 64), db(m, 64);
    da.upload_state(a);
    db.upload_state(b);
    robot_step_gpu(dm.view(), da.view(), da.work(), 0.002f, 50);
    robot_step_gpu(dm.view(), db.view(), db.work(), 0.002f, 50);
    CUDA_CHECK(cudaDeviceSynchronize());
    da.download_state(a);
    db.download_state(b);
    for (int e = 0; e < 64; ++e) {
        if (e == 17) continue;
        for (int i = 0; i < m.nq; ++i) ASSERT_EQ(a.env_qpos(e)[i], b.env_qpos(e)[i]) << "env " << e;
    }
}

TEST(RobotGpu, BrokenModelReportsStatus) {
    RobotModel m = load("quadruped", true);
    m.mass[m.body_index("fl_shin")] = 0.0f;
    for (int k = 0; k < 9; ++k) m.inertia[9 * m.body_index("fl_shin") + k] = 0.0f;
    RobotBatchHost h(m, 4);
    RobotModelDevice dm(m);
    RobotBatchDevice d(m, 4);
    d.upload_state(h);
    robot_step_gpu(dm.view(), d.view(), d.work(), 0.002f, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    d.download_state(h);
    for (int s : h.status) EXPECT_EQ(s, 1);
    for (float a : h.qacc) EXPECT_EQ(a, 0.0f);
}
