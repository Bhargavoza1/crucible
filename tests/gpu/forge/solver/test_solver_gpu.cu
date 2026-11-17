// Phase 6b: the constraint solver on the GPU. Every test runs both kernels:
// one thread per env, and one warp per env (scratch in shared memory).
//   1. GPU vs real MuJoCo on the scene reference states (each state = one env)
//   2. GPU vs the CPU answer key on 1024 different envs (one call)
//   3. GPU vs CPU trajectories on 1024 envs
//
// Why GPU and CPU differ at all: nvcc fuses a*b+c into one FMA, g++ does not.
// The limits below are about 10x the difference between an FMA and a no-FMA CPU
// build of the same code on the same states (measured: qacc normwise 1.6e-5).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/solver/heightfield.h"

#include "forge/dynamics/robot_work.h"

#include "forge/env/tasks/g1_stand.h"

#include "forge/env/terrain.h"

#include "common/mujoco_setup.h"
#include "common/robot_states.h"
#include "forge/dynamics/robot_batch_host.h"
#include "forge/dynamics/robot_gpu.h"
#include "forge/robots/actuators.h"
#include "forge/robots/urdf.h"
#include "forge/solver/constraint_batch_host.h"
#include "forge/solver/solver_gpu.h"

using namespace forge;

namespace {

// ||a_e - b_e|| / max(1, ||b_e||) for env e; blocks of 'len' numbers.
double normwise_env(const std::vector<float>& a, const std::vector<float>& b, int len, int e) {
    double d = 0, s = 0;
    for (int i = 0; i < len; ++i) {
        const double x = a[size_t(e) * len + i], y = b[size_t(e) * len + i];
        d += (x - y) * (x - y);
        s += y * y;
    }
    return std::sqrt(d) / std::max(1.0, std::sqrt(s));
}

ConstraintModel make_constraints(const RobotModel& m, bool floor) {
    ConstraintModel c(m);
    if (floor) c.add_ground();
    return c;
}

// Everything for one robot (optional floor and PD actuators), on the GPU and CPU.
struct Rig {
    RobotModel m;
    ConstraintModel cm;
    Actuators act;
    RobotModelDevice dm;
    ConstraintModelDevice dcm;
    ActuatorsDevice dact;

    explicit Rig(RobotModel model, bool floor, bool actuators)
        : m(std::move(model)), cm(make_constraints(m, floor)),
          act(actuators ? Actuators::position_all(m, 80, 2, -40, 40) : Actuators{}), dm(m), dcm(cm), dact(act) {}
};

RobotModel load(const std::string& name, bool floating, bool capsules = true) {
    UrdfOptions u;
    u.floating_base = floating;
    u.capsules = capsules;
    return load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/" + name + ".urdf", u);
}

const char* kernel_name(GpuSolverKernel k) { return k == GpuSolverKernel::kWarpPerEnv ? "warp" : "thread"; }

// ---- 1. GPU vs MuJoCo ------------------------------------------------------------
void check_scene_vs_mujoco(const std::string& name, GpuSolverKernel kernel) {
    testutil::Setup s = testutil::load_mujoco_setup("scene_" + name);
    ConstraintModel cm(s.model);
    if (s.ref.floor) cm.add_ground();
    Actuators act;
    for (const auto& j : s.ref.act_joints)
        act.add_position(s.model, j, float(s.ref.act_kp), float(s.ref.act_kv), -float(s.ref.act_force),
                         float(s.ref.act_force));
    const int n = static_cast<int>(s.ref.cases.size()), nv = s.model.nv;
    RobotBatchHost h(s.model, n);
    ConstraintBatchHost ch(cm, act.nu(), n);
    for (int e = 0; e < n; ++e) {
        const auto& rc = s.ref.cases[e];
        const auto q = testutil::to_ours(rc.qpos, s.q_mj), v = testutil::to_ours(rc.qvel, s.d_mj);
        std::copy(q.begin(), q.end(), h.env_qpos(e));
        std::copy(v.begin(), v.end(), h.env_qvel(e));
        for (int k = 0; k < act.nu(); ++k) ch.ctrl[size_t(e) * act.nu() + k] = float(rc.ctrl[k]);
    }
    RobotModelDevice dm(s.model);
    ConstraintModelDevice dcm(cm);
    ActuatorsDevice dact(act);
    RobotBatchDevice d(s.model, n);
    ConstraintBatchDevice dc(cm, act.nu(), n);
    d.upload_state(h);
    dc.upload_ctrl(ch);
    robot_dynamics_constrained_gpu(dm.view(), dcm.view(), dact.view(), d.view(), d.work(), dc.view(), nullptr, kernel);
    CUDA_CHECK(cudaDeviceSynchronize());
    d.download_state(h);
    dc.download_all(ch);

    double worst = 0;
    for (int e = 0; e < n; ++e) {
        const auto& rc = s.ref.cases[e];
        ASSERT_EQ(h.status[e], 0) << name << " env " << e;
        ASSERT_EQ(ch.nefc[e], rc.nefc) << name << " env " << e << ": number of constraint rows";
        double dd = 0, ss = 0;
        for (int i = 0; i < nv; ++i) {
            const double x = h.qacc[size_t(e) * nv + i], y = rc.qacc_constrained[s.d_mj[i]];
            dd += (x - y) * (x - y);
            ss += y * y;
        }
        const double err = std::sqrt(dd) / std::max(1.0, std::sqrt(ss));
        worst = std::max(worst, err);
        EXPECT_LE(err, 2e-4) << name << " env " << e << " qacc";
    }
    std::printf("GPU (%s kernel) %s vs MuJoCo, %d envs: worst qacc normwise %.2e\n", kernel == GpuSolverKernel::kWarpPerEnv ? "warp" : "thread", name.c_str(), n, worst);
}



class SolverGpu : public ::testing::TestWithParam<GpuSolverKernel> {};

}  // namespace

TEST_P(SolverGpu, BallMatchesMujoco) { check_scene_vs_mujoco("ball", GetParam()); }
TEST_P(SolverGpu, QuadrupedMatchesMujoco) { check_scene_vs_mujoco("quadruped", GetParam()); }
TEST_P(SolverGpu, Arm3LimitsMatchMujoco) { check_scene_vs_mujoco("arm3", GetParam()); }
TEST_P(SolverGpu, G1MatchesMujoco) { check_scene_vs_mujoco("g1_12dof", GetParam()); }  // Phase 8
TEST_P(SolverGpu, Go2MatchesMujoco) { check_scene_vs_mujoco("go2", GetParam()); }       // Phase 8e
// Phase 9a: box contacts (a free cube next to a ball; the Panda's finger pads on a cube)
TEST_P(SolverGpu, BallCubeMatchesMujoco) { check_scene_vs_mujoco("ball_cube", GetParam()); }
TEST_P(SolverGpu, PandaCubeMatchesMujoco) { check_scene_vs_mujoco("panda_cube", GetParam()); }
// Phase 10: OpenArm's capsules vs the floor and vs a cube
TEST_P(SolverGpu, OpenArmMatchesMujoco) { check_scene_vs_mujoco("openarm", GetParam()); }
TEST_P(SolverGpu, OpenArmCubeMatchesMujoco) { check_scene_vs_mujoco("openarm_cube", GetParam()); }
// Phase 10: G1 and Go2 lying on the floor: body and leg capsules touch it
TEST_P(SolverGpu, G1FallenMatchesMujoco) { check_scene_vs_mujoco("g1_fallen", GetParam()); }
TEST_P(SolverGpu, Go2FallenMatchesMujoco) { check_scene_vs_mujoco("go2_fallen", GetParam()); }

// Phase 8d: the G1 standing on the terrain heightfield (feet 2 mm into the ground at
// 64 spots: smooth, rough, hills, valleys), one forward dynamics: GPU vs CPU.
// Spots go out to 20 m from the origin. There float32 holds a position only to
// ~2 um, and on a slope that enters the contact distance (stiffness ~2770), so FMA
// (GPU) vs no FMA (CPU) rounding differs more far out. Two CPU builds, with and
// without -mfma, on these 64 states: worst 4.6e-4 within 6 m, 2.6e-3 beyond 12 m.
// So the limit is 1e-3 within 5 m and grows with distance (4.1e-3 at 20.4 m).
TEST_P(SolverGpu, G1OnTerrainMatchesCpu) {
    UrdfOptions u;
    u.floating_base = true;
    G1StandTask::urdf_options(u);
    const RobotModel m = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/g1/g1_12dof.urdf", u);
    ConstraintModel cm(m);
    const Terrain t = make_terrain();
    cm.set_heightfield(t.nx, t.ny, t.cell, t.x0, t.y0, t.h);
    const Actuators act = G1StandTask::make_actuators(m);
    const auto p = G1StandTask::make_params(m);
    const HeightfieldView hv{t.nx, t.ny, t.cell, t.x0, t.y0, t.h.data()};
    const int n = 64, nv = m.nv;
    RobotBatchHost h(m, n);
    ConstraintBatchHost ch(cm, act.nu(), n);
    RobotWork work(m);
    DynamicsWork w = work.view();
    for (int e = 0; e < n; ++e) {
        float* q = h.env_qpos(e);
        for (int i = 0; i < m.nq; ++i) q[i] = 0.0f;
        q[p.base_q + 0] = -14.0f + 28.0f * float(e % 8) / 7.0f + 0.013f * e;
        q[p.base_q + 1] = -14.0f + 28.0f * float(e / 8) / 7.0f;
        q[p.base_q + 3] = 1.0f;
        for (int k = 0; k < 12; ++k) {
            q[p.q[k]] = p.q0[k] + 0.04f * float((e + k) % 5 - 2);
            ch.ctrl[size_t(e) * act.nu() + k] = p.q0[k];
            h.env_qvel(e)[p.d[k]] = 0.1f * float((e + 2 * k) % 7 - 3);
        }
        forward_kinematics(m.view(), q, w);
        float gap = 1e9f;
        for (const auto& sp : m.spheres) {
            const Vec3 c = load3(w.xpos + 3 * sp.body) + rotate(load_quat(w.xquat + 4 * sp.body), sp.pos);
            gap = std::min(gap, c.z - sp.radius - hfield_height(hv, c.x, c.y));
        }
        q[p.base_q + 2] -= gap + 0.002f;  // lowest foot 2 mm into the ground
        h.env_qvel(e)[p.base_d] = 0.1f;
    }
    RobotBatchHost hc = h;
    ConstraintBatchHost cc = ch;
    robot_dynamics_constrained_cpu(m.view(), cm.view(), act.view(), hc.view(), hc.work(), cc.view());
    RobotModelDevice dm(m);
    ConstraintModelDevice dcm(cm);
    ActuatorsDevice dact(act);
    RobotBatchDevice d(m, n);
    ConstraintBatchDevice dc(cm, act.nu(), n);
    d.upload_state(h);
    dc.upload_ctrl(ch);
    robot_dynamics_constrained_gpu(dm.view(), dcm.view(), dact.view(), d.view(), d.work(), dc.view(), nullptr,
                                   GetParam());
    CUDA_CHECK(cudaDeviceSynchronize());
    d.download_state(h);
    dc.download_all(ch);
    double worst = 0;
    int rows = 0;
    for (int e = 0; e < n; ++e) {
        ASSERT_EQ(h.status[e], 0) << "env " << e;
        EXPECT_EQ(ch.nefc[e], cc.nefc[e]) << "env " << e << ": constraint rows GPU vs CPU";
        rows += cc.nefc[e];
        double dd = 0, ss = 0;
        for (int i = 0; i < nv; ++i) {
            const double x = h.qacc[size_t(e) * nv + i], y = hc.qacc[size_t(e) * nv + i];
            dd += (x - y) * (x - y);
            ss += y * y;
        }
        const double err = std::sqrt(dd) / std::max(1.0, std::sqrt(ss));
        worst = std::max(worst, err);
        const float* q = h.env_qpos(e);
        const double r = std::hypot(double(q[p.base_q]), double(q[p.base_q + 1]));
        EXPECT_LE(err, 1e-3 * std::max(1.0, r / 5.0)) << "env " << e << " (" << r << " m from origin) qacc GPU vs CPU";
    }
    std::printf("GPU (%s kernel) G1 on terrain vs CPU, %d envs, %.1f rows per env: worst qacc normwise %.2e\n",
                kernel_name(GetParam()), n, double(rows) / n, worst);
}

// ---- 2. GPU vs CPU, one call, 1024 envs ---------------------------------------------
TEST_P(SolverGpu, DynamicsMatchCpuManyEnvs) {
    Rig r(load("quadruped", true), true, true);
    const int N = 1024, nv = r.m.nv;
    RobotBatchHost cpu(r.m, N);
    ConstraintBatchHost ccpu(r.cm, r.act.nu(), N);
    testutil::randomize_on_floor(r.m, cpu, ccpu.ctrl, r.act.nu(), 0.5f, 11);
    RobotBatchHost gpu = cpu;
    ConstraintBatchHost cgpu = ccpu;
    RobotBatchDevice d(r.m, N);
    ConstraintBatchDevice dc(r.cm, r.act.nu(), N);
    d.upload_state(gpu);
    dc.upload_ctrl(cgpu);
    robot_dynamics_constrained_gpu(r.dm.view(), r.dcm.view(), r.dact.view(), d.view(), d.work(), dc.view(), nullptr,
                                   GetParam());
    CUDA_CHECK(cudaDeviceSynchronize());
    d.download_state(gpu);
    dc.download_all(cgpu);
    robot_dynamics_constrained_cpu(r.m.view(), r.cm.view(), r.act.view(), cpu.view(), cpu.work(), ccpu.view());

    // A contact exists when distance <= 0. A sphere at distance ~1e-8 can be in on
    // one side and out on the other; allow a few such envs, skip them below.
    int nefc_diff = 0, rows = 0, max_iter = 0;
    double worst = 0;
    for (int e = 0; e < N; ++e) {
        ASSERT_EQ(gpu.status[e], 0) << "env " << e;
        rows += ccpu.nefc[e];
        max_iter = std::max(max_iter, cgpu.iterations[e]);
        if (cgpu.nefc[e] != ccpu.nefc[e]) {
            ++nefc_diff;
            continue;
        }
        const double err = normwise_env(gpu.qacc, cpu.qacc, nv, e);
        worst = std::max(worst, err);
        EXPECT_LE(err, 2e-4) << "env " << e << " qacc";
    }
    std::printf("[%s] GPU vs CPU, %d envs (%.1f rows per env): worst qacc normwise %.2e, %d envs with a different "
                "number of rows, Newton iterations <= %d\n",
                kernel_name(GetParam()), N, double(rows) / N, worst, nefc_diff, max_iter);
    EXPECT_LE(nefc_diff, 2);
    EXPECT_GT(rows, 4 * N) << "the test states should touch the floor";
}

// ---- 3. GPU vs CPU trajectories, 1024 envs x 100 steps (0.2 s) -----------------------
// A foot that touches down one step earlier on one side gets a different push,
// and from there that env's motion is different (the CPU build with FMA shows
// the same: 1 of 1024 envs after 150 steps). That is how contact works, not a
// bug. So: every env must stay finite with no errors, and 99% of envs must agree.
TEST_P(SolverGpu, TrajectoriesMatchCpu) {
    Rig r(load("quadruped", true), true, true);
    const int N = 1024, steps = 100;
    RobotBatchHost cpu(r.m, N);
    ConstraintBatchHost ccpu(r.cm, r.act.nu(), N);
    testutil::randomize_on_floor(r.m, cpu, ccpu.ctrl, r.act.nu(), 0.5f, 7);
    RobotBatchHost gpu = cpu;
    ConstraintBatchHost cgpu = ccpu;
    RobotBatchDevice d(r.m, N);
    ConstraintBatchDevice dc(r.cm, r.act.nu(), N);
    d.upload_state(gpu);
    dc.upload_ctrl(cgpu);
    robot_step_constrained_gpu(r.dm.view(), r.dcm.view(), r.dact.view(), d.view(), d.work(), dc.view(), 0.002f, steps,
                               nullptr, GetParam());
    CUDA_CHECK(cudaDeviceSynchronize());
    d.download_state(gpu);
    robot_step_constrained_cpu(r.m.view(), r.cm.view(), r.act.view(), cpu.view(), cpu.work(), ccpu.view(), 0.002f,
                               steps);
    const int nq = r.m.nq;
    int agree = 0;
    std::vector<double> err(N);
    for (int e = 0; e < N; ++e) {
        ASSERT_EQ(gpu.status[e], 0) << "env " << e;
        double w = 0;
        for (int i = 0; i < nq; ++i) {
            ASSERT_TRUE(std::isfinite(gpu.env_qpos(e)[i])) << "env " << e;
            w = std::max(w, double(std::fabs(gpu.env_qpos(e)[i] - cpu.env_qpos(e)[i])));
        }
        err[e] = w;
        agree += w <= 1e-4;
    }
    std::sort(err.begin(), err.end());
    std::printf("[%s] %d envs x %d steps: qpos GPU vs CPU median %.2e, 99th percentile %.2e, max %.2e; %d envs within 1e-4\n",
                kernel_name(GetParam()), N, steps, err[N / 2], err[N * 99 / 100], err.back(), agree);
    EXPECT_GE(agree, N * 99 / 100);
}

// A broken env (NaN velocity) must not change any other env.
TEST_P(SolverGpu, EnvsAreIndependent) {
    Rig r(load("quadruped", true), true, true);
    const int N = 64;
    RobotBatchHost a(r.m, N);
    ConstraintBatchHost ca(r.cm, r.act.nu(), N);
    testutil::randomize_on_floor(r.m, a, ca.ctrl, r.act.nu(), 0.5f, 5);
    RobotBatchHost b = a;
    b.env_qvel(17)[2] = std::nanf("");
    RobotBatchDevice da(r.m, N), db(r.m, N);
    ConstraintBatchDevice dca(r.cm, r.act.nu(), N), dcb(r.cm, r.act.nu(), N);
    da.upload_state(a);
    db.upload_state(b);
    dca.upload_ctrl(ca);
    dcb.upload_ctrl(ca);
    robot_step_constrained_gpu(r.dm.view(), r.dcm.view(), r.dact.view(), da.view(), da.work(), dca.view(), 0.002f, 50,
                               nullptr, GetParam());
    robot_step_constrained_gpu(r.dm.view(), r.dcm.view(), r.dact.view(), db.view(), db.work(), dcb.view(), 0.002f, 50,
                               nullptr, GetParam());
    CUDA_CHECK(cudaDeviceSynchronize());
    da.download_state(a);
    db.download_state(b);
    for (int e = 0; e < N; ++e) {
        if (e == 17) continue;
        for (int i = 0; i < r.m.nq; ++i) ASSERT_EQ(a.env_qpos(e)[i], b.env_qpos(e)[i]) << "env " << e;
    }
}

INSTANTIATE_TEST_SUITE_P(Kernels, SolverGpu,
                         ::testing::Values(GpuSolverKernel::kThreadPerEnv, GpuSolverKernel::kWarpPerEnv),
                         [](const ::testing::TestParamInfo<GpuSolverKernel>& p) { return kernel_name(p.param); });

// The quadruped's solver scratch must fit in shared memory, or the warp tests
// above silently test the thread kernel twice.
TEST(SolverGpuSetup, QuadrupedFitsWarpKernel) {
    const RobotModel m = load("quadruped", true);
    ConstraintModel cm(m);
    cm.add_ground();
    EXPECT_GT(warp_solver_shared_bytes(m.nv, cm.max_rows()), 0u);
}

// G1 legs: nv 18; rows reserved: 2 per joint limit (one per side) x 12 + 8 foot
// spheres x 4 = 56, ~10.7 KB per env, 42.7 KB per block of 4 envs (limit 48 KB).
// Loaded as the training tasks load it: feet only, no capsules (Phase 10: the 23
// body capsules reserve 2 x 4 rows each on the floor, 240 rows in all, too many
// for the warp kernel: scenes with them use the thread kernel).
TEST(SolverGpuSetup, G1FitsWarpKernel) {
    const RobotModel m = load("g1/g1_12dof", true, false);
    ConstraintModel cm(m);
    cm.add_ground();
    EXPECT_EQ(cm.max_rows(), 56);
    EXPECT_GT(warp_solver_shared_bytes(m.nv, cm.max_rows()), 0u);
}

// Panda + free cube: nv 15; rows: 2 x 9 joint limits + 4 x (6 pad spheres on the
// floor + 6 on the cube + 4 cube corners on the floor) = 82, 11.3 KB per env, 45.3 KB
// per block of 4 envs (limit 48 KB; 4 pad spheres per finger would need 51.7 KB).
TEST(SolverGpuSetup, PandaCubeFitsWarpKernel) {
    RobotModel m = load("panda/panda", false);
    add_free_box(m, "cube", {0.025f, 0.025f, 0.025f}, 0.1f);
    ConstraintModel cm(m);
    cm.add_ground();
    EXPECT_EQ(cm.max_rows(), 82);
    EXPECT_GT(warp_solver_shared_bytes(m.nv, cm.max_rows()), 0u);
}

// Go2: nv 18; rows: 2 per joint limit x 12 + 4 foot spheres x 4 = 40 (feet only,
// as the training tasks load it; see G1FitsWarpKernel).
TEST(SolverGpuSetup, Go2FitsWarpKernel) {
    const RobotModel m = load("go2/go2", true, false);
    ConstraintModel cm(m);
    cm.add_ground();
    EXPECT_EQ(cm.max_rows(), 40);
    EXPECT_GT(warp_solver_shared_bytes(m.nv, cm.max_rows()), 0u);
}
