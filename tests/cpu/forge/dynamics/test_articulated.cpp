// Phase 5 done check: our FK, mass matrix, bias forces and forward dynamics vs
// real MuJoCo on the same URDF robots (doc 9: "match within 1e-4").
// The numbers come from tests/data/mujoco_*.txt (tools/gen_mujoco_refs.py).
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/mujoco_ref.h"
#include "forge/dynamics/articulated.h"
#include "forge/dynamics/robot_work.h"
#include "forge/robots/urdf.h"

using namespace forge;
using testutil::MujocoRef;

namespace {

constexpr double kTol = 1e-4;  // doc: positions, mass matrix and bias forces within 1e-4

// |a - b| <= tol * max(1, |b|): absolute for small numbers, relative for big ones.
bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }

// Our model + the index maps between our qpos/qvel and MuJoCo's (by joint name).
struct Setup {
    MujocoRef ref;
    RobotModel model;
    std::vector<int> q_mj;  // our qpos index -> MuJoCo qpos index
    std::vector<int> d_mj;  // our dof index -> MuJoCo dof index
    std::vector<int> b_mj;  // our body index -> MuJoCo body index (without the world)
};

Setup load(const std::string& robot) {
    Setup s;
    s.ref = testutil::load_mujoco_ref(std::string(CRUCIBLE_SOURCE_DIR) + "/tests/data/mujoco_" + robot + ".txt");
    UrdfOptions opt;
    opt.floating_base = s.ref.floating;
    s.model = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/" + s.ref.urdf, opt);
    EXPECT_EQ(s.model.nq, s.ref.nq);
    EXPECT_EQ(s.model.nv, s.ref.nv);
    s.q_mj.assign(s.model.nq, -1);
    s.d_mj.assign(s.model.nv, -1);
    for (const auto& j : s.ref.joints) {
        const int b = s.model.joint_index(j.name);
        EXPECT_GE(b, 0) << "joint " << j.name << " missing in our model";
        if (b < 0) continue;
        for (int k = 0; k < joint_nq(s.model.jtype[b]); ++k) s.q_mj[s.model.qadr[b] + k] = j.qadr + k;
        for (int k = 0; k < joint_nv(s.model.jtype[b]); ++k) s.d_mj[s.model.dadr[b] + k] = j.dadr + k;
    }
    for (int b = 0; b < s.model.nbody(); ++b) {
        int found = -1;
        for (size_t k = 0; k < s.ref.bodies.size(); ++k)
            if (s.ref.bodies[k] == s.model.body_name[b]) found = static_cast<int>(k);
        EXPECT_GE(found, 0) << "body " << s.model.body_name[b] << " missing in MuJoCo";
        s.b_mj.push_back(found);
    }
    return s;
}

std::vector<float> to_ours(const std::vector<double>& mj, const std::vector<int>& map) {
    std::vector<float> v(map.size());
    for (size_t i = 0; i < map.size(); ++i) v[i] = static_cast<float>(mj[map[i]]);
    return v;
}

// Runs every reference case and checks one quantity with 'check'.
void check_all_cases(const std::string& robot) {
    Setup s = load(robot);
    const ModelView m = s.model.view();
    RobotWork work(s.model);
    DynamicsWork w = work.view();
    const int nv = s.model.nv;
    double worst_pos = 0, worst_M = 0, worst_bias = 0, worst_qacc = 0;

    for (size_t c = 0; c < s.ref.cases.size(); ++c) {
        SCOPED_TRACE(robot + " case " + std::to_string(c));
        const auto& rc = s.ref.cases[c];
        const std::vector<float> qpos = to_ours(rc.qpos, s.q_mj);
        const std::vector<float> qvel = to_ours(rc.qvel, s.d_mj);
        const std::vector<float> tau = to_ours(rc.tau, s.d_mj);
        std::vector<float> qacc(nv);
        ASSERT_TRUE(forward_dynamics(m, qpos.data(), qvel.data(), tau.data(), w, qacc.data()));

        for (int b = 0; b < s.model.nbody(); ++b) {
            const int k = s.b_mj[b];
            for (int a = 0; a < 3; ++a) {
                const double ours = work.xpos[3 * b + a], mj = rc.xpos[3 * k + a];
                worst_pos = std::max(worst_pos, std::fabs(ours - mj));
                EXPECT_TRUE(close(ours, mj, kTol)) << "xpos body " << s.model.body_name[b] << " axis " << a << ": " << ours << " vs " << mj;
            }
            // q and -q are the same rotation: compare |dot|.
            const double d = work.xquat[4 * b + 3] * rc.xquat[4 * k] + work.xquat[4 * b] * rc.xquat[4 * k + 1] +
                             work.xquat[4 * b + 1] * rc.xquat[4 * k + 2] + work.xquat[4 * b + 2] * rc.xquat[4 * k + 3];
            EXPECT_NEAR(std::fabs(d), 1.0, kTol) << "xquat body " << s.model.body_name[b];
        }
        for (int i = 0; i < nv; ++i) {
            for (int j = 0; j < nv; ++j) {
                const double ours = work.M[i * nv + j], mj = rc.M[s.d_mj[i] * nv + s.d_mj[j]];
                worst_M = std::max(worst_M, std::fabs(ours - mj) / std::max(1.0, std::fabs(mj)));
                EXPECT_TRUE(close(ours, mj, kTol)) << "M[" << i << "][" << j << "]: " << ours << " vs " << mj;
            }
            const double b_ours = work.bias[i], b_mj = rc.bias[s.d_mj[i]];
            worst_bias = std::max(worst_bias, std::fabs(b_ours - b_mj) / std::max(1.0, std::fabs(b_mj)));
            EXPECT_TRUE(close(b_ours, b_mj, kTol)) << "bias[" << i << "]: " << b_ours << " vs " << b_mj;
            const double a_mj = rc.qacc[s.d_mj[i]];
            worst_qacc = std::max(worst_qacc, std::fabs(qacc[i] - a_mj) / std::max(1.0, std::fabs(a_mj)));
            EXPECT_TRUE(close(qacc[i], a_mj, 1e-3)) << "qacc[" << i << "]: " << qacc[i] << " vs " << a_mj;
        }
    }
    std::printf("%s vs MuJoCo, %zu states: worst xpos %.2g m, M %.2g, bias %.2g, qacc %.2g (relative)\n",
                robot.c_str(), s.ref.cases.size(), worst_pos, worst_M, worst_bias, worst_qacc);
}

// Integrate with our Euler and compare with MuJoCo's trajectory.
void check_trajectory(const std::string& robot, double tol) {
    Setup s = load(robot);
    const ModelView m = s.model.view();
    RobotWork work(s.model);
    DynamicsWork w = work.view();
    std::vector<float> qpos = to_ours(s.ref.traj_qpos0, s.q_mj);
    std::vector<float> qvel = to_ours(s.ref.traj_qvel0, s.d_mj);
    std::vector<float> qacc(s.model.nv);
    const float dt = static_cast<float>(s.ref.traj_dt);
    int step = 0;
    double worst = 0;
    for (const auto& p : s.ref.traj) {
        for (; step < p.step; ++step) {
            ASSERT_TRUE(forward_dynamics(m, qpos.data(), qvel.data(), nullptr, w, qacc.data()));
            integrate_euler(m, qpos.data(), qvel.data(), qacc.data(), dt);
        }
        for (int i = 0; i < s.model.nq; ++i) {
            worst = std::max(worst, std::fabs(qpos[i] - p.qpos[s.q_mj[i]]));
            EXPECT_NEAR(qpos[i], p.qpos[s.q_mj[i]], tol) << robot << " step " << p.step << " qpos[" << i << "]";
        }
    }
    std::printf("%s trajectory, %d steps of %.3f s: worst qpos difference %.2g\n", robot.c_str(), step, dt, worst);
}

}  // namespace

TEST(ArticulatedVsMujoco, Pendulum) { check_all_cases("pendulum"); }
TEST(ArticulatedVsMujoco, Arm3) { check_all_cases("arm3"); }
TEST(ArticulatedVsMujoco, QuadrupedFloatingBase) { check_all_cases("quadruped"); }

// 0.5 s of free motion. FP32 (ours) vs FP64 (MuJoCo): small differences add up,
// so the tolerance is looser than for one state.
TEST(ArticulatedVsMujoco, PendulumTrajectory) { check_trajectory("pendulum", 1e-3); }
TEST(ArticulatedVsMujoco, Arm3Trajectory) { check_trajectory("arm3", 1e-3); }
TEST(ArticulatedVsMujoco, QuadrupedTrajectory) { check_trajectory("quadruped", 1e-3); }

// Physics facts that do not need MuJoCo.
TEST(Articulated, MassMatrixIsSymmetricPositiveDefinite) {
    RobotModel model = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/quadruped.urdf", {true});
    RobotWork work(model);
    DynamicsWork w = work.view();
    std::vector<float> qpos = model.qpos0(), qvel(model.nv, 0.0f), qacc(model.nv);
    for (int i = 7; i < model.nq; ++i) qpos[i] = 0.3f * float(i % 5) - 0.6f;
    ASSERT_TRUE(forward_dynamics(model.view(), qpos.data(), qvel.data(), nullptr, w, qacc.data()));
    const int nv = model.nv;
    for (int i = 0; i < nv; ++i)
        for (int j = 0; j < nv; ++j) EXPECT_EQ(work.M[i * nv + j], work.M[j * nv + i]);
    // Translation block of a free base = total mass * identity.
    float total = 0;
    for (float mb : model.mass) total += mb;
    for (int a = 0; a < 3; ++a) EXPECT_NEAR(work.M[a * nv + a], total, 1e-4f * total);
}

// A free robot at rest in zero gravity: no bias force at all.
// With gravity: the base bias is exactly the weight, pointing up (it is the force
// needed to hold the robot still).
TEST(Articulated, GravityBiasOfFreeRobotIsItsWeight) {
    RobotModel model = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/quadruped.urdf", {true});
    RobotWork work(model);
    DynamicsWork w = work.view();
    std::vector<float> qpos = model.qpos0(), qvel(model.nv, 0.0f), qacc(model.nv);
    forward_dynamics(model.view(), qpos.data(), qvel.data(), nullptr, w, qacc.data());
    float total = 0;
    for (float mb : model.mass) total += mb;
    EXPECT_NEAR(work.bias[2], total * 9.81f, 1e-3f);
    EXPECT_NEAR(qacc[2], -9.81f, 1e-4f);  // free fall
    model.gravity = {0, 0, 0};
    forward_dynamics(model.view(), qpos.data(), qvel.data(), nullptr, w, qacc.data());
    for (int i = 0; i < model.nv; ++i) EXPECT_NEAR(work.bias[i], 0.0f, 1e-6f);
}

// Free flight in zero gravity: linear and angular momentum are conserved.
// Kinetic energy too (Euler integration only drifts a little in 0.5 s).
TEST(Articulated, FreeFlightConservesEnergy) {
    RobotModel model = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/quadruped.urdf", {true});
    model.gravity = {0, 0, 0};
    const ModelView m = model.view();
    RobotWork work(model);
    DynamicsWork w = work.view();
    std::vector<float> qpos = model.qpos0(), qvel(model.nv), qacc(model.nv);
    for (int i = 0; i < model.nv; ++i) qvel[i] = 0.5f * std::sin(1.7f * i + 0.3f);
    auto energy = [&] {
        forward_dynamics(m, qpos.data(), qvel.data(), nullptr, w, qacc.data());
        double e = 0;
        for (int i = 0; i < model.nv; ++i)
            for (int j = 0; j < model.nv; ++j) e += 0.5 * qvel[i] * work.M[i * model.nv + j] * qvel[j];
        return e;
    };
    const double e0 = energy();
    for (int s = 0; s < 500; ++s) {
        forward_dynamics(m, qpos.data(), qvel.data(), nullptr, w, qacc.data());
        integrate_euler(m, qpos.data(), qvel.data(), qacc.data(), 0.001f);
    }
    EXPECT_NEAR(energy(), e0, 0.01 * e0);
}
