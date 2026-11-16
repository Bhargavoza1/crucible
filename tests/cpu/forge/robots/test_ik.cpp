// Phase 9a: inverse kinematics (forge/robots/ik.h).
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/dynamics/articulated.h"
#include "forge/dynamics/robot_work.h"
#include "forge/robots/ik.h"
#include "forge/robots/urdf.h"

using namespace forge;

namespace {

// The Panda's hand from Menagerie's home pose down to just above the floor, same
// orientation: the same answer as tools/gen_mujoco_refs.py (panda_ik, MuJoCo), and
// every joint inside its limits.
TEST(Ik, PandaHandDownToTheFloorAsTheReferenceGenerator) {
    const RobotModel m = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/panda/panda.urdf");
    const int hand = m.body_index("hand");
    ASSERT_GE(hand, 0);
    std::vector<int> arm;
    for (int k = 1; k <= 7; ++k) arm.push_back(m.joint_index("joint" + std::to_string(k)));
    std::vector<float> q(m.nq, 0.0f);
    const float home[7] = {0.0f, 0.0f, 0.0f, -1.57079f, 0.0f, 1.57079f, -0.7853f};
    for (int k = 0; k < 7; ++k) q[m.qadr[arm[k]]] = home[k];
    RobotWork work(m);
    DynamicsWork w = work.view();
    forward_kinematics(m.view(), q.data(), w);
    const Quat down = load_quat(w.xquat + 4 * hand);
    const IkResult r = solve_ik(m, hand, {0.5f, 0.0f, 0.025f + 0.0584f + 0.0445f}, down, arm, q);
    EXPECT_LT(r.pos_error, 1e-5f);
    EXPECT_LT(r.rot_error, 1e-5f);
    // tools/gen_mujoco_refs.py: arm q [0, 0.475, 0, -2.361, 0, 2.836, -0.785]
    const float ref[7] = {0.0f, 0.475f, 0.0f, -2.361f, 0.0f, 2.836f, -0.785f};
    for (int k = 0; k < 7; ++k) {
        const float x = q[m.qadr[arm[k]]];
        EXPECT_NEAR(x, ref[k], 2e-3f) << "joint" << k + 1;
        EXPECT_GE(x, m.joint_lower[arm[k]]);
        EXPECT_LE(x, m.joint_upper[arm[k]]);
    }
}

// A target out of reach: no crash, joints stay inside their limits, the error stays.
TEST(Ik, OutOfReachStaysInsideLimits) {
    const RobotModel m = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/panda/panda.urdf");
    std::vector<int> arm;
    for (int k = 1; k <= 7; ++k) arm.push_back(m.joint_index("joint" + std::to_string(k)));
    std::vector<float> q(m.nq, 0.0f);
    const IkResult r = solve_ik(m, m.body_index("hand"), {3.0f, 0.0f, 0.5f}, quat_identity(), arm, q);
    EXPECT_GT(r.pos_error, 1.0f);
    for (int j : arm) {
        EXPECT_GE(q[m.qadr[j]], m.joint_lower[j]);
        EXPECT_LE(q[m.qadr[j]], m.joint_upper[j]);
    }
}

}  // namespace

namespace {

// The posture (null-space) term: 400 small random hand moves (like a policy's
// end-effector actions), 3 IK steps each, from the start posture above the table.
// Plain IK lets the free elbow motion drift; with the posture term the joints stay
// near the start posture, and the hand still reaches its targets as well. Drift is
// measured after the hand is sent back to its start pose: the joint difference
// left then is the free elbow motion only.
TEST(Ik, PostureTermKeepsTheElbowNatural) {
    const RobotModel m = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/panda/panda.urdf");
    const int hand = m.body_index("hand");
    std::vector<int> arm;
    for (int k = 1; k <= 7; ++k) arm.push_back(m.joint_index("joint" + std::to_string(k)));
    std::vector<float> q0(m.nq, 0.0f);
    const float home[7] = {0.0f, 0.0f, 0.0f, -1.57079f, 0.0f, 1.57079f, -0.7853f};
    for (int k = 0; k < 7; ++k) q0[m.qadr[arm[k]]] = home[k];
    RobotWork work(m);
    DynamicsWork w = work.view();
    forward_kinematics(m.view(), q0.data(), w);
    const Quat down = load_quat(w.xquat + 4 * hand);
    solve_ik(m, hand, {0.5f, 0.0f, 0.25f}, down, arm, q0);  // start: hand above the table
    std::vector<float> posture;
    for (int j : arm) posture.push_back(q0[m.qadr[j]]);

    auto run = [&](float gain, float& drift, float& worst_err) {
        std::vector<float> q = q0;
        const IkJoints jt = make_ik_joints(m, arm, posture);
        IkOptions o;
        o.posture_gain = gain;
        Vec3 target{0.5f, 0.0f, 0.25f};
        float yaw = 0.0f;
        uint32_t s = 12345;
        auto rnd = [&s]() { s = s * 1664525u + 1013904223u; return float(s >> 8) / float(1u << 24) * 2.0f - 1.0f; };
        worst_err = 0.0f;
        for (int t = 0; t < 400; ++t) {
            target.x = std::fmin(0.7f, std::fmax(0.3f, target.x + 0.02f * rnd()));
            target.y = std::fmin(0.3f, std::fmax(-0.3f, target.y + 0.02f * rnd()));
            target.z = std::fmin(0.45f, std::fmax(0.12f, target.z + 0.02f * rnd()));
            yaw = std::fmin(1.2f, std::fmax(-1.2f, yaw + 0.05f * rnd()));
            const Quat rot = quat_from_axis_angle({0, 0, 1}, yaw) * down;
            IkResult r;
            for (int k = 0; k < 3; ++k) {
                forward_kinematics(m.view(), q.data(), w);
                ik_step(m.view(), w, hand, target, rot, jt, q.data(), o, r);
            }
            forward_kinematics(m.view(), q.data(), w);
            IkJoints none = jt;
            none.n = 0;
            ik_step(m.view(), w, hand, target, rot, none, q.data(), o, r);
            if (t >= 10) worst_err = std::fmax(worst_err, r.pos_error);
        }
        // Back to the start pose: the hand is where it began, so any joint difference
        // left is the free elbow motion (drift).
        for (int k = 0; k < 60; ++k) {
            forward_kinematics(m.view(), q.data(), w);
            IkResult r;
            ik_step(m.view(), w, hand, {0.5f, 0.0f, 0.25f}, down, jt, q.data(), o, r);
        }
        drift = 0.0f;
        for (int c = 0; c < 7; ++c) drift = std::fmax(drift, std::fabs(q[m.qadr[arm[c]]] - posture[c]));
    };
    float drift_plain, err_plain, drift_posture, err_posture;
    run(0.0f, drift_plain, err_plain);
    run(0.2f, drift_posture, err_posture);
    std::printf("400 random hand moves, then back to the start pose: joints off the start posture by %.3f rad "
                "without the posture term, %.3f rad with it; worst hand error while moving %.1f mm vs %.1f mm\n",
                drift_plain, drift_posture, 1000.0f * err_plain, 1000.0f * err_posture);
    EXPECT_LT(drift_posture, 0.005f) << "back to the start posture";
    EXPECT_GT(drift_plain, 0.03f) << "plain IK keeps the drifted elbow (measured 0.068 rad; else this test shows nothing)";
    EXPECT_LT(err_posture, 0.003f) << "and the hand still follows the target";
}

// Gravity compensation = MuJoCo's bias forces at zero velocity (gravity only).
TEST(Ik, GravityTorquesAreTheBiasAtRest) {
    RobotModel m = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/panda/panda.urdf");
    const int cube = add_free_box(m, "cube", {0.025f, 0.025f, 0.025f}, 0.1f);
    std::vector<float> q = m.qpos0(), zero(m.nv, 0.0f), tau(m.nv, 7.0f);
    for (int i = 0; i < 9; ++i) q[i] = 0.3f * float(i % 3) - 0.2f;
    q[m.qadr[cube] + 2] = 0.3f;
    RobotWork work(m);
    DynamicsWork w = work.view();
    forward_kinematics(m.view(), q.data(), w);
    body_inertias(m.view(), w);
    bias_forces(m.view(), zero.data(), w);
    gravity_torques(m.view(), w, 0, tau.data());
    for (int i = 0; i < 9; ++i) EXPECT_NEAR(tau[i], w.bias[i], 1e-4f * std::fmax(1.0f, std::fabs(w.bias[i]))) << "dof " << i;
    for (int i = 9; i < m.nv; ++i) EXPECT_EQ(tau[i], 7.0f) << "the cube's dofs are not touched";
}

}  // namespace
