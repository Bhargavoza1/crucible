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
