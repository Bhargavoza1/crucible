// Phase 9a: collision boxes, free bodies next to a robot, sphere-box and plane-box
// contacts (forge/solver/constraints.h). The contacts themselves are checked against
// MuJoCo in test_constraints_mujoco.cpp (scenes ball_cube and panda_cube); these
// tests cover the parts around them.
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "forge/dynamics/robot_work.h"
#include "forge/robots/actuators.h"
#include "forge/robots/urdf.h"
#include "forge/solver/constrained_dynamics.h"
#include "forge/solver/constraint_model.h"
#include "forge/solver/constraint_work.h"

using namespace forge;

namespace {

const char* kBallUrdf = R"(<robot name="b"><link name="ball">
  <inertial><mass value="1"/><inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/></inertial>
  <collision><geometry><sphere radius="0.1"/></geometry></collision>
  <collision><origin xyz="0 0 0.2" rpy="0 0 1.5707963"/><geometry><box size="0.1 0.2 0.3"/></geometry></collision>
</link></robot>)";

// URDF boxes: half sizes and pose; a free box after a floating robot: a second root
// with its own free joint, last in qpos and qvel, solid-box inertia.
TEST(BoxModel, UrdfBoxesAndFreeBox) {
    UrdfOptions u;
    u.floating_base = true;
    RobotModel m = parse_urdf(kBallUrdf, u);
    ASSERT_EQ(m.boxes.size(), 1u);
    EXPECT_FLOAT_EQ(m.boxes[0].half.x, 0.05f);
    EXPECT_FLOAT_EQ(m.boxes[0].half.z, 0.15f);
    EXPECT_FLOAT_EQ(m.boxes[0].pos.z, 0.2f);
    EXPECT_NEAR(m.boxes[0].rot.z, std::sin(0.785398f), 1e-6f);
    const int b = add_free_box(m, "cube", {0.03f, 0.04f, 0.05f}, 0.2f);
    EXPECT_EQ(b, 1);
    EXPECT_EQ(m.parent[b], -1);
    EXPECT_EQ(m.jtype[b], kJointFree);
    EXPECT_EQ(m.qadr[b], 7);
    EXPECT_EQ(m.dadr[b], 6);
    EXPECT_EQ(m.nq, 14);
    EXPECT_EQ(m.nv, 12);
    EXPECT_NEAR(m.inertia[9 * b + 0], 0.2f * (0.04f * 0.04f + 0.05f * 0.05f) / 3.0f, 1e-9f);
    EXPECT_NEAR(m.inertia[9 * b + 8], 0.2f * (0.03f * 0.03f + 0.04f * 0.04f) / 3.0f, 1e-9f);
    const auto q0 = m.qpos0();
    EXPECT_EQ(q0[3], 1.0f);
    EXPECT_EQ(q0[10], 1.0f) << "the cube's quaternion w";
    EXPECT_THROW(add_free_box(m, "bad", {0.0f, 0.1f, 0.1f}, 1.0f), std::runtime_error);
}

// sphere_box: outside (face, edge), inside (out through the nearest face), apart.
TEST(BoxContact, SphereBoxGeometry) {
    const Mat3 I = quat_to_mat3(quat_identity());
    const Vec3 h{0.03f, 0.04f, 0.05f};
    float dist;
    Vec3 n;
    ASSERT_TRUE(sphere_box({0.076f, 0.01f, -0.02f}, 0.05f, {0, 0, 0}, I, h, dist, n));
    EXPECT_NEAR(dist, -0.004f, 1e-6f);
    EXPECT_NEAR(n.x, 1.0f, 1e-6f);
    ASSERT_TRUE(sphere_box({0.06f, 0.07f, 0.0f}, 0.05f, {0, 0, 0}, I, h, dist, n));
    EXPECT_NEAR(dist, std::sqrt(2.0f) * 0.03f - 0.05f, 1e-6f);
    EXPECT_NEAR(n.x, 0.7071068f, 1e-6f);
    EXPECT_NEAR(n.y, 0.7071068f, 1e-6f);
    ASSERT_TRUE(sphere_box({0.02f, 0.0f, 0.0f}, 0.05f, {0, 0, 0}, I, h, dist, n));
    EXPECT_NEAR(dist, -0.06f, 1e-6f) << "MuJoCo: -(0.01 to the +x face) - radius";
    EXPECT_NEAR(n.x, 1.0f, 1e-6f);
    EXPECT_FALSE(sphere_box({0.2f, 0.0f, 0.0f}, 0.05f, {0, 0, 0}, I, h, dist, n));
}

// A box sunk flat into the floor touches with its 4 bottom corners only (MuJoCo's
// mjc_PlaneBox stops at 4), and a box resting on the floor stays (qacc ~ 0).
TEST(BoxContact, BoxOnFloorFourCornersAndRests) {
    RobotModel m;
    m.name = "only_cube";
    add_free_box(m, "cube", {0.05f, 0.05f, 0.05f}, 1.0f);
    ConstraintOptions o;
    o.timestep = 0.002f;
    ConstraintModel cm(m, o);
    cm.add_ground();
    EXPECT_EQ(cm.max_rows(), 4 * 4);
    RobotWork work(m);
    DynamicsWork w = work.view();
    ConstraintWorkHost cwh(cm);
    ConstraintWork cw = cwh.view();
    std::vector<float> q = m.qpos0(), qd(6, 0.0f), qacc(6);
    q[2] = 0.0f;  // center at the floor: the whole lower half is inside
    ASSERT_EQ(forward_dynamics_constrained(m.view(), cm.view(), ActuatorView{}, q.data(), qd.data(), nullptr, nullptr,
                                           w, cw, qacc.data()),
              kDynamicsOk);
    EXPECT_EQ(cw.nefc, 16) << "4 corners x 4 pyramid rows";
    // Let it settle from 1 mm above the floor, then it must stay put.
    q[2] = 0.051f;
    for (int t = 0; t < 1000; ++t)
        ASSERT_EQ(step_constrained(m.view(), cm.view(), ActuatorView{}, q.data(), qd.data(), nullptr, nullptr, w, cw,
                                   qacc.data(), 0.002f),
                  kDynamicsOk);
    EXPECT_NEAR(q[2], 0.05f, 1e-3f) << "resting on the floor (soft contact sinks it a little)";
    EXPECT_NEAR(q[0], 0.0f, 1e-6f);
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(qd[i], 0.0f, 1e-3f) << "dof " << i;
}

}  // namespace
