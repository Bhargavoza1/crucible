// Phase 9a: the Panda picks up a cube (the grasp scene's control loop, scripted):
// IK moves the hand down around a cube on the floor, the fingers close, the hand
// lifts. Friction at the 6 finger-pad contacts must carry the cube's weight.
// impratio 10 (MuJoCo's usual setting for grasping): with the default 1, MuJoCo's
// soft friction lets a held cube creep down ~3 mm/s (MuJoCo itself, same Panda,
// same gains: -2.96 mm in 1 s; ours in this test: -2.95 mm); with 10, ~0.4 mm.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/dynamics/articulated.h"
#include "forge/dynamics/robot_batch_host.h"
#include "forge/dynamics/robot_work.h"
#include "forge/robots/actuators.h"
#include "forge/robots/ik.h"
#include "forge/robots/urdf.h"
#include "forge/solver/constraint_batch_host.h"
#include "forge/solver/constraint_model.h"

using namespace forge;

namespace {

struct PickResult {
    float cube_z = 0.0f, finger[2] = {0.0f, 0.0f};
    float slip_hold = 0.0f;  // cube height minus pad height: change over the 1 s hold (slip)
    float offset = 0.0f;     // cube center below the pad center at the end
    int status = 0;
};

// Phases (s): 0-1 hand down to the cube, 1-1.5 close, 1.5-3 lift to 'lift_to', 3-4 hold.
PickResult scripted_pick(float grip_kp, float friction, float lift_to, float impratio) {
    UrdfOptions u;
    u.armature = 0.1f;
    u.damping = 1.0f;
    RobotModel m = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/panda/panda.urdf", u);
    const float half = 0.025f;
    const int cube = add_free_box(m, "cube", {half, half, half}, 0.1f);
    const int hand = m.body_index("hand");
    std::vector<int> arm;
    for (int k = 1; k <= 7; ++k) arm.push_back(m.joint_index("joint" + std::to_string(k)));
    ConstraintOptions o;
    o.timestep = 0.002f;
    o.friction = friction;
    o.impratio = impratio;
    ConstraintModel cm(m, o);
    cm.add_ground();
    Actuators act;
    for (int j : arm) act.add_position(m, m.joint_name[j], 1000.0f, 30.0f, 0.0f, 0.0f);
    for (const char* f : {"finger_joint1", "finger_joint2"}) act.add_position(m, f, grip_kp, 10.0f, 0.0f, 0.0f);
    RobotBatchHost host(m, 1);
    ConstraintBatchHost ch(cm, act.nu(), 1);
    RobotWork work(m);
    DynamicsWork w = work.view();

    const float home[7] = {0.0f, 0.0f, 0.0f, -1.57079f, 0.0f, 1.57079f, -0.7853f};
    std::vector<float> plan(m.nq, 0.0f);
    for (int k = 0; k < 7; ++k) plan[m.qadr[arm[k]]] = home[k];
    forward_kinematics(m.view(), plan.data(), w);
    const Quat down = load_quat(w.xquat + 4 * hand);
    const float pad_z = 0.0584f + 0.0445f;
    const Vec3 above{0.5f, 0.0f, 0.25f};
    solve_ik(m, hand, above, down, arm, plan);
    float* q = host.env_qpos(0);
    for (int j : arm) q[m.qadr[j]] = plan[m.qadr[j]];
    q[m.qadr[m.joint_index("finger_joint1")]] = q[m.qadr[m.joint_index("finger_joint2")]] = 0.04f;
    forward_kinematics(m.view(), q, w);
    const Mat3 Rh = quat_to_mat3(load_quat(w.xquat + 4 * hand));
    const float cy = std::atan2(Rh(1, 0), Rh(0, 0));
    const int c = m.qadr[cube];
    q[c + 0] = 0.5f;
    q[c + 2] = half;
    q[c + 3] = std::cos(0.5f * cy);
    q[c + 6] = std::sin(0.5f * cy);

    std::vector<float> zero(m.nv, 0.0f);
    const float dt = 0.002f;
    float hold_rel = 0.0f, slip = 0.0f, offset = 0.0f;
    for (int step = 0; step < 2000; ++step) {
        const float t = step * dt;
        const float z = t < 1.0f ? above.z + (pad_z + half - above.z) * t
                                 : (t < 1.5f ? pad_z + half : std::fmin(lift_to, pad_z + half + (lift_to - pad_z - half) * (t - 1.5f) / 1.5f));
        IkOptions ik;
        ik.max_iterations = 10;
        solve_ik(m, hand, {above.x, above.y, z}, down, arm, plan, ik);
        for (int k = 0; k < 7; ++k) ch.ctrl[k] = plan[m.qadr[arm[k]]];
        ch.ctrl[7] = ch.ctrl[8] = t < 1.0f ? 0.04f : 0.0f;
        forward_kinematics(m.view(), q, w);
        body_inertias(m.view(), w);
        bias_forces(m.view(), zero.data(), w);
        for (int i = 0; i < m.nv; ++i) host.tau[i] = m.dof_body[i] == cube ? 0.0f : w.bias[i];
        robot_step_constrained_cpu(m.view(), cm.view(), act.view(), host.view(), host.work(), ch.view(), dt, 1);
        if (step == 1500 || step == 1999) {
            forward_kinematics(m.view(), q, w);
            const Vec3 hp = load3(w.xpos + 3 * hand);
            const float rel = q[c + 2] - (hp.z - pad_z);
            if (step == 1500) hold_rel = rel;
            else {
                slip = rel - hold_rel;
                offset = -rel;
            }
        }
    }
    PickResult r;
    r.slip_hold = slip;
    r.offset = offset;
    r.cube_z = q[c + 2];
    r.finger[0] = q[m.qadr[m.joint_index("finger_joint1")]];
    r.finger[1] = q[m.qadr[m.joint_index("finger_joint2")]];
    r.status = host.status[0];
    return r;
}

TEST(Grasp, PandaPicksUpTheCube) {
    const PickResult r = scripted_pick(200.0f, 1.0f, 0.25f, 10.0f);
    std::printf("pick: cube center at %.3f m after the lift (start %.3f), fingers %.4f %.4f m; cube %.1f mm below "
                "the pad center, slip during the last second %.2f mm\n", r.cube_z, 0.025f, r.finger[0], r.finger[1],
                1000.0f * r.offset, 1000.0f * r.slip_hold);
    ASSERT_EQ(r.status, 0);
    EXPECT_GT(r.cube_z, 0.10f) << "lifted (the hand went up to 0.25 m)";
    EXPECT_LT(std::fabs(r.slip_hold), 0.001f) << "held, not sliding out";
    EXPECT_NEAR(r.finger[0], 0.025f - 0.0015f, 0.003f) << "pads on the cube's faces";
    EXPECT_NEAR(r.finger[1], 0.025f - 0.0015f, 0.003f);
}

// Without friction the fingers cannot hold it: the cube stays on the floor.
TEST(Grasp, NoFrictionNoPick) {
    const PickResult r = scripted_pick(200.0f, 0.0f, 0.25f, 10.0f);
    ASSERT_EQ(r.status, 0);
    EXPECT_LT(r.cube_z, 0.04f);
}

}  // namespace
