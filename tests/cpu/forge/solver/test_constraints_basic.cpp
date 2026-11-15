// Phase 6a: simple physics checks of the constraint solver, without MuJoCo.
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/dynamics/robot_work.h"
#include "forge/robots/actuators.h"
#include "forge/robots/urdf.h"
#include "forge/solver/constrained_dynamics.h"
#include "forge/solver/constraint_model.h"
#include "forge/solver/constraint_work.h"

using namespace forge;

namespace {

std::string asset(const std::string& name) { return std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/" + name; }

// Runs 'steps' steps of dt; returns false on a dynamics error.
bool run(const RobotModel& model, const ConstraintModel& cm, const Actuators& act, std::vector<float>& qpos,
         std::vector<float>& qvel, const std::vector<float>& ctrl, int steps, float dt,
         float* min_q = nullptr, int min_q_index = 0) {
    RobotWork work(model);
    DynamicsWork w = work.view();
    ConstraintWorkHost cwork(cm);
    ConstraintWork cw = cwork.view();
    std::vector<float> qacc(model.nv);
    for (int s = 0; s < steps; ++s) {
        if (step_constrained(model.view(), cm.view(), act.view(), qpos.data(), qvel.data(), nullptr,
                             ctrl.empty() ? nullptr : ctrl.data(), w, cw, qacc.data(), dt) != kDynamicsOk)
            return false;
        if (min_q) *min_q = std::fmin(*min_q, qpos[min_q_index]);
    }
    return true;
}

// No friction: the floor pushes only along its normal, so horizontal velocity stays.
TEST(Constraints, FrictionlessBallKeepsSliding) {
    UrdfOptions opt;
    opt.floating_base = true;
    const RobotModel ball = load_urdf(asset("ball.urdf"), opt);
    ConstraintOptions co;
    co.friction = 0.0f;
    ConstraintModel cm(ball, co);
    cm.add_ground();
    std::vector<float> qpos = ball.qpos0(), qvel(ball.nv, 0.0f);
    qpos[2] = 0.1f;   // resting on the floor (radius 0.1)
    qvel[0] = 1.0f;   // sliding along x
    ASSERT_TRUE(run(ball, cm, Actuators{}, qpos, qvel, {}, 500, 0.002f));
    EXPECT_NEAR(qvel[0], 1.0f, 1e-4f);
    EXPECT_NEAR(qpos[0], 1.0f, 1e-3f);
    EXPECT_NEAR(qpos[2], 0.1f, 5e-3f);  // soft contact: a few mm of penetration at rest
    EXPECT_NEAR(qvel[2], 0.0f, 1e-3f);
}

// With friction a sliding ball slows down and starts to roll: v = w r (no slip).
TEST(Constraints, SlidingBallStartsRolling) {
    UrdfOptions opt;
    opt.floating_base = true;
    const RobotModel ball = load_urdf(asset("ball.urdf"), opt);
    ConstraintModel cm(ball);  // MuJoCo default friction 1
    cm.add_ground();
    std::vector<float> qpos = ball.qpos0(), qvel(ball.nv, 0.0f);
    qpos[2] = 0.1f;
    qvel[0] = 1.0f;
    ASSERT_TRUE(run(ball, cm, Actuators{}, qpos, qvel, {}, 500, 0.002f));
    // Rolling along +x: angular velocity about +y = v / r (body frame = world frame
    // up to the ball's rotation about y, so use the speed of the contact point).
    // Solid ball: the final speed is 5/7 of the start speed.
    EXPECT_NEAR(qvel[0], 5.0f / 7.0f, 0.02f);
}

// The arm's slider (j2, range -0.5..0.5) falls onto its lower limit and stays there.
TEST(Constraints, JointLimitHolds) {
    const RobotModel arm = load_urdf(asset("arm3.urdf"));
    ConstraintModel cm(arm);
    const int j2 = arm.joint_index("j2");
    ASSERT_GE(j2, 0);
    ASSERT_TRUE(arm.joint_limited[j2]);
    std::vector<float> qpos = arm.qpos0(), qvel(arm.nv, 0.0f);
    qpos[arm.qadr[arm.joint_index("j1")]] = 0.3f;
    qpos[arm.qadr[j2]] = 0.2f;
    qpos[arm.qadr[arm.joint_index("j3")]] = 2.5f;
    float min_q = 1e9f;
    ASSERT_TRUE(run(arm, cm, Actuators{}, qpos, qvel, {}, 1500, 0.002f, &min_q, arm.qadr[j2]));
    EXPECT_GT(min_q, -0.5f - 0.02f) << "soft limit: small overshoot only";
    EXPECT_LT(qpos[arm.qadr[j2]], -0.45f) << "the slider should rest at its lower limit";
}

TEST(Actuators, PdForceAndClamp) {
    const RobotModel arm = load_urdf(asset("arm3.urdf"));
    Actuators act;
    act.add_position(arm, "j3", 10.0f, 2.0f, -5.0f, 5.0f);
    const ActuatorView v = act.view();
    std::vector<float> qpos = {0, 0, 0.1f}, qvel = {0, 0, 0.5f}, qfrc(3, 0.0f);
    float ctrl = 0.3f;
    apply_actuators(v, &ctrl, qpos.data(), qvel.data(), qfrc.data());
    EXPECT_FLOAT_EQ(qfrc[arm.dadr[arm.joint_index("j3")]], 10.0f * 0.2f - 2.0f * 0.5f);
    ctrl = 10.0f;  // would be 98.0 - 1.0: clamped to 5
    qfrc.assign(3, 0.0f);
    apply_actuators(v, &ctrl, qpos.data(), qvel.data(), qfrc.data());
    EXPECT_FLOAT_EQ(qfrc[arm.dadr[arm.joint_index("j3")]], 5.0f);
    EXPECT_THROW(act.add_position(arm, "tool_mount", 1, 0, 0, 0), std::runtime_error);  // fixed joint
    EXPECT_THROW(act.add_position(arm, "nope", 1, 0, 0, 0), std::runtime_error);
}

}  // namespace
