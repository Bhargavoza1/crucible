#pragma once
// PD position actuators (doc 3.5), the same as MuJoCo's <position> actuator:
//
//   force = kp * (ctrl - q) - kv * qvel      then clamped to [force_lo, force_hi]
//
// One actuator drives one revolute or prismatic joint. ctrl is the target
// position (rad or m). The force goes into the joint's dof, like tau.
// No clamp when force_lo >= force_hi (MuJoCo: forcelimited = false).
#include <string>
#include <vector>

#include "forge/math/hd.h"
#include "forge/robots/model.h"

namespace forge {

struct ActuatorView {
    int nu = 0;
    const int* qadr = nullptr;  // [nu] qpos index of the driven joint
    const int* dadr = nullptr;  // [nu] dof index of the driven joint
    const float* kp = nullptr;  // [nu] N/m or Nm/rad
    const float* kv = nullptr;  // [nu] N s/m or Nm s/rad
    const float* force_lo = nullptr;
    const float* force_hi = nullptr;
};

// qfrc[dof] += actuator force, for every actuator. ctrl has nu entries.
FORGE_HD void apply_actuators(const ActuatorView& a, const float* ctrl, const float* qpos, const float* qvel,
                              float* qfrc) {
    for (int k = 0; k < a.nu; ++k) {
        float f = a.kp[k] * (ctrl[k] - qpos[a.qadr[k]]) - a.kv[k] * qvel[a.dadr[k]];
        if (a.force_lo[k] < a.force_hi[k]) f = fminf(fmaxf(f, a.force_lo[k]), a.force_hi[k]);
        qfrc[a.dadr[k]] += f;
    }
}

struct Actuators {
    std::vector<std::string> joint;
    std::vector<int> qadr, dadr;
    std::vector<float> kp, kv, force_lo, force_hi;

    int nu() const { return static_cast<int>(qadr.size()); }

    // Adds one actuator. Throws if the joint is missing or is not revolute / prismatic.
    void add_position(const RobotModel& m, const std::string& joint_name, float kp, float kv, float force_lo,
                      float force_hi);

    // One actuator on every revolute and prismatic joint, in body order.
    static Actuators position_all(const RobotModel& m, float kp, float kv, float force_lo, float force_hi);

    ActuatorView view() const;
};

}  // namespace forge
