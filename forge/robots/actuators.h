#pragma once
// PD position actuators (doc 3.5), the same as MuJoCo's <position> actuator:
//
//   force = kp * (ctrl - q) - kv * qvel      then clamped to [force_lo, force_hi]
//
// One actuator drives one revolute or prismatic joint. ctrl is the target
// position (rad or m). The force goes into the joint's dof, like tau.
// No clamp when force_lo >= force_hi (MuJoCo: forcelimited = false).
// add_position also caps the range by the joint's URDF effort limit, as MuJoCo's
// URDF import does (joint actuatorfrcrange).
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
// kp_scale, kv_scale: per-env PD gain factors; strength: per-env motor strength
// (scales the whole force, limit included). Domain randomization (doc 6.2); 1 = none.
FORGE_HD void apply_actuators(const ActuatorView& a, const float* ctrl, const float* qpos, const float* qvel,
                              float* qfrc, float kp_scale = 1.0f, float kv_scale = 1.0f, float strength = 1.0f) {
    for (int k = 0; k < a.nu; ++k) {
        float f = kp_scale * a.kp[k] * (ctrl[k] - qpos[a.qadr[k]]) - kv_scale * a.kv[k] * qvel[a.dadr[k]];
        if (a.force_lo[k] < a.force_hi[k]) f = fminf(fmaxf(f, a.force_lo[k]), a.force_hi[k]);
        qfrc[a.dadr[k]] += strength * f;
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
