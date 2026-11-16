#include "forge/robots/actuators.h"

#include <algorithm>

#include <stdexcept>

namespace forge {

void Actuators::add_position(const RobotModel& m, const std::string& joint_name, float kp_, float kv_,
                             float lo, float hi) {
    const int b = m.joint_index(joint_name);
    if (b < 0) throw std::runtime_error("actuator: no joint named '" + joint_name + "'");
    if (m.jtype[b] != kJointRevolute && m.jtype[b] != kJointPrismatic)
        throw std::runtime_error("actuator: joint '" + joint_name + "' is not revolute or prismatic");
    joint.push_back(joint_name);
    qadr.push_back(m.qadr[b]);
    dadr.push_back(m.dadr[b]);
    kp.push_back(kp_);
    kv.push_back(kv_);
    // The URDF <limit effort> also caps the actuator force on the joint, as
    // MuJoCo's URDF import does (joint actuatorfrcrange). One actuator per joint
    // here, so capping each actuator is the same as capping their sum.
    const float effort = m.joint_effort[b];
    if (effort > 0.0f) {
        if (lo >= hi) {  // the actuator itself has no limit
            lo = -effort;
            hi = effort;
        } else {
            lo = std::max(lo, -effort);
            hi = std::min(hi, effort);
        }
    }
    force_lo.push_back(lo);
    force_hi.push_back(hi);
}

Actuators Actuators::position_all(const RobotModel& m, float kp, float kv, float lo, float hi) {
    Actuators a;
    for (int b = 0; b < m.nbody(); ++b)
        if (m.jtype[b] == kJointRevolute || m.jtype[b] == kJointPrismatic)
            a.add_position(m, m.joint_name[b], kp, kv, lo, hi);
    return a;
}

ActuatorView Actuators::view() const {
    return {nu(), qadr.data(), dadr.data(), kp.data(), kv.data(), force_lo.data(), force_hi.data()};
}

}  // namespace forge
