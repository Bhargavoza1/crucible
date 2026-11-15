#pragma once
// URDF reader (doc 3.5). Reads the file once on the CPU and builds a RobotModel.
//
// Supported: links with <inertial> and sphere <collision>; joints of type
// revolute, continuous, prismatic, fixed, with <origin>, <axis>, <limit>,
// <dynamics damping>. Not supported (error): floating/planar joints, loops.
// Ignored: <visual>, non-sphere collision shapes (counted in a warning),
// the <mujoco> extension tag (it is only for MuJoCo's own loader).
//
// Body order: depth first from the root link, children in the order their
// joints appear in the file. MuJoCo uses the same order.
#include <string>

#include "forge/robots/model.h"

namespace forge {

struct UrdfOptions {
    bool floating_base = false;             // give the root link a free joint
    std::string free_joint_name = "root";  // name of that joint
};

// Throws std::runtime_error with the file name and a clear message on bad input.
RobotModel load_urdf(const std::string& path, const UrdfOptions& opt = {});
RobotModel parse_urdf(const std::string& xml, const UrdfOptions& opt = {}, const std::string& source = "<string>");

// URDF rpy (fixed-axis roll about x, then pitch about y, then yaw about z) to a
// quaternion: R = Rz(yaw) * Ry(pitch) * Rx(roll).
Quat quat_from_rpy(float roll, float pitch, float yaw);

}  // namespace forge
