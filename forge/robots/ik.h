#pragma once
// Inverse kinematics (Phase 9a): joint angles that put a body's frame at a target
// position and orientation. Damped least squares on the position and rotation
// error, steps of at most max_step rad (or m), clipped to the joint limits after
// every step (the method of tools/gen_mujoco_refs.py panda_ik). Host only, small
// models; the scripted expert of the pick task (Phase 9b) uses it too.
#include <vector>

#include "forge/math/quat.h"
#include "forge/math/vec3.h"
#include "forge/robots/model.h"

namespace forge {

struct IkOptions {
    int max_iterations = 300;
    double damping = 1e-4;      // lambda in J^T (J J^T + lambda I)^-1 e
    float max_step = 0.1f;      // largest joint change per iteration
    float tolerance = 1e-6f;    // stop when |position error| + |rotation error| is below
    bool rotation = true;       // false: position only (3 rows)
};

struct IkResult {
    int iterations = 0;
    float pos_error = 0.0f;  // m
    float rot_error = 0.0f;  // rad (about)
};

// Moves the listed joints (body indices of revolute or prismatic joints) in qpos so
// body 'body' reaches target_pos / target_rot (world). Other qpos entries stay.
IkResult solve_ik(const RobotModel& m, int body, Vec3 target_pos, Quat target_rot, const std::vector<int>& joints,
                  std::vector<float>& qpos, const IkOptions& opt = {});

}  // namespace forge
