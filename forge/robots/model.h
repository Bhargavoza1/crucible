#pragma once
// Robot model: the joint tree, built once from a URDF (doc 3.5).
//
// Bodies are stored in topological order: parent index < child index, and the
// root's parent is -1 (the world). Each body has exactly one joint that
// connects it to its parent: free, revolute, prismatic or fixed.
//
// Coordinates follow MuJoCo exactly, so our numbers can be compared with it:
//   free joint  qpos: x y z, qw qx qy qz   (7)  quaternion with w FIRST
//               qvel: vx vy vz (world frame, velocity of the body origin),
//                     wx wy wz (BODY frame angular velocity)          (6)
//   revolute    qpos: angle (rad), qvel: rate             (1, 1)
//   prismatic   qpos: offset (m),  qvel: rate             (1, 1)
//   fixed       nothing                                   (0, 0)
// A joint sits at the body's frame origin (URDF: joint frame = child frame).
//
// ModelView is a struct of pointers (host or device memory), like ParticleView,
// so the dynamics code is written once and runs on the CPU and the GPU.
#include <string>
#include <vector>

#include "forge/math/hd.h"
#include "forge/math/mat3.h"
#include "forge/math/quat.h"
#include "forge/math/vec3.h"

namespace forge {

enum JointType : int { kJointFree = 0, kJointRevolute = 1, kJointPrismatic = 2, kJointFixed = 3 };

constexpr int kMaxBodies = 64;
constexpr int kMaxDofs = 64;

FORGE_HD int joint_nq(int type) { return type == kJointFree ? 7 : (type == kJointFixed ? 0 : 1); }
FORGE_HD int joint_nv(int type) { return type == kJointFree ? 6 : (type == kJointFixed ? 0 : 1); }

struct ModelView {
    int nbody = 0, nq = 0, nv = 0;
    const int* parent = nullptr;       // [nbody], -1 = world
    const int* jtype = nullptr;        // [nbody], JointType
    const int* qadr = nullptr;         // [nbody], first qpos index of the joint (-1 if fixed)
    const int* dadr = nullptr;         // [nbody], first dof index of the joint (-1 if fixed)
    const float* jaxis = nullptr;      // [3 * nbody], joint axis in the body frame (unit)
    const float* body_pos = nullptr;   // [3 * nbody], body frame in the parent frame, at q = 0
    const float* body_quat = nullptr;  // [4 * nbody], same, Quat order x y z w
    const float* mass = nullptr;       // [nbody], kg
    const float* com = nullptr;        // [3 * nbody], center of mass in the body frame
    const float* inertia = nullptr;    // [9 * nbody], about the com, body axes, row major
    const int* dof_body = nullptr;     // [nv], body that owns each dof
    Vec3 gravity{0, 0, -9.81f};
};

struct RobotModel {
    std::string name;
    std::vector<std::string> body_name, joint_name;  // joint_name[b] = joint into body b
    std::vector<int> parent, jtype, qadr, dadr, dof_body;
    std::vector<float> jaxis, body_pos, body_quat, mass, com, inertia;
    std::vector<float> joint_lower, joint_upper, joint_effort, joint_damping;  // per body (unused for free/fixed)
    std::vector<bool> joint_limited;

    // Collision spheres (Phase 5 collides only spheres and planes).
    struct Sphere {
        int body;
        Vec3 pos;  // in the body frame
        float radius;
    };
    std::vector<Sphere> spheres;

    Vec3 gravity{0, 0, -9.81f};
    int nq = 0, nv = 0;

    int nbody() const { return static_cast<int>(parent.size()); }
    int body_index(const std::string& name) const;   // -1 if not found
    int joint_index(const std::string& name) const;  // body index of the joint, -1 if not found

    // qpos for the zero pose: joints at 0, free base at the origin, identity quaternion.
    std::vector<float> qpos0() const;

    // Pointers into the vectors. Invalid after the model changes.
    ModelView view() const;

    // Throws if the arrays do not make a valid tree (order, sizes, addresses).
    void validate() const;
};

}  // namespace forge
