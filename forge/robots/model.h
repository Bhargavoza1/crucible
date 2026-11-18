#pragma once
// Robot model: the joint tree, built once from a URDF (doc 3.5).
//
// Bodies are stored in topological order: parent index < child index, and a
// root's parent is -1 (the world). Each body has exactly one joint that
// connects it to its parent: free, revolute, prismatic or fixed. A model may have
// several roots (Phase 9: an arm and a free cube); a free joint is only allowed
// on a root.
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
    // Per dof, may be null (= 0). Like MuJoCo: armature is added to the diagonal
    // of M (motor rotor inertia); damping is a passive force -damping * qvel.
    const float* dof_armature = nullptr;  // [nv]
    const float* dof_damping = nullptr;   // [nv]
    Vec3 gravity{0, 0, -9.81f};
};

struct RobotModel {
    std::string name;
    std::vector<std::string> body_name, joint_name;  // joint_name[b] = joint into body b
    std::vector<int> parent, jtype, qadr, dadr, dof_body;
    std::vector<float> jaxis, body_pos, body_quat, mass, com, inertia;
    std::vector<float> joint_lower, joint_upper, joint_effort, joint_damping;  // per body (unused for free/fixed)
    std::vector<bool> joint_limited;
    std::vector<float> dof_armature, dof_damping;  // [nv] each, or empty (= 0)

    // Collision spheres (Phase 5 collides only spheres and planes).
    // Collision filter of every sphere, box and capsule (Phase 10d, MuJoCo's rule):
    // two shapes may touch when (contype1 & conaffinity2) || (contype2 & conaffinity1).
    // Planes and the heightfield: contype 0, conaffinity 1 (they touch every shape
    // with contype bit 1). The URDF loader gives a robot's shapes contype 1,
    // conaffinity 0: they touch the ground and free objects (add_free_box: 1, 1) but
    // not each other (as the MuJoCo reference scenes); a task may change that.
    struct Sphere {
        int body;
        Vec3 pos;  // in the body frame
        float radius;
        int contype = 1, conaffinity = 1;
    };
    std::vector<Sphere> spheres;
    // Collision boxes (Phase 9a): touch planes and spheres of other bodies.
    struct Box {
        int body;
        Vec3 pos;   // center, in the body frame
        Quat rot;   // box axes in the body frame
        Vec3 half;  // half sizes
        bool drawn = true;  // viewers draw it (false: a task draws it its own way)
        int contype = 1, conaffinity = 1;
        float rgb[3] = {0.95f, 0.45f, 0.1f};  // viewers' color (Phase 11: openarm_stack's cubes)
    };
    std::vector<Box> boxes;
    // Collision capsules (Phase 10): a segment along the local z axis from -half_length
    // to +half_length, swept by a sphere of 'radius' (URDF <cylinder>, see urdf.h).
    // Touch planes and boxes of other bodies.
    struct Capsule {
        int body;
        Vec3 pos;           // center, in the body frame
        Quat rot;           // capsule frame in the body frame (axis = its z)
        float radius;
        float half_length;  // of the segment (the cylinder part), >= 0
        int contype = 1, conaffinity = 1;
    };
    std::vector<Capsule> capsules;
    // Visual-only boxes in the world (Phase 10: a task's table, tray, stand). Not
    // used by the physics: viewers draw them.
    struct VisualBox {
        Vec3 center, half;
        float rgb[3];
        int obs_xy = -1;  // >= 0: a box that moves per env: center x y relative to obs[obs_xy], obs[obs_xy + 1]
    };
    std::vector<VisualBox> visuals;
    // A task's suggested viewer camera (Phase 10): orbit target, distance (0 = the
    // viewer picks), yaw and pitch (rad, as flux::OrbitCamera).
    Vec3 view_target{0, 0, 0};
    float view_distance = 0.0f, view_yaw = 0.0f, view_pitch = 0.0f;

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

// Appends a free-floating solid box of uniform density: a new root body with a
// free joint (named 'name', like the body) and one collision box. Its qpos and
// dofs come last, the order MuJoCo gives a body added after the robot. Returns
// the new body's index.
int add_free_box(RobotModel& m, const std::string& name, Vec3 half, float mass);

}  // namespace forge
