#pragma once
// What the constraint solver needs besides the robot model (doc 3.4):
// joint limits, collision spheres, ground planes, and MuJoCo's soft-constraint
// constants. Built once on the host; ConstraintView is the pointer struct the
// FORGE_HD code reads (host or device memory).
//
// Contacts: collision spheres vs planes and the heightfield (Phase 6, 8d),
// (Phase 9a) boxes vs planes and spheres vs boxes of other bodies, and (Phase 10)
// capsules vs planes and boxes of other bodies. No sphere-sphere, sphere-capsule,
// capsule-capsule, box-box, or capsule/box-heightfield contacts. In MuJoCo terms:
// robot spheres and capsules touch the floor and the boxes, boxes touch the floor.
#include <vector>

#include "forge/solver/heightfield.h"

#include "forge/math/hd.h"
#include "forge/math/vec3.h"
#include "forge/robots/model.h"

namespace forge {

// Defaults are MuJoCo's defaults, so a model with no settings matches MuJoCo.
struct ConstraintOptions {
    float timestep = 0.002f;                          // MuJoCo raises solref[0] to at least 2 * timestep
    float solref[2] = {0.02f, 1.0f};                  // timeconst (s), dampratio
    float solimp[5] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};  // dmin, dmax, width (m), mid, power
    float friction = 1.0f;   // sliding friction sphere vs plane (MuJoCo: max of the two geoms). 0 = frictionless
    float impratio = 1.0f;   // friction-to-normal impedance ratio (MuJoCo opt.impratio)
    int max_iterations = 50; // Newton iterations; it usually stops after 1-5
};

struct ConstraintView {
    int nbody = 0, nv = 0;
    int nlimit = 0;                        // limited revolute / prismatic joints
    const int* limit_qadr = nullptr;       // [nlimit]
    const int* limit_dadr = nullptr;       // [nlimit]
    const float* limit_lower = nullptr;    // [nlimit]
    const float* limit_upper = nullptr;    // [nlimit]
    int nsphere = 0;                       // spheres on bodies that can move
    const int* sphere_body = nullptr;      // [nsphere]
    const float* sphere_pos = nullptr;     // [3 * nsphere] in the body frame
    const float* sphere_radius = nullptr;  // [nsphere]
    int nplane = 0;
    const float* plane_normal = nullptr;   // [3 * nplane] unit
    const float* plane_point = nullptr;    // [3 * nplane] a point on the plane
    HeightfieldView hfield;                // terrain (Phase 8d); hfield.h null = none
    int nbox = 0;                          // boxes on bodies that can move (Phase 9a)
    const int* box_body = nullptr;         // [nbox]
    const float* box_pos = nullptr;        // [3 * nbox] center in the body frame
    const float* box_quat = nullptr;       // [4 * nbox] box axes in the body frame, x y z w
    const float* box_half = nullptr;       // [3 * nbox] half sizes
    int ncapsule = 0;                      // capsules (Phase 10), also on bodies welded to the world
    const int* capsule_body = nullptr;     // [ncapsule]
    const float* capsule_pos = nullptr;    // [3 * ncapsule] center in the body frame
    const float* capsule_quat = nullptr;   // [4 * ncapsule] capsule frame (axis = z) in the body frame, x y z w
    const float* capsule_size = nullptr;   // [2 * ncapsule] radius, half length
    const float* body_invweight0 = nullptr;  // [2 * nbody] MuJoCo body_invweight0 (translation, rotation)
    const float* dof_invweight0 = nullptr;   // [nv]
    // Soft-constraint constants, from ConstraintOptions (same for all rows).
    float K = 0, B = 0;      // aref = -B * vel - K * imp * pos
    float solimp[5] = {};
    float friction = 0, impratio = 1;
    int max_rows = 0;        // most rows one env can have
    int max_iterations = 0;
};

class ConstraintModel {
public:
    ConstraintModel(const RobotModel& m, const ConstraintOptions& opt = {});

    // Adds a plane (normal is normalized). Call before view().
    void add_plane(Vec3 normal, Vec3 point);
    void add_ground() { add_plane({0, 0, 1}, {0, 0, 0}); }  // z = 0, Z up
    // Heightfield terrain (see heightfield.h): heights[j * nx + i] at (x0 + i cell, y0 + j cell).
    // Every sphere can touch it (one contact per sphere). Use instead of add_ground().
    void set_heightfield(int nx, int ny, float cell, float x0, float y0, std::vector<float> heights);

    ConstraintView view() const;
    int max_rows() const;

    ConstraintOptions opt;
    std::vector<int> limit_qadr, limit_dadr;
    std::vector<float> limit_lower, limit_upper;
    std::vector<int> sphere_body;
    std::vector<float> sphere_pos, sphere_radius;
    std::vector<float> plane_normal, plane_point;
    std::vector<int> box_body;
    std::vector<float> box_pos, box_quat, box_half;
    std::vector<int> capsule_body;
    std::vector<float> capsule_pos, capsule_quat, capsule_size;
    int hf_nx = 0, hf_ny = 0;
    float hf_cell = 0.0f, hf_x0 = 0.0f, hf_y0 = 0.0f;
    std::vector<float> hf_heights;
    std::vector<float> body_invweight0, dof_invweight0;
    int nbody = 0, nv = 0;
};

// MuJoCo's body_invweight0 and dof_invweight0 (engine_setconst.c, setInertia0):
// at qpos0, A = J M^-1 J^T for each body (J at the body's center of mass) and
// for each joint; the averages of A's diagonal blocks. Computed in double.
void compute_invweight0(const RobotModel& m, std::vector<float>& body_invweight0, std::vector<float>& dof_invweight0);

}  // namespace forge
