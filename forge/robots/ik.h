#pragma once
// Inverse kinematics (Phase 9a, 9b): joint angles that put a body's frame at a
// target position and orientation, plus gravity compensation torques.
//
// ik_step (FORGE_HD: CPU and GPU, the pick task's controller runs it per env):
// one damped-least-squares step on the position and rotation error,
//   dq = J^T (J J^T + lambda I)^-1 e  +  (I - J^T (J J^T + lambda I)^-1 J) z,
//   z = posture_gain (q_posture - q)
// The second part is the posture term: an arm with more joints than the 6 numbers
// of a pose (the Panda has 7) has a free "elbow" motion that does not move the
// hand; z pulls the joints toward a natural posture along that motion only, so
// the hand still goes to the target. Steps of at most max_step, clipped to the
// joint limits (the same method as tools/gen_mujoco_refs.py panda_ik, plus the
// posture term).
#include <vector>

#include "forge/dynamics/articulated.h"
#include "forge/math/hd.h"
#include "forge/math/quat.h"
#include "forge/math/vec3.h"
#include "forge/robots/model.h"

namespace forge {

constexpr int kIkMaxJoints = 8;

struct IkJoints {
    int n = 0;
    int body[kIkMaxJoints] = {};     // body index of each joint (revolute or prismatic, above the target body)
    float lower[kIkMaxJoints] = {};  // limits; lower >= upper = no limit
    float upper[kIkMaxJoints] = {};
    float posture[kIkMaxJoints] = {};  // natural posture (used when posture_gain > 0)
};

struct IkOptions {
    int max_iterations = 300;
    float damping = 1e-4f;       // lambda
    float max_step = 0.1f;       // largest joint change per iteration
    float tolerance = 1e-6f;     // stop when |position error| + |rotation error| is below
    float posture_gain = 0.0f;   // 0 = no posture term (plain damped least squares)
};

struct IkResult {
    int iterations = 0;
    float pos_error = 0.0f;  // m
    float rot_error = 0.0f;  // rad (about)
};

// Cholesky solve of the 6 x 6 symmetric positive definite A x = b (A is overwritten).
FORGE_HD void ik_solve6(float A[36], float b[6]) {
    for (int j = 0; j < 6; ++j) {
        float s = A[j * 6 + j];
        for (int k = 0; k < j; ++k) s -= A[j * 6 + k] * A[j * 6 + k];
        A[j * 6 + j] = sqrtf(fmaxf(s, 1e-20f));
        for (int i = j + 1; i < 6; ++i) {
            float t = A[i * 6 + j];
            for (int k = 0; k < j; ++k) t -= A[i * 6 + k] * A[j * 6 + k];
            A[i * 6 + j] = t / A[j * 6 + j];
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < i; ++k) b[i] -= A[i * 6 + k] * b[k];
        b[i] /= A[i * 6 + i];
    }
    for (int i = 5; i >= 0; --i) {
        for (int k = i + 1; k < 6; ++k) b[i] -= A[k * 6 + i] * b[k];
        b[i] /= A[i * 6 + i];
    }
}

// One IK step. Needs forward_kinematics(m, qpos, w) already run (w.xpos, w.xquat,
// w.S). Moves the joints in qpos; fills the errors before the step into 'res'.
FORGE_HD void ik_step(const ModelView& m, const DynamicsWork& w, int body, Vec3 target_pos, Quat target_rot,
                      const IkJoints& jt, float* qpos, const IkOptions& opt, IkResult& res) {
    const Vec3 p = load3(w.xpos + 3 * body);
    const Mat3 R = quat_to_mat3(load_quat(w.xquat + 4 * body)), R0 = quat_to_mat3(target_rot);
    const Vec3 ep = target_pos - p;
    // rotation error: 0.5 sum_i (current axis i) x (target axis i)
    const Vec3 er = (cross(col(R, 0), col(R0, 0)) + cross(col(R, 1), col(R0, 1)) + cross(col(R, 2), col(R0, 2))) * 0.5f;
    res.pos_error = length(ep);
    res.rot_error = length(er);
    const int n = jt.n;
    float J[6][kIkMaxJoints];
    for (int c = 0; c < n; ++c) {
        const int jb = jt.body[c];
        bool above = false;
        for (int k = body; k >= 0; k = m.parent[k]) above = above || k == jb;
        const SpatialVec& S = w.S[m.dadr[jb]];
        const Vec3 v = above ? S.lin + cross(S.ang, p) : Vec3{0, 0, 0};
        const Vec3 a = above ? S.ang : Vec3{0, 0, 0};
        J[0][c] = v.x, J[1][c] = v.y, J[2][c] = v.z, J[3][c] = a.x, J[4][c] = a.y, J[5][c] = a.z;
    }
    float A[36], L[36];
    for (int r = 0; r < 6; ++r)
        for (int s = 0; s < 6; ++s) {
            float sum = r == s ? opt.damping : 0.0f;
            for (int c = 0; c < n; ++c) sum += J[r][c] * J[s][c];
            A[r * 6 + s] = sum;
        }
    for (int i = 0; i < 36; ++i) L[i] = A[i];
    float y[6] = {ep.x, ep.y, ep.z, er.x, er.y, er.z};
    ik_solve6(L, y);
    float dq[kIkMaxJoints];
    for (int c = 0; c < n; ++c) {
        dq[c] = 0.0f;
        for (int r = 0; r < 6; ++r) dq[c] += J[r][c] * y[r];
    }
    if (opt.posture_gain > 0.0f) {  // + (I - J^T A^-1 J) z
        float z[kIkMaxJoints], u[6];
        for (int c = 0; c < n; ++c) z[c] = opt.posture_gain * (jt.posture[c] - qpos[m.qadr[jt.body[c]]]);
        for (int r = 0; r < 6; ++r) {
            u[r] = 0.0f;
            for (int c = 0; c < n; ++c) u[r] += J[r][c] * z[c];
        }
        for (int i = 0; i < 36; ++i) L[i] = A[i];
        ik_solve6(L, u);
        for (int c = 0; c < n; ++c) {
            float ju = 0.0f;
            for (int r = 0; r < 6; ++r) ju += J[r][c] * u[r];
            dq[c] += z[c] - ju;
        }
    }
    float biggest = 0.0f;
    for (int c = 0; c < n; ++c) biggest = fmaxf(biggest, fabsf(dq[c]));
    const float scale = biggest > opt.max_step ? opt.max_step / biggest : 1.0f;
    for (int c = 0; c < n; ++c) {
        float& q = qpos[m.qadr[jt.body[c]]];
        q += scale * dq[c];
        if (jt.lower[c] < jt.upper[c]) q = fminf(fmaxf(q, jt.lower[c]), jt.upper[c]);
    }
}

// Joint list for ik_step from body indices (limits and posture from the model and 'posture').
IkJoints make_ik_joints(const RobotModel& m, const std::vector<int>& joints, const std::vector<float>& posture = {});

// Repeats ik_step until the error is below opt.tolerance or max_iterations. Host.
IkResult solve_ik(const RobotModel& m, int body, Vec3 target_pos, Quat target_rot, const std::vector<int>& joints,
                  std::vector<float>& qpos, const IkOptions& opt = {}, const std::vector<float>& posture = {});

// Joint torques that hold the robot against gravity at rest (MuJoCo's gravcomp, or
// the real Panda's controller): out[i] = - sum over bodies b moved by dof i of
// m_b g . (velocity of b's center of mass for unit qvel_i). Needs forward_kinematics.
// Only bodies of the tree under 'root' (a root body index); other dofs untouched.
FORGE_HD void gravity_torques(const ModelView& m, const DynamicsWork& w, int root, float* out) {
    for (int b = 0; b < m.nbody; ++b) {
        int r = b;
        while (m.parent[r] >= 0) r = m.parent[r];
        if (r == root)
            for (int k = b; k >= 0; k = m.parent[k])
                for (int i = m.dadr[k]; m.jtype[k] != kJointFixed && i < m.dadr[k] + joint_nv(m.jtype[k]); ++i)
                    out[i] = 0.0f;
    }
    for (int b = 0; b < m.nbody; ++b) {
        int r = b;
        while (m.parent[r] >= 0) r = m.parent[r];
        if (r != root || m.mass[b] <= 0.0f) continue;
        const Vec3 c = load3(w.xpos + 3 * b) + rotate(load_quat(w.xquat + 4 * b), load3(m.com + 3 * b));
        const Vec3 f = m.gravity * m.mass[b];
        for (int k = b; k >= 0; k = m.parent[k]) {
            if (m.jtype[k] == kJointFixed) continue;
            for (int i = m.dadr[k]; i < m.dadr[k] + joint_nv(m.jtype[k]); ++i) {
                const SpatialVec& S = w.S[i];
                out[i] -= dot(f, S.lin + cross(S.ang, c));
            }
        }
    }
}

}  // namespace forge
