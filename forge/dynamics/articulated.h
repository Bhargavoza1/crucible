#pragma once
// Rigid-body tree dynamics in joint coordinates (doc 3.3, 3.5). FORGE_HD: the
// same code runs on the CPU (answer key, checked against MuJoCo) and on the GPU.
//
// One step of forward dynamics:
//   1. forward_kinematics  qpos -> body poses (xpos, xquat) and the motion
//                          vector S of every dof (world frame, world origin)
//   2. body_inertias       spatial inertia of every body in the world frame
//   3. mass_matrix         M (nv x nv) with CRBA: composite inertia of each
//                          subtree, M[i][j] = S_i . (Ic S_j)
//   4. bias_forces         c(q, qvel) = Coriolis + centrifugal + gravity, with
//                          RNEA at zero joint acceleration
//   5. solve               qacc = M^-1 (tau - c)     (Cholesky, M is SPD)
//   6. integrate_euler     qvel += dt qacc, then qpos from the NEW qvel
//                          (semi-implicit, same as MuJoCo's Euler)
// Equation of motion:  M(q) qacc + c(q, qvel) = tau.
#include "forge/dynamics/spatial.h"
#include "forge/math/hd.h"
#include "forge/math/mat3.h"
#include "forge/math/quat.h"
#include "forge/robots/model.h"

namespace forge {

// Scratch and results for one robot (one env). All pointers sized from the model.
struct DynamicsWork {
    float* xpos;          // [3 * nbody] body frame origin, world
    float* xquat;         // [4 * nbody] body orientation, Quat order x y z w
    SpatialVec* S;        // [nv] motion vector of each dof
    SpatialInertia* I;    // [nbody] body inertia, world frame
    SpatialInertia* Ic;   // [nbody] composite inertia (body + subtree)
    SpatialVec* V;        // [nbody] body velocity
    SpatialVec* A;        // [nbody] body acceleration (bias part)
    SpatialVec* F;        // [nbody] force on body from its parent joint (RNEA)
    float* M;             // [nv * nv] mass matrix, row major
    float* L;             // [nv * nv] Cholesky factor (lower triangle used)
    float* bias;          // [nv]
};

// ---- small helpers ---------------------------------------------------------------
FORGE_HD Vec3 load3(const float* p) { return {p[0], p[1], p[2]}; }
FORGE_HD void store3(float* p, Vec3 v) {
    p[0] = v.x;
    p[1] = v.y;
    p[2] = v.z;
}
FORGE_HD Quat load_quat(const float* p) { return {p[0], p[1], p[2], p[3]}; }  // x y z w
FORGE_HD void store_quat(float* p, Quat q) {
    p[0] = q.x;
    p[1] = q.y;
    p[2] = q.z;
    p[3] = q.w;
}
// MuJoCo qpos quaternions are w x y z.
FORGE_HD Quat load_quat_wxyz(const float* p) { return {p[1], p[2], p[3], p[0]}; }
FORGE_HD void store_quat_wxyz(float* p, Quat q) {
    p[0] = q.w;
    p[1] = q.x;
    p[2] = q.y;
    p[3] = q.z;
}
FORGE_HD Mat3 load_mat3(const float* p) {
    Mat3 m;
    for (int i = 0; i < 9; ++i) m.m[i] = p[i];
    return m;
}

// ---- 1. forward kinematics + motion vectors ----------------------------------------
FORGE_HD void forward_kinematics(const ModelView& m, const float* qpos, DynamicsWork& w) {
    for (int b = 0; b < m.nbody; ++b) {
        const int p = m.parent[b];
        const int type = m.jtype[b];
        Vec3 pos;
        Quat rot;
        if (type == kJointFree) {
            // The free joint IS the body pose (MuJoCo ignores body_pos for it).
            const float* q = qpos + m.qadr[b];
            pos = load3(q);
            rot = normalize(load_quat_wxyz(q + 3));
            const Mat3 R = quat_to_mat3(rot);
            const int d = m.dadr[b];
            for (int k = 0; k < 3; ++k) {
                const Vec3 e{k == 0 ? 1.0f : 0.0f, k == 1 ? 1.0f : 0.0f, k == 2 ? 1.0f : 0.0f};
                w.S[d + k] = {{0, 0, 0}, e};          // translation along world axis k
                const Vec3 r = col(R, k);             // body axis k, in world
                w.S[d + 3 + k] = {r, cross(pos, r)};  // rotation about body axis k, through the body origin
            }
        } else {
            const Vec3 pp = p < 0 ? Vec3{0, 0, 0} : load3(w.xpos + 3 * p);
            const Quat pq = p < 0 ? quat_identity() : load_quat(w.xquat + 4 * p);
            pos = pp + rotate(pq, load3(m.body_pos + 3 * b));
            rot = pq * load_quat(m.body_quat + 4 * b);
            if (type == kJointRevolute || type == kJointPrismatic) {
                const Vec3 axis_body = load3(m.jaxis + 3 * b);
                const Vec3 a = rotate(rot, axis_body);  // joint axis, world (before the joint moves)
                const float q = qpos[m.qadr[b]];
                const int d = m.dadr[b];
                if (type == kJointRevolute) {
                    w.S[d] = {a, cross(pos, a)};  // rotation about axis a through the joint (body origin)
                    rot = rot * quat_from_axis_angle(axis_body, q);
                } else {
                    w.S[d] = {{0, 0, 0}, a};  // translation along a
                    pos = pos + a * q;
                }
            }
        }
        store3(w.xpos + 3 * b, pos);
        store_quat(w.xquat + 4 * b, rot);
    }
}

// ---- 2. body inertias in the world frame -------------------------------------------
FORGE_HD void body_inertias(const ModelView& m, DynamicsWork& w) {
    for (int b = 0; b < m.nbody; ++b) {
        const Mat3 R = quat_to_mat3(load_quat(w.xquat + 4 * b));
        const Vec3 c = load3(w.xpos + 3 * b) + R * load3(m.com + 3 * b);
        const Mat3 Ic = rotate_inertia(R, load_mat3(m.inertia + 9 * b));
        w.I[b] = body_inertia(m.mass[b], c, Ic);
    }
}

// ---- 3. mass matrix (CRBA) ---------------------------------------------------------
FORGE_HD void mass_matrix(const ModelView& m, DynamicsWork& w) {
    for (int b = 0; b < m.nbody; ++b) w.Ic[b] = w.I[b];
    for (int b = m.nbody - 1; b >= 0; --b)  // children come after parents: add each subtree up
        if (m.parent[b] >= 0) w.Ic[m.parent[b]] = w.Ic[m.parent[b]] + w.Ic[b];

    const int nv = m.nv;
    // Dofs on different branches (two legs) never meet below, so their entries
    // stay 0. Set every entry here: never trust the memory to start at 0
    // (shared memory and cudaMalloc are not zeroed).
    for (int k = 0; k < nv * nv; ++k) w.M[k] = 0.0f;
    for (int j = 0; j < nv; ++j) {
        const int bj = m.dof_body[j];
        const SpatialVec F = w.Ic[bj] * w.S[j];  // force needed to move dof j's subtree with unit qacc_j
        // dof j only moves bodies in its subtree; dofs of bj and its ancestors feel F.
        for (int k = bj; k >= 0; k = m.parent[k]) {
            if (m.jtype[k] == kJointFixed) continue;
            const int d0 = m.dadr[k], dn = joint_nv(m.jtype[k]);
            for (int i = d0; i < d0 + dn; ++i) {
                const float v = dot(w.S[i], F);
                w.M[i * nv + j] = v;
                w.M[j * nv + i] = v;
            }
        }
    }
    if (m.dof_armature)  // motor rotor inertia, as MuJoCo
        for (int i = 0; i < nv; ++i) w.M[i * nv + i] += m.dof_armature[i];
}

// Passive joint forces: f -= damping * qvel (MuJoCo's joint damping; MuJoCo's
// Euler integrator would treat it implicitly unless eulerdamp is disabled).
FORGE_HD void add_passive_forces(const ModelView& m, const float* qvel, float* f) {
    if (!m.dof_damping) return;
    for (int i = 0; i < m.nv; ++i) f[i] -= m.dof_damping[i] * qvel[i];
}

// ---- 4. bias forces (RNEA with qacc = 0) -------------------------------------------
FORGE_HD void bias_forces(const ModelView& m, const float* qvel, DynamicsWork& w) {
    // Gravity as an upward acceleration of the base: same forces, simpler code.
    const SpatialVec a_base{{0, 0, 0}, {-m.gravity.x, -m.gravity.y, -m.gravity.z}};
    for (int b = 0; b < m.nbody; ++b) {
        const int p = m.parent[b];
        const SpatialVec Vp = p < 0 ? spatial_zero() : w.V[p];
        const SpatialVec Ap = p < 0 ? a_base : w.A[p];
        SpatialVec V = Vp, A = Ap;
        const int type = m.jtype[b];
        if (type == kJointFree) {
            const int d = m.dadr[b];
            for (int k = 0; k < 6; ++k) V += w.S[d + k] * qvel[d + k];
            // Velocity-product term of a free body: with qacc = 0 the body origin
            // has no acceleration and the body-frame angular velocity is constant,
            // which gives [0; v_origin x w] about the world origin.
            const Vec3 v_origin{qvel[d], qvel[d + 1], qvel[d + 2]};
            A += SpatialVec{{0, 0, 0}, cross(v_origin, V.ang)};
        } else if (type != kJointFixed) {
            const int d = m.dadr[b];
            const SpatialVec Sq = w.S[d] * qvel[d];
            V += Sq;
            A += cross_motion(Vp, Sq);  // dS/dt * qvel: the axis moves with the parent
        }
        w.V[b] = V;
        w.A[b] = A;
        w.F[b] = w.I[b] * A + cross_force(V, w.I[b] * V);  // Newton-Euler for this body
    }
    for (int b = m.nbody - 1; b >= 0; --b)  // pass forces from children to parents
        if (m.parent[b] >= 0) w.F[m.parent[b]] += w.F[b];
    for (int i = 0; i < m.nv; ++i) w.bias[i] = dot(w.S[i], w.F[m.dof_body[i]]);
}

// ---- 5. solve M x = rhs (Cholesky, M symmetric positive definite) ----------------------
// Returns false if M is not positive definite (a model error: zero mass or inertia).
// Profile (skyline) Cholesky: row i of L is 0 left of row i's first nonzero in M (f),
// so the sums start there. Exact (the skipped terms are 0), and much less work for
// models of several trees (two arms + free cubes: M is block diagonal).
FORGE_HD bool cholesky_solve(const float* M, int n, const float* rhs, float* x, float* L) {
    for (int i = 0; i < n; ++i) {
        int f = 0;
        while (f < i && M[i * n + f] == 0.0f) ++f;
        for (int j = 0; j < f; ++j) L[i * n + j] = 0.0f;
        for (int j = f; j <= i; ++j) {
            float s = M[i * n + j];
            for (int k = f; k < j; ++k) s -= L[i * n + k] * L[j * n + k];
            if (i == j) {
                if (s <= 0.0f) return false;
                L[i * n + i] = sqrtf(s);
            } else {
                L[i * n + j] = s / L[j * n + j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {  // L y = rhs
        float s = rhs[i];
        for (int k = 0; k < i; ++k) s -= L[i * n + k] * x[k];
        x[i] = s / L[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {  // L^T x = y
        float s = x[i];
        for (int k = i + 1; k < n; ++k) s -= L[k * n + i] * x[k];
        x[i] = s / L[i * n + i];
    }
    return true;
}

// Steps 1-5. tau may be null (no applied force). Returns false if M is not SPD.
FORGE_HD bool forward_dynamics(const ModelView& m, const float* qpos, const float* qvel, const float* tau,
                               DynamicsWork& w, float* qacc) {
    forward_kinematics(m, qpos, w);
    body_inertias(m, w);
    mass_matrix(m, w);
    bias_forces(m, qvel, w);
    for (int i = 0; i < m.nv; ++i) qacc[i] = (tau ? tau[i] : 0.0f) - w.bias[i];
    add_passive_forces(m, qvel, qacc);
    return cholesky_solve(w.M, m.nv, qacc, qacc, w.L);  // in place: rhs is read before x is written
}

// ---- 6. semi-implicit Euler (MuJoCo's Euler integrator) ----------------------------
FORGE_HD void integrate_euler(const ModelView& m, float* qpos, float* qvel, const float* qacc, float dt) {
    for (int i = 0; i < m.nv; ++i) qvel[i] += dt * qacc[i];
    for (int b = 0; b < m.nbody; ++b) {
        const int type = m.jtype[b];
        if (type == kJointFixed) continue;
        float* q = qpos + m.qadr[b];
        const float* v = qvel + m.dadr[b];
        if (type == kJointFree) {
            q[0] += dt * v[0];
            q[1] += dt * v[1];
            q[2] += dt * v[2];
            // Exact rotation by the body-frame angular velocity over dt (exponential
            // map, like MuJoCo's mju_quatIntegrate): q = q * exp(w dt / 2).
            const Vec3 wb{v[3], v[4], v[5]};
            const float speed = length(wb);
            Quat rot = load_quat_wxyz(q + 3);
            if (speed > 0.0f) rot = rot * quat_from_axis_angle(wb / speed, speed * dt);
            store_quat_wxyz(q + 3, normalize(rot));
        } else {
            q[0] += dt * v[0];
        }
    }
}

}  // namespace forge
