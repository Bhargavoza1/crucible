#pragma once
// Constraint rows, MuJoCo's soft model (doc 3.4). FORGE_HD: CPU and GPU.
//
// Each row i is one scalar constraint with a Jacobian row J_i (nv numbers),
// a distance pos_i (negative = penetration) and these MuJoCo quantities
// (engine_core_constraint.c), so our numbers can be checked against MuJoCo's:
//
//   imp   impedance in (0, 1), from solimp and pos   (getimpedance)
//   R     regularizer = (1 - imp) / imp * diagA      (mj_makeImpedance)
//         diagA = approximate inverse inertia of the row, from invweight0
//   D     = 1 / R
//   aref  reference acceleration = -B * (J qvel) - K * imp * pos
//
// Rows (MuJoCo order: limits first, then contacts):
//   joint limit, lower side: pos = q - lower,  J = +1 at the joint's dof
//   joint limit, upper side: pos = upper - q,  J = -1
//     (a row exists only while pos < 0)
//   sphere-plane contact (pos = signed distance <= 0), pyramidal friction cone:
//     4 rows  J_n + mu J_t1,  J_n - mu J_t1,  J_n + mu J_t2,  J_n - mu J_t2
//     n = plane normal, t1 t2 from MuJoCo's mju_makeFrame. All 4 rows share
//     R = 2 mu^2 R0 / impratio (MuJoCo's pyramidal R), R0 from diagA = tran (1 + mu^2)
//     friction == 0: one frictionless row J_n instead
//
// The solver (newton.h) finds the forces f_i >= 0 from these rows.
#include "forge/dynamics/articulated.h"
#include "forge/math/hd.h"
#include "forge/robots/model.h"
#include "forge/solver/constraint_model.h"

namespace forge {

enum ConstraintRowType : int { kRowLimit = 0, kRowContact = 1 };

// Scratch and results for one env. Sized by ConstraintView::max_rows and nv.
struct ConstraintWork {
    int nefc;          // rows in use
    int iterations;    // Newton iterations of the last solve
    int* type;         // [max_rows] ConstraintRowType
    int* id;           // [max_rows] limit index, or sphere * nplane + plane
    float* J;          // [max_rows * nv]
    float* pos;        // [max_rows]
    float* aref;       // [max_rows]
    float* R;          // [max_rows]
    float* D;          // [max_rows]
    float* jar;        // [max_rows] J qacc - aref
    float* Jd;         // [max_rows] J * search direction
    float* force;      // [max_rows] result, >= 0
    int* state;        // [max_rows] 1 = active (jar < 0)
    float* qfrc_smooth;  // [nv] tau + actuators - bias
    float* qacc_smooth;  // [nv] M^-1 qfrc_smooth (no constraints)
    float* H;          // [nv * nv] Newton Hessian
    float* LH;         // [nv * nv] its Cholesky factor
    float* grad;       // [nv]
    float* dir;        // [nv]
    float* Ma;         // [nv]
    float* Mdir;       // [nv]
};

// MuJoCo getimpedance (solimp = dmin, dmax, width, mid, power; margin = 0).
FORGE_HD float constraint_impedance(const float* solimp, float pos) {
    const float dmin = solimp[0], dmax = solimp[1], width = solimp[2], mid = solimp[3], p = solimp[4];
    if (dmin == dmax || width <= 1e-15f) return 0.5f * (dmin + dmax);
    const float x = fabsf(pos / width);
    if (x >= 1.0f || x <= 0.0f) return x >= 1.0f ? dmax : dmin;
    float y;
    if (p == 1.0f) {
        y = x;
    } else if (x <= mid) {
        y = powf(x, p) / powf(mid, p - 1.0f);
    } else {
        y = 1.0f - powf(1.0f - x, p) / powf(1.0f - mid, p - 1.0f);
    }
    return dmin + y * (dmax - dmin);
}

// MuJoCo mju_makeFrame: tangents t1, t2 for a unit normal n.
FORGE_HD void contact_frame(Vec3 n, Vec3& t1, Vec3& t2) {
    Vec3 y = fabsf(n.y) < 0.5f ? Vec3{0, 1, 0} : Vec3{0, 0, 1};
    y = y - n * dot(n, y);
    t1 = y / length(y);
    t2 = cross(n, t1);
}

// Jacobian row of a point p (world) on body b, along direction u:
//   out[i] = u . (velocity of p for unit qvel_i)   for every dof i.
FORGE_HD void point_jacobian_row(const ModelView& m, const DynamicsWork& w, int b, Vec3 p, Vec3 u, float* out) {
    for (int i = 0; i < m.nv; ++i) out[i] = 0.0f;
    for (int k = b; k >= 0; k = m.parent[k]) {
        if (m.jtype[k] == kJointFixed) continue;
        const int d0 = m.dadr[k], dn = joint_nv(m.jtype[k]);
        for (int i = d0; i < d0 + dn; ++i) {
            const SpatialVec& S = w.S[i];
            out[i] = dot(u, S.lin + cross(S.ang, p));
        }
    }
}

// Builds all rows for the current state. Needs forward_kinematics already run
// (w.xpos, w.xquat, w.S). qvel gives the row velocity for aref.
FORGE_HD void make_constraints(const ModelView& m, const ConstraintView& c, const float* qpos, const float* qvel,
                               const DynamicsWork& w, ConstraintWork& cw) {
    const int nv = m.nv;
    int r = 0;

    // ---- joint limits ----
    for (int l = 0; l < c.nlimit; ++l) {
        const float q = qpos[c.limit_qadr[l]];
        for (int side = -1; side <= 1; side += 2) {
            const float dist = side < 0 ? q - c.limit_lower[l] : c.limit_upper[l] - q;
            if (!(dist < 0.0f) || r >= c.max_rows) continue;
            float* J = cw.J + r * nv;
            for (int i = 0; i < nv; ++i) J[i] = 0.0f;
            J[c.limit_dadr[l]] = side < 0 ? 1.0f : -1.0f;
            const float imp = constraint_impedance(c.solimp, dist);
            const float diagA = c.dof_invweight0[c.limit_dadr[l]];
            const float R = fmaxf(1e-15f, (1.0f - imp) * diagA / imp);
            cw.type[r] = kRowLimit;
            cw.id[r] = l;
            cw.pos[r] = dist;
            cw.R[r] = R;
            cw.D[r] = 1.0f / R;
            const float vel = J[c.limit_dadr[l]] * qvel[c.limit_dadr[l]];
            cw.aref[r] = -c.B * vel - c.K * imp * dist;
            ++r;
        }
    }

    // ---- sphere-plane contacts ----
    const bool frictional = c.friction > 0.0f;
    const int nrow = frictional ? 4 : 1;
    for (int s = 0; s < c.nsphere; ++s) {
        const int b = c.sphere_body[s];
        const Mat3 Rb = quat_to_mat3(load_quat(w.xquat + 4 * b));
        const Vec3 center = load3(w.xpos + 3 * b) + Rb * load3(c.sphere_pos + 3 * s);
        const float radius = c.sphere_radius[s];
        for (int p = 0; p < c.nplane; ++p) {
            const Vec3 n = load3(c.plane_normal + 3 * p);
            const float dist = dot(center - load3(c.plane_point + 3 * p), n) - radius;
            if (dist > 0.0f || r + nrow > c.max_rows) continue;  // MuJoCo: contact when dist <= margin (0)
            const Vec3 point = center - n * (0.5f * dist + radius);  // midway in the overlap (MuJoCo)
            const float imp = constraint_impedance(c.solimp, dist);
            const float tran = c.body_invweight0[2 * b];  // the plane (world) adds 0
            float* J0 = cw.J + r * nv;
            if (!frictional) {
                point_jacobian_row(m, w, b, point, n, J0);
                cw.R[r] = fmaxf(1e-15f, (1.0f - imp) * tran / imp);
            } else {
                Vec3 t1, t2;
                contact_frame(n, t1, t2);
                // Build J_n, J_t1, J_t2 in rows r, r+2, r+3, then mix them into the 4 pyramid rows.
                float* Jn = cw.J + (r + 0) * nv;
                float* Ja = cw.J + (r + 2) * nv;  // t1 for now
                float* Jb = cw.J + (r + 3) * nv;  // t2 for now
                point_jacobian_row(m, w, b, point, n, Jn);
                point_jacobian_row(m, w, b, point, t1, Ja);
                point_jacobian_row(m, w, b, point, t2, Jb);
                const float mu = c.friction;
                float* J1 = cw.J + (r + 1) * nv;
                for (int i = 0; i < nv; ++i) {
                    const float jn = Jn[i], ja = Ja[i], jb = Jb[i];
                    Jn[i] = jn + mu * ja;  // row r:   n + mu t1
                    J1[i] = jn - mu * ja;  // row r+1: n - mu t1
                    Ja[i] = jn + mu * jb;  // row r+2: n + mu t2
                    Jb[i] = jn - mu * jb;  // row r+3: n - mu t2
                }
                const float diagA = tran + mu * mu * tran;
                const float R0 = fmaxf(1e-15f, (1.0f - imp) * diagA / imp);
                const float R1 = R0 / fmaxf(1e-15f, c.impratio);
                const float mu_reg = mu * sqrtf(R1 / R0);
                const float Rpy = 2.0f * mu_reg * mu_reg * R0;
                for (int k = 0; k < 4; ++k) cw.R[r + k] = Rpy;
            }
            for (int k = 0; k < nrow; ++k) {
                const float* Jk = cw.J + (r + k) * nv;
                float vel = 0.0f;
                for (int i = 0; i < nv; ++i) vel += Jk[i] * qvel[i];
                cw.type[r + k] = kRowContact;
                cw.id[r + k] = s * c.nplane + p;
                cw.pos[r + k] = dist;
                cw.D[r + k] = 1.0f / cw.R[r + k];
                cw.aref[r + k] = -c.B * vel - c.K * imp * dist;
            }
            r += nrow;
        }
    }
    cw.nefc = r;
}

}  // namespace forge
