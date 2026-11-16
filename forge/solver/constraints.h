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
//   contact between geom 1 on body b1 and geom 2 on body b2 (-1 = the world),
//   pos = signed distance <= 0, n = normal from geom 1 to geom 2 (MuJoCo), so
//   J_n = (J_b2 - J_b1) . n at the contact point is the separation speed.
//   Pyramidal friction cone:
//     4 rows  J_n + mu J_t1,  J_n - mu J_t1,  J_n + mu J_t2,  J_n - mu J_t2
//     t1 t2 from MuJoCo's mju_makeFrame(n). All 4 rows share
//     R = 2 mu^2 R0 / impratio (MuJoCo's pyramidal R), R0 from diagA = tran (1 + mu^2),
//     tran = body_invweight0 translation of b1 + of b2 (the world adds 0)
//     friction == 0: one frictionless row J_n instead
//   Contact kinds and order (the generator sorts MuJoCo's contacts the same way):
//     per sphere s: each plane, the heightfield, each box on another body
//                   (not its parent or child; MuJoCo filters parent-child pairs)
//     then per box k: each plane, up to 4 corners in the plane (mjc_PlaneBox order)
//   sphere-plane: point = center - n (dist / 2 + radius)
//   sphere-box (mjc_SphereBox): geom 1 = sphere, n = from the sphere into the box:
//     minus the outward direction from the box's nearest point to the center, or,
//     with the center inside, minus the normal of the face nearest the center;
//     point = center + n (dist / 2 + radius)
//   plane-box: one contact per box corner below the plane, point = corner - n dist / 2
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

// Adds scale * u . (velocity of p for unit qvel_i) to out[i] for the dofs that move body b.
FORGE_HD void point_jacobian_add(const ModelView& m, const DynamicsWork& w, int b, Vec3 p, Vec3 u, float scale,
                                 float* out) {
    for (int k = b; k >= 0; k = m.parent[k]) {
        if (m.jtype[k] == kJointFixed) continue;
        const int d0 = m.dadr[k], dn = joint_nv(m.jtype[k]);
        for (int i = d0; i < d0 + dn; ++i) {
            const SpatialVec& S = w.S[i];
            out[i] += scale * dot(u, S.lin + cross(S.ang, p));
        }
    }
}

// Sphere (center c, radius r) vs box (center bc, axes R, half sizes h), MuJoCo
// mjc_SphereBox: dist = signed distance, n_out = unit direction from the box to
// the sphere. Returns false when they do not touch (dist > 0).
FORGE_HD bool sphere_box(Vec3 c, float r, Vec3 bc, const Mat3& R, Vec3 h, float& dist, Vec3& n_out) {
    const Vec3 p = transpose(R) * (c - bc);  // sphere center in the box frame
    const Vec3 q{fminf(fmaxf(p.x, -h.x), h.x), fminf(fmaxf(p.y, -h.y), h.y), fminf(fmaxf(p.z, -h.z), h.z)};
    const Vec3 d = p - q;
    const float len = length(d);
    Vec3 nl;
    if (len > 0.0f) {  // center outside: nearest point on the box
        dist = len - r;
        if (dist > 0.0f) return false;
        nl = d / len;
    } else {  // center inside: out through the nearest face
        const float gx = h.x - fabsf(p.x), gy = h.y - fabsf(p.y), gz = h.z - fabsf(p.z);
        float g = gx;
        nl = Vec3{p.x < 0.0f ? -1.0f : 1.0f, 0.0f, 0.0f};
        if (gy < g) {
            g = gy;
            nl = Vec3{0.0f, p.y < 0.0f ? -1.0f : 1.0f, 0.0f};
        }
        if (gz < g) {
            g = gz;
            nl = Vec3{0.0f, 0.0f, p.z < 0.0f ? -1.0f : 1.0f};
        }
        dist = -g - r;
    }
    n_out = R * nl;
    return true;
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

    // ---- contacts ----
    const bool frictional = c.friction > 0.0f;
    const int nrow = frictional ? 4 : 1;
    // Rows of one contact: geom 1 on body b1, geom 2 on body b2 (-1 = world), normal n
    // from geom 1 to geom 2, at 'point', signed distance 'dist'.
    auto contact = [&](int b1, int b2, Vec3 point, Vec3 n, float dist, int id) {
        if (dist > 0.0f || r + nrow > c.max_rows) return;  // MuJoCo: contact when dist <= margin (0)
        const float imp = constraint_impedance(c.solimp, dist);
        const float tran = (b1 >= 0 ? c.body_invweight0[2 * b1] : 0.0f) + (b2 >= 0 ? c.body_invweight0[2 * b2] : 0.0f);
        // (J_b2 - J_b1) . u into 'out'
        auto jrow = [&](Vec3 u, float* out) {
            for (int i = 0; i < nv; ++i) out[i] = 0.0f;
            if (b2 >= 0) point_jacobian_add(m, w, b2, point, u, 1.0f, out);
            if (b1 >= 0) point_jacobian_add(m, w, b1, point, u, -1.0f, out);
        };
        float* J0 = cw.J + r * nv;
        if (!frictional) {
            jrow(n, J0);
            cw.R[r] = fmaxf(1e-15f, (1.0f - imp) * tran / imp);
        } else {
            Vec3 t1, t2;
            contact_frame(n, t1, t2);
            // Build J_n, J_t1, J_t2 in rows r, r+2, r+3, then mix them into the 4 pyramid rows.
            float* Jn = cw.J + (r + 0) * nv;
            float* Ja = cw.J + (r + 2) * nv;  // t1 for now
            float* Jb = cw.J + (r + 3) * nv;  // t2 for now
            jrow(n, Jn);
            jrow(t1, Ja);
            jrow(t2, Jb);
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
            cw.id[r + k] = id;
            cw.pos[r + k] = dist;
            cw.D[r + k] = 1.0f / cw.R[r + k];
            cw.aref[r + k] = -c.B * vel - c.K * imp * dist;
        }
        r += nrow;
    };
    // Sphere vs a plane (n, plane_point): plane = geom 1 (the world).
    auto sphere_plane = [&](int b, Vec3 center, float radius, Vec3 n, Vec3 plane_point, int id) {
        const float dist = dot(center - plane_point, n) - radius;
        if (dist > 0.0f) return;
        contact(-1, b, center - n * (0.5f * dist + radius), n, dist, id);  // midway in the overlap (MuJoCo)
    };
    const int id_sphere_box = c.nsphere * c.nplane + c.nsphere;
    const int id_box_plane = id_sphere_box + c.nsphere * c.nbox;
    for (int s = 0; s < c.nsphere; ++s) {
        const int b = c.sphere_body[s];
        const Mat3 Rb = quat_to_mat3(load_quat(w.xquat + 4 * b));
        const Vec3 center = load3(w.xpos + 3 * b) + Rb * load3(c.sphere_pos + 3 * s);
        const float radius = c.sphere_radius[s];
        for (int p = 0; p < c.nplane; ++p)
            sphere_plane(b, center, radius, load3(c.plane_normal + 3 * p), load3(c.plane_point + 3 * p), s * c.nplane + p);
        if (c.hfield.h) {  // the plane of the terrain triangle under the sphere's center
            Vec3 hp, hn;
            hfield_plane(c.hfield, center.x, center.y, hp, hn);
            sphere_plane(b, center, radius, hn, hp, c.nsphere * c.nplane + s);
        }
        for (int k = 0; k < c.nbox; ++k) {
            const int bb = c.box_body[k];
            if (bb == b || m.parent[bb] == b || m.parent[b] == bb) continue;
            const Quat qb = load_quat(w.xquat + 4 * bb);
            const Vec3 bc = load3(w.xpos + 3 * bb) + rotate(qb, load3(c.box_pos + 3 * k));
            const Mat3 R = quat_to_mat3(qb * load_quat(c.box_quat + 4 * k));
            float dist;
            Vec3 n_out;
            if (!sphere_box(center, radius, bc, R, load3(c.box_half + 3 * k), dist, n_out)) continue;
            const Vec3 n = -n_out;  // sphere (geom 1) -> box (geom 2)
            contact(b, bb, center + n * (0.5f * dist + radius), n, dist, id_sphere_box + s * c.nbox + k);
        }
    }
    for (int k = 0; k < c.nbox; ++k) {
        const int bb = c.box_body[k];
        const Quat qb = load_quat(w.xquat + 4 * bb);
        const Vec3 bc = load3(w.xpos + 3 * bb) + rotate(qb, load3(c.box_pos + 3 * k));
        const Mat3 R = quat_to_mat3(qb * load_quat(c.box_quat + 4 * k));
        const Vec3 h = load3(c.box_half + 3 * k);
        for (int p = 0; p < c.nplane; ++p) {
            const Vec3 n = load3(c.plane_normal + 3 * p), pp = load3(c.plane_point + 3 * p);
            int found = 0;
            for (int i = 0; i < 8 && found < 4; ++i) {
                const Vec3 corner = bc + R * Vec3{i & 1 ? h.x : -h.x, i & 2 ? h.y : -h.y, i & 4 ? h.z : -h.z};
                const float dist = dot(corner - pp, n);
                if (dist > 0.0f) continue;
                contact(-1, bb, corner - n * (0.5f * dist), n, dist, id_box_plane + (k * c.nplane + p) * 8 + i);
                ++found;
            }
        }
    }
    cw.nefc = r;
}

}  // namespace forge
