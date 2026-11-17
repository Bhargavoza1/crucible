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
//     then per capsule (below)
//   sphere-plane: point = center - n (dist / 2 + radius)
//   sphere-box (mjc_SphereBox): geom 1 = sphere, n = from the sphere into the box:
//     minus the outward direction from the box's nearest point to the center, or,
//     with the center inside, minus the normal of the face nearest the center;
//     point = center + n (dist / 2 + radius)
//   plane-box: one contact per box corner below the plane, point = corner - n dist / 2
//   then (Phase 10) per capsule c (also on bodies welded to the world, which only
//     touch boxes): each plane (2 contacts, the +axis end then the
//     -axis end: sphere-plane at each end, contact frame t1 = the axis, as
//     mjc_PlaneCapsule), then each box on another body (mjc_CapsuleBox: up to 2
//     points on the segment, then sphere-box at each; geom 1 = the capsule)
//   then (Phase 10b) per box pair k < j on different bodies (not parent and child):
//     mjraw_BoxBox, up to 8 contacts, geom 1 = box k, n from box k to box j
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

// mju_makeFrame with a given first tangent (MuJoCo's capsule-plane contacts use
// the capsule axis): t1 = the hint made orthogonal to n, t2 = n x t1. A hint
// shorter than 0.5 is ignored (the default frame above).
FORGE_HD void contact_frame(Vec3 n, Vec3 hint, Vec3& t1, Vec3& t2) {
    if (dot(hint, hint) < 0.25f) {
        contact_frame(n, t1, t2);
        return;
    }
    const Vec3 y = hint - n * dot(n, hint);
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

// Capsule vs box, MuJoCo mjraw_CapsuleBox (engine_collision_box.c, Apache-2.0,
// Copyright DeepMind Technologies Limited), ported to float. Capsule: center pc,
// unit axis a (world), half length hl; box: center bc, axes R, half sizes h. It
// finds the point of the capsule's segment closest to the box (from the faces and
// the 12 edges) and, when the capsule lies along a face or an edge, a second point;
// writes their segment positions (in units of hl, -1 .. 1, from the center along a)
// to s[0], s[1] and returns how many (0, 1 or 2). The caller collides a sphere of
// the capsule's radius at each point with the box (mjraw_SphereBox), as MuJoCo.
// cut: MuJoCo's first distance bound, (margin + 2 (r + hl + hx + hy + hz))^2.
FORGE_HD int capsule_box_points(Vec3 pc, Vec3 a, float hl, float r, Vec3 bc, const Mat3& R, Vec3 h, float* s) {
    const float size2[3] = {h.x, h.y, h.z};
    const Vec3 pl = transpose(R) * (pc - bc);
    const Vec3 al = transpose(R) * a;
    const float pos[3] = {pl.x, pl.y, pl.z}, axis[3] = {al.x, al.y, al.z};
    const float halfaxis[3] = {axis[0] * hl, axis[1] * hl, axis[2] * hl};
    int axisdir = 0;
    if (halfaxis[0] > 0) axisdir += 1;
    if (halfaxis[1] > 0) axisdir += 2;
    if (halfaxis[2] > 0) axisdir += 4;
    float bestdistmax = 2.0f * (r + hl + h.x + h.y + h.z);
    bestdistmax *= bestdistmax;
    float bestdist = bestdistmax, bestsegmentpos = 0.0f, bestboxpos = 0.0f, secondpos = -4.0f;
    int cltype = -4, clface = -1, clcorner = 0, cledge = 0;
    // a face of the box closest to one end of the capsule?
    for (int i = -1; i <= 1; i += 2) {
        float t1[3], t2[3];
        int c1 = 0, c2 = -1;
        for (int j = 0; j < 3; ++j) {
            t1[j] = pos[j] + halfaxis[j] * float(i);
            t2[j] = t1[j];
            if (t1[j] < -size2[j]) {
                ++c1;
                c2 = j;
                t1[j] = -size2[j];
            } else if (t1[j] > size2[j]) {
                ++c1;
                c2 = j;
                t1[j] = size2[j];
            }
        }
        if (c1 > 1) continue;
        const float dx = t1[0] - t2[0], dy = t1[1] - t2[1], dz = t1[2] - t2[2];
        const float dist = dx * dx + dy * dy + dz * dz;
        if (dist < bestdist) {
            bestdist = dist;
            bestsegmentpos = float(i);
            cltype = -2 + i;
            clface = c2;
        }
    }
    // the 12 edges of the box
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 8; ++i) {
            if (i & (1 << j)) continue;
            float t3[3] = {((i & 1) ? 1.0f : -1.0f) * size2[0], ((i & 2) ? 1.0f : -1.0f) * size2[1],
                           ((i & 4) ? 1.0f : -1.0f) * size2[2]};
            t3[j] = 0.0f;
            float dif[3] = {t3[0] - pos[0], t3[1] - pos[1], t3[2] - pos[2]};
            const float ma = size2[j] * size2[j], mb = -size2[j] * halfaxis[j], mc = hl * hl;
            const float u = -size2[j] * dif[j];
            const float v = halfaxis[0] * dif[0] + halfaxis[1] * dif[1] + halfaxis[2] * dif[2];
            const float det = ma * mc - mb * mb;
            if (fabsf(det) < 1e-15f) continue;
            const float idet = 1.0f / det;
            float x1 = (mc * u - mb * v) * idet, x2 = (ma * v - mb * u) * idet;
            int s1 = 1, s2 = 1;
            if (x1 > 1) {
                x1 = 1, s1 = 2;
                x2 = (v - mb) * (1.0f / mc);
            } else if (x1 < -1) {
                x1 = -1, s1 = 0;
                x2 = (v + mb) * (1.0f / mc);
            }
            if (x2 > 1) {
                x2 = 1, s2 = 2;
                x1 = (u - mb) * (1.0f / ma);
                if (x1 > 1)
                    x1 = 1, s1 = 2;
                else if (x1 < -1)
                    x1 = -1, s1 = 0;
            } else if (x2 < -1) {
                x2 = -1, s2 = 0;
                x1 = (u + mb) * (1.0f / ma);
                if (x1 > 1)
                    x1 = 1, s1 = 2;
                else if (x1 < -1)
                    x1 = -1, s1 = 0;
            }
            for (int k = 0; k < 3; ++k) dif[k] = t3[k] - pos[k] - halfaxis[k] * x2;
            dif[j] += size2[j] * x1;
            const float d2 = dif[0] * dif[0] + dif[1] * dif[1] + dif[2] * dif[2];
            const int c1 = s1 * 3 + s2;
            if (d2 < bestdist - 1e-15f) {
                bestdist = d2;
                bestsegmentpos = x2;
                bestboxpos = x1;
                clcorner = i + (1 << j) * (c1 / 6);
                cledge = j;
                cltype = c1;
            }
        }
    if (cltype == -4) return 0;
    float mul = 1.0f, de = 0.0f, dp = 0.0f, e1, e2;
    int ax = 0, ax1 = 1, ax2 = 2;
    bool second = true;
    if (cltype >= 0 && cltype / 3 != 1) {  // closest to a corner of the box
        int c1 = axisdir ^ clcorner;
        if (c1 == 0 || c1 == 7) {
            second = false;  // pointing to or away from the corner
        } else {
            if (c1 == 1 || c1 == 2 || c1 == 4) {
                mul = 1;
                de = 1 - bestsegmentpos;
                dp = 1 + bestsegmentpos;
            } else {  // 3, 5, 6
                mul = -1;
                c1 = 7 - c1;
                dp = 1 - bestsegmentpos;
                de = 1 + bestsegmentpos;
            }
            if (c1 == 1) ax = 0, ax1 = 1, ax2 = 2;
            if (c1 == 2) ax = 1, ax1 = 2, ax2 = 0;
            if (c1 == 4) ax = 2, ax1 = 0, ax2 = 1;
            if (axis[ax] * axis[ax] > 0.5f) {  // second point along the edge of the box
                secondpos = de;
                e1 = 2 * size2[ax] / fabsf(halfaxis[ax]);
                if (e1 < secondpos) secondpos = e1;
                secondpos *= mul;
            } else {  // second point along a face of the box
                secondpos = dp;
                e1 = 2 * size2[ax1] / fabsf(halfaxis[ax1]);
                if (e1 < secondpos) secondpos = e1;
                e1 = 2 * size2[ax2] / fabsf(halfaxis[ax2]);
                if (e1 < secondpos) secondpos = e1;
                secondpos *= -mul;
            }
        }
    } else if (cltype >= 0 && cltype / 3 == 1) {  // on an edge of the box
        int c1 = axisdir ^ clcorner;
        c1 &= 7 - (1 << cledge);
        if (c1 != 1 && c1 != 2 && c1 != 4) {
            second = false;  // a T configuration
        } else {
            if (cledge == 0) ax1 = 1, ax2 = 2;
            if (cledge == 1) ax1 = 2, ax2 = 0;
            if (cledge == 2) ax1 = 0, ax2 = 1;
            ax = cledge;
            if (fabsf(axis[ax1]) > fabsf(axis[ax2])) ax1 = ax2;
            ax2 = 3 - ax - ax1;
            if (c1 & (1 << ax2)) {
                mul = 1;
                secondpos = 1 - bestsegmentpos;
            } else {
                mul = -1;
                secondpos = 1 + bestsegmentpos;
            }
            e1 = 2 * size2[ax2] / fabsf(halfaxis[ax2]);
            if (e1 < secondpos) secondpos = e1;
            if (((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0))
                e2 = 1 - bestboxpos;
            else
                e2 = 1 + bestboxpos;
            e1 = size2[ax] * e2 / fabsf(halfaxis[ax]);
            if (e1 < secondpos) secondpos = e1;
            secondpos *= mul;
        }
    } else if (cltype < 0) {  // one end of the capsule closest to a face
        if (clface == -1) {
            second = false;  // that end is inside the box
        } else {
            mul = cltype == -3 ? 1.0f : -1.0f;
            secondpos = 2;
            float t1[3];
            for (int k = 0; k < 3; ++k) t1[k] = pos[k] - halfaxis[k] * mul;
            for (int i = 0; i < 3; ++i) {
                if (i == clface) continue;
                e1 = (size2[i] - t1[i]) / halfaxis[i] * mul;
                if (e1 > 0 && e1 < secondpos) secondpos = e1;
                e1 = (-size2[i] - t1[i]) / halfaxis[i] * mul;
                if (e1 > 0 && e1 < secondpos) secondpos = e1;
            }
            secondpos *= mul;
        }
    }
    s[0] = bestsegmentpos;
    if (second && secondpos > -3.0f) {
        s[1] = secondpos + bestsegmentpos;
        return 2;
    }
    return 1;
}

// Box vs box, MuJoCo mjraw_BoxBox (engine_collision_box.c, Apache-2.0, Copyright
// DeepMind Technologies Limited; MuJoCo 4.0), ported to float with MuJoCo's
// single-precision tolerances. Stage 1: the separating-axis test over the 15 axes
// (3 faces of each box, 9 edge pairs; faces win near-ties). Stage 2: a face axis
// clips the incident box's face against the reference face (up to 8 points), an
// edge axis gives one point midway between the closest points of the two edges.
// Box k: center c_k, axes R_k (columns), half sizes h_k. Writes up to 8 contacts:
// pos (world), dist (<= 0: touching; MuJoCo margin 0) and one normal n (unit,
// from box 1 to box 2, the same for all). Returns how many.
struct BoxBoxContacts {
    Vec3 pos[8];
    float dist[8];
    Vec3 n;
};
FORGE_HD int box_box(Vec3 c1, const Mat3& R1, Vec3 h1, Vec3 c2, const Mat3& R2, Vec3 h2, BoxBoxContacts& out) {
    const float SEPEPS = 1e-6f, PAREPS = 1e-7f, SGNEPS = 1e-5f, DUPEPS = 1e-10f, EDGEBIAS = 1e-6f;
    const float size1[3] = {h1.x, h1.y, h1.z}, size2[3] = {h2.x, h2.y, h2.z};
    // rot[3 r + c] = (R1^T R2)(r, c): box 2's axis c in box 1's frame
    const Mat3 Rr = transpose(R1) * R2;
    float rot[9], rotabs[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) rot[3 * r + c] = Rr(r, c), rotabs[3 * r + c] = fabsf(Rr(r, c));
    const Vec3 p21v = transpose(R1) * (c2 - c1), p12v = transpose(R2) * (c1 - c2);
    const float pos21[3] = {p21v.x, p21v.y, p21v.z}, pos12[3] = {p12v.x, p12v.y, p12v.z};
    const float septol = SEPEPS * (size1[0] + size1[1] + size1[2] + size2[0] + size2[1] + size2[2]);
    float sep_best = -3.0e38f;
    int code = -1;
    for (int i = 0; i < 3; ++i) {
        const float radius2 = rotabs[3 * i] * size2[0] + rotabs[3 * i + 1] * size2[1] + rotabs[3 * i + 2] * size2[2];
        const float sep = fabsf(pos21[i]) - size1[i] - radius2;
        if (sep > septol) return 0;
        if (sep > sep_best) sep_best = sep, code = i;
    }
    for (int j = 0; j < 3; ++j) {
        const float radius1 = rotabs[j] * size1[0] + rotabs[3 + j] * size1[1] + rotabs[6 + j] * size1[2];
        const float sep = fabsf(pos12[j]) - size2[j] - radius1;
        if (sep > septol) return 0;
        if (sep > sep_best) sep_best = sep, code = 3 + j;
    }
    const float sep_face = sep_best;
    const int code_face = code;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
            float ax1 = -rot[3 * i2 + j], ax2 = rot[3 * i1 + j];
            const float norm2 = ax1 * ax1 + ax2 * ax2;
            if (norm2 < PAREPS) continue;
            const float inv = 1.0f / sqrtf(norm2);
            ax1 *= inv, ax2 *= inv;
            const float radius1 = size1[i1] * fabsf(ax1) + size1[i2] * fabsf(ax2);
            const int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
            const float a21 = ax1 * rot[3 * i1 + j1] + ax2 * rot[3 * i2 + j1];
            const float a22 = ax1 * rot[3 * i1 + j2] + ax2 * rot[3 * i2 + j2];
            const float radius2 = size2[j1] * fabsf(a21) + size2[j2] * fabsf(a22);
            const float sep = fabsf(ax1 * pos21[i1] + ax2 * pos21[i2]) - radius1 - radius2;
            if (sep > septol) return 0;
            if (sep - EDGEBIAS * fabsf(sep) > sep_best && sep > sep_face) sep_best = sep, code = 6 + 3 * i + j;
        }
    if (code < 0) return 0;
    auto edge_axis = [&](int c, float* axis) {  // unit axis e_i x (box 2 axis j), box 1 frame
        const int i = (c - 6) / 3, j = (c - 6) % 3, i1 = (i + 1) % 3, i2 = (i + 2) % 3;
        axis[i] = 0.0f;
        axis[i1] = -rot[3 * i2 + j];
        axis[i2] = rot[3 * i1 + j];
        const float l = sqrtf(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        if (l > 1e-15f) axis[0] /= l, axis[1] /= l, axis[2] /= l;
    };
    if (code >= 6) {  // an edge axis nearly parallel to the best face axis: use the face
        float axis[3];
        edge_axis(code, axis);
        float face_dot;
        if (code_face < 3) {
            face_dot = fabsf(axis[code_face]);
        } else {
            const int f = code_face - 3;
            face_dot = fabsf(axis[0] * rot[f] + axis[1] * rot[3 + f] + axis[2] * rot[6 + f]);
        }
        if (face_dot > 0.99f && sep_best < sep_face + 0.05f * fabsf(sep_face) + 1e-15f) code = code_face;
    }
    if (code >= 6) {  // edge-edge: one contact
        const int i = (code - 6) / 3, j = (code - 6) % 3;
        const int i1 = (i + 1) % 3, i2 = (i + 2) % 3, j1 = (j + 1) % 3, j2 = (j + 2) % 3;
        float axis[3];
        edge_axis(code, axis);
        if (axis[0] * pos21[0] + axis[1] * pos21[1] + axis[2] * pos21[2] < 0.0f)
            axis[0] = -axis[0], axis[1] = -axis[1], axis[2] = -axis[2];
        const float a2[3] = {axis[0] * rot[0] + axis[1] * rot[3] + axis[2] * rot[6],
                             axis[0] * rot[1] + axis[1] * rot[4] + axis[2] * rot[7],
                             axis[0] * rot[2] + axis[1] * rot[5] + axis[2] * rot[8]};
        int amb1 = -1, amb2 = -1;
        if (fabsf(axis[i1]) < SGNEPS)
            amb1 = i1;
        else if (fabsf(axis[i2]) < SGNEPS)
            amb1 = i2;
        if (fabsf(a2[j1]) < SGNEPS)
            amb2 = j1;
        else if (fabsf(a2[j2]) < SGNEPS)
            amb2 = j2;
        const float d2[3] = {rot[j], rot[3 + j], rot[6 + j]};
        const float b = d2[i], denom = 1.0f - b * b;
        float w1[3] = {0, 0, 0}, w2[3] = {0, 0, 0}, best_d2 = 3.0e38f;
        for (int v1 = 0; v1 < (amb1 >= 0 ? 2 : 1); ++v1)
            for (int v2 = 0; v2 < (amb2 >= 0 ? 2 : 1); ++v2) {
                float cA[3];
                cA[i] = 0.0f;
                cA[i1] = axis[i1] >= 0.0f ? size1[i1] : -size1[i1];
                cA[i2] = axis[i2] >= 0.0f ? size1[i2] : -size1[i2];
                if (amb1 >= 0 && v1) cA[amb1] = -cA[amb1];
                float cc[3];
                cc[j] = 0.0f;
                cc[j1] = a2[j1] >= 0.0f ? -size2[j1] : size2[j1];
                cc[j2] = a2[j2] >= 0.0f ? -size2[j2] : size2[j2];
                if (amb2 >= 0 && v2) cc[amb2] = -cc[amb2];
                float cB[3];
                for (int r = 0; r < 3; ++r)
                    cB[r] = rot[3 * r] * cc[0] + rot[3 * r + 1] * cc[1] + rot[3 * r + 2] * cc[2] + pos21[r];
                const float e[3] = {cB[0] - cA[0], cB[1] - cA[1], cB[2] - cA[2]};
                const float d1e = e[i], d2e = d2[0] * e[0] + d2[1] * e[1] + d2[2] * e[2];
                float sp = denom < 1e-15f ? 0.0f : (d1e - b * d2e) / denom;
                sp = fminf(fmaxf(sp, -size1[i]), size1[i]);
                const float t = fminf(fmaxf(b * sp - d2e, -size2[j]), size2[j]);
                sp = fminf(fmaxf(d1e + b * t, -size1[i]), size1[i]);
                float p1[3] = {cA[0], cA[1], cA[2]}, p2[3];
                p1[i] += sp;
                for (int r = 0; r < 3; ++r) p2[r] = cB[r] + d2[r] * t;
                const float g2 = (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]) +
                                 (p2[2] - p1[2]) * (p2[2] - p1[2]);
                if (g2 < best_d2) {
                    best_d2 = g2;
                    for (int r = 0; r < 3; ++r) w1[r] = p1[r], w2[r] = p2[r];
                }
            }
        const float dist = (w2[0] - w1[0]) * axis[0] + (w2[1] - w1[1]) * axis[1] + (w2[2] - w1[2]) * axis[2];
        if (dist > septol) return 0;
        const Vec3 mid{0.5f * (w1[0] + w2[0]), 0.5f * (w1[1] + w2[1]), 0.5f * (w1[2] + w2[2])};
        out.pos[0] = R1 * mid + c1;
        out.dist[0] = dist;
        out.n = R1 * Vec3{axis[0], axis[1], axis[2]};
        return 1;
    }
    // face contact: the reference box owns the winning face; the incident box's face
    // most opposed to it is clipped against the reference face's four sides
    const bool ref1 = code < 3;
    const int a = ref1 ? code : code - 3;
    const float* sizeref = ref1 ? size1 : size2;
    const float* sizeinc = ref1 ? size2 : size1;
    const float* posoi = ref1 ? pos21 : pos12;
    const Mat3& matref = ref1 ? R1 : R2;
    const Vec3 posref = ref1 ? c1 : c2;
    float rinc[9];  // rinc[3 r + c]: component r of incident axis c, reference frame
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) rinc[3 * r + c] = ref1 ? rot[3 * r + c] : rot[3 * c + r];
    const float sgn = posoi[a] >= 0.0f ? 1.0f : -1.0f;
    int binc = 0;
    for (int k = 1; k < 3; ++k)
        if (fabsf(rinc[3 * a + k]) > fabsf(rinc[3 * a + binc])) binc = k;
    const float tinc = sgn * rinc[3 * a + binc] > 0.0f ? -1.0f : 1.0f;
    const int ax = (a + 1) % 3, ay = (a + 2) % 3, bu = (binc + 1) % 3, bv = (binc + 2) % 3;
    float cx[3], du[3], dv[3];
    for (int r = 0; r < 3; ++r) {
        const int c = r == 0 ? ax : (r == 1 ? ay : a);
        cx[r] = posoi[c] + tinc * sizeinc[binc] * rinc[3 * c + binc];
        du[r] = sizeinc[bu] * rinc[3 * c + bu];
        dv[r] = sizeinc[bv] * rinc[3 * c + bv];
    }
    cx[2] = sgn * cx[2] - sizeref[a];
    du[2] *= sgn;
    dv[2] *= sgn;
    float poly[2][12][3];
    const float cs[4][2] = {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
    for (int k = 0; k < 4; ++k)
        for (int r = 0; r < 3; ++r) poly[0][k][r] = cx[r] + cs[k][0] * du[r] + cs[k][1] * dv[r];
    int cur = 0, nvert = 4;
    auto clip = [&](int coord, float sign, float limit) {  // keep sign * v[coord] <= limit
        float d[12];
        bool all_inside = true;
        for (int k = 0; k < nvert; ++k) {
            d[k] = sign * poly[cur][k][coord] - limit;
            all_inside = all_inside && d[k] <= 0.0f;
        }
        if (all_inside) return;
        const int o = 1 - cur;
        int nout = 0;
        for (int k = 0; k < nvert; ++k) {
            const int k1 = k + 1 == nvert ? 0 : k + 1;
            const float dp = d[k], dq = d[k1];
            if (dp <= 0.0f && nout < 12) {
                for (int r = 0; r < 3; ++r) poly[o][nout][r] = poly[cur][k][r];
                ++nout;
            }
            if (((dp < 0.0f && dq > 0.0f) || (dp > 0.0f && dq < 0.0f)) && nout < 12) {
                const float t = dp / (dp - dq);
                for (int r = 0; r < 3; ++r) poly[o][nout][r] = poly[cur][k][r] + t * (poly[cur][k1][r] - poly[cur][k][r]);
                ++nout;
            }
        }
        cur = o;
        nvert = nout;
    };
    clip(0, 1.0f, sizeref[ax]);
    clip(0, -1.0f, sizeref[ax]);
    clip(1, 1.0f, sizeref[ay]);
    clip(1, -1.0f, sizeref[ay]);
    const float dupe2 = DUPEPS * (sizeref[ax] * sizeref[ax] + sizeref[ay] * sizeref[ay]);
    float acc[12][3];
    int na = 0;
    for (int k = 0; k < nvert; ++k) {
        if (poly[cur][k][2] > 0.0f) continue;  // margin 0
        bool dupe = false;
        for (int q = 0; q < na && !dupe; ++q) {
            const float dx = acc[q][0] - poly[cur][k][0], dy = acc[q][1] - poly[cur][k][1];
            dupe = dx * dx + dy * dy < dupe2;
        }
        if (!dupe)
            for (int r = 0; r < 3; ++r) acc[na][r] = poly[cur][k][r];
        if (!dupe) ++na;
    }
    if (na == 0) return 0;
    const float nsign = ref1 ? sgn : -sgn;
    out.n = Vec3{matref(0, a), matref(1, a), matref(2, a)} * nsign;
    if (na > 8) na = 8;  // a 4-gon clipped by 4 half-planes has at most 8 corners
    for (int k = 0; k < na; ++k) {
        float pc[3];
        pc[ax] = acc[k][0];
        pc[ay] = acc[k][1];
        pc[a] = sgn * (sizeref[a] + 0.5f * acc[k][2]);
        out.pos[k] = matref * Vec3{pc[0], pc[1], pc[2]} + posref;
        out.dist[k] = acc[k][2];
    }
    return na;
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
    auto contact = [&](int b1, int b2, Vec3 point, Vec3 n, float dist, int id, Vec3 tangent = Vec3{0, 0, 0}) {
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
            contact_frame(n, tangent, t1, t2);
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
    // capsules (Phase 10): per capsule, each plane (2 contacts: the +axis end, then
    // the -axis end, mjc_PlaneCapsule, tangent = the axis), then each box on another
    // body (up to 2 contacts, mjc_CapsuleBox; capsule = geom 1)
    const int id_cap_plane = id_box_plane + c.nbox * c.nplane * 8;
    const int id_cap_box = id_cap_plane + c.ncapsule * c.nplane * 2;
    for (int k = 0; k < c.ncapsule; ++k) {
        const int b = c.capsule_body[k];
        const Quat qb = load_quat(w.xquat + 4 * b);
        const Vec3 pc = load3(w.xpos + 3 * b) + rotate(qb, load3(c.capsule_pos + 3 * k));
        const Vec3 a = rotate(qb * load_quat(c.capsule_quat + 4 * k), Vec3{0, 0, 1});
        const float radius = c.capsule_size[2 * k], hl = c.capsule_size[2 * k + 1];
        bool welded = true;  // to the world: no plane contacts (MuJoCo skips world-world pairs)
        for (int q = b; q >= 0 && welded; q = m.parent[q]) welded = m.jtype[q] == kJointFixed;
        for (int p = 0; p < c.nplane && !welded; ++p) {
            const Vec3 n = load3(c.plane_normal + 3 * p), pp = load3(c.plane_point + 3 * p);
            for (int e = 0; e < 2; ++e) {
                const Vec3 end = pc + a * (e == 0 ? hl : -hl);
                const float dist = dot(end - pp, n) - radius;
                if (dist > 0.0f) continue;
                contact(-1, b, end - n * (0.5f * dist + radius), n, dist, id_cap_plane + (k * c.nplane + p) * 2 + e, a);
            }
        }
        for (int j = 0; j < c.nbox; ++j) {
            const int bb = c.box_body[j];
            if (bb == b || m.parent[bb] == b || m.parent[b] == bb) continue;
            const Quat qbb = load_quat(w.xquat + 4 * bb);
            const Vec3 bc = load3(w.xpos + 3 * bb) + rotate(qbb, load3(c.box_pos + 3 * j));
            const Mat3 R = quat_to_mat3(qbb * load_quat(c.box_quat + 4 * j));
            const Vec3 h = load3(c.box_half + 3 * j);
            float sp[2];
            const int np = capsule_box_points(pc, a, hl, radius, bc, R, h, sp);
            for (int i = 0; i < np; ++i) {
                const Vec3 center = pc + a * (hl * sp[i]);
                float dist;
                Vec3 n_out;
                if (!sphere_box(center, radius, bc, R, h, dist, n_out)) continue;
                const Vec3 n = -n_out;  // capsule (geom 1) -> box (geom 2)
                contact(b, bb, center + n * (0.5f * dist + radius), n, dist, id_cap_box + (k * c.nbox + j) * 2 + i);
            }
        }
    }
    // box vs box (Phase 10b): per pair k < j on different bodies (not parent and child),
    // up to 8 contacts (mjraw_BoxBox), geom 1 = box k, n from box k to box j
    const int id_box_box = id_cap_box + c.ncapsule * c.nbox * 2;
    for (int k = 0; k < c.nbox; ++k) {
        const int b1 = c.box_body[k];
        const Quat q1 = load_quat(w.xquat + 4 * b1);
        const Vec3 c1 = load3(w.xpos + 3 * b1) + rotate(q1, load3(c.box_pos + 3 * k));
        const Mat3 R1 = quat_to_mat3(q1 * load_quat(c.box_quat + 4 * k));
        const Vec3 h1 = load3(c.box_half + 3 * k);
        for (int j = k + 1; j < c.nbox; ++j) {
            const int b2 = c.box_body[j];
            if (b2 == b1 || m.parent[b2] == b1 || m.parent[b1] == b2) continue;
            const Quat q2 = load_quat(w.xquat + 4 * b2);
            const Vec3 c2 = load3(w.xpos + 3 * b2) + rotate(q2, load3(c.box_pos + 3 * j));
            const Vec3 h2 = load3(c.box_half + 3 * j);
            // bounding spheres apart: no contact (cheap test before the full one)
            if (length(c2 - c1) > length(h1) + length(h2)) continue;
            const Mat3 R2 = quat_to_mat3(q2 * load_quat(c.box_quat + 4 * j));
            BoxBoxContacts bb;
            const int nc = box_box(c1, R1, h1, c2, R2, h2, bb);
            for (int i = 0; i < nc; ++i)
                if (bb.dist[i] <= 0.0f) contact(b1, b2, bb.pos[i], bb.n, bb.dist[i], id_box_box + (k * c.nbox + j) * 8 + i);
        }
    }
    cw.nefc = r;
}

}  // namespace forge
