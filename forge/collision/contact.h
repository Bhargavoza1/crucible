#pragma once
// Soft contact force (spring + damper + friction), used by the CPU answer key
// and the GPU kernels, so both run the same math.
//
// The contact is set by two numbers, like MuJoCo's solref:
//   timeconst  how fast a contact pushes bodies apart (s). Smaller = stiffer.
//   dampratio  1 = critical damping (no bounce), < 1 = some bounce.
// Per unit of effective mass:
//   k = 1 / timeconst^2          (spring)
//   b = 2 * dampratio / timeconst (damper)
//   normal force  fn = m_eff * (k * depth - b * vn),   never negative (contacts push, never pull)
//   friction      ft = min(mu * fn, m_eff * b * |vt|), against the sliding direction
// The friction cap "m_eff * b * |vt|" makes friction smooth near zero speed
// (no jitter). The price: a resting body can creep very slowly on a slope.
//
// Stability (explicit integration): keep dt well below timeconst. With many
// contacts on one body the springs and dampers add up, so dense piles need a
// small step (see sandbox: 10 substeps per physics step).
#include "forge/math/hd.h"
#include "forge/math/vec3.h"

namespace forge {

struct ContactParams {
    float timeconst = 0.004f;  // s
    float dampratio = 0.5f;
    float friction = 0.5f;     // Coulomb mu
};

struct ContactCoeffs {
    float k;   // 1/s^2
    float b;   // 1/s
    float mu;
};

FORGE_HD ContactCoeffs contact_coeffs(const ContactParams& c) {
    const float w = 1.0f / c.timeconst;
    return {w * w, 2.0f * c.dampratio * w, c.friction};
}

// Force on body A from one contact.
//   n      unit normal, pointing from B toward A (the direction A is pushed)
//   depth  penetration, > 0 when touching
//   v_rel  velocity of A minus velocity of B at the contact
//   m_eff  effective mass: 1 / (inv_mass_A + inv_mass_B)
// The force on B is exactly the negative: swap A and B and every term flips sign.
FORGE_HD Vec3 contact_force(Vec3 n, float depth, Vec3 v_rel, float m_eff, const ContactCoeffs& cc) {
    const float vn = dot(v_rel, n);
    const float fn = m_eff * (cc.k * depth - cc.b * vn);
    if (fn <= 0.0f) return {0.0f, 0.0f, 0.0f};
    Vec3 f = fn * n;
    const Vec3 vt = v_rel - vn * n;
    const float vt_len = length(vt);
    if (vt_len > 1e-9f) {
        const float ft = fminf(cc.mu * fn, m_eff * cc.b * vt_len);
        f -= (ft / vt_len) * vt;
    }
    return f;
}

// ---- Planes ------------------------------------------------------------------
// A static plane: points with dot(normal, p) >= offset are on the free side.
// Floor: normal (0, 0, 1), offset 0. Planes are env-local, like positions.
struct Plane {
    Vec3 normal{0.0f, 0.0f, 1.0f};  // unit length
    float offset = 0.0f;            // m
};

// Force on a sphere (center p, radius r, velocity v, mass m) from a plane.
FORGE_HD Vec3 plane_contact_force(const Plane& pl, Vec3 p, float r, Vec3 v, float m, const ContactCoeffs& cc) {
    const float depth = r - (dot(pl.normal, p) - pl.offset);
    if (depth <= 0.0f) return {0.0f, 0.0f, 0.0f};
    return contact_force(pl.normal, depth, v, m, cc);
}

// True if two spheres overlap (and are not exactly on top of each other).
// Also returns the squared center distance. sphere_contact_force uses this same
// test, so a kernel can call it first to skip loading data for far neighbors
// and still get exactly the same result.
FORGE_HD bool spheres_overlap(Vec3 pi, float ri, Vec3 pj, float rj, float& dist2) {
    const Vec3 d = pi - pj;
    dist2 = dot(d, d);
    const float rsum = ri + rj;
    return dist2 < rsum * rsum && dist2 > 0.0f;
}

// Force on sphere i from sphere j. inv_m_i > 0 is assumed (static i never moves).
FORGE_HD Vec3 sphere_contact_force(Vec3 pi, float ri, Vec3 vi, float inv_mi, Vec3 pj, float rj, Vec3 vj,
                                   float inv_mj, const ContactCoeffs& cc) {
    float dist2;
    if (!spheres_overlap(pi, ri, pj, rj, dist2)) return {0.0f, 0.0f, 0.0f};  // apart, or exactly on top
    const float dist = sqrtf(dist2);
    const Vec3 n = (pi - pj) / dist;
    const float m_eff = 1.0f / (inv_mi + inv_mj);
    return contact_force(n, ri + rj - dist, vi - vj, m_eff, cc);
}

}  // namespace forge
