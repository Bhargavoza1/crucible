#pragma once
#include "forge/math/mat3.h"

namespace forge {

// Unit quaternion for rotation. Hamilton convention, w is the scalar part.
// Field order x, y, z, w matches the SoA arrays qx, qy, qz, qw (doc 3.1).
// Note: MuJoCo stores quaternions as (w, x, y, z). Convert when comparing.
struct Quat {
    float x, y, z, w;
};

FORGE_HD Quat quat_identity() { return Quat{0.0f, 0.0f, 0.0f, 1.0f}; }

FORGE_HD Quat operator*(Quat a, Quat b) {
    return Quat{a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

FORGE_HD Quat conjugate(Quat q) { return Quat{-q.x, -q.y, -q.z, q.w}; }

FORGE_HD float norm_sq(Quat q) { return q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w; }
FORGE_HD float norm(Quat q) { return sqrtf(norm_sq(q)); }

// Returns identity for a zero quaternion (no NaN).
FORGE_HD Quat normalize(Quat q) {
    float n2 = norm_sq(q);
    if (n2 <= 0.0f) return quat_identity();
    float s = 1.0f / sqrtf(n2);
    return Quat{q.x * s, q.y * s, q.z * s, q.w * s};
}

// Full inverse, also correct for non-unit quaternions.
FORGE_HD Quat inverse(Quat q) {
    float n2 = norm_sq(q);
    Quat c = conjugate(q);
    float s = 1.0f / n2;
    return Quat{c.x * s, c.y * s, c.z * s, c.w * s};
}

// Axis must be unit length. Angle in radians.
FORGE_HD Quat quat_from_axis_angle(Vec3 axis, float angle) {
    float h = 0.5f * angle;
    float s = sinf(h);
    return Quat{axis.x * s, axis.y * s, axis.z * s, cosf(h)};
}

// Rotate vector v by unit quaternion q: q * v * q^-1, written in the fast form
// v' = v + 2w (u x v) + 2 u x (u x v), with u = (x, y, z).
FORGE_HD Vec3 rotate(Quat q, Vec3 v) {
    Vec3 u{q.x, q.y, q.z};
    Vec3 t = 2.0f * cross(u, v);
    return v + q.w * t + cross(u, t);
}

// Rotate by the inverse of a unit quaternion (world -> body).
FORGE_HD Vec3 rotate_inv(Quat q, Vec3 v) { return rotate(conjugate(q), v); }

// Rotation matrix of a unit quaternion.
FORGE_HD Mat3 quat_to_mat3(Quat q) {
    float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    return Mat3{{1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz),        2.0f * (xz + wy),
                 2.0f * (xy + wz),        1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx),
                 2.0f * (xz - wy),        2.0f * (yz + wx),        1.0f - 2.0f * (xx + yy)}};
}

// Quaternion from a rotation matrix (Shepperd's method: pick the largest
// diagonal term so we never divide by a small number). Output has w >= 0.
FORGE_HD Quat mat3_to_quat(const Mat3& R) {
    float tr = R(0, 0) + R(1, 1) + R(2, 2);
    Quat q;
    if (tr > 0.0f) {
        float s = sqrtf(tr + 1.0f) * 2.0f;  // s = 4w
        q.w = 0.25f * s;
        q.x = (R(2, 1) - R(1, 2)) / s;
        q.y = (R(0, 2) - R(2, 0)) / s;
        q.z = (R(1, 0) - R(0, 1)) / s;
    } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
        float s = sqrtf(1.0f + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0f;  // s = 4x
        q.w = (R(2, 1) - R(1, 2)) / s;
        q.x = 0.25f * s;
        q.y = (R(0, 1) + R(1, 0)) / s;
        q.z = (R(0, 2) + R(2, 0)) / s;
    } else if (R(1, 1) > R(2, 2)) {
        float s = sqrtf(1.0f + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0f;  // s = 4y
        q.w = (R(0, 2) - R(2, 0)) / s;
        q.x = (R(0, 1) + R(1, 0)) / s;
        q.y = 0.25f * s;
        q.z = (R(1, 2) + R(2, 1)) / s;
    } else {
        float s = sqrtf(1.0f + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0f;  // s = 4z
        q.w = (R(1, 0) - R(0, 1)) / s;
        q.x = (R(0, 2) + R(2, 0)) / s;
        q.y = (R(1, 2) + R(2, 1)) / s;
        q.z = 0.25f * s;
    }
    if (q.w < 0.0f) q = Quat{-q.x, -q.y, -q.z, -q.w};
    return normalize(q);
}

// One integration step of orientation with world-frame angular velocity omega:
// q_new = normalize(q + 0.5 * dt * (omega, 0) * q). Used by the integrator.
FORGE_HD Quat integrate(Quat q, Vec3 omega, float dt) {
    Quat w{omega.x, omega.y, omega.z, 0.0f};
    Quat dq = w * q;
    float h = 0.5f * dt;
    return normalize(Quat{q.x + h * dq.x, q.y + h * dq.y, q.z + h * dq.z, q.w + h * dq.w});
}

// True when a and b are the same rotation within tol (q and -q are the same rotation).
FORGE_HD bool same_rotation(Quat a, Quat b, float tol) {
    float d = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    if (d < 0.0f) d = -d;
    return d >= 1.0f - tol;
}

}  // namespace forge
