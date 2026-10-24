#pragma once
#include "forge/math/vec3.h"

namespace forge {

// 3x3 matrix, row major: m[row * 3 + col].
struct Mat3 {
    float m[9];
    FORGE_HD float& operator()(int r, int c) { return m[r * 3 + c]; }
    FORGE_HD float operator()(int r, int c) const { return m[r * 3 + c]; }
};

FORGE_HD Mat3 mat3_zero() { return Mat3{{0, 0, 0, 0, 0, 0, 0, 0, 0}}; }
FORGE_HD Mat3 mat3_identity() { return Mat3{{1, 0, 0, 0, 1, 0, 0, 0, 1}}; }
FORGE_HD Mat3 mat3_diag(float a, float b, float c) { return Mat3{{a, 0, 0, 0, b, 0, 0, 0, c}}; }

FORGE_HD Mat3 mat3_from_rows(Vec3 r0, Vec3 r1, Vec3 r2) {
    return Mat3{{r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z}};
}

FORGE_HD Vec3 row(const Mat3& a, int r) { return {a(r, 0), a(r, 1), a(r, 2)}; }
FORGE_HD Vec3 col(const Mat3& a, int c) { return {a(0, c), a(1, c), a(2, c)}; }

FORGE_HD Mat3 operator+(const Mat3& a, const Mat3& b) {
    Mat3 r;
    for (int i = 0; i < 9; ++i) r.m[i] = a.m[i] + b.m[i];
    return r;
}
FORGE_HD Mat3 operator-(const Mat3& a, const Mat3& b) {
    Mat3 r;
    for (int i = 0; i < 9; ++i) r.m[i] = a.m[i] - b.m[i];
    return r;
}
FORGE_HD Mat3 operator*(const Mat3& a, float s) {
    Mat3 r;
    for (int i = 0; i < 9; ++i) r.m[i] = a.m[i] * s;
    return r;
}

FORGE_HD Vec3 operator*(const Mat3& a, Vec3 v) {
    return {a(0, 0) * v.x + a(0, 1) * v.y + a(0, 2) * v.z,
            a(1, 0) * v.x + a(1, 1) * v.y + a(1, 2) * v.z,
            a(2, 0) * v.x + a(2, 1) * v.y + a(2, 2) * v.z};
}

FORGE_HD Mat3 operator*(const Mat3& a, const Mat3& b) {
    Mat3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r(i, j) = a(i, 0) * b(0, j) + a(i, 1) * b(1, j) + a(i, 2) * b(2, j);
    return r;
}

FORGE_HD Mat3 transpose(const Mat3& a) {
    return Mat3{{a(0, 0), a(1, 0), a(2, 0),
                 a(0, 1), a(1, 1), a(2, 1),
                 a(0, 2), a(1, 2), a(2, 2)}};
}

FORGE_HD float det(const Mat3& a) {
    return a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1))
         - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0))
         + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
}

// Matrix of the cross product: skew(a) * b == cross(a, b).
FORGE_HD Mat3 skew(Vec3 a) {
    return Mat3{{0.0f, -a.z, a.y,
                 a.z, 0.0f, -a.x,
                 -a.y, a.x, 0.0f}};
}

// Outer product a * b^T.
FORGE_HD Mat3 outer(Vec3 a, Vec3 b) {
    return Mat3{{a.x * b.x, a.x * b.y, a.x * b.z,
                 a.y * b.x, a.y * b.y, a.y * b.z,
                 a.z * b.x, a.z * b.y, a.z * b.z}};
}

// ---- Symmetric 3x3 packed in 6 floats (inertia storage, doc section 3.1) ----
// Order: xx, yy, zz, xy, xz, yz.
FORGE_HD Mat3 sym6_to_mat3(const float* s) {
    return Mat3{{s[0], s[3], s[4],
                 s[3], s[1], s[5],
                 s[4], s[5], s[2]}};
}

// Reads the upper triangle. Input should already be symmetric.
FORGE_HD void mat3_to_sym6(const Mat3& a, float* s) {
    s[0] = a(0, 0); s[1] = a(1, 1); s[2] = a(2, 2);
    s[3] = a(0, 1); s[4] = a(0, 2); s[5] = a(1, 2);
}

// Inertia from body frame to world frame: R * I * R^T.
FORGE_HD Mat3 rotate_inertia(const Mat3& R, const Mat3& I_body) {
    return R * I_body * transpose(R);
}

}  // namespace forge
