#pragma once
#include "forge/math/hd.h"

namespace forge {

// 3D vector, FP32. Used inside kernels after reading SoA arrays (px, py, pz).
struct Vec3 {
    float x, y, z;
};

FORGE_HD Vec3 make_vec3(float x, float y, float z) { return Vec3{x, y, z}; }

FORGE_HD Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
FORGE_HD Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
FORGE_HD Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
FORGE_HD Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
FORGE_HD Vec3 operator*(float s, Vec3 a) { return {a.x * s, a.y * s, a.z * s}; }
FORGE_HD Vec3 operator/(Vec3 a, float s) { return a * (1.0f / s); }
FORGE_HD Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }
FORGE_HD Vec3& operator-=(Vec3& a, Vec3 b) { a = a - b; return a; }
FORGE_HD Vec3& operator*=(Vec3& a, float s) { a = a * s; return a; }

FORGE_HD float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

FORGE_HD Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

FORGE_HD float length_sq(Vec3 a) { return dot(a, a); }
FORGE_HD float length(Vec3 a) { return sqrtf(dot(a, a)); }

// Returns zero vector for zero input (no NaN).
FORGE_HD Vec3 normalize(Vec3 a) {
    float l2 = dot(a, a);
    if (l2 <= 0.0f) return {0.0f, 0.0f, 0.0f};
    return a * (1.0f / sqrtf(l2));
}

}  // namespace forge
