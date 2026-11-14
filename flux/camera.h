#pragma once
// Camera math for the viewer. Host only. World is Z-up (same as forge).
//
// Mat4 is column major (m[col * 4 + row]), the same layout as GLSL mat4, so it
// can be copied into push constants with memcpy.
// Vulkan clip space: x right, y DOWN, depth 0..1. perspective() handles that.
#include <cmath>

#include "forge/math/vec3.h"

namespace flux {

using forge::Vec3;

struct Mat4 {
    float m[16];
    float& at(int row, int col) { return m[col * 4 + row]; }
    float at(int row, int col) const { return m[col * 4 + row]; }
};

inline Mat4 mat4_zero() { return Mat4{{0}}; }

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r = mat4_zero();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) r.at(i, j) += a.at(i, k) * b.at(k, j);
    return r;
}

// Right-handed view matrix: camera looks down its own -Z axis.
inline Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up) {
    const Vec3 f = forge::normalize(target - eye);
    const Vec3 s = forge::normalize(forge::cross(f, up));
    const Vec3 u = forge::cross(s, f);
    Mat4 v = mat4_zero();
    v.at(0, 0) = s.x; v.at(0, 1) = s.y; v.at(0, 2) = s.z; v.at(0, 3) = -forge::dot(s, eye);
    v.at(1, 0) = u.x; v.at(1, 1) = u.y; v.at(1, 2) = u.z; v.at(1, 3) = -forge::dot(u, eye);
    v.at(2, 0) = -f.x; v.at(2, 1) = -f.y; v.at(2, 2) = -f.z; v.at(2, 3) = forge::dot(f, eye);
    v.at(3, 3) = 1.0f;
    return v;
}

// Perspective for Vulkan: flips y and maps depth to [0, 1] (near -> 0, far -> 1).
inline Mat4 perspective(float fovy_rad, float aspect, float z_near, float z_far) {
    const float f = 1.0f / std::tan(0.5f * fovy_rad);
    Mat4 p = mat4_zero();
    p.at(0, 0) = f / aspect;
    p.at(1, 1) = -f;
    p.at(2, 2) = z_far / (z_near - z_far);
    p.at(2, 3) = z_near * z_far / (z_near - z_far);
    p.at(3, 2) = -1.0f;
    return p;
}

// Orbit camera around a target point. Yaw turns around world Z, pitch tilts up.
struct OrbitCamera {
    Vec3 target{0.0f, 0.0f, 1.0f};
    float distance = 10.0f;  // m
    float yaw = 0.6f;        // rad
    float pitch = 0.35f;     // rad, > 0 looks down at the target
    float fovy = 0.9f;       // rad (about 52 degrees)

    Vec3 eye() const {
        const float c = std::cos(pitch);
        return target + Vec3{distance * c * std::cos(yaw), distance * c * std::sin(yaw),
                             distance * std::sin(pitch)};
    }
};

}  // namespace flux
