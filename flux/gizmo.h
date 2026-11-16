#pragma once
// World-axes gizmo made of flat quads (doc 4.1 "debug lines"), drawn after the
// robots so it stays visible: +X red, +Y green, +Z blue. Each arrow is a thin
// strip with a wider tip. The X and Y arrows lie just above the floor; the Z
// arrow is two crossed vertical strips, so it is seen from every side.
#include <cmath>
#include <vector>

#include "flux/renderer.h"

namespace flux {

inline void add_axes_gizmo(SceneQuads& q, Vec3 origin, float length, float width = 0.02f) {
    auto strip = [&](Vec3 a, Vec3 b, Vec3 side, const float rgba[4]) {
        const Vec3 c = (a + b) * 0.5f, u = (b - a) * 0.5f;
        q.after.push_back({{c.x, c.y, c.z}, {u.x, u.y, u.z}, {side.x, side.y, side.z},
                           {rgba[0], rgba[1], rgba[2], rgba[3]}, 0.0f});
    };
    const float red[4] = {0.95f, 0.2f, 0.2f, 1.0f}, green[4] = {0.2f, 0.85f, 0.25f, 1.0f},
                blue[4] = {0.25f, 0.45f, 1.0f, 1.0f};
    const float tip = 0.18f * length, h = 0.004f;  // tip length; height above the floor
    const Vec3 o{origin.x, origin.y, origin.z + h};
    // +X
    strip(o, o + Vec3{length - tip, 0, 0}, Vec3{0, width, 0}, red);
    strip(o + Vec3{length - tip, 0, 0}, o + Vec3{length, 0, 0}, Vec3{0, 3 * width, 0}, red);
    // +Y
    strip(o, o + Vec3{0, length - tip, 0}, Vec3{width, 0, 0}, green);
    strip(o + Vec3{0, length - tip, 0}, o + Vec3{0, length, 0}, Vec3{3 * width, 0, 0}, green);
    // +Z: two crossed vertical strips
    for (const Vec3 side : {Vec3{width, 0, 0}, Vec3{0, width, 0}}) {
        strip(o, o + Vec3{0, 0, length - tip}, side, blue);
        strip(o + Vec3{0, 0, length - tip}, o + Vec3{0, 0, length}, side * 3.0f, blue);
    }
}

// 3D axes for the corner gizmo: each axis is a round shaft with a cone at the end,
// in pure RGB (+X (1,0,0), +Y (0,1,0), +Z (0,0,1)), lit from one fixed direction so
// the round shape shows. Built from many small opaque quads (the renderer draws
// them with depth write on, so near faces hide far ones). Unit length from origin.
inline std::vector<Quad> axes_gizmo_3d(float length = 1.0f, int sides = 20) {
    std::vector<Quad> out;
    const Vec3 light = normalize(Vec3{0.35f, -0.45f, 0.82f});
    const float pi2 = 6.2831853f;
    // One ring of 'sides' patches around 'axis', from height h0 (radius r0) to h1 (radius r1).
    auto ring = [&](Vec3 axis, Vec3 e1, Vec3 e2, float h0, float r0, float h1, float r1, const float rgb[3]) {
        for (int k = 0; k < sides; ++k) {
            const float a = (k + 0.5f) * pi2 / sides;
            const Vec3 radial = e1 * std::cos(a) + e2 * std::sin(a);
            const Vec3 tangent = e2 * std::cos(a) - e1 * std::sin(a);
            const float rm = 0.5f * (r0 + r1);
            const Vec3 c = axis * (0.5f * (h0 + h1)) + radial * rm;
            const Vec3 u = tangent * (1.08f * rm * std::tan(0.5f * pi2 / sides));  // slight overlap: no gaps
            const Vec3 v = (axis * (h1 - h0) + radial * (r1 - r0)) * 0.5f;
            // outward normal of the patch: perpendicular to the slope, pointing out
            Vec3 n = radial * (h1 - h0) - axis * (r1 - r0);
            n = normalize(n);
            const float lit = 0.65f + 0.35f * std::fmax(0.0f, dot(n, light));
            out.push_back({{c.x, c.y, c.z}, {u.x, u.y, u.z}, {v.x, v.y, v.z},
                           {rgb[0] * lit, rgb[1] * lit, rgb[2] * lit, 1.0f}, 0.0f});
        }
    };
    struct Axis { Vec3 dir, e1, e2; float rgb[3]; };
    const Axis axes[3] = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}},
                          {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}},
                          {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    const float shaft_r = 0.035f * length, head_r = 0.09f * length, head_len = 0.25f * length;
    const float shaft_end = length - head_len;
    for (const Axis& ax : axes) {
        ring(ax.dir, ax.e1, ax.e2, 0.0f, shaft_r, shaft_end, shaft_r, ax.rgb);                // shaft
        ring(ax.dir, ax.e1, ax.e2, shaft_end, shaft_r, shaft_end, head_r, ax.rgb);            // under the cone
        const int steps = 8;                                                                  // cone, in bands
        for (int i = 0; i < steps; ++i) {
            const float t0 = float(i) / steps, t1 = float(i + 1) / steps;
            ring(ax.dir, ax.e1, ax.e2, shaft_end + t0 * head_len, head_r * (1 - t0), shaft_end + t1 * head_len,
                 head_r * (1 - t1), ax.rgb);
        }
    }
    return out;
}

}  // namespace flux
