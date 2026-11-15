#pragma once
// World-axes gizmo made of flat quads (doc 4.1 "debug lines"), drawn after the
// robots so it stays visible: +X red, +Y green, +Z blue. Each arrow is a thin
// strip with a wider tip. The X and Y arrows lie just above the floor; the Z
// arrow is two crossed vertical strips, so it is seen from every side.
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

}  // namespace flux
