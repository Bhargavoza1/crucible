#pragma once
// Spatial (6D) algebra for rigid-body dynamics (Featherstone).
//
// Every spatial quantity here is written in the WORLD frame, about the WORLD
// origin. Then adding the motion of a parent and a child needs no transforms:
// it is a plain sum. The price: numbers grow with distance from the origin,
// which is fine for robots that stay within a few meters of it.
//
//   Motion  m = [w; v]   w = angular velocity, v = linear velocity of the
//                        (imaginary) body point at the world origin
//   Force   f = [n; f]   n = moment about the world origin, f = force
//
// Power = dot(motion, force) = w.n + v.f
#include "forge/math/hd.h"
#include "forge/math/mat3.h"
#include "forge/math/vec3.h"

namespace forge {

struct SpatialVec {
    Vec3 ang;  // w (motion) or n (force)
    Vec3 lin;  // v (motion) or f (force)
};

FORGE_HD SpatialVec spatial_zero() { return {{0, 0, 0}, {0, 0, 0}}; }
FORGE_HD SpatialVec operator+(const SpatialVec& a, const SpatialVec& b) { return {a.ang + b.ang, a.lin + b.lin}; }
FORGE_HD SpatialVec operator-(const SpatialVec& a, const SpatialVec& b) { return {a.ang - b.ang, a.lin - b.lin}; }
FORGE_HD SpatialVec operator*(const SpatialVec& a, float s) { return {a.ang * s, a.lin * s}; }
FORGE_HD SpatialVec& operator+=(SpatialVec& a, const SpatialVec& b) {
    a = a + b;
    return a;
}

// Motion . force (power), or motion . motion when used for S^T F.
FORGE_HD float dot(const SpatialVec& a, const SpatialVec& b) { return dot(a.ang, b.ang) + dot(a.lin, b.lin); }

// a x m: how motion vector m changes when the frame moves with velocity a.
FORGE_HD SpatialVec cross_motion(const SpatialVec& a, const SpatialVec& m) {
    return {cross(a.ang, m.ang), cross(a.ang, m.lin) + cross(a.lin, m.ang)};
}

// a x* f: the same for a force vector f.
FORGE_HD SpatialVec cross_force(const SpatialVec& a, const SpatialVec& f) {
    return {cross(a.ang, f.ang) + cross(a.lin, f.lin), cross(a.ang, f.lin)};
}

// Spatial inertia about the world origin, stored compactly (10 numbers):
//   I = [ I_O       skew(h) ]     I_O = rotational inertia about the world origin
//       [ -skew(h)  m * 1   ]     h   = m * c (c = center of mass, world)
// So  I * [w; v] = [ I_O w + h x v ;  m v - h x w ].
// Composite inertias (CRBA) are plain sums of these.
struct SpatialInertia {
    Mat3 I_O;
    Vec3 h;
    float m;
};

FORGE_HD SpatialInertia inertia_zero() { return {mat3_zero(), {0, 0, 0}, 0.0f}; }

FORGE_HD SpatialInertia operator+(const SpatialInertia& a, const SpatialInertia& b) {
    return {a.I_O + b.I_O, a.h + b.h, a.m + b.m};
}

// Body of mass m, center of mass c (world), rotational inertia Ic about its
// center of mass (world axes). Parallel axis theorem moves Ic to the origin:
//   I_O = Ic - m * skew(c) * skew(c)
FORGE_HD SpatialInertia body_inertia(float m, Vec3 c, const Mat3& Ic) {
    const Mat3 cx = skew(c);
    return {Ic - (cx * cx) * m, c * m, m};
}

FORGE_HD SpatialVec operator*(const SpatialInertia& I, const SpatialVec& v) {
    return {I.I_O * v.ang + cross(I.h, v.lin), v.lin * I.m - cross(I.h, v.ang)};
}

}  // namespace forge
