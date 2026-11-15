#include <cmath>

#include <gtest/gtest.h>

#include "common/test_helpers.h"
#include "forge/collision/contact.h"

using namespace forge;
using testutil::expect_vec_near;

TEST(Contact, CoeffsFromTimeconst) {
    ContactParams c;
    c.timeconst = 0.01f;
    c.dampratio = 0.5f;
    const ContactCoeffs cc = contact_coeffs(c);
    EXPECT_FLOAT_EQ(cc.k, 10000.0f);  // 1 / 0.01^2
    EXPECT_FLOAT_EQ(cc.b, 100.0f);    // 2 * 0.5 / 0.01
}

TEST(Contact, PushesAlongNormalWhenResting) {
    const ContactCoeffs cc{10000.0f, 100.0f, 0.5f};
    const Vec3 f = contact_force({0, 0, 1}, 0.001f, {0, 0, 0}, 2.0f, cc);
    expect_vec_near(f, {0, 0, 2.0f * 10000.0f * 0.001f});  // m * k * depth = 20 N
}

TEST(Contact, NeverPulls) {
    const ContactCoeffs cc{10000.0f, 100.0f, 0.5f};
    // Separating fast: the damper would pull, but contacts only push.
    const Vec3 f = contact_force({0, 0, 1}, 0.001f, {0, 0, 5.0f}, 1.0f, cc);
    expect_vec_near(f, {0, 0, 0});
}

TEST(Contact, FrictionIsCappedByCoulomb) {
    const ContactCoeffs cc{10000.0f, 100.0f, 0.5f};
    const float depth = 0.001f;  // fn = 10 N
    // Fast sliding: friction = mu * fn = 5 N, against the motion.
    const Vec3 fast = contact_force({0, 0, 1}, depth, {3.0f, 0, 0}, 1.0f, cc);
    EXPECT_FLOAT_EQ(fast.z, 10.0f);
    EXPECT_FLOAT_EQ(fast.x, -5.0f);
    // Very slow sliding: friction = m * b * |vt| (smooth, below the Coulomb cap).
    const Vec3 slow = contact_force({0, 0, 1}, depth, {0.01f, 0, 0}, 1.0f, cc);
    EXPECT_FLOAT_EQ(slow.x, -1.0f);  // 1 * 100 * 0.01
}

// Newton's third law, exactly: force on i from j == -(force on j from i), bit for bit.
TEST(Contact, SphereForceIsExactlyAntisymmetric) {
    testutil::Rng r(5);
    const ContactCoeffs cc = contact_coeffs(ContactParams{});
    for (int k = 0; k < 1000; ++k) {
        const Vec3 pi = r.vec() * 0.03f, pj = r.vec() * 0.03f;  // close enough to touch
        const Vec3 vi = r.vec(), vj = r.vec();
        const float inv_i = 1.0f + 0.5f * std::fabs(r.normal());
        const float inv_j = 1.0f + 0.5f * std::fabs(r.normal());
        const Vec3 fij = sphere_contact_force(pi, 0.05f, vi, inv_i, pj, 0.05f, vj, inv_j, cc);
        const Vec3 fji = sphere_contact_force(pj, 0.05f, vj, inv_j, pi, 0.05f, vi, inv_i, cc);
        EXPECT_EQ(fij.x, -fji.x);
        EXPECT_EQ(fij.y, -fji.y);
        EXPECT_EQ(fij.z, -fji.z);
    }
}

TEST(Contact, SpheresApartOrOnTopGiveZero) {
    const ContactCoeffs cc = contact_coeffs(ContactParams{});
    expect_vec_near(sphere_contact_force({0, 0, 0}, 0.05f, {}, 1, {0.2f, 0, 0}, 0.05f, {}, 1, cc), {0, 0, 0});
    // Exactly the same center: no normal direction, so no force (and no NaN).
    const Vec3 f = sphere_contact_force({1, 1, 1}, 0.05f, {}, 1, {1, 1, 1}, 0.05f, {}, 1, cc);
    expect_vec_near(f, {0, 0, 0});
}

TEST(Contact, PlaneOnlyWhenTouching) {
    const ContactCoeffs cc = contact_coeffs(ContactParams{});
    const Plane floor{{0, 0, 1}, 0.0f};
    expect_vec_near(plane_contact_force(floor, {0, 0, 0.06f}, 0.05f, {}, 1.0f, cc), {0, 0, 0});
    EXPECT_GT(plane_contact_force(floor, {0, 0, 0.04f}, 0.05f, {}, 1.0f, cc).z, 0.0f);
    // A wall at x = 1 facing -x: a sphere at x = 0.97 is pushed toward -x.
    const Plane wall{{-1, 0, 0}, -1.0f};
    EXPECT_LT(plane_contact_force(wall, {0.97f, 0, 0}, 0.05f, {}, 1.0f, cc).x, 0.0f);
}
