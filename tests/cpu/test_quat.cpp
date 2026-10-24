#include "test_helpers.h"

using namespace forge;
using namespace testutil;

namespace {
constexpr float kPi = 3.14159265358979f;
}

TEST(Quat, TimesInverseIsIdentity) {
    Rng r;
    for (int i = 0; i < 1000; ++i) {
        Quat q = r.quat();
        Quat p = q * inverse(q);
        EXPECT_TRUE(same_rotation(p, quat_identity(), kTol));
        EXPECT_NEAR(p.w, 1.0f, kTol);
    }
}

TEST(Quat, InverseOfNonUnit) {
    Quat q{1.0f, 2.0f, 3.0f, 4.0f};  // not normalized
    Quat p = q * inverse(q);
    EXPECT_NEAR(p.x, 0.0f, kTol);
    EXPECT_NEAR(p.y, 0.0f, kTol);
    EXPECT_NEAR(p.z, 0.0f, kTol);
    EXPECT_NEAR(p.w, 1.0f, kTol);
}

TEST(Quat, AxisAngleRotation) {
    // 90 degrees about z maps x to y.
    Quat q = quat_from_axis_angle({0, 0, 1}, 0.5f * kPi);
    expect_vec_near(rotate(q, {1, 0, 0}), {0, 1, 0});
    expect_vec_near(rotate(q, {0, 0, 1}), {0, 0, 1});
    // 180 degrees about x maps y to -y.
    Quat p = quat_from_axis_angle({1, 0, 0}, kPi);
    expect_vec_near(rotate(p, {0, 1, 0}), {0, -1, 0});
}

TEST(Quat, RotateMatchesMatrix) {
    Rng r;
    for (int i = 0; i < 1000; ++i) {
        Quat q = r.quat();
        Vec3 v = r.vec();
        expect_vec_near(rotate(q, v), quat_to_mat3(q) * v, 1e-5f);
        expect_vec_near(rotate_inv(q, rotate(q, v)), v, 1e-5f);
    }
}

TEST(Quat, ProductIsComposition) {
    // (a * b) rotates by b first, then a.
    Rng r;
    for (int i = 0; i < 1000; ++i) {
        Quat a = r.quat(), b = r.quat();
        Vec3 v = r.vec();
        expect_vec_near(rotate(a * b, v), rotate(a, rotate(b, v)), 1e-5f);
    }
}

TEST(Quat, MatrixIsOrthogonal) {
    // R * R^T and det(R) chain about 10 FP32 operations, so error reaches ~1e-6.
    // 1e-6 is right for single operations; chained ones get kTolChain.
    constexpr float kTolChain = 4e-6f;
    Rng r;
    for (int i = 0; i < 1000; ++i) {
        Mat3 R = quat_to_mat3(r.quat());
        expect_mat_near(R * transpose(R), mat3_identity(), kTolChain);
        EXPECT_NEAR(det(R), 1.0f, kTolChain);
    }
}

TEST(Quat, MatrixRoundTrip) {
    Rng r;
    for (int i = 0; i < 1000; ++i) {
        Quat q = r.quat();
        Quat back = mat3_to_quat(quat_to_mat3(q));
        EXPECT_TRUE(same_rotation(q, back, 1e-6f)) << "i=" << i;
        EXPECT_GE(back.w, 0.0f);
    }
    // All four Shepperd branches: identity, and 180 deg about x, y, z (trace = -1).
    const Quat cases[] = {quat_identity(),
                          quat_from_axis_angle({1, 0, 0}, kPi),
                          quat_from_axis_angle({0, 1, 0}, kPi),
                          quat_from_axis_angle({0, 0, 1}, kPi)};
    for (const Quat& q : cases) EXPECT_TRUE(same_rotation(q, mat3_to_quat(quat_to_mat3(q)), 1e-6f));
}

TEST(Quat, NormalizeZeroIsIdentity) {
    Quat q = normalize(Quat{0, 0, 0, 0});
    EXPECT_FLOAT_EQ(q.w, 1.0f);
}

TEST(Quat, IntegrateConstantSpin) {
    // Spin about z at 1 rad/s for 1 s in small steps: about 1 rad of rotation.
    Quat q = quat_identity();
    const Vec3 w{0, 0, 1};
    const float dt = 0.005f;
    for (int i = 0; i < 200; ++i) q = integrate(q, w, dt);
    Quat expected = quat_from_axis_angle({0, 0, 1}, 1.0f);
    EXPECT_TRUE(same_rotation(q, expected, 1e-5f));
    EXPECT_NEAR(norm(q), 1.0f, kTol);  // invariant: unit norm
}
