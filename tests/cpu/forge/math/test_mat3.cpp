#include "common/test_helpers.h"

using namespace forge;
using namespace testutil;

TEST(Mat3, IdentityAndTranspose) {
    Mat3 a{{1, 2, 3, 4, 5, 6, 7, 8, 10}};
    expect_mat_near(a * mat3_identity(), a);
    expect_mat_near(mat3_identity() * a, a);
    expect_mat_near(transpose(transpose(a)), a);
    EXPECT_FLOAT_EQ(transpose(a)(0, 1), a(1, 0));
}

TEST(Mat3, Determinant) {
    EXPECT_FLOAT_EQ(det(mat3_identity()), 1.0f);
    EXPECT_FLOAT_EQ(det(mat3_diag(2, 3, 4)), 24.0f);
    Mat3 a{{1, 2, 3, 4, 5, 6, 7, 8, 10}};
    EXPECT_NEAR(det(a), -3.0f, 1e-5f);
}

TEST(Mat3, MatVecMatchesRows) {
    Mat3 a{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    Vec3 v{1, -1, 2};
    expect_vec_near(a * v, {dot(row(a, 0), v), dot(row(a, 1), v), dot(row(a, 2), v)});
}

TEST(Mat3, SkewIsCross) {
    Rng r;
    for (int i = 0; i < 1000; ++i) {
        Vec3 a = r.vec(), b = r.vec();
        expect_vec_near(skew(a) * b, cross(a, b), 1e-5f);
    }
}

TEST(Mat3, Sym6RoundTrip) {
    const float s[6] = {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f};  // xx yy zz xy xz yz
    Mat3 m = sym6_to_mat3(s);
    EXPECT_FLOAT_EQ(m(0, 1), 0.1f);
    EXPECT_FLOAT_EQ(m(1, 0), 0.1f);
    EXPECT_FLOAT_EQ(m(0, 2), 0.2f);
    EXPECT_FLOAT_EQ(m(1, 2), 0.3f);
    float back[6];
    mat3_to_sym6(m, back);
    for (int i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(back[i], s[i]);
}

TEST(Mat3, RotateInertiaKeepsTraceAndSymmetry) {
    Rng r;
    const Mat3 I = mat3_diag(0.5f, 1.5f, 2.0f);
    for (int i = 0; i < 1000; ++i) {
        Mat3 R = quat_to_mat3(r.quat());
        Mat3 Iw = rotate_inertia(R, I);
        // Trace is invariant under rotation, and the result stays symmetric.
        EXPECT_NEAR(Iw(0, 0) + Iw(1, 1) + Iw(2, 2), 4.0f, 1e-5f);
        expect_mat_near(Iw, transpose(Iw), 1e-6f);
        // Rotating back gives the body inertia.
        expect_mat_near(rotate_inertia(transpose(R), Iw), I, 1e-5f);
    }
}
