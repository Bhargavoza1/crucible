#include "test_helpers.h"

using namespace forge;
using namespace testutil;

TEST(Vec3, BasicOps) {
    Vec3 a{1, 2, 3}, b{4, -5, 6};
    expect_vec_near(a + b, {5, -3, 9});
    expect_vec_near(a - b, {-3, 7, -3});
    expect_vec_near(a * 2.0f, {2, 4, 6});
    expect_vec_near(2.0f * a, {2, 4, 6});
    expect_vec_near(a / 2.0f, {0.5f, 1.0f, 1.5f});
    expect_vec_near(-a, {-1, -2, -3});
    EXPECT_FLOAT_EQ(dot(a, b), 1 * 4 + 2 * -5 + 3 * 6);
}

TEST(Vec3, CrossOfAxes) {
    expect_vec_near(cross({1, 0, 0}, {0, 1, 0}), {0, 0, 1});
    expect_vec_near(cross({0, 1, 0}, {0, 0, 1}), {1, 0, 0});
    expect_vec_near(cross({0, 0, 1}, {1, 0, 0}), {0, 1, 0});
}

TEST(Vec3, CrossIsPerpendicular) {
    Rng r;
    for (int i = 0; i < 1000; ++i) {
        Vec3 a = r.vec(), b = r.vec();
        Vec3 c = cross(a, b);
        EXPECT_NEAR(dot(c, a), 0.0f, 1e-5f);
        EXPECT_NEAR(dot(c, b), 0.0f, 1e-5f);
    }
}

TEST(Vec3, NormalizeIsUnitAndZeroIsSafe) {
    Rng r;
    for (int i = 0; i < 1000; ++i) EXPECT_NEAR(length(normalize(r.vec())), 1.0f, kTol);
    Vec3 z = normalize(Vec3{0, 0, 0});
    expect_vec_near(z, {0, 0, 0});
}
