#pragma once
#include <random>

#include <gtest/gtest.h>

#include "forge/math/math.h"

namespace testutil {

constexpr float kTol = 1e-6f;  // doc 11.2: math unit tests

inline void expect_vec_near(forge::Vec3 a, forge::Vec3 b, float tol = kTol) {
    EXPECT_NEAR(a.x, b.x, tol);
    EXPECT_NEAR(a.y, b.y, tol);
    EXPECT_NEAR(a.z, b.z, tol);
}

inline void expect_mat_near(const forge::Mat3& a, const forge::Mat3& b, float tol = kTol) {
    for (int i = 0; i < 9; ++i) EXPECT_NEAR(a.m[i], b.m[i], tol) << "element " << i;
}

// Fixed seed, written inside the test (doc 11.1).
struct Rng {
    std::mt19937 gen{12345};
    std::normal_distribution<float> g{0.0f, 1.0f};
    float normal() { return g(gen); }
    forge::Vec3 vec() { return {normal(), normal(), normal()}; }
    forge::Quat quat() { return forge::normalize(forge::Quat{normal(), normal(), normal(), normal()}); }
};

}  // namespace testutil
