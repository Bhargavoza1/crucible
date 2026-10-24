// GPU vs CPU test for the forge math headers (doc 11.1: every kernel has a CPU
// version and the test compares GPU against CPU). The same header functions run
// in a kernel and on the host; results must match within FP32 tolerance.
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "forge/math/math.h"
#include "forge/util/device_buffer.h"

using namespace forge;

namespace {

constexpr int kN = 4096;
// FP32. GPU and CPU may use FMA differently, so allow a few ulp more than 1e-6.
constexpr float kTol = 1e-5f;

struct Inputs {
    std::vector<Quat> qa, qb;
    std::vector<Vec3> v, omega;
};

Inputs make_inputs(unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    Inputs in;
    for (int i = 0; i < kN; ++i) {
        in.qa.push_back(normalize(Quat{g(rng), g(rng), g(rng), g(rng)}));
        in.qb.push_back(normalize(Quat{g(rng), g(rng), g(rng), g(rng)}));
        in.v.push_back(Vec3{g(rng), g(rng), g(rng)});
        in.omega.push_back(Vec3{g(rng), g(rng), g(rng)} * 5.0f);
    }
    return in;
}

// One thread per element. Writes the result of several math functions.
__global__ void math_kernel(int n, const Quat* qa, const Quat* qb, const Vec3* v,
                            const Vec3* omega, float dt, Quat* out_mul, Vec3* out_rot,
                            Mat3* out_mat, Quat* out_int) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_mul[i] = qa[i] * qb[i];
    out_rot[i] = rotate(qa[i], v[i]);
    out_mat[i] = quat_to_mat3(qa[i]);
    out_int[i] = integrate(qa[i], omega[i], dt);
}

template <typename T>
DeviceBuffer<T> to_device(const std::vector<T>& h) {
    DeviceBuffer<T> d;
    d.upload(h);
    return d;
}

void expect_near(Quat a, Quat b, int i) {
    EXPECT_NEAR(a.x, b.x, kTol) << "i=" << i;
    EXPECT_NEAR(a.y, b.y, kTol) << "i=" << i;
    EXPECT_NEAR(a.z, b.z, kTol) << "i=" << i;
    EXPECT_NEAR(a.w, b.w, kTol) << "i=" << i;
}

void expect_near(Vec3 a, Vec3 b, int i) {
    EXPECT_NEAR(a.x, b.x, kTol) << "i=" << i;
    EXPECT_NEAR(a.y, b.y, kTol) << "i=" << i;
    EXPECT_NEAR(a.z, b.z, kTol) << "i=" << i;
}

}  // namespace

TEST(MathGpu, MatchesCpu) {
    const Inputs in = make_inputs(42);
    const float dt = 0.005f;

    auto d_qa = to_device(in.qa);
    auto d_qb = to_device(in.qb);
    auto d_v = to_device(in.v);
    auto d_w = to_device(in.omega);
    DeviceBuffer<Quat> d_mul(kN), d_int(kN);
    DeviceBuffer<Vec3> d_rot(kN);
    DeviceBuffer<Mat3> d_mat(kN);

    const int block = 256;
    math_kernel<<<(kN + block - 1) / block, block>>>(kN, d_qa.data(), d_qb.data(), d_v.data(),
                                                     d_w.data(), dt, d_mul.data(), d_rot.data(),
                                                     d_mat.data(), d_int.data());
    KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());  // fine in a test, never in the training loop

    const auto mul = d_mul.download();
    const auto rot = d_rot.download();
    const auto mat = d_mat.download();
    const auto itg = d_int.download();

    for (int i = 0; i < kN; ++i) {
        expect_near(mul[i], in.qa[i] * in.qb[i], i);
        expect_near(rot[i], rotate(in.qa[i], in.v[i]), i);
        const Mat3 m = quat_to_mat3(in.qa[i]);
        for (int k = 0; k < 9; ++k) EXPECT_NEAR(mat[i].m[k], m.m[k], kTol) << "i=" << i;
        expect_near(itg[i], integrate(in.qa[i], in.omega[i], dt), i);
        // Invariants: no NaN, unit norm after integrate.
        ASSERT_FALSE(std::isnan(itg[i].w)) << "i=" << i;
        EXPECT_NEAR(norm(itg[i]), 1.0f, 1e-5f) << "i=" << i;
    }
}
