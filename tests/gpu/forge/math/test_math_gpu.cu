// GPU vs CPU test for the forge math headers (doc 11.1: every kernel has a CPU
// version and the test compares GPU against CPU). The same header functions run
// in a kernel and on the host; results must match within FP32 tolerance.
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "common/test_helpers.h"
#include "forge/math/math.h"
#include "forge/util/device_buffer.h"

using namespace forge;
using testutil::expect_quat_near;
using testutil::expect_vec_near;

namespace {

constexpr int kN = 4096;
// GPU and CPU may use FMA differently, so allow a few ulp more than 1e-6.
constexpr float kTol = 1e-5f;

struct Inputs {
    std::vector<Quat> qa, qb;
    std::vector<Vec3> v, omega;
};

Inputs make_inputs(unsigned seed) {
    testutil::Rng r(seed);
    Inputs in;
    for (int i = 0; i < kN; ++i) {
        in.qa.push_back(r.quat());
        in.qb.push_back(r.quat());
        in.v.push_back(r.vec());
        in.omega.push_back(r.vec() * 5.0f);
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
        SCOPED_TRACE("i=" + std::to_string(i));
        expect_quat_near(mul[i], in.qa[i] * in.qb[i], kTol);
        expect_vec_near(rot[i], rotate(in.qa[i], in.v[i]), kTol);
        testutil::expect_mat_near(mat[i], quat_to_mat3(in.qa[i]), kTol);
        expect_quat_near(itg[i], integrate(in.qa[i], in.omega[i], dt), kTol);
        // Invariants: no NaN, unit norm after integrate.
        ASSERT_FALSE(std::isnan(itg[i].w));
        EXPECT_NEAR(norm(itg[i]), 1.0f, 1e-5f);
    }
}
