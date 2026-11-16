// Phase 7b: temper/nn and temper/optim on the GPU (doc 5.7).
//   GEMM: CUTLASS == cuBLAS (the oracle), all four transpose cases, odd sizes
//   Linear, ELU, MLP forward and backward == the CPU reference (double), which
//   itself is checked against finite differences in tests_cpu
//   Adam and gradient clipping == their formulas on the CPU
//   A small MLP learns a function (the loss goes down a lot)
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "temper/nn/gemm.h"
#include "temper/nn/mlp.h"
#include "temper/nn/reference_cpu.h"
#include "temper/optim/adam.h"

using namespace temper;

namespace {

std::vector<float> random_vec(size_t n, unsigned seed, float scale = 1.0f) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = u(g);
    return v;
}

std::vector<double> to_double(const std::vector<float>& v) { return {v.begin(), v.end()}; }

// max |a - b| / max(1, max |b|): relative to the size of the numbers.
double rel_err(const std::vector<float>& a, const std::vector<double>& b) {
    double e = 0, s = 1;
    for (size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, std::fabs(double(a[i]) - b[i]));
        s = std::max(s, std::fabs(b[i]));
    }
    return e / s;
}

#define CUBLAS_CHECK(x) ASSERT_EQ((x), CUBLAS_STATUS_SUCCESS)

// cuBLAS is column major: row-major D = op(A) op(B) is column-major D^T = op(B)^T op(A)^T,
// so swap A and B and keep the transpose flags.
void cublas_gemm(cublasHandle_t h, Op oa, Op ob, int M, int N, int K, const float* A, int lda, const float* B,
                 int ldb, float* D, int ldd) {
    const float one = 1.0f, zero = 0.0f;
    const cublasOperation_t ta = oa == Op::T ? CUBLAS_OP_T : CUBLAS_OP_N, tb = ob == Op::T ? CUBLAS_OP_T : CUBLAS_OP_N;
    ASSERT_EQ(cublasSgemm(h, tb, ta, N, M, K, &one, B, ldb, A, lda, &zero, D, ldd), CUBLAS_STATUS_SUCCESS);
}

}  // namespace

// Both backends against a plain cuBLAS call and against double sums.
class GemmBackends : public ::testing::TestWithParam<GemmBackend> {
protected:
    void SetUp() override { set_gemm_backend(GetParam()); }
    void TearDown() override { set_gemm_backend(GemmBackend::kCublas); }
};
INSTANTIATE_TEST_SUITE_P(Gemm, GemmBackends, ::testing::Values(GemmBackend::kCublas, GemmBackend::kCutlass),
                         [](const auto& info) { return std::string(gemm_backend_name(info.param)); });

TEST_P(GemmBackends, MatchesCublasAllTransposes) {
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    // FP32 SIMT on both sides: only the summation order differs.
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));
    const int sizes[][3] = {{37, 53, 29}, {1, 256, 33}, {1024, 256, 256}, {300, 1, 7}};
    for (auto [M, N, K] : sizes)
        for (Op oa : {Op::N, Op::T})
            for (Op ob : {Op::N, Op::T}) {
                const int lda = oa == Op::N ? K : M, ldb = ob == Op::N ? N : K;
                Tensor A(1, M * K), B(1, K * N), D1(1, M * N), D2(1, M * N);
                A.upload(random_vec(size_t(M) * K, 1));
                B.upload(random_vec(size_t(K) * N, 2));
                gemm(oa, ob, M, N, K, 1.0f, A.data(), lda, B.data(), ldb, 0.0f, D1.data(), N, D1.data(), N, nullptr);
                cublas_gemm(h, oa, ob, M, N, K, A.data(), lda, B.data(), ldb, D2.data(), N);
                const auto d1 = D1.download(), d2 = D2.download();
                double e = 0;
                for (size_t i = 0; i < d1.size(); ++i) e = std::max(e, double(std::fabs(d1[i] - d2[i])));
                EXPECT_LT(e, 1e-4 * std::sqrt(double(K))) << M << "x" << N << "x" << K << " op " << int(oa) << int(ob);
            }
    cublasDestroy(h);
}

TEST_P(GemmBackends, BiasRowBroadcast) {
    // D = A B + bias on every row (ldc = 0), the Linear forward trick.
    const int M = 5, N = 3, K = 4;
    Tensor A(M, K), B(K, N), bias(1, N), D(M, N);
    const auto a = random_vec(M * K, 3), b = random_vec(K * N, 4), c = random_vec(N, 5);
    A.upload(a);
    B.upload(b);
    bias.upload(c);
    gemm(Op::N, Op::N, M, N, K, 1.0f, A.data(), K, B.data(), N, 1.0f, bias.data(), 0, D.data(), N, nullptr);
    const auto d = D.download();
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double s = c[j];
            for (int k = 0; k < K; ++k) s += double(a[i * K + k]) * b[k * N + j];
            EXPECT_NEAR(d[i * N + j], s, 1e-5);
        }
    // C a full matrix (ldc = N), not in place: D = 2 A B + 0.5 C
    Tensor Cm(M, N), D2(M, N);
    const auto cm = random_vec(M * N, 6);
    Cm.upload(cm);
    gemm(Op::N, Op::N, M, N, K, 2.0f, A.data(), K, B.data(), N, 0.5f, Cm.data(), N, D2.data(), N, nullptr);
    const auto d2 = D2.download();
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.5 * cm[i * N + j];
            for (int k = 0; k < K; ++k) s += 2.0 * double(a[i * K + k]) * b[k * N + j];
            EXPECT_NEAR(d2[i * N + j], s, 1e-5);
        }
}

// The policy-size MLP (doc 5.3: 3 hidden layers of 256, ELU), forward and every
// gradient vs the CPU reference in double.
TEST(Mlp, ForwardBackwardMatchReference) {
    const std::vector<int> dims = {33, 256, 256, 256, 8};
    const int batch = 512;
    Mlp net(dims, batch, 7);
    const auto x = random_vec(size_t(batch) * 33, 8);
    Tensor X(batch, 33), dY(batch, 8), dX(batch, 33);
    X.upload(x);
    const float* y = net.forward(X.data(), batch, nullptr);
    std::vector<float> yh(size_t(batch) * 8);
    CUDA_CHECK(cudaMemcpy(yh.data(), y, yh.size() * 4, cudaMemcpyDeviceToHost));
    const auto dy = random_vec(size_t(batch) * 8, 9);
    dY.upload(dy);
    net.backward(dY.data(), batch, nullptr, dX.data());

    ref::Mlp r;
    r.dims = dims;
    for (Linear* l : net.linears()) {
        r.W.push_back(to_double(l->W.download()));
        r.b.push_back(to_double(l->b.download()));
    }
    const auto& yr = r.forward(to_double(x), batch);
    r.backward(to_double(dy));
    double worst_y = rel_err(yh, yr), worst_g = 0;
    const auto lin = net.linears();
    for (size_t l = 0; l < lin.size(); ++l) {
        worst_g = std::max(worst_g, rel_err(lin[l]->dW.download(), r.dW[l]));
        worst_g = std::max(worst_g, rel_err(lin[l]->db.download(), r.db[l]));
    }
    const double worst_x = rel_err(dX.download(), r.dx);
    std::printf("MLP 33-256-256-256-8, batch %d vs CPU double: output %.2e, param grads %.2e, input grad %.2e\n",
                batch, worst_y, worst_g, worst_x);
    EXPECT_LT(worst_y, 1e-5);
    EXPECT_LT(worst_g, 1e-5);
    EXPECT_LT(worst_x, 1e-5);
}

TEST(Mlp, OrthogonalInit) {
    Mlp net({6, 16, 3}, 4, 1);
    net.init_orthogonal(0.01f, 2);
    const auto lin = net.linears();
    // First layer W (16 x 6): columns orthonormal times sqrt(2).
    const auto w = lin[0]->W.download();
    for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b) {
            double d = 0;
            for (int o = 0; o < 16; ++o) d += double(w[o * 6 + a]) * w[o * 6 + b];
            EXPECT_NEAR(d, a == b ? 2.0 : 0.0, 1e-5);
        }
    // Last layer W (3 x 16): rows orthonormal times 0.01. Biases 0.
    const auto w2 = lin[1]->W.download();
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            double d = 0;
            for (int i = 0; i < 16; ++i) d += double(w2[a * 16 + i]) * w2[b * 16 + i];
            EXPECT_NEAR(d, a == b ? 1e-4 : 0.0, 1e-9);
        }
    for (float v : lin[1]->b.download()) EXPECT_EQ(v, 0.0f);
}

TEST(Adam, MatchesFormula) {
    Tensor p(1, 5), g(1, 5);
    std::vector<double> ph = {0.5, -1.0, 2.0, 0.0, 3.0}, m(5, 0.0), v(5, 0.0);
    p.upload(std::vector<float>(ph.begin(), ph.end()));
    Adam adam({&p}, {&g}, {0.01f, 0.9f, 0.999f, 1e-8f});
    for (int t = 1; t <= 20; ++t) {
        std::vector<float> gh = random_vec(5, 100 + t);
        g.upload(gh);
        adam.step(nullptr);
        for (int i = 0; i < 5; ++i) {
            m[i] = 0.9 * m[i] + 0.1 * gh[i];
            v[i] = 0.999 * v[i] + 0.001 * double(gh[i]) * gh[i];
            ph[i] -= 0.01 * (m[i] / (1 - std::pow(0.9, t))) / (std::sqrt(v[i] / (1 - std::pow(0.999, t))) + 1e-8);
        }
    }
    const auto out = p.download();
    for (int i = 0; i < 5; ++i) EXPECT_NEAR(out[i], ph[i], 1e-5);
}

TEST(GradClip, ScalesToMaxNorm) {
    Tensor a(1, 300), b(1, 7);
    const auto av = random_vec(300, 1), bv = random_vec(7, 2);
    a.upload(av);
    b.upload(bv);
    double s = 0;
    for (float x : av) s += double(x) * x;
    for (float x : bv) s += double(x) * x;
    const double norm = std::sqrt(s);
    GradClip clip({&a, &b});
    for (float max_norm : {1.0f, 1000.0f}) {
        clip.compute(max_norm, nullptr);
        float out[2];
        CUDA_CHECK(cudaMemcpy(out, clip.norm(), 8, cudaMemcpyDeviceToHost));
        EXPECT_NEAR(out[0], norm, 1e-4 * norm);
        EXPECT_NEAR(out[1], std::min(1.0, max_norm / (norm + 1e-6)), 1e-6);
    }
}

// End to end: an MLP + Adam fit y = sin(3 x0) + x1^2 (2 in, 1 out). Mean squared
// error must fall below 2% of the start (the variance of y).
TEST(Mlp, LearnsASmoothFunction) {
    const int n = 4096;
    Mlp net({2, 64, 64, 1}, n, 3);
    std::vector<Tensor*> p, g;
    net.params(p, g);
    Adam adam(p, g, {1e-3f});
    GradClip clip(g);
    const auto x = random_vec(size_t(n) * 2, 11);
    std::vector<float> y(n);
    for (int i = 0; i < n; ++i) y[i] = std::sin(3.0f * x[2 * i]) + x[2 * i + 1] * x[2 * i + 1];
    Tensor X(n, 2), dY(n, 1);
    X.upload(x);
    auto mse_and_grad = [&](bool grad) {
        const float* out = net.forward(X.data(), n, nullptr);
        std::vector<float> o(n);
        CUDA_CHECK(cudaMemcpy(o.data(), out, n * 4, cudaMemcpyDeviceToHost));
        double e = 0;
        std::vector<float> d(n);
        for (int i = 0; i < n; ++i) {
            e += double(o[i] - y[i]) * (o[i] - y[i]);
            d[i] = 2.0f * (o[i] - y[i]) / n;  // d(mean squared error)/d(out)
        }
        if (grad) {
            dY.upload(d);
            net.backward(dY.data(), n, nullptr);
        }
        return e / n;
    };
    double var = 0, mean = 0;
    for (float v : y) mean += v / n;
    for (float v : y) var += (v - mean) * (v - mean) / n;
    double mse = 0;
    for (int it = 0; it < 1500; ++it) {
        mse = mse_and_grad(true);
        clip.compute(1.0f, nullptr);
        adam.step(nullptr, clip.scale());
    }
    mse = mse_and_grad(false);
    std::printf("fit sin(3 x0) + x1^2: variance of y %.3f, MSE after 1500 Adam steps %.4f (%.2f%%)\n", var, mse,
                100 * mse / var);
    EXPECT_LT(mse, 0.02 * var);
}
