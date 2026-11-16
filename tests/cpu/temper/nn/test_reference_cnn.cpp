// Phase 9d: CPU reference for Conv2d and LayerNorm (temper/nn/reference_cpu.h), the
// answer key for the GPU layers. Gradients against finite differences; a small CNN
// (conv, ReLU, conv, ReLU, Linear, LayerNorm) against PyTorch (tests/data/torch_cnn.txt,
// tools/gen_torch_cnn_ref.py).
#include <cmath>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "temper/nn/reference_cpu.h"

using namespace temper;

namespace {

std::vector<double> randn(size_t n, std::mt19937& g) {
    std::normal_distribution<double> d(0.0, 1.0);
    std::vector<double> v(n);
    for (auto& x : v) x = d(g);
    return v;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// Loss L = sum(y * r). Checks every analytic gradient entry against (L(p + h) - L(p - h)) / 2h.
template <class Forward>
void check_grad(std::vector<double>& p, const std::vector<double>& analytic, Forward f, const std::vector<double>& r,
                const char* name) {
    const double h = 1e-6;
    double worst = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        const double keep = p[i];
        p[i] = keep + h;
        const double up = dot(f(), r);
        p[i] = keep - h;
        const double down = dot(f(), r);
        p[i] = keep;
        const double num = (up - down) / (2 * h);
        worst = std::max(worst, std::fabs(num - analytic[i]) / std::max(1.0, std::fabs(num)));
    }
    EXPECT_LT(worst, 1e-6) << name;
}

TEST(ReferenceConv, GradientsMatchFiniteDifferences) {
    std::mt19937 g(7);
    for (const ref::ConvShape s : {ref::ConvShape{2, 7, 6, 3, 3, 1, 0}, ref::ConvShape{3, 9, 8, 4, 3, 2, 1},
                                   ref::ConvShape{2, 11, 9, 2, 5, 2, 0}}) {
        const int batch = 2;
        std::vector<double> x = randn(size_t(batch) * s.C * s.H * s.W, g);
        std::vector<double> W = randn(size_t(s.K) * s.C * s.k * s.k, g), b = randn(s.K, g);
        const std::vector<double> r = randn(size_t(batch) * s.K * s.Ho() * s.Wo(), g);
        std::vector<double> y, dW, db, dx;
        auto f = [&]() {
            ref::conv2d_forward(x, W, b, batch, s, y);
            return y;
        };
        f();
        ref::conv2d_backward(x, W, r, batch, s, dW, db, dx);
        check_grad(W, dW, f, r, "conv dW");
        check_grad(b, db, f, r, "conv db");
        check_grad(x, dx, f, r, "conv dx");
    }
}

TEST(ReferenceLayerNorm, GradientsMatchFiniteDifferences) {
    std::mt19937 g(8);
    const int batch = 3, D = 7;
    std::vector<double> x = randn(size_t(batch) * D, g), gamma = randn(D, g), beta = randn(D, g);
    const std::vector<double> r = randn(size_t(batch) * D, g);
    std::vector<double> y, dg, dbeta, dx;
    auto f = [&]() {
        ref::layernorm_forward(x, gamma, beta, batch, D, 1e-5, y);
        return y;
    };
    f();
    ref::layernorm_backward(x, gamma, r, batch, D, 1e-5, dg, dbeta, dx);
    check_grad(gamma, dg, f, r, "layernorm dgamma");
    check_grad(beta, dbeta, f, r, "layernorm dbeta");
    check_grad(x, dx, f, r, "layernorm dx");
}

std::map<std::string, std::vector<double>> load_torch_ref() {
    std::ifstream in(std::string(CRUCIBLE_SOURCE_DIR) + "/tests/data/torch_cnn.txt");
    std::map<std::string, std::vector<double>> m;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        double v;
        while (ss >> v) m[key].push_back(v);
    }
    return m;
}

// The small CNN of tools/gen_torch_cnn_ref.py, forward and backward with the CPU
// reference: output and every gradient as PyTorch (double precision both).
TEST(ReferenceCnn, MatchesPyTorch) {
    auto t = load_torch_ref();
    ASSERT_FALSE(t.empty()) << "tests/data/torch_cnn.txt missing";
    const int batch = 2;
    const ref::ConvShape c1{3, 13, 11, 4, 5, 2, 0}, c2{4, 5, 4, 5, 3, 2, 1};
    ASSERT_EQ(c1.Ho(), 5);
    ASSERT_EQ(c1.Wo(), 4);
    ASSERT_EQ(c2.Ho() * c2.Wo() * c2.K, 30);
    std::vector<double> a1, h1, a2, h2, z, y;
    ref::conv2d_forward(t["x"], t["conv1_w"], t["conv1_b"], batch, c1, a1);
    h1 = a1;
    for (auto& v : h1) v = std::max(v, 0.0);
    ref::conv2d_forward(h1, t["conv2_w"], t["conv2_b"], batch, c2, a2);
    h2 = a2;
    for (auto& v : h2) v = std::max(v, 0.0);
    ref::linear_forward(h2, t["fc_w"], t["fc_b"], batch, 30, 6, z);
    ref::layernorm_forward(z, t["ln_g"], t["ln_b"], batch, 6, 1e-5, y);
    auto expect_close = [](const std::vector<double>& a, const std::vector<double>& b, const char* name) {
        ASSERT_EQ(a.size(), b.size()) << name;
        double worst = 0;
        for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]) / std::max(1.0, std::fabs(b[i])));
        EXPECT_LT(worst, 1e-10) << name;
    };
    expect_close(y, t["y"], "y");
    std::vector<double> dg, dbeta, dz, dW3, db3, dh2, dW2, db2, dh1, dW1, db1, dx;
    ref::layernorm_backward(z, t["ln_g"], t["dy"], batch, 6, 1e-5, dg, dbeta, dz);
    ref::linear_backward(h2, t["fc_w"], dz, batch, 30, 6, dW3, db3, dh2);
    for (size_t i = 0; i < dh2.size(); ++i) dh2[i] *= a2[i] > 0 ? 1.0 : 0.0;
    ref::conv2d_backward(h1, t["conv2_w"], dh2, batch, c2, dW2, db2, dh1);
    for (size_t i = 0; i < dh1.size(); ++i) dh1[i] *= a1[i] > 0 ? 1.0 : 0.0;
    ref::conv2d_backward(t["x"], t["conv1_w"], dh1, batch, c1, dW1, db1, dx);
    expect_close(dg, t["ln_dg"], "ln dgamma");
    expect_close(dbeta, t["ln_db"], "ln dbeta");
    expect_close(dW3, t["fc_dw"], "fc dW");
    expect_close(db3, t["fc_db"], "fc db");
    expect_close(dW2, t["conv2_dw"], "conv2 dW");
    expect_close(db2, t["conv2_db"], "conv2 db");
    expect_close(dW1, t["conv1_dw"], "conv1 dW");
    expect_close(db1, t["conv1_db"], "conv1 db");
    expect_close(dx, t["dx"], "dx");
}

}  // namespace
