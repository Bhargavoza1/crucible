// The CPU reference MLP (the answer key for the GPU layers) checked against
// finite differences (doc 5.7), in double precision.
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "temper/nn/reference_cpu.h"

using temper::ref::Mlp;

namespace {

// loss = 0.5 * sum(y^2) so dL/dy = y.
double loss(Mlp& net, const std::vector<double>& x, int batch) {
    const auto& y = net.forward(x, batch);
    double s = 0;
    for (double v : y) s += 0.5 * v * v;
    return s;
}

TEST(ReferenceMlp, GradientsMatchFiniteDifferences) {
    std::mt19937 g(5);
    std::normal_distribution<double> n(0.0, 0.7);
    Mlp net;
    net.dims = {5, 7, 6, 3};
    for (size_t l = 0; l + 1 < net.dims.size(); ++l) {
        net.W.emplace_back(size_t(net.dims[l + 1]) * net.dims[l]);
        net.b.emplace_back(net.dims[l + 1]);
        for (auto& w : net.W.back()) w = n(g);
        for (auto& w : net.b.back()) w = n(g);
    }
    const int batch = 4;
    std::vector<double> x(size_t(batch) * 5);
    for (auto& v : x) v = n(g);
    net.forward(x, batch);
    net.backward(net.post.back());  // dL/dy = y
    const auto dW = net.dW, db = net.db;
    const auto dx = net.dx;

    const double h = 1e-6;
    double worst = 0;
    auto check = [&](double& param, double analytic) {
        const double keep = param;
        param = keep + h;
        const double lp = loss(net, x, batch);
        param = keep - h;
        const double lm = loss(net, x, batch);
        param = keep;
        const double fd = (lp - lm) / (2 * h);
        worst = std::max(worst, std::fabs(fd - analytic) / std::max(1.0, std::fabs(fd)));
    };
    for (size_t l = 0; l < net.W.size(); ++l) {
        for (size_t k = 0; k < net.W[l].size(); ++k) check(net.W[l][k], dW[l][k]);
        for (size_t k = 0; k < net.b[l].size(); ++k) check(net.b[l][k], db[l][k]);
    }
    for (size_t k = 0; k < x.size(); ++k) check(x[k], dx[k]);
    EXPECT_LT(worst, 1e-7) << "finite differences vs backward";
}

}  // namespace
