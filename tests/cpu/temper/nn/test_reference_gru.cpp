// Phase 10: CPU reference for the GRU (temper/nn/reference_cpu.h), the answer key for
// temper::Gru. Every gradient (weights, biases, inputs, initial state) against finite
// differences, with episode resets inside the sequence and a gradient into the last
// state. (Also checked against PyTorch's torch.nn.GRUCell unrolled with the same
// resets: equal to 1e-15.)
#include <cmath>
#include <functional>
#include <random>
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

TEST(ReferenceGru, GradientsMatchFiniteDifferences) {
    std::mt19937 g(11);
    const ref::GruShape s{5, 3, 4, 6};
    const int G = 3 * s.H;
    std::vector<double> x = randn(size_t(s.T) * s.B * s.in, g), h0 = randn(size_t(s.B) * s.H, g);
    std::vector<double> Wih = randn(size_t(G) * s.in, g), Whh = randn(size_t(G) * s.H, g);
    std::vector<double> bih = randn(G, g), bhh = randn(G, g);
    for (auto* v : {&Wih, &Whh, &bih, &bhh})
        for (auto& w : *v) w *= 0.5;
    std::vector<double> reset(size_t(s.T) * s.B, 0.0);
    reset[2 * s.B + 1] = 1.0;  // env 1 starts an episode at step 2, env 0 at step 4
    reset[4 * s.B + 0] = 1.0;
    const std::vector<double> dy = randn(size_t(s.T) * s.B * s.H, g), dhT = randn(size_t(s.B) * s.H, g);
    // L = sum(y dy) + sum(h_T dhT)
    auto loss = [&]() {
        std::vector<double> y;
        ref::gru_forward(x, h0, reset, Wih, Whh, bih, bhh, s, y);
        double L = 0;
        for (size_t i = 0; i < y.size(); ++i) L += y[i] * dy[i];
        for (size_t i = 0; i < dhT.size(); ++i) L += y[size_t(s.T - 1) * s.B * s.H + i] * dhT[i];
        return L;
    };
    std::vector<double> dWih, dWhh, dbih, dbhh, dx, dh0;
    ref::gru_backward(x, h0, reset, Wih, Whh, bih, bhh, s, dy, dhT, dWih, dWhh, dbih, dbhh, dx, dh0);
    auto check = [&](std::vector<double>& p, const std::vector<double>& analytic, const char* name) {
        const double h = 1e-6;
        double worst = 0;
        for (size_t i = 0; i < p.size(); ++i) {
            const double keep = p[i];
            p[i] = keep + h;
            const double up = loss();
            p[i] = keep - h;
            const double down = loss();
            p[i] = keep;
            const double num = (up - down) / (2 * h);
            worst = std::max(worst, std::fabs(num - analytic[i]) / std::max(1.0, std::fabs(num)));
        }
        EXPECT_LT(worst, 1e-6) << name;
    };
    check(Wih, dWih, "dW_ih");
    check(Whh, dWhh, "dW_hh");
    check(bih, dbih, "db_ih");
    check(bhh, dbhh, "db_hh");
    check(x, dx, "dx");
    check(h0, dh0, "dh0");
}

}  // namespace
