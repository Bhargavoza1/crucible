// SAC actor math (temper/algos/sac/sac_math.h) vs finite differences, in double.
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "temper/algos/sac/sac_math.h"

using namespace temper;

namespace {

// A smooth stand-in for min(Q1, Q2): Q(a) = sum c_k a_k + 0.5 w_k a_k^2.
struct FakeQ {
    std::vector<double> c, w;
    double value(const std::vector<double>& a) const {
        double q = 0;
        for (size_t k = 0; k < a.size(); ++k) q += c[k] * a[k] + 0.5 * w[k] * a[k] * a[k];
        return q;
    }
};

// The loss (alpha log pi - Q) in double, straight from the definition.
double loss(const std::vector<double>& mu, const std::vector<double>& raw, const std::vector<double>& eps,
            double alpha, const FakeQ& q) {
    std::vector<double> a(mu.size());
    double logp = 0;
    for (size_t k = 0; k < mu.size(); ++k) {
        const double ls = -5.0 + 0.5 * 7.0 * (std::tanh(raw[k]) + 1.0);
        const double u = mu[k] + std::exp(ls) * eps[k];
        a[k] = std::tanh(u);
        logp += -0.5 * eps[k] * eps[k] - ls - 0.5 * std::log(2 * M_PI) - std::log(1.0 / std::cosh(u) / std::cosh(u));
    }
    return alpha * logp - q.value(a);
}

TEST(SacMath, ActorGradientMatchesFiniteDifferences) {
    std::mt19937 g(4);
    std::normal_distribution<double> n(0, 1);
    const int A = 4;
    double worst = 0;
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<double> mu(A), raw(A), eps(A);
        FakeQ q{std::vector<double>(A), std::vector<double>(A)};
        for (int k = 0; k < A; ++k) {
            mu[k] = 0.5 * n(g);
            raw[k] = 0.8 * n(g);
            eps[k] = n(g);
            q.c[k] = n(g);
            q.w[k] = n(g);
        }
        const double alpha = 0.2;
        // Analytic, float, as the GPU does it.
        std::vector<float> fmu(mu.begin(), mu.end()), fraw(raw.begin(), raw.end()), feps(eps.begin(), eps.end());
        std::vector<float> a(A), dq(A), dmu(A), draw(A);
        sac_squash(fmu.data(), fraw.data(), feps.data(), A, a.data());
        for (int k = 0; k < A; ++k) dq[k] = float(q.c[k] + q.w[k] * a[k]);
        sac_actor_grad(fmu.data(), fraw.data(), feps.data(), dq.data(), float(alpha), 1.0f, A, dmu.data(), draw.data());
        const double h = 1e-6;
        for (int k = 0; k < A; ++k) {
            for (auto [vec, an] : {std::pair{&mu, dmu[k]}, std::pair{&raw, draw[k]}}) {
                const double keep = (*vec)[k];
                (*vec)[k] = keep + h;
                const double lp = loss(mu, raw, eps, alpha, q);
                (*vec)[k] = keep - h;
                const double lm = loss(mu, raw, eps, alpha, q);
                (*vec)[k] = keep;
                const double fd = (lp - lm) / (2 * h);
                worst = std::max(worst, std::fabs(fd - an) / std::max(1.0, std::fabs(fd)));
            }
        }
    }
    EXPECT_LT(worst, 1e-4) << "SAC actor gradient vs finite differences (float analytic)";
}

TEST(SacMath, SquashMatchesDefinition) {
    const float mu[2] = {0.3f, -1.2f}, raw[2] = {0.0f, 2.0f}, eps[2] = {0.5f, -0.7f};
    float a[2];
    const float logp = sac_squash(mu, raw, eps, 2, a);
    double ref = 0;
    for (int k = 0; k < 2; ++k) {
        const double ls = -5.0 + 3.5 * (std::tanh(double(raw[k])) + 1.0);
        const double u = mu[k] + std::exp(ls) * eps[k];
        EXPECT_NEAR(a[k], std::tanh(u), 1e-6);
        ref += -0.5 * eps[k] * eps[k] - ls - 0.5 * std::log(2 * M_PI) + 2.0 * std::log(std::cosh(u));  // -log(sech^2 u)
    }
    EXPECT_NEAR(logp, ref, 1e-4);
    EXPECT_LE(std::fabs(a[0]), 1.0f);
}

}  // namespace
