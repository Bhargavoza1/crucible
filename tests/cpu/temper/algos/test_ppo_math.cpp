// PPO math (temper/algos/ppo/ppo_math.h) checked on the CPU: the loss gradients
// against finite differences, GAE against a plain reference, the running
// normalizer against a direct mean and variance.
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "temper/algos/ppo/ppo_math.h"

using namespace temper;

namespace {

// The full minibatch loss in double, straight from its definition.
struct Batch {
    int B, A;
    std::vector<double> a, logp_old, adv, ret;
};

double loss(const Batch& b, const std::vector<double>& mu, const std::vector<double>& log_std,
            const std::vector<double>& v, double eps, double vf, double ent) {
    double L = 0;
    for (int i = 0; i < b.B; ++i) {
        double lp = 0;
        for (int k = 0; k < b.A; ++k) {
            const double z = (b.a[i * b.A + k] - mu[i * b.A + k]) / std::exp(log_std[k]);
            lp += -0.5 * z * z - log_std[k] - 0.5 * std::log(2 * M_PI);
        }
        const double r = std::exp(lp - b.logp_old[i]);
        const double rc = std::min(std::max(r, 1 - eps), 1 + eps);
        L += -std::min(r * b.adv[i], rc * b.adv[i]) / b.B;
        L += vf * 0.5 * (v[i] - b.ret[i]) * (v[i] - b.ret[i]) / b.B;
    }
    double H = 0;
    for (int k = 0; k < b.A; ++k) H += log_std[k] + 0.5 * std::log(2 * M_PI) + 0.5;
    return L - ent * H;
}

TEST(PpoMath, LossGradientsMatchFiniteDifferences) {
    std::mt19937 g(3);
    std::normal_distribution<double> n(0, 1);
    Batch b{16, 3, {}, {}, {}, {}};
    std::vector<double> mu(b.B * b.A), log_std = {-0.3, 0.1, 0.4}, v(b.B);
    for (int i = 0; i < b.B * b.A; ++i) {
        mu[i] = n(g);
        b.a.push_back(mu[i] + 0.8 * n(g));
    }
    for (int i = 0; i < b.B; ++i) {
        // logp_old near logp_new, so some ratios are inside and some outside the clip range.
        double lp = 0;
        for (int k = 0; k < b.A; ++k) {
            const double z = (b.a[i * b.A + k] - mu[i * b.A + k]) / std::exp(log_std[k]);
            lp += -0.5 * z * z - log_std[k] - 0.5 * std::log(2 * M_PI);
        }
        b.logp_old.push_back(lp + 0.4 * n(g));
        b.adv.push_back(n(g));
        b.ret.push_back(n(g));
        v[i] = n(g);
    }
    const double eps = 0.2, vf = 0.5, ent = 0.01;

    // Analytic gradients from ppo_sample (float), as the GPU computes them.
    std::vector<double> dmu(b.B * b.A), dls(b.A, -ent), dv(b.B);
    std::vector<float> lsf(log_std.begin(), log_std.end());
    for (int i = 0; i < b.B; ++i) {
        std::vector<float> ai(b.a.begin() + i * b.A, b.a.begin() + (i + 1) * b.A);
        std::vector<float> mi(mu.begin() + i * b.A, mu.begin() + (i + 1) * b.A);
        const float lp = gaussian_logp(ai.data(), mi.data(), lsf.data(), b.A);
        const PpoSampleOut o = ppo_sample(lp, float(b.logp_old[i]), float(b.adv[i]), float(v[i]), float(b.ret[i]),
                                          float(eps), float(vf), 1.0f / b.B);
        dv[i] = o.dv;
        for (int k = 0; k < b.A; ++k) {
            const double s2 = std::exp(2 * log_std[k]), d = b.a[i * b.A + k] - mu[i * b.A + k];
            dmu[i * b.A + k] = o.g_logp * d / s2;
            dls[k] += o.g_logp * (d * d / s2 - 1);
        }
    }

    const double h = 1e-6;
    double worst = 0;
    auto fd = [&](double& x, double analytic) {
        const double keep = x;
        x = keep + h;
        const double lp = loss(b, mu, log_std, v, eps, vf, ent);
        x = keep - h;
        const double lm = loss(b, mu, log_std, v, eps, vf, ent);
        x = keep;
        worst = std::max(worst, std::fabs((lp - lm) / (2 * h) - analytic));
    };
    for (int i = 0; i < b.B * b.A; ++i) fd(mu[i], dmu[i]);
    for (int k = 0; k < b.A; ++k) fd(log_std[k], dls[k]);
    for (int i = 0; i < b.B; ++i) fd(v[i], dv[i]);
    EXPECT_LT(worst, 1e-5) << "PPO loss gradient vs finite differences";
}

TEST(PpoMath, GaeMatchesReferenceWithTerminationAndTruncation) {
    // 2 envs x 6 steps, stored [t][env]. Env 0 terminates at t = 2; env 1 is
    // truncated at t = 3 (its reward there already holds gamma * V(final_obs)).
    const int T = 6, N = 2;
    const float gamma = 0.99f, lambda = 0.95f;
    std::vector<float> r = {1, 0.5f, 1, 2, 0.2f, 1, 1, 1, 1, 0.3f, 1, 1}, val(T * N);
    std::vector<int> done(T * N, 0);
    for (int i = 0; i < T * N; ++i) val[i] = 0.1f * float(i % 5) - 0.2f;
    done[2 * N + 0] = 1;
    done[3 * N + 1] = 1;
    const float last[N] = {0.7f, -0.4f};
    std::vector<float> adv(T * N), ret(T * N);
    for (int e = 0; e < N; ++e)
        gae_env(r.data() + e, done.data() + e, val.data() + e, last[e], T, N, gamma, lambda, adv.data() + e,
                ret.data() + e);
    for (int e = 0; e < N; ++e) {
        // Reference: direct sum of discounted deltas inside each episode piece.
        for (int t = 0; t < T; ++t) {
            double a = 0, w = 1;
            for (int k = t; k < T; ++k) {
                const int i = k * N + e;
                const double nv = done[i] ? 0.0 : (k + 1 < T ? val[(k + 1) * N + e] : last[e]);
                a += w * (r[i] + gamma * nv - val[i]);
                if (done[i]) break;
                w *= gamma * lambda;
            }
            EXPECT_NEAR(adv[t * N + e], a, 1e-5) << "env " << e << " t " << t;
            EXPECT_NEAR(ret[t * N + e], a + val[t * N + e], 1e-5);
        }
    }
}

TEST(PpoMath, RunningMergeEqualsDirectStatistics) {
    std::mt19937 g(9);
    std::normal_distribution<float> n(3.0f, 2.0f);
    std::vector<float> all;
    float mean = 0, var = 1, count = 1e-4f;  // gym's RunningMeanStd start
    for (int batch = 0; batch < 50; ++batch) {
        std::vector<float> x(256);
        for (auto& v : x) v = n(g);
        double bm = 0, bv = 0;
        for (float v : x) bm += v / 256.0;
        for (float v : x) bv += (v - bm) * (v - bm) / 256.0;
        running_merge(mean, var, count, float(bm), float(bv), 256.0f);
        count += 256.0f;
        all.insert(all.end(), x.begin(), x.end());
    }
    double m = 0, v = 0;
    for (float x : all) m += x / all.size();
    for (float x : all) v += (x - m) * (x - m) / all.size();
    EXPECT_NEAR(mean, m, 1e-3);
    EXPECT_NEAR(var, v, 1e-2);
    EXPECT_FLOAT_EQ(normalize_obs(100.0f, 0.0f, 1.0f, 5.0f), 5.0f);  // clipped
}

}  // namespace
