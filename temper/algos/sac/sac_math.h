#pragma once
// SAC math per sample (FORGE_HD): the same code in the GPU kernels and in the
// CPU tests (finite differences).
//
// Actor output per action k: mean mu_k and raw_k.
//   log_std_k = LOG_STD_MIN + 0.5 (LOG_STD_MAX - LOG_STD_MIN) (tanh(raw_k) + 1)   (CleanRL)
//   u_k = mu_k + exp(log_std_k) eps_k,   a_k = tanh(u_k)
//   log pi(a) = sum_k [ -0.5 eps_k^2 - log_std_k - 0.5 log(2 pi) - log(1 - tanh(u_k)^2) ]
//   with the stable, exact form log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))
//   (CleanRL writes log(1 - a^2 + 1e-6); in float that loses all digits once tanh
//   saturates, so we use the stable form, and so does tools/sac_baseline.py)
// Actor loss over a batch of B: mean_b (alpha log pi(a_b) - min(Q1, Q2)(s_b, a_b))
#include <cmath>

#include "forge/math/hd.h"

namespace temper {

constexpr float kSacLogStdMin = -5.0f, kSacLogStdMax = 2.0f;

// log(1 - tanh(u)^2), stable for any u.
FORGE_HD float log1m_tanh2(float u) {
    const float x = -2.0f * u;
    const float softplus = x > 20.0f ? x : log1pf(expf(x));
    return 2.0f * (0.69314718055994531f - u - softplus);
}

FORGE_HD float sac_log_std(float raw) {
    return kSacLogStdMin + 0.5f * (kSacLogStdMax - kSacLogStdMin) * (tanhf(raw) + 1.0f);
}

// Action and log-prob for one sample.
FORGE_HD float sac_squash(const float* mu, const float* raw, const float* eps, int A, float* a) {
    float logp = 0.0f;
    for (int k = 0; k < A; ++k) {
        const float ls = sac_log_std(raw[k]);
        const float u = mu[k] + expf(ls) * eps[k];
        a[k] = tanhf(u);
        logp += -0.5f * eps[k] * eps[k] - ls - 0.91893853320467274f - log1m_tanh2(u);
    }
    return logp;
}

// Gradient of one sample's actor loss term (alpha log pi(a) - Qmin(a)) / B with
// respect to mu and raw, given dq_da = dQmin/da (from the critic's input
// gradient). eps is held fixed (reparameterization).
FORGE_HD void sac_actor_grad(const float* mu, const float* raw, const float* eps, const float* dq_da, float alpha,
                             float inv_b, int A, float* dmu, float* draw) {
    for (int k = 0; k < A; ++k) {
        const float t = tanhf(raw[k]);
        const float ls = kSacLogStdMin + 0.5f * (kSacLogStdMax - kSacLogStdMin) * (t + 1.0f);
        const float s = expf(ls);
        const float ak = tanhf(mu[k] + s * eps[k]);
        // d/du of (alpha log pi - Q(a)): -log(1 - tanh(u)^2) gives +2 tanh(u); a = tanh(u)
        const float dl_du = alpha * 2.0f * ak - dq_da[k] * (1.0f - ak * ak);
        dmu[k] = dl_du * inv_b;                               // u = mu + s eps
        const float dl_dls = dl_du * s * eps[k] - alpha;      // u depends on ls; log pi has -ls
        draw[k] = dl_dls * 0.5f * (kSacLogStdMax - kSacLogStdMin) * (1.0f - t * t) * inv_b;
    }
}

}  // namespace temper
