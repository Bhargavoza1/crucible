#pragma once
// PPO math per sample / per env (FORGE_HD): the same code in the GPU kernels and
// in the CPU tests (finite differences, reference GAE).
//
// Policy: Gaussian, mean from the actor MLP, std = exp(log_std) (a learned vector).
//   log p(a) = sum_k [ -0.5 ((a_k - mu_k) / s_k)^2 - log s_k - 0.5 log(2 pi) ]
// Loss over a minibatch of B samples (CleanRL's PPO):
//   L = mean_b[ -min(r A, clip(r, 1-eps, 1+eps) A) ] - ent_coef * H + vf_coef * 0.5 * mean_b (V - R)^2
//   r = exp(logp_new - logp_old), H = sum_k (log s_k + 0.5 log(2 pi e)) (same for every sample)
#include <cmath>

#include "forge/math/hd.h"

namespace temper {

constexpr float kHalfLog2Pi = 0.91893853320467274f;  // 0.5 * log(2 pi)

FORGE_HD float gaussian_logp(const float* a, const float* mu, const float* log_std, int A) {
    float s = 0.0f;
    for (int k = 0; k < A; ++k) {
        const float z = (a[k] - mu[k]) * expf(-log_std[k]);
        s += -0.5f * z * z - log_std[k] - kHalfLog2Pi;
    }
    return s;
}

// Entropy of the Gaussian (the same for every sample).
FORGE_HD float gaussian_entropy(const float* log_std, int A) {
    float s = 0.0f;
    for (int k = 0; k < A; ++k) s += log_std[k] + kHalfLog2Pi + 0.5f;
    return s;
}

struct PpoSampleOut {
    float policy_loss;  // this sample's term before the 1/B mean
    float g_logp;       // dL/d(logp_new) for this sample (includes the 1/B)
    float dv;           // dL/dV (includes the 1/B and vf_coef)
    float value_loss;   // 0.5 (V - R)^2
    float ratio;
    float approx_kl;    // (r - 1) - log r   (always >= 0)
    float clipped;      // 1 if |r - 1| > eps
};

// One sample. The gradients for mu and log_std follow from g_logp:
//   dL/dmu_k     = g_logp * (a_k - mu_k) / s_k^2
//   dL/dlog_s_k  = g_logp * ((a_k - mu_k)^2 / s_k^2 - 1)     (+ the entropy term, once)
FORGE_HD PpoSampleOut ppo_sample(float logp_new, float logp_old, float adv, float v, float ret, float clip_eps,
                                        float vf_coef, float inv_batch) {
    PpoSampleOut o;
    const float logr = logp_new - logp_old;
    const float r = expf(logr);
    const float lo = 1.0f - clip_eps, hi = 1.0f + clip_eps;
    const float rc = fminf(fmaxf(r, lo), hi);
    const float unclipped = r * adv, clipped = rc * adv;
    o.policy_loss = -fminf(unclipped, clipped);
    // The min picks the unclipped term unless the clipped one is smaller; then r
    // is outside [lo, hi] and the term does not depend on r: gradient 0.
    const bool use_unclipped = unclipped <= clipped;
    o.g_logp = use_unclipped ? -adv * r * inv_batch : 0.0f;  // d(-r A)/dlogp = -r A
    o.value_loss = 0.5f * (v - ret) * (v - ret);
    o.dv = vf_coef * (v - ret) * inv_batch;
    o.ratio = r;
    o.approx_kl = (r - 1.0f) - logr;
    o.clipped = fabsf(r - 1.0f) > clip_eps ? 1.0f : 0.0f;
    return o;
}

// GAE for one env over T steps (doc 3.6 rules). Arrays are indexed [t].
//   reward[t] already includes gamma * V(final_obs) where step t was truncated
//   (bootstrap at a time limit); done[t] = terminated or truncated (cuts the trace).
//   value[t] = V(obs_t), last_value = V(obs after step T-1).
//   adv[t] = delta_t + gamma lambda (1 - done_t) adv[t+1]
//   delta_t = r_t + gamma (1 - done_t) V_{t+1} - V_t,   ret[t] = adv[t] + value[t]
// stride: distance between t and t+1 in the arrays (num_envs for [T x N] storage).
FORGE_HD void gae_env(const float* reward, const int* done, const float* value, float last_value, int T,
                             int stride, float gamma, float lambda, float* adv, float* ret) {
    float next_adv = 0.0f, next_value = last_value;
    for (int t = T - 1; t >= 0; --t) {
        const size_t i = size_t(t) * stride;
        const float notdone = done[i] ? 0.0f : 1.0f;
        const float delta = reward[i] + gamma * notdone * next_value - value[i];
        next_adv = delta + gamma * lambda * notdone * next_adv;
        adv[i] = next_adv;
        ret[i] = next_adv + value[i];
        next_value = value[i];
    }
}

// Running mean / variance (population variance), merged with a new batch
// (Chan et al.), like gym's RunningMeanStd. count is a float: plenty for
// billions of samples. Mean and var are per dimension.
FORGE_HD void running_merge(float& mean, float& var, float count, float b_mean, float b_var, float b_count) {
    const float total = count + b_count;
    const float delta = b_mean - mean;
    const float new_mean = mean + delta * (b_count / total);
    const float m2 = var * count + b_var * b_count + delta * delta * count * b_count / total;
    mean = new_mean;
    var = m2 / total;
}

// Normalized and clipped observation value.
FORGE_HD float normalize_obs(float x, float mean, float var, float clip) {
    const float z = (x - mean) / sqrtf(var + 1e-8f);
    return fminf(fmaxf(z, -clip), clip);
}

}  // namespace temper
