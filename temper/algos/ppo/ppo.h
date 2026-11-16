#pragma once
// PPO (doc 5.1 - 5.4), on the GPU. Actor and critic are separate MLPs
// (3 x 256, ELU); log_std is a learned vector; one Adam over all of them.
//
// One iteration (doc 5.2), nothing leaves the GPU except the stats at the end:
//   for t in 0..horizon-1:
//       ppo.act(env.obs(), actions)      normalizer update, policy forward, sample
//       env.step(actions)
//       ppo.observe(env)                 reward, done flags, bootstrap at time limits
//   stats = ppo.update(env.obs())        GAE, epochs x minibatches of PPO loss, Adam
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "forge/util/config.h"
#include "forge/util/device_buffer.h"
#include "temper/common/checkpoint.h"
#include "temper/common/obs_normalizer.h"
#include "temper/nn/mlp.h"
#include "temper/optim/adam.h"

namespace temper {

struct PpoConfig {
    int horizon = 24;          // steps per env per iteration
    int epochs = 5;
    int minibatches = 4;
    float lr = 3e-4f;
    float clip = 0.2f;
    float gamma = 0.99f;
    float lambda = 0.95f;
    float vf_coef = 0.5f;      // CleanRL
    float ent_coef = 0.0f;
    float max_grad_norm = 1.0f;
    float obs_clip = 5.0f;
    float init_log_std = 0.0f; // std 1 at the start
    // Learning rate schedule. "fixed": lr all the time (CleanRL). "adaptive" (rsl_rl,
    // used by unitree_rl_gym): after each minibatch's forward pass, KL > 2 desired_kl
    // -> lr / 1.5, KL < desired_kl / 2 -> lr * 1.5, kept in [lr_min, lr_max]. Our KL is
    // the ratio estimator (r - 1) - log r (rsl_rl: the exact Gaussian KL). On the GPU,
    // no sync.
    std::string schedule = "fixed";
    float desired_kl = 0.01f, lr_min = 1e-5f, lr_max = 1e-2f;
    std::vector<int> hidden = {256, 256, 256};
    uint64_t seed = 1;

    // Keys with the prefix ppo. (doc 7.2): ppo.horizon, ppo.epochs, ppo.minibatches,
    // ppo.lr, ppo.clip, ppo.gamma, ppo.lambda, ppo.vf_coef, ppo.ent_coef,
    // ppo.max_grad_norm, ppo.obs_clip, ppo.init_log_std, ppo.schedule (fixed |
    // adaptive), ppo.desired_kl; seed.
    static PpoConfig from_config(const forge::Config& cfg);
};

struct PpoStats {  // means over all minibatches of one update
    float policy_loss = 0, value_loss = 0, entropy = 0, approx_kl = 0, clip_fraction = 0, grad_norm = 0, std = 0;
    float lr = 0;  // learning rate at the end of the update
};

class Ppo {
public:
    Ppo(int obs_dim, int act_dim, int num_envs, const PpoConfig& cfg, cudaStream_t stream);

    // Rollout step t (0..horizon-1). raw_obs and actions are device arrays
    // [num_envs x obs_dim], [num_envs x act_dim]. Actions are not clipped here
    // (the Env clips them; the log-prob is of the unclipped sample).
    void act(const float* raw_obs, float* actions);
    // After env.step: reward, terminated, truncated, final_obs (device arrays).
    void observe(const float* reward, const int* terminated, const int* truncated, const float* final_obs);
    // After 'horizon' act/observe pairs. raw_obs = the obs after the last step.
    // Syncs once at the end to read the stats.
    PpoStats update(const float* raw_obs);

    // Play mode: mean action of the normalized obs (no sampling, no normalizer update).
    void act_deterministic(const float* raw_obs, float* actions, int n);

    Checkpoint checkpoint(const std::string& task) const;   // weights, normalizer, Adam, iteration
    void load(const Checkpoint& c, bool with_optimizer);     // checks the sizes

    int64_t iteration() const { return iteration_; }
    float current_lr() const { return lr_now_; }
    bool rollout_full() const { return t_ == cfg_.horizon; }
    int obs_dim() const { return obs_dim_; }
    int act_dim() const { return act_dim_; }
    int num_envs() const { return num_envs_; }
    const PpoConfig& config() const { return cfg_; }

private:
    void collect_params();

    int obs_dim_, act_dim_, num_envs_;
    PpoConfig cfg_;
    cudaStream_t stream_;
    int batch_, mb_;            // horizon * num_envs, batch / minibatches
    int t_ = 0;                 // rollout step inside the iteration
    int64_t iteration_ = 0;

    Mlp actor_, critic_;
    Tensor log_std_, dlog_std_;
    ObsNormalizer norm_;
    std::vector<Tensor*> params_, grads_;
    std::unique_ptr<Adam> adam_;
    std::unique_ptr<GradClip> clip_;

    // Rollout buffer [horizon x num_envs] (doc 5.6: about 20 MB at 4096 envs)
    Tensor obs_, act_, logp_, value_, reward_, adv_, ret_;
    forge::DeviceBuffer<int> done_;
    Tensor fobs_norm_;          // num_envs x obs_dim: normalized final_obs / last obs
    // Minibatch scratch
    Tensor mb_obs_, mb_act_, mb_logp_, mb_adv_, mb_ret_, dmu_, dv_, dls_part_, stat_part_;
    Tensor stats_;              // accumulated per update: 5 numbers + grad norm
    Tensor lr_;                 // the learning rate, on the GPU (adaptive schedule changes it)
    Tensor kl_mb_;              // this minibatch's mean KL estimate
    float lr_now_ = 0.0f;       // lr_ as of the last update (host copy)
    forge::DeviceBuffer<uint32_t> keys_in_, keys_out_;
    forge::DeviceBuffer<int> idx_in_, idx_out_;
    forge::DeviceBuffer<unsigned char> sort_tmp_;
    size_t sort_tmp_bytes_ = 0;
};

}  // namespace temper
