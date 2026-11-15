#pragma once
// SAC (doc 5.1), on the GPU. Same algorithm as tools/sac_baseline.py:
//   actor (obs -> 256 -> 256 -> 2 x act_dim, ReLU): mean and raw log_std, tanh-Gaussian
//   two critics and two target critics (obs + act -> 256 -> 256 -> 1, ReLU)
//   y = r + gamma (1 - terminated) (min Q'(s', a') - alpha log pi(a'|s'))
//   critic loss mean (Q1 - y)^2 + mean (Q2 - y)^2; actor loss mean(alpha log pi - min Q)
//   learned alpha (target entropy -act_dim); soft target update tau every gradient step
// Many envs: every env step adds num_envs transitions; update() does
// updates_per_step gradient steps on random batches. Random actions until the
// buffer holds learning_starts transitions. No obs normalization (like CleanRL).
#include <memory>

#include "forge/util/device_buffer.h"
#include "temper/algos/algorithm.h"
#include "temper/buffers/replay/replay_buffer.h"
#include "temper/nn/mlp.h"
#include "temper/optim/adam.h"

namespace temper {

struct SacConfig {
    int buffer_size = 1000000;
    int batch_size = 512;
    int updates_per_step = 4;
    int learning_starts = 2000;  // transitions
    int steps_per_iteration = 24;
    float gamma = 0.99f, tau = 0.005f;
    float q_lr = 1e-3f, policy_lr = 3e-4f, alpha_lr = 1e-3f;
    std::vector<int> hidden = {256, 256};
    uint64_t seed = 1;
    // Keys (prefix sac.): sac.buffer_size, sac.batch_size, sac.updates_per_step,
    // sac.learning_starts, sac.steps_per_iteration, sac.gamma, sac.tau, sac.q_lr,
    // sac.policy_lr, sac.alpha_lr; seed.
    static SacConfig from_config(const forge::Config* cfg);
};

class Sac : public Algorithm {
public:
    Sac(int obs_dim, int act_dim, int num_envs, const SacConfig& cfg, cudaStream_t stream);

    const char* name() const override { return "sac"; }
    int steps_per_iteration() const override { return cfg_.steps_per_iteration; }
    void act(const float* obs, float* actions) override;
    void act_deterministic(const float* obs, float* actions, int n) override;
    void observe(const TransitionView& t) override;
    bool ready_to_update() const override { return replay_.size() >= cfg_.learning_starts; }
    void update(const float* obs) override;
    std::vector<std::string> stat_names() const override;
    std::vector<double> read_stats() override;
    double learning_rate() const override { return cfg_.policy_lr; }
    Checkpoint checkpoint(const std::string& task) const override;
    void load(const Checkpoint& c, bool with_optimizer) override;
    int64_t iteration() const override { return iteration_; }

private:
    void gradient_step();

    int D_, A_, N_;
    SacConfig cfg_;
    cudaStream_t stream_;
    int64_t iteration_ = 0, env_steps_ = 0, grad_steps_ = 0;
    ReplayBuffer replay_;
    Mlp actor_, q1_, q2_, q1t_, q2t_;
    Tensor log_alpha_, dlog_alpha_;
    std::vector<Tensor*> pq_, gq_, pa_, ga_;
    std::vector<Tensor*> pqt_;  // target critic params, same order as pq_
    std::unique_ptr<Adam> q_opt_, pi_opt_, alpha_opt_;
    // per env step
    Tensor obs_now_, act_now_;
    // per gradient step (batch rows)
    Tensor b_obs_, b_act_, b_rew_, b_next_, b_done_, eps_, act_new_, logp_, y_, qin_, dq1_, dq2_, dx1_, dx2_, dout_,
        part_;
    Tensor stats_;  // sums: q_loss, actor_loss, alpha, entropy, q_mean, count
};

}  // namespace temper
