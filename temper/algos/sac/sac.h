#pragma once
// SAC and TQC (doc 5.1), on the GPU. One class: TQC is SAC with quantile critics.
//
// Shared (same as tools/sac_baseline.py):
//   actor (obs -> 256 -> 256 -> 2 x act_dim, ReLU): mean and raw log_std, tanh-Gaussian
//   n critics + n target critics (obs + act -> 256 -> 256 -> n_quantiles, ReLU)
//   learned alpha (target entropy -act_dim); soft target update tau every gradient step
//   many envs: every env step adds num_envs transitions; update() does
//   updates_per_step gradient steps on random batches; random actions until the
//   buffer holds learning_starts transitions. No obs normalization (like CleanRL).
//
// SAC  (n_critics 2, n_quantiles 1, like CleanRL):
//   y = r + gamma (1 - terminated) (min_c Q'_c(s', a') - alpha log pi(a'|s'))
//   critic loss sum_c mean_b (Q_c - y)^2;  actor loss mean(alpha log pi - min_c Q_c)
// TQC  (n_critics 5, n_quantiles 25, drop 2 per critic, like SB3-contrib):
//   target atoms: all n x N target quantiles at (s', a'), sorted, the top
//   n x drop dropped -> M atoms;  y_j = r + gamma (1 - terminated) (atom_j - alpha log pi')
//   critic loss: quantile Huber (kappa 1), mean over batch, critics, N quantiles, M atoms:
//     |tau_k - 1(y_j - z_ck < 0)| * huber(y_j - z_ck),  tau_k = (k + 0.5) / N
//   actor loss mean(alpha log pi - mean over critics and quantiles of Z(s, a))
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "temper/algos/algorithm.h"
#include "temper/buffers/replay/replay_buffer.h"
#include "temper/nn/mlp.h"
#include "temper/optim/adam.h"

namespace temper {

struct SacConfig {
    bool tqc = false;
    int n_critics = 2;
    int n_quantiles = 1;
    int drop_per_critic = 0;  // TQC: top quantiles dropped per critic
    int buffer_size = 1000000;
    int batch_size = 512;
    int updates_per_step = 4;
    int learning_starts = 2000;  // transitions
    int steps_per_iteration = 24;
    float gamma = 0.99f, tau = 0.005f;
    float q_lr = 1e-3f, policy_lr = 3e-4f, alpha_lr = 1e-3f;
    std::vector<int> hidden = {256, 256};
    uint64_t seed = 1;

    static SacConfig sac_defaults() { return {}; }
    static SacConfig tqc_defaults();
    // Keys with prefix sac. (algo sac) or tqc. (algo tqc): buffer_size, batch_size,
    // updates_per_step, learning_starts, steps_per_iteration, gamma, tau, q_lr,
    // policy_lr, alpha_lr; tqc. only: n_critics, n_quantiles, drop_per_critic. And seed.
    static SacConfig from_config(const forge::Config* cfg, bool tqc = false);
    int kept_atoms() const { return n_critics * (n_quantiles - drop_per_critic); }  // TQC M
};

// Debug: everything one gradient step used and made (batch, noise, targets,
// gradients), so tools/sac_check_step.py can redo the step in PyTorch and compare.
struct SacStepCapture {
    std::map<std::string, std::vector<float>> arrays;
};

class Sac : public Algorithm {
public:
    Sac(int obs_dim, int act_dim, int num_envs, const SacConfig& cfg, cudaStream_t stream);

    // The next gradient step fills *c (slow: syncs several times). One step only.
    void capture_next_step(SacStepCapture* c) { capture_ = c; }

    const char* name() const override { return cfg_.tqc ? "tqc" : "sac"; }
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
    // with_optimizer = false (play mode): loads the actor only.
    void load(const Checkpoint& c, bool with_optimizer) override;
    int64_t iteration() const override { return iteration_; }
    const SacConfig& config() const { return cfg_; }

private:
    void gradient_step();
    void grab(const std::string& name, const Tensor& t, size_t n);

    int D_, A_, N_;
    SacConfig cfg_;
    cudaStream_t stream_;
    int64_t iteration_ = 0, env_steps_ = 0, grad_steps_ = 0;
    ReplayBuffer replay_;
    Mlp actor_;
    std::vector<std::unique_ptr<Mlp>> q_, qt_;  // critics and target critics
    Tensor log_alpha_, dlog_alpha_;
    std::vector<Tensor*> pq_, gq_, pqt_, pa_, ga_;  // pqt_: same order as pq_
    std::unique_ptr<Adam> q_opt_, pi_opt_, alpha_opt_;
    // per env step
    Tensor obs_now_, act_now_;
    // per gradient step (batch rows)
    Tensor b_obs_, b_act_, b_rew_, b_next_, b_done_, eps_, act_new_, logp_, y_, qin_, dout_, part_;
    Tensor zt_;                  // target critic outputs [n x B x N]
    Tensor z_;                   // critic outputs [n x B x N] (copied from each net)
    Tensor dz_;                  // gradient of the loss wrt z [n x B x N]
    Tensor dx_, dx_sum_;         // critic input gradients [B x (D + A)]
    Tensor qloss_elem_;          // TQC: loss per (b, c, k)
    SacStepCapture* capture_ = nullptr;
    Tensor stats_;  // sums: q_loss, actor_loss, alpha, entropy, q_mean, count
};

}  // namespace temper
