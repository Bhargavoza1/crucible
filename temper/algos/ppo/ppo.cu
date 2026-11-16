#include "temper/algos/ppo/ppo.h"

#include <cub/cub.cuh>

#include <cmath>
#include <stdexcept>

#include "forge/util/cuda_check.h"
#include "forge/util/nvtx.h"
#include "forge/util/rng.h"
#include "temper/algos/ppo/ppo_math.h"

namespace temper {

namespace {

constexpr int kThreads = 256;
int blocks_for(size_t n) { return int((n + kThreads - 1) / kThreads); }

// Two standard normal numbers (Box-Muller).
__device__ void gaussian2(forge::Rng& r, float& a, float& b) {
    const float u1 = fmaxf(r.uniform(), 1e-12f), u2 = r.uniform();
    const float rad = sqrtf(-2.0f * logf(u1));
    a = rad * cosf(6.2831853f * u2);
    b = rad * sinf(6.2831853f * u2);
}

// a = mu + std * eps; log-prob, value and the action into the rollout at step t.
__global__ void sample_kernel(int N, int A, const float* mu, const float* log_std, const float* v, uint64_t seed,
                              uint64_t stream_id, float* act_t, float* logp_t, float* value_t, float* actions) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= N) return;
    forge::Rng rng(seed ^ 0x5050505050ull, stream_id, uint64_t(e));
    float spare = 0.0f;
    for (int k = 0; k < A; ++k) {
        float eps;
        if (k % 2 == 0) gaussian2(rng, eps, spare);
        else eps = spare;
        const float a = mu[size_t(e) * A + k] + expf(log_std[k]) * eps;
        act_t[size_t(e) * A + k] = a;
        actions[size_t(e) * A + k] = a;
    }
    logp_t[e] = gaussian_logp(act_t + size_t(e) * A, mu + size_t(e) * A, log_std, A);
    value_t[e] = v[e];
}

// Reward with the time-limit bootstrap (doc 3.6): r + gamma V(final_obs) where truncated.
__global__ void observe_kernel(int N, const float* reward, const int* term, const int* trunc, const float* v_final,
                               float gamma, float* reward_t, int* done_t) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= N) return;
    reward_t[e] = reward[e] + (trunc[e] ? gamma * v_final[e] : 0.0f);
    done_t[e] = (term[e] || trunc[e]) ? 1 : 0;
}

__global__ void gae_kernel(int N, int T, const float* reward, const int* done, const float* value,
                           const float* last_value, float gamma, float lambda, float* adv, float* ret) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= N) return;
    gae_env(reward + e, done + e, value + e, last_value[e], T, N, gamma, lambda, adv + e, ret + e);
}

__global__ void keys_kernel(int n, uint64_t seed, uint64_t stream_id, uint32_t* keys, int* idx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    forge::Rng rng(seed ^ 0x7777777777ull, stream_id, uint64_t(i));
    keys[i] = uint32_t(rng.next() >> 32);
    idx[i] = i;
}

__global__ void gather_kernel(int mb, int D, int A, const int* idx, const float* obs, const float* act,
                              const float* logp, const float* adv, const float* ret, float* mb_obs, float* mb_act,
                              float* mb_logp, float* mb_adv, float* mb_ret) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mb) return;
    const int j = idx[i];
    for (int k = 0; k < D; ++k) mb_obs[size_t(i) * D + k] = obs[size_t(j) * D + k];
    for (int k = 0; k < A; ++k) mb_act[size_t(i) * A + k] = act[size_t(j) * A + k];
    mb_logp[i] = logp[j];
    mb_adv[i] = adv[j];
    mb_ret[i] = ret[j];
}

// Block-wide sum of one value per thread (fixed order: deterministic).
__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
        __syncthreads();
    }
    const float s = sh[0];
    __syncthreads();
    return s;
}

// Advantages of one minibatch: mean 0, std 1 (CleanRL: unbiased std, per minibatch).
__global__ void adv_norm_kernel(int n, float* adv) {
    __shared__ float sh[1024];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += adv[i];
    const float mean = block_sum(s, sh) / float(n);
    float q = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) q += (adv[i] - mean) * (adv[i] - mean);
    const float var = block_sum(q, sh) / float(n > 1 ? n - 1 : 1);
    const float inv = 1.0f / (sqrtf(var) + 1e-8f);
    for (int i = threadIdx.x; i < n; i += blockDim.x) adv[i] = (adv[i] - mean) * inv;
}

// Per sample: PPO loss terms and gradients (ppo_math.h).
__global__ void loss_kernel(int mb, int A, const float* mu, const float* v, const float* log_std, const float* act,
                            const float* logp_old, const float* adv, const float* ret, float clip, float vf_coef,
                            float* dmu, float* dv, float* dls_part, float* stat_part) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mb) return;
    const float* a = act + size_t(i) * A;
    const float* m = mu + size_t(i) * A;
    const float lp = gaussian_logp(a, m, log_std, A);
    const PpoSampleOut o = ppo_sample(lp, logp_old[i], adv[i], v[i], ret[i], clip, vf_coef, 1.0f / float(mb));
    for (int k = 0; k < A; ++k) {
        const float inv_var = expf(-2.0f * log_std[k]);
        const float d = a[k] - m[k];
        dmu[size_t(i) * A + k] = o.g_logp * d * inv_var;
        dls_part[size_t(i) * A + k] = o.g_logp * (d * d * inv_var - 1.0f);
    }
    dv[i] = o.dv;
    float* s = stat_part + size_t(i) * 4;
    s[0] = o.policy_loss;
    s[1] = o.value_loss;
    s[2] = o.approx_kl;
    s[3] = o.clipped;
}

// One block: dlog_std = sum over samples - ent_coef; the 4 stats (means) are
// added to the update's totals.
__global__ void reduce_kernel(int mb, int A, const float* dls_part, const float* stat_part, float ent_coef,
                              float* dlog_std, float* stats, float* kl_mb) {
    __shared__ float sh[1024];
    for (int k = 0; k < A; ++k) {
        float s = 0.0f;
        for (int i = threadIdx.x; i < mb; i += blockDim.x) s += dls_part[size_t(i) * A + k];
        s = block_sum(s, sh);
        if (threadIdx.x == 0) dlog_std[k] = s - ent_coef;
    }
    for (int j = 0; j < 4; ++j) {
        float s = 0.0f;
        for (int i = threadIdx.x; i < mb; i += blockDim.x) s += stat_part[size_t(i) * 4 + j];
        s = block_sum(s, sh);
        if (threadIdx.x == 0) stats[j] += s / float(mb);
        if (threadIdx.x == 0 && j == 2) *kl_mb = s / float(mb);
    }
}

// rsl_rl's adaptive learning rate (see PpoConfig::schedule).
__global__ void adapt_lr_kernel(const float* kl, float* lr, float desired, float lo, float hi) {
    const float k = *kl;
    if (k > 2.0f * desired) *lr = fmaxf(*lr / 1.5f, lo);
    else if (k < 0.5f * desired && k > 0.0f) *lr = fminf(*lr * 1.5f, hi);
}

__global__ void add_kernel(float* dst, const float* src) { *dst += *src; }

}  // namespace

PpoConfig PpoConfig::from_config(const forge::Config& c) {
    PpoConfig p;
    p.horizon = c.get_int("ppo.horizon", p.horizon);
    p.epochs = c.get_int("ppo.epochs", p.epochs);
    p.minibatches = c.get_int("ppo.minibatches", p.minibatches);
    p.lr = c.get_float("ppo.lr", p.lr);
    p.clip = c.get_float("ppo.clip", p.clip);
    p.gamma = c.get_float("ppo.gamma", p.gamma);
    p.lambda = c.get_float("ppo.lambda", p.lambda);
    p.vf_coef = c.get_float("ppo.vf_coef", p.vf_coef);
    p.ent_coef = c.get_float("ppo.ent_coef", p.ent_coef);
    p.max_grad_norm = c.get_float("ppo.max_grad_norm", p.max_grad_norm);
    p.obs_clip = c.get_float("ppo.obs_clip", p.obs_clip);
    p.init_log_std = c.get_float("ppo.init_log_std", p.init_log_std);
    p.schedule = c.get_string("ppo.schedule", p.schedule);
    p.desired_kl = c.get_float("ppo.desired_kl", p.desired_kl);
    if (p.schedule != "fixed" && p.schedule != "adaptive")
        throw forge::ConfigError("ppo.schedule must be fixed or adaptive, not '" + p.schedule + "'");
    p.seed = static_cast<uint64_t>(c.get_int("seed", 1));
    return p;
}

static std::vector<int> dims_of(int in, const std::vector<int>& hidden, int out) {
    std::vector<int> d = {in};
    d.insert(d.end(), hidden.begin(), hidden.end());
    d.push_back(out);
    return d;
}

Ppo::Ppo(int obs_dim, int act_dim, int num_envs, const PpoConfig& cfg, cudaStream_t stream)
    : obs_dim_(obs_dim), act_dim_(act_dim), num_envs_(num_envs), cfg_(cfg), stream_(stream),
      batch_(cfg.horizon * num_envs), mb_(batch_ / std::max(1, cfg.minibatches)),
      actor_(dims_of(obs_dim, cfg.hidden, act_dim), std::max(num_envs, batch_ / std::max(1, cfg.minibatches)), cfg.seed),
      critic_(dims_of(obs_dim, cfg.hidden, 1), std::max(num_envs, batch_ / std::max(1, cfg.minibatches)), cfg.seed + 7),
      log_std_(1, act_dim), dlog_std_(1, act_dim), norm_(obs_dim, cfg.obs_clip) {
    if (cfg.horizon <= 0 || cfg.epochs <= 0 || cfg.minibatches <= 0)
        throw std::runtime_error("ppo: horizon, epochs and minibatches must be > 0");
    if (batch_ % cfg.minibatches != 0)
        throw std::runtime_error("ppo: horizon * num_envs must divide by ppo.minibatches");
    actor_.init_orthogonal(0.01f, cfg.seed);   // CleanRL: small actor output at the start
    critic_.init_orthogonal(1.0f, cfg.seed + 7);
    log_std_.upload(std::vector<float>(act_dim, cfg.init_log_std));
    dlog_std_.zero(stream_);
    collect_params();
    adam_ = std::make_unique<Adam>(params_, grads_, AdamOptions{cfg.lr});
    lr_.resize(1, 1);
    lr_.upload({cfg.lr});
    kl_mb_.resize(1, 1);
    kl_mb_.zero(stream_);
    lr_now_ = cfg.lr;
    adam_->set_lr_device(lr_.data());
    clip_ = std::make_unique<GradClip>(grads_);

    const int N = num_envs, T = cfg.horizon, D = obs_dim, A = act_dim;
    obs_.resize(T * N, D);
    act_.resize(T * N, A);
    logp_.resize(T, N);
    value_.resize(T, N);
    reward_.resize(T, N);
    adv_.resize(T, N);
    ret_.resize(T, N);
    done_.resize(size_t(T) * N);
    fobs_norm_.resize(N, D);
    mb_obs_.resize(mb_, D);
    mb_act_.resize(mb_, A);
    mb_logp_.resize(1, mb_);
    mb_adv_.resize(1, mb_);
    mb_ret_.resize(1, mb_);
    dmu_.resize(mb_, A);
    dv_.resize(mb_, 1);
    dls_part_.resize(mb_, A);
    stat_part_.resize(mb_, 4);
    stats_.resize(1, 5);
    keys_in_.resize(batch_);
    keys_out_.resize(batch_);
    idx_in_.resize(batch_);
    idx_out_.resize(batch_);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sort_tmp_bytes_, keys_in_.data(), keys_out_.data(),
                                               idx_in_.data(), idx_out_.data(), batch_, 0, 32, stream_));
    sort_tmp_.resize(sort_tmp_bytes_);
}

void Ppo::collect_params() {
    params_.clear();
    grads_.clear();
    actor_.params(params_, grads_);
    critic_.params(params_, grads_);
    params_.push_back(&log_std_);
    grads_.push_back(&dlog_std_);
}

void Ppo::act(const float* raw_obs, float* actions) {
    if (t_ >= cfg_.horizon) throw std::runtime_error("ppo: act() called more than horizon times before update()");
    const int N = num_envs_, D = obs_dim_, A = act_dim_;
    float* obs_t = obs_.data() + size_t(t_) * N * D;
    norm_.update(raw_obs, N, stream_);
    norm_.normalize(raw_obs, obs_t, N, stream_);
    const float* mu = actor_.forward(obs_t, N, stream_);
    const float* v = critic_.forward(obs_t, N, stream_);
    sample_kernel<<<blocks_for(N), kThreads, 0, stream_>>>(
        N, A, mu, log_std_.data(), v, cfg_.seed, uint64_t(iteration_) * cfg_.horizon + t_,
        act_.data() + size_t(t_) * N * A, logp_.data() + size_t(t_) * N, value_.data() + size_t(t_) * N, actions);
    KERNEL_CHECK();
}

void Ppo::observe(const float* reward, const int* terminated, const int* truncated, const float* final_obs) {
    const int N = num_envs_;
    norm_.normalize(final_obs, fobs_norm_.data(), N, stream_);
    const float* v_final = critic_.forward(fobs_norm_.data(), N, stream_);
    observe_kernel<<<blocks_for(N), kThreads, 0, stream_>>>(N, reward, terminated, truncated, v_final, cfg_.gamma,
                                                           reward_.data() + size_t(t_) * N,
                                                           done_.data() + size_t(t_) * N);
    KERNEL_CHECK();
    ++t_;
}

PpoStats Ppo::update(const float* raw_obs) {
    if (t_ != cfg_.horizon) throw std::runtime_error("ppo: update() needs exactly horizon act/observe steps");
    FORGE_NVTX_RANGE("ppo update");
    const int N = num_envs_, T = cfg_.horizon, D = obs_dim_, A = act_dim_;
    norm_.normalize(raw_obs, fobs_norm_.data(), N, stream_);
    const float* last_v = critic_.forward(fobs_norm_.data(), N, stream_);
    gae_kernel<<<blocks_for(N), kThreads, 0, stream_>>>(N, T, reward_.data(), done_.data(), value_.data(), last_v,
                                                       cfg_.gamma, cfg_.lambda, adv_.data(), ret_.data());
    KERNEL_CHECK();
    stats_.zero(stream_);

    int updates = 0;
    for (int ep = 0; ep < cfg_.epochs; ++ep) {
        // Random order of all horizon x num_envs samples (CleanRL shuffles each epoch).
        keys_kernel<<<blocks_for(batch_), kThreads, 0, stream_>>>(
            batch_, cfg_.seed, uint64_t(iteration_) * cfg_.epochs + ep, keys_in_.data(), idx_in_.data());
        KERNEL_CHECK();
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(sort_tmp_.data(), sort_tmp_bytes_, keys_in_.data(),
                                                   keys_out_.data(), idx_in_.data(), idx_out_.data(), batch_, 0, 32,
                                                   stream_));
        for (int k = 0; k < cfg_.minibatches; ++k) {
            gather_kernel<<<blocks_for(mb_), kThreads, 0, stream_>>>(
                mb_, D, A, idx_out_.data() + size_t(k) * mb_, obs_.data(), act_.data(), logp_.data(), adv_.data(),
                ret_.data(), mb_obs_.data(), mb_act_.data(), mb_logp_.data(), mb_adv_.data(), mb_ret_.data());
            KERNEL_CHECK();
            adv_norm_kernel<<<1, 1024, 0, stream_>>>(mb_, mb_adv_.data());
            KERNEL_CHECK();
            const float* mu = actor_.forward(mb_obs_.data(), mb_, stream_);
            const float* v = critic_.forward(mb_obs_.data(), mb_, stream_);
            loss_kernel<<<blocks_for(mb_), kThreads, 0, stream_>>>(mb_, A, mu, v, log_std_.data(), mb_act_.data(),
                                                                  mb_logp_.data(), mb_adv_.data(), mb_ret_.data(),
                                                                  cfg_.clip, cfg_.vf_coef, dmu_.data(), dv_.data(),
                                                                  dls_part_.data(), stat_part_.data());
            KERNEL_CHECK();
            reduce_kernel<<<1, 1024, 0, stream_>>>(mb_, A, dls_part_.data(), stat_part_.data(), cfg_.ent_coef,
                                                   dlog_std_.data(), stats_.data(), kl_mb_.data());
            KERNEL_CHECK();
            if (cfg_.schedule == "adaptive") {  // before this minibatch's step, as rsl_rl
                adapt_lr_kernel<<<1, 1, 0, stream_>>>(kl_mb_.data(), lr_.data(), cfg_.desired_kl, cfg_.lr_min,
                                                      cfg_.lr_max);
                KERNEL_CHECK();
            }
            actor_.backward(dmu_.data(), mb_, stream_);
            critic_.backward(dv_.data(), mb_, stream_);
            clip_->compute(cfg_.max_grad_norm, stream_);
            add_kernel<<<1, 1, 0, stream_>>>(stats_.data() + 4, clip_->norm());
            KERNEL_CHECK();
            adam_->step(stream_, clip_->scale());
            ++updates;
        }
    }

    // The one sync of the iteration: read the stats (and log_std for entropy and std).
    std::vector<float> s(5), ls(A);
    CUDA_CHECK(cudaMemcpyAsync(s.data(), stats_.data(), 5 * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(&lr_now_, lr_.data(), sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(ls.data(), log_std_.data(), A * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    PpoStats out;
    out.policy_loss = s[0] / updates;
    out.value_loss = s[1] / updates;
    out.approx_kl = s[2] / updates;
    out.clip_fraction = s[3] / updates;
    out.grad_norm = s[4] / updates;
    out.entropy = gaussian_entropy(ls.data(), A);
    float mean_std = 0.0f;
    for (float x : ls) mean_std += std::exp(x) / A;
    out.std = mean_std;
    out.lr = lr_now_;
    t_ = 0;
    ++iteration_;
    return out;
}

void Ppo::act_deterministic(const float* raw_obs, float* actions, int n) {
    if (n > num_envs_) throw std::runtime_error("ppo: act_deterministic with more envs than the policy was made for");
    norm_.normalize(raw_obs, fobs_norm_.data(), n, stream_);
    const float* mu = actor_.forward(fobs_norm_.data(), n, stream_);
    CUDA_CHECK(cudaMemcpyAsync(actions, mu, size_t(n) * act_dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
}

Checkpoint Ppo::checkpoint(const std::string& task) const {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    Checkpoint c;
    c.algo = "ppo";
    c.task = task;
    c.obs_dim = obs_dim_;
    c.act_dim = act_dim_;
    c.layer_sizes["actor"] = const_cast<Mlp&>(actor_).dims();
    c.layer_sizes["critic"] = const_cast<Mlp&>(critic_).dims();
    c.iteration = iteration_;
    c.seed = cfg_.seed;
    c.optimizer_steps = adam_->steps();
    auto& self = const_cast<Ppo&>(*this);
    for (const char* net : {"actor", "critic"}) {
        Mlp& m = std::string(net) == "actor" ? self.actor_ : self.critic_;
        const auto lin = m.linears();
        for (size_t i = 0; i < lin.size(); ++i) {
            c.arrays[std::string(net) + "." + std::to_string(i) + ".W"] = lin[i]->W.download();
            c.arrays[std::string(net) + "." + std::to_string(i) + ".b"] = lin[i]->b.download();
        }
    }
    c.arrays["log_std"] = log_std_.download();
    c.arrays["lr"] = {lr_now_};
    c.arrays["obs.mean"] = norm_.mean.download();
    c.arrays["obs.var"] = norm_.var.download();
    c.arrays["obs.count"] = norm_.count.download();
    for (size_t i = 0; i < params_.size(); ++i) {
        c.arrays["adam.m." + std::to_string(i)] = self.adam_->m()[i].download();
        c.arrays["adam.v." + std::to_string(i)] = self.adam_->v()[i].download();
    }
    return c;
}

void Ppo::load(const Checkpoint& c, bool with_optimizer) {
    if (c.algo != "ppo") throw std::runtime_error("checkpoint is for algorithm '" + c.algo + "', not ppo");
    if (c.obs_dim != obs_dim_ || c.act_dim != act_dim_)
        throw std::runtime_error("checkpoint obs/act sizes " + std::to_string(c.obs_dim) + "/" +
                                 std::to_string(c.act_dim) + " do not match the task " + std::to_string(obs_dim_) +
                                 "/" + std::to_string(act_dim_));
    if (c.layer_sizes.at("actor") != actor_.dims() || c.layer_sizes.at("critic") != critic_.dims())
        throw std::runtime_error("checkpoint network sizes do not match");
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (const char* net : {"actor", "critic"}) {
        Mlp& m = std::string(net) == "actor" ? actor_ : critic_;
        const auto lin = m.linears();
        for (size_t i = 0; i < lin.size(); ++i) {
            lin[i]->W.upload(c.array(std::string(net) + "." + std::to_string(i) + ".W"));
            lin[i]->b.upload(c.array(std::string(net) + "." + std::to_string(i) + ".b"));
        }
    }
    log_std_.upload(c.array("log_std"));
    norm_.mean.upload(c.array("obs.mean"));
    norm_.var.upload(c.array("obs.var"));
    norm_.count.upload(c.array("obs.count"));
    if (with_optimizer) {
        for (size_t i = 0; i < params_.size(); ++i) {
            adam_->m()[i].upload(c.array("adam.m." + std::to_string(i)));
            adam_->v()[i].upload(c.array("adam.v." + std::to_string(i)));
        }
        adam_->set_steps(c.optimizer_steps);
        if (c.arrays.count("lr")) {  // the adaptive schedule's learning rate goes on where it was
            lr_now_ = c.arrays.at("lr").at(0);
            lr_.upload({lr_now_});
        }
        iteration_ = c.iteration;
    }
}

}  // namespace temper
