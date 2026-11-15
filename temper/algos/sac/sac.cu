#include "temper/algos/sac/sac.h"

#include <cmath>
#include <stdexcept>

#include "forge/util/cuda_check.h"
#include "forge/util/nvtx.h"
#include "forge/util/rng.h"
#include "temper/algos/sac/sac_math.h"

namespace temper {

namespace {

constexpr int kT = 128;
int nb(int n) { return (n + kT - 1) / kT; }

__device__ float gaussian(forge::Rng& r) {
    const float u1 = fmaxf(r.uniform(), 1e-12f), u2 = r.uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

__global__ void uniform_actions_kernel(int n, int A, uint64_t seed, uint64_t sid, float* act, float* out) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    forge::Rng rng(seed ^ 0x0A11CEull, sid, uint64_t(e));
    for (int k = 0; k < A; ++k) {
        const float a = rng.uniform(-1.0f, 1.0f);
        act[size_t(e) * A + k] = a;
        if (out) out[size_t(e) * A + k] = a;
    }
}

// Actor output [n x 2A] (mean | raw) -> eps, tanh-Gaussian action, log-prob.
__global__ void policy_sample_kernel(int n, int A, const float* out, uint64_t seed, uint64_t sid, float* eps,
                                     float* act, float* logp, float* act_copy) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    forge::Rng rng(seed ^ 0x5AC5ACull, sid, uint64_t(i));
    float* e = eps + size_t(i) * A;
    for (int k = 0; k < A; ++k) e[k] = gaussian(rng);
    const float* o = out + size_t(i) * 2 * A;
    float* a = act + size_t(i) * A;
    const float lp = sac_squash(o, o + A, e, A, a);
    if (logp) logp[i] = lp;
    if (act_copy)
        for (int k = 0; k < A; ++k) act_copy[size_t(i) * A + k] = a[k];
}

__global__ void mean_action_kernel(int n, int A, const float* out, float* act) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int k = 0; k < A; ++k) act[size_t(i) * A + k] = tanhf(out[size_t(i) * 2 * A + k]);
}

__global__ void concat_kernel(int n, int D, int A, const float* obs, const float* act, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* o = out + size_t(i) * (D + A);
    for (int k = 0; k < D; ++k) o[k] = obs[size_t(i) * D + k];
    for (int k = 0; k < A; ++k) o[D + k] = act[size_t(i) * A + k];
}

__global__ void target_kernel(int B, const float* r, const float* done, const float* t1, const float* t2,
                              const float* nlogp, const float* log_alpha, float gamma, float* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    const float alpha = expf(*log_alpha);
    y[i] = r[i] + gamma * (1.0f - done[i]) * (fminf(t1[i], t2[i]) - alpha * nlogp[i]);
}

// part[i] = {q_loss, q_mean, actor_loss, log_pi}
__global__ void critic_grad_kernel(int B, const float* v1, const float* v2, const float* y, float* dq1, float* dq2,
                                   float* part) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    const float e1 = v1[i] - y[i], e2 = v2[i] - y[i];
    dq1[i] = 2.0f * e1 / float(B);
    dq2[i] = 2.0f * e2 / float(B);
    part[size_t(i) * 4 + 0] = e1 * e1 + e2 * e2;
    part[size_t(i) * 4 + 1] = v1[i];
}

// d min(Q1, Q2) / d Qj: 1 for the smaller head, 0 for the other.
__global__ void min_mask_kernel(int B, const float* a1, const float* a2, float* dy1, float* dy2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    const bool first = a1[i] <= a2[i];
    dy1[i] = first ? 1.0f : 0.0f;
    dy2[i] = first ? 0.0f : 1.0f;
}

__global__ void actor_grad_kernel(int B, int D, int A, const float* out, const float* eps, const float* dx1,
                                  const float* dx2, const float* a1, const float* a2, const float* logp,
                                  const float* log_alpha, float* dout, float* part) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    const float alpha = expf(*log_alpha);
    float dq[16];  // act_dim <= 16 (checked on the host)
    for (int k = 0; k < A; ++k) dq[k] = dx1[size_t(i) * (D + A) + D + k] + dx2[size_t(i) * (D + A) + D + k];
    const float* o = out + size_t(i) * 2 * A;
    float* g = dout + size_t(i) * 2 * A;
    sac_actor_grad(o, o + A, eps + size_t(i) * A, dq, alpha, 1.0f / float(B), A, g, g + A);
    part[size_t(i) * 4 + 2] = alpha * logp[i] - fminf(a1[i], a2[i]);
    part[size_t(i) * 4 + 3] = logp[i];
}

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

// One block, fixed order. stats += {q_loss, actor_loss, alpha, entropy, q_mean, 1};
// alpha gradient: d/dlog_alpha mean(-alpha (log pi + target_entropy)).
__global__ void reduce_kernel(int B, const float* part, const float* log_alpha, float target_entropy,
                              float* dlog_alpha, float* stats) {
    __shared__ float sh[512];
    float s[4];
    for (int c = 0; c < 4; ++c) {
        float v = 0.0f;
        for (int i = threadIdx.x; i < B; i += blockDim.x) v += part[size_t(i) * 4 + c];
        s[c] = block_sum(v, sh) / float(B);
    }
    if (threadIdx.x == 0) {
        const float alpha = expf(*log_alpha);
        *dlog_alpha = -alpha * (s[3] + target_entropy);
        stats[0] += s[0];
        stats[1] += s[2];
        stats[2] += alpha;
        stats[3] += -s[3];
        stats[4] += s[1];
        stats[5] += 1.0f;
    }
}

__global__ void polyak_kernel(size_t n, float* target, const float* src, float tau) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) target[i] = (1.0f - tau) * target[i] + tau * src[i];
}

std::vector<int> dims_of(int in, const std::vector<int>& hidden, int out) {
    std::vector<int> d = {in};
    d.insert(d.end(), hidden.begin(), hidden.end());
    d.push_back(out);
    return d;
}

}  // namespace

SacConfig SacConfig::from_config(const forge::Config* c) {
    SacConfig s;
    if (!c) return s;
    s.buffer_size = c->get_int("sac.buffer_size", s.buffer_size);
    s.batch_size = c->get_int("sac.batch_size", s.batch_size);
    s.updates_per_step = c->get_int("sac.updates_per_step", s.updates_per_step);
    s.learning_starts = c->get_int("sac.learning_starts", s.learning_starts);
    s.steps_per_iteration = c->get_int("sac.steps_per_iteration", s.steps_per_iteration);
    s.gamma = c->get_float("sac.gamma", s.gamma);
    s.tau = c->get_float("sac.tau", s.tau);
    s.q_lr = c->get_float("sac.q_lr", s.q_lr);
    s.policy_lr = c->get_float("sac.policy_lr", s.policy_lr);
    s.alpha_lr = c->get_float("sac.alpha_lr", s.alpha_lr);
    s.seed = static_cast<uint64_t>(c->get_int("seed", 1));
    return s;
}

Sac::Sac(int D, int A, int N, const SacConfig& cfg, cudaStream_t stream)
    : D_(D), A_(A), N_(N), cfg_(cfg), stream_(stream), replay_(cfg.buffer_size, D, A),
      actor_(dims_of(D, cfg.hidden, 2 * A), std::max(N, cfg.batch_size), cfg.seed, Activation::kRelu),
      q1_(dims_of(D + A, cfg.hidden, 1), cfg.batch_size, cfg.seed + 1, Activation::kRelu),
      q2_(dims_of(D + A, cfg.hidden, 1), cfg.batch_size, cfg.seed + 2, Activation::kRelu),
      q1t_(dims_of(D + A, cfg.hidden, 1), cfg.batch_size, cfg.seed + 1, Activation::kRelu),
      q2t_(dims_of(D + A, cfg.hidden, 1), cfg.batch_size, cfg.seed + 2, Activation::kRelu), log_alpha_(1, 1),
      dlog_alpha_(1, 1) {
    if (A > 16) throw std::runtime_error("sac: act_dim > 16 is not supported (actor_grad_kernel)");
    if (cfg.batch_size <= 0 || cfg.updates_per_step <= 0) throw std::runtime_error("sac: batch_size, updates_per_step > 0");
    q1t_.copy_from(q1_, stream_);
    q2t_.copy_from(q2_, stream_);
    log_alpha_.upload({0.0f});  // alpha 1 at the start (CleanRL)
    dlog_alpha_.zero(stream_);
    q1_.params(pq_, gq_);
    q2_.params(pq_, gq_);
    std::vector<Tensor*> tg;
    q1t_.params(pqt_, tg);
    q2t_.params(pqt_, tg);
    std::vector<Tensor*> pa, ga;
    actor_.params(pa, ga);
    q_opt_ = std::make_unique<Adam>(pq_, gq_, AdamOptions{cfg.q_lr});
    pi_opt_ = std::make_unique<Adam>(pa, ga, AdamOptions{cfg.policy_lr});
    pa_ = {&log_alpha_};
    ga_ = {&dlog_alpha_};
    alpha_opt_ = std::make_unique<Adam>(pa_, ga_, AdamOptions{cfg.alpha_lr});
    const int B = cfg.batch_size;
    obs_now_.resize(N, D);
    act_now_.resize(N, A);
    b_obs_.resize(B, D);
    b_act_.resize(B, A);
    b_rew_.resize(1, B);
    b_next_.resize(B, D);
    b_done_.resize(1, B);
    eps_.resize(std::max(N, B), A);
    act_new_.resize(std::max(N, B), A);
    logp_.resize(1, std::max(N, B));
    y_.resize(1, B);
    qin_.resize(B, D + A);
    dq1_.resize(B, 1);
    dq2_.resize(B, 1);
    dx1_.resize(B, D + A);
    dx2_.resize(B, D + A);
    dout_.resize(B, 2 * A);
    part_.resize(B, 4);
    stats_.resize(1, 6);
    stats_.zero(stream_);
}

void Sac::act(const float* obs, float* actions) {
    CUDA_CHECK(cudaMemcpyAsync(obs_now_.data(), obs, size_t(N_) * D_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    if (!ready_to_update()) {
        uniform_actions_kernel<<<nb(N_), kT, 0, stream_>>>(N_, A_, cfg_.seed, uint64_t(env_steps_), act_now_.data(), actions);
    } else {
        const float* out = actor_.forward(obs_now_.data(), N_, stream_);
        policy_sample_kernel<<<nb(N_), kT, 0, stream_>>>(N_, A_, out, cfg_.seed ^ 0xE1Full, uint64_t(env_steps_),
                                                        eps_.data(), act_now_.data(), nullptr, actions);
    }
    KERNEL_CHECK();
    ++env_steps_;
}

void Sac::act_deterministic(const float* obs, float* actions, int n) {
    if (n > N_) throw std::runtime_error("sac: act_deterministic with more envs than the policy was made for");
    const float* out = actor_.forward(obs, n, stream_);
    mean_action_kernel<<<nb(n), kT, 0, stream_>>>(n, A_, out, actions);
    KERNEL_CHECK();
}

void Sac::observe(const TransitionView& t) {
    replay_.add(obs_now_.data(), act_now_.data(), t.reward, t.next_obs, t.final_obs, t.terminated, t.truncated, N_,
                stream_);
}

void Sac::update(const float*) {
    FORGE_NVTX_RANGE("sac update");
    for (int g = 0; g < cfg_.updates_per_step; ++g) gradient_step();
}

void Sac::gradient_step() {
    const int B = cfg_.batch_size, D = D_, A = A_;
    const uint64_t g = uint64_t(grad_steps_++);
    replay_.sample(B, cfg_.seed, g, b_obs_, b_act_, b_rew_, b_next_, b_done_, stream_);

    // Critic target with the target critics and a fresh action at s'.
    const float* out_n = actor_.forward(b_next_.data(), B, stream_);
    policy_sample_kernel<<<nb(B), kT, 0, stream_>>>(B, A, out_n, cfg_.seed, 2 * g, eps_.data(), act_new_.data(),
                                                    logp_.data(), nullptr);
    concat_kernel<<<nb(B), kT, 0, stream_>>>(B, D, A, b_next_.data(), act_new_.data(), qin_.data());
    const float* t1 = q1t_.forward(qin_.data(), B, stream_);
    const float* t2 = q2t_.forward(qin_.data(), B, stream_);
    target_kernel<<<nb(B), kT, 0, stream_>>>(B, b_rew_.data(), b_done_.data(), t1, t2, logp_.data(), log_alpha_.data(),
                                            cfg_.gamma, y_.data());
    KERNEL_CHECK();

    // Critics.
    concat_kernel<<<nb(B), kT, 0, stream_>>>(B, D, A, b_obs_.data(), b_act_.data(), qin_.data());
    const float* v1 = q1_.forward(qin_.data(), B, stream_);
    const float* v2 = q2_.forward(qin_.data(), B, stream_);
    critic_grad_kernel<<<nb(B), kT, 0, stream_>>>(B, v1, v2, y_.data(), dq1_.data(), dq2_.data(), part_.data());
    KERNEL_CHECK();
    q1_.backward(dq1_.data(), B, stream_);
    q2_.backward(dq2_.data(), B, stream_);
    q_opt_->step(stream_);

    // Actor through the (updated) critics; the critic gradients made here are
    // never applied (the next critic backward overwrites them).
    const float* out = actor_.forward(b_obs_.data(), B, stream_);
    policy_sample_kernel<<<nb(B), kT, 0, stream_>>>(B, A, out, cfg_.seed, 2 * g + 1, eps_.data(), act_new_.data(),
                                                    logp_.data(), nullptr);
    concat_kernel<<<nb(B), kT, 0, stream_>>>(B, D, A, b_obs_.data(), act_new_.data(), qin_.data());
    const float* a1 = q1_.forward(qin_.data(), B, stream_);
    const float* a2 = q2_.forward(qin_.data(), B, stream_);
    min_mask_kernel<<<nb(B), kT, 0, stream_>>>(B, a1, a2, dq1_.data(), dq2_.data());
    KERNEL_CHECK();
    q1_.backward(dq1_.data(), B, stream_, dx1_.data());
    q2_.backward(dq2_.data(), B, stream_, dx2_.data());
    actor_grad_kernel<<<nb(B), kT, 0, stream_>>>(B, D, A, out, eps_.data(), dx1_.data(), dx2_.data(), a1, a2,
                                                logp_.data(), log_alpha_.data(), dout_.data(), part_.data());
    KERNEL_CHECK();
    actor_.backward(dout_.data(), B, stream_);
    pi_opt_->step(stream_);

    // Alpha, stats, target critics.
    reduce_kernel<<<1, 512, 0, stream_>>>(B, part_.data(), log_alpha_.data(), -float(A), dlog_alpha_.data(),
                                          stats_.data());
    KERNEL_CHECK();
    alpha_opt_->step(stream_);
    for (size_t i = 0; i < pq_.size(); ++i) {
        Tensor* t = pqt_[i];
        polyak_kernel<<<int((t->size() + 255) / 256), 256, 0, stream_>>>(t->size(), t->data(), pq_[i]->data(), cfg_.tau);
    }
    KERNEL_CHECK();
}

std::vector<std::string> Sac::stat_names() const {
    return {"q_loss", "actor_loss", "alpha", "entropy", "q_mean", "updates"};
}

std::vector<double> Sac::read_stats() {
    float s[6];
    CUDA_CHECK(cudaMemcpyAsync(s, stats_.data(), sizeof(s), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    stats_.zero(stream_);
    ++iteration_;
    const double n = s[5];
    if (n <= 0) return {0, 0, 0, 0, 0, 0};  // warmup: no updates yet
    return {s[0] / n, s[1] / n, s[2] / n, s[3] / n, s[4] / n, n};
}

Checkpoint Sac::checkpoint(const std::string& task) const {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto& self = const_cast<Sac&>(*this);
    Checkpoint c;
    c.algo = "sac";
    c.task = task;
    c.obs_dim = D_;
    c.act_dim = A_;
    c.layer_sizes["actor"] = actor_.dims();
    c.layer_sizes["critic"] = q1_.dims();
    c.iteration = iteration_;
    c.seed = cfg_.seed;
    c.optimizer_steps = grad_steps_;
    for (auto [name, net] : {std::pair{"actor", &self.actor_}, std::pair{"q1", &self.q1_}, std::pair{"q2", &self.q2_},
                             std::pair{"q1t", &self.q1t_}, std::pair{"q2t", &self.q2t_}}) {
        const auto lin = net->linears();
        for (size_t i = 0; i < lin.size(); ++i) {
            c.arrays[std::string(name) + "." + std::to_string(i) + ".W"] = lin[i]->W.download();
            c.arrays[std::string(name) + "." + std::to_string(i) + ".b"] = lin[i]->b.download();
        }
    }
    c.arrays["log_alpha"] = log_alpha_.download();
    for (auto [name, opt] : {std::pair{"adam_q", self.q_opt_.get()}, std::pair{"adam_pi", self.pi_opt_.get()},
                             std::pair{"adam_alpha", self.alpha_opt_.get()}}) {
        for (size_t i = 0; i < opt->m().size(); ++i) {
            c.arrays[std::string(name) + ".m." + std::to_string(i)] = opt->m()[i].download();
            c.arrays[std::string(name) + ".v." + std::to_string(i)] = opt->v()[i].download();
        }
        c.arrays[std::string(name) + ".steps"] = {float(opt->steps())};
    }
    return c;
}

void Sac::load(const Checkpoint& c, bool with_optimizer) {
    if (c.algo != "sac") throw std::runtime_error("checkpoint is for algorithm '" + c.algo + "', not sac");
    if (c.obs_dim != D_ || c.act_dim != A_)
        throw std::runtime_error("checkpoint obs/act sizes " + std::to_string(c.obs_dim) + "/" +
                                 std::to_string(c.act_dim) + " do not match the task " + std::to_string(D_) + "/" +
                                 std::to_string(A_));
    if (c.layer_sizes.at("actor") != actor_.dims() || c.layer_sizes.at("critic") != q1_.dims())
        throw std::runtime_error("checkpoint network sizes do not match");
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (auto [name, net] : {std::pair{"actor", &actor_}, std::pair{"q1", &q1_}, std::pair{"q2", &q2_},
                             std::pair{"q1t", &q1t_}, std::pair{"q2t", &q2t_}}) {
        const auto lin = net->linears();
        for (size_t i = 0; i < lin.size(); ++i) {
            lin[i]->W.upload(c.array(std::string(name) + "." + std::to_string(i) + ".W"));
            lin[i]->b.upload(c.array(std::string(name) + "." + std::to_string(i) + ".b"));
        }
    }
    log_alpha_.upload(c.array("log_alpha"));
    if (with_optimizer) {
        for (auto [name, opt] : {std::pair{"adam_q", q_opt_.get()}, std::pair{"adam_pi", pi_opt_.get()},
                                 std::pair{"adam_alpha", alpha_opt_.get()}}) {
            for (size_t i = 0; i < opt->m().size(); ++i) {
                opt->m()[i].upload(c.array(std::string(name) + ".m." + std::to_string(i)));
                opt->v()[i].upload(c.array(std::string(name) + ".v." + std::to_string(i)));
            }
            opt->set_steps(static_cast<long long>(c.array(std::string(name) + ".steps").at(0)));
        }
        iteration_ = c.iteration;
        grad_steps_ = c.optimizer_steps;
    }
}

}  // namespace temper
