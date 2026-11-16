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

// SAC target: y = r + gamma (1 - d) (min_c zt_c - alpha logp')
__global__ void target_sac_kernel(int B, int n, const float* zt, const float* r, const float* done,
                                  const float* nlogp, const float* log_alpha, float gamma, float* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    float m = zt[i];
    for (int c = 1; c < n; ++c) m = fminf(m, zt[size_t(c) * B + i]);
    y[i] = r[i] + gamma * (1.0f - done[i]) * (m - expf(*log_alpha) * nlogp[i]);
}

constexpr int kMaxAtoms = 256;  // n_critics x n_quantiles (checked on the host)

// TQC target: sort all n x N target atoms of sample i, keep the lowest M.
__global__ void target_tqc_kernel(int B, int n, int N, int M, const float* zt, const float* r, const float* done,
                                  const float* nlogp, const float* log_alpha, float gamma, float* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    float atoms[kMaxAtoms];
    int cnt = 0;
    for (int c = 0; c < n; ++c)
        for (int k = 0; k < N; ++k) {
            const float v = zt[(size_t(c) * B + i) * N + k];
            int j = cnt++;
            while (j > 0 && atoms[j - 1] > v) {  // insertion sort
                atoms[j] = atoms[j - 1];
                --j;
            }
            atoms[j] = v;
        }
    const float base = r[i], disc = gamma * (1.0f - done[i]), ent = expf(*log_alpha) * nlogp[i];
    for (int j = 0; j < M; ++j) y[size_t(i) * M + j] = base + disc * (atoms[j] - ent);
}

// SAC critic loss gradient. part[i] = {q_loss, q_mean, actor_loss, log_pi}
__global__ void critic_grad_sac_kernel(int B, int n, const float* z, const float* y, float* dz, float* part) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    float l = 0.0f;
    for (int c = 0; c < n; ++c) {
        const float e = z[size_t(c) * B + i] - y[i];
        dz[size_t(c) * B + i] = 2.0f * e / float(B);
        l += e * e;
    }
    part[size_t(i) * 4 + 0] = l;
    part[size_t(i) * 4 + 1] = z[i];
}

// TQC quantile Huber loss gradient, one thread per (sample, critic, quantile).
// loss = mean over (b, c, k, j) of |tau_k - 1(d < 0)| huber(d), d = y_bj - z_bck
__global__ void critic_grad_tqc_kernel(int B, int n, int N, int M, const float* z, const float* y, float* dz,
                                       float* qloss_elem) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * n * N) return;
    const int k = t % N, c = (t / N) % n, i = t / (N * n);
    const float zk = z[(size_t(c) * B + i) * N + k];
    const float tau = (k + 0.5f) / float(N);
    float g = 0.0f, l = 0.0f;
    for (int j = 0; j < M; ++j) {
        const float d = y[size_t(i) * M + j] - zk;
        const float w = fabsf(tau - (d < 0.0f ? 1.0f : 0.0f));
        const float ad = fabsf(d);
        l += w * (ad > 1.0f ? ad - 0.5f : 0.5f * d * d);
        g += w * fminf(fmaxf(d, -1.0f), 1.0f);
    }
    dz[(size_t(c) * B + i) * N + k] = -g / (float(B) * n * N * M);
    qloss_elem[(size_t(i) * n + c) * N + k] = l / (float(n) * N * M);
}

__global__ void tqc_part_kernel(int B, int n, int N, const float* qloss_elem, const float* z, float* part) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    float l = 0.0f, q = 0.0f;
    for (int e = 0; e < n * N; ++e) l += qloss_elem[size_t(i) * n * N + e];
    for (int k = 0; k < N; ++k) q += z[size_t(i) * N + k];
    part[size_t(i) * 4 + 0] = l;
    part[size_t(i) * 4 + 1] = q / float(N);
}

// Actor pass: aggregate Q (SAC: min over critics; TQC: mean over critics and
// quantiles), its gradient wrt each critic output, and the actor loss terms.
__global__ void actor_q_kernel(int B, int n, int N, bool tqc, const float* z, const float* logp,
                               const float* log_alpha, float* dz, float* part) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    float q;
    if (tqc) {
        q = 0.0f;
        for (int c = 0; c < n; ++c)
            for (int k = 0; k < N; ++k) {
                q += z[(size_t(c) * B + i) * N + k];
                dz[(size_t(c) * B + i) * N + k] = 1.0f / float(n * N);
            }
        q /= float(n * N);
    } else {
        int best = 0;
        q = z[i];
        for (int c = 1; c < n; ++c)
            if (z[size_t(c) * B + i] < q) {
                q = z[size_t(c) * B + i];
                best = c;
            }
        for (int c = 0; c < n; ++c) dz[size_t(c) * B + i] = c == best ? 1.0f : 0.0f;
    }
    part[size_t(i) * 4 + 2] = expf(*log_alpha) * logp[i] - q;
    part[size_t(i) * 4 + 3] = logp[i];
}

__global__ void add_kernel(size_t n, float* dst, const float* src, bool first) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = first ? src[i] : dst[i] + src[i];
}

__global__ void actor_grad_kernel(int B, int D, int A, const float* out, const float* eps, const float* dx_sum,
                                  const float* log_alpha, float* dout) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    const float alpha = expf(*log_alpha);
    float dq[16];  // act_dim <= 16 (checked on the host)
    for (int k = 0; k < A; ++k) dq[k] = dx_sum[size_t(i) * (D + A) + D + k];
    const float* o = out + size_t(i) * 2 * A;
    float* g = dout + size_t(i) * 2 * A;
    sac_actor_grad(o, o + A, eps + size_t(i) * A, dq, alpha, 1.0f / float(B), A, g, g + A);
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

SacConfig SacConfig::tqc_defaults() {
    SacConfig s;
    s.tqc = true;
    s.n_critics = 5;
    s.n_quantiles = 25;
    s.drop_per_critic = 2;
    return s;
}

SacConfig SacConfig::from_config(const forge::Config* c, bool tqc) {
    SacConfig s = tqc ? tqc_defaults() : sac_defaults();
    if (!c) return s;
    const std::string p = tqc ? "tqc." : "sac.";
    s.buffer_size = c->get_int(p + "buffer_size", s.buffer_size);
    s.batch_size = c->get_int(p + "batch_size", s.batch_size);
    s.updates_per_step = c->get_int(p + "updates_per_step", s.updates_per_step);
    s.learning_starts = c->get_int(p + "learning_starts", s.learning_starts);
    s.steps_per_iteration = c->get_int(p + "steps_per_iteration", s.steps_per_iteration);
    s.gamma = c->get_float(p + "gamma", s.gamma);
    s.tau = c->get_float(p + "tau", s.tau);
    s.q_lr = c->get_float(p + "q_lr", s.q_lr);
    s.policy_lr = c->get_float(p + "policy_lr", s.policy_lr);
    s.alpha_lr = c->get_float(p + "alpha_lr", s.alpha_lr);
    if (tqc) {
        s.n_critics = c->get_int("tqc.n_critics", s.n_critics);
        s.n_quantiles = c->get_int("tqc.n_quantiles", s.n_quantiles);
        s.drop_per_critic = c->get_int("tqc.drop_per_critic", s.drop_per_critic);
    }
    s.seed = static_cast<uint64_t>(c->get_int("seed", 1));
    return s;
}

Sac::Sac(int D, int A, int N, const SacConfig& cfg, cudaStream_t stream)
    : D_(D), A_(A), N_(N), cfg_(cfg), stream_(stream), replay_(cfg.buffer_size, D, A),
      actor_(dims_of(D, cfg.hidden, 2 * A), std::max(N, cfg.batch_size), cfg.seed, Activation::kRelu),
      log_alpha_(1, 1), dlog_alpha_(1, 1) {
    if (A > 16) throw std::runtime_error("sac: act_dim > 16 is not supported (actor_grad_kernel)");
    if (cfg.batch_size <= 0 || cfg.updates_per_step <= 0) throw std::runtime_error("sac: batch_size, updates_per_step > 0");
    if (cfg.n_critics < 1 || cfg.n_quantiles < 1) throw std::runtime_error("sac: n_critics, n_quantiles >= 1");
    if (!cfg.tqc && cfg.n_quantiles != 1) throw std::runtime_error("sac: n_quantiles must be 1 (use tqc)");
    if (cfg.tqc && (cfg.kept_atoms() <= 0 || cfg.n_critics * cfg.n_quantiles > kMaxAtoms))
        throw std::runtime_error("tqc: need 0 <= drop_per_critic < n_quantiles and n_critics x n_quantiles <= 256");
    const int Q = cfg.n_quantiles, B = cfg.batch_size, n = cfg.n_critics;
    for (int c = 0; c < n; ++c) {
        // targets start as copies (same seed, then copy_from)
        q_.push_back(std::make_unique<Mlp>(dims_of(D + A, cfg.hidden, Q), B, cfg.seed + 1 + c, Activation::kRelu));
        qt_.push_back(std::make_unique<Mlp>(dims_of(D + A, cfg.hidden, Q), B, cfg.seed + 1 + c, Activation::kRelu));
        qt_.back()->copy_from(*q_.back(), stream_);
    }
    log_alpha_.upload({0.0f});  // alpha 1 at the start (CleanRL)
    dlog_alpha_.zero(stream_);
    std::vector<Tensor*> tg;
    for (int c = 0; c < n; ++c) {
        q_[c]->params(pq_, gq_);
        qt_[c]->params(pqt_, tg);
    }
    std::vector<Tensor*> pa, ga;
    actor_.params(pa, ga);
    q_opt_ = std::make_unique<Adam>(pq_, gq_, AdamOptions{cfg.q_lr});
    pi_opt_ = std::make_unique<Adam>(pa, ga, AdamOptions{cfg.policy_lr});
    pa_ = {&log_alpha_};
    ga_ = {&dlog_alpha_};
    alpha_opt_ = std::make_unique<Adam>(pa_, ga_, AdamOptions{cfg.alpha_lr});
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
    y_.resize(B, cfg.tqc ? cfg.kept_atoms() : 1);
    qin_.resize(B, D + A);
    zt_.resize(n * B, Q);
    z_.resize(n * B, Q);
    dz_.resize(n * B, Q);
    dx_.resize(B, D + A);
    dx_sum_.resize(B, D + A);
    if (cfg.tqc) qloss_elem_.resize(B, n * Q);
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

void Sac::grab(const std::string& name, const Tensor& t, size_t n) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    std::vector<float> v = t.download();
    v.resize(n);
    capture_->arrays[name] = std::move(v);
}

void Sac::gradient_step() {
    const int B = cfg_.batch_size, D = D_, A = A_, n = cfg_.n_critics, Q = cfg_.n_quantiles;
    const int M = cfg_.tqc ? cfg_.kept_atoms() : 1;
    const size_t zsz = size_t(B) * Q;  // one critic's output
    const uint64_t g = uint64_t(grad_steps_++);
    replay_.sample(B, cfg_.seed, g, b_obs_, b_act_, b_rew_, b_next_, b_done_, stream_);

    // Critic target with the target critics and a fresh action at s'.
    const float* out_n = actor_.forward(b_next_.data(), B, stream_);
    policy_sample_kernel<<<nb(B), kT, 0, stream_>>>(B, A, out_n, cfg_.seed, 2 * g, eps_.data(), act_new_.data(),
                                                    logp_.data(), nullptr);
    if (capture_) {
        capture_->arrays["config"] = {cfg_.tqc ? 1.0f : 0.0f, float(n), float(Q), float(cfg_.drop_per_critic)};
        grab("b_obs", b_obs_, size_t(B) * D);
        grab("b_act", b_act_, size_t(B) * A);
        grab("b_rew", b_rew_, B);
        grab("b_next", b_next_, size_t(B) * D);
        grab("b_done", b_done_, B);
        grab("eps_next", eps_, size_t(B) * A);
        grab("logp_next", logp_, B);
    }
    concat_kernel<<<nb(B), kT, 0, stream_>>>(B, D, A, b_next_.data(), act_new_.data(), qin_.data());
    for (int c = 0; c < n; ++c) {
        const float* t = qt_[c]->forward(qin_.data(), B, stream_);
        CUDA_CHECK(cudaMemcpyAsync(zt_.data() + c * zsz, t, zsz * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    }
    if (cfg_.tqc)
        target_tqc_kernel<<<nb(B), kT, 0, stream_>>>(B, n, Q, M, zt_.data(), b_rew_.data(), b_done_.data(),
                                                     logp_.data(), log_alpha_.data(), cfg_.gamma, y_.data());
    else
        target_sac_kernel<<<nb(B), kT, 0, stream_>>>(B, n, zt_.data(), b_rew_.data(), b_done_.data(), logp_.data(),
                                                     log_alpha_.data(), cfg_.gamma, y_.data());
    KERNEL_CHECK();

    // Critics.
    concat_kernel<<<nb(B), kT, 0, stream_>>>(B, D, A, b_obs_.data(), b_act_.data(), qin_.data());
    for (int c = 0; c < n; ++c) {
        const float* v = q_[c]->forward(qin_.data(), B, stream_);
        CUDA_CHECK(cudaMemcpyAsync(z_.data() + c * zsz, v, zsz * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    }
    if (cfg_.tqc) {
        const int T = B * n * Q;
        critic_grad_tqc_kernel<<<nb(T), kT, 0, stream_>>>(B, n, Q, M, z_.data(), y_.data(), dz_.data(),
                                                         qloss_elem_.data());
        tqc_part_kernel<<<nb(B), kT, 0, stream_>>>(B, n, Q, qloss_elem_.data(), z_.data(), part_.data());
    } else {
        critic_grad_sac_kernel<<<nb(B), kT, 0, stream_>>>(B, n, z_.data(), y_.data(), dz_.data(), part_.data());
    }
    KERNEL_CHECK();
    for (int c = 0; c < n; ++c) q_[c]->backward(dz_.data() + c * zsz, B, stream_);
    if (capture_) {
        grab("y", y_, size_t(B) * M);
        for (size_t i = 0; i < gq_.size(); ++i) grab("grad_q." + std::to_string(i), *gq_[i], gq_[i]->size());
    }
    q_opt_->step(stream_);

    // Actor through the (updated) critics; the critic gradients made here are
    // never applied (the next critic backward overwrites them).
    const float* out = actor_.forward(b_obs_.data(), B, stream_);
    policy_sample_kernel<<<nb(B), kT, 0, stream_>>>(B, A, out, cfg_.seed, 2 * g + 1, eps_.data(), act_new_.data(),
                                                    logp_.data(), nullptr);
    if (capture_) {
        grab("eps_pi", eps_, size_t(B) * A);
        grab("logp_pi", logp_, B);
    }
    concat_kernel<<<nb(B), kT, 0, stream_>>>(B, D, A, b_obs_.data(), act_new_.data(), qin_.data());
    for (int c = 0; c < n; ++c) {
        const float* v = q_[c]->forward(qin_.data(), B, stream_);
        CUDA_CHECK(cudaMemcpyAsync(z_.data() + c * zsz, v, zsz * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    }
    actor_q_kernel<<<nb(B), kT, 0, stream_>>>(B, n, Q, cfg_.tqc, z_.data(), logp_.data(), log_alpha_.data(),
                                              dz_.data(), part_.data());
    KERNEL_CHECK();
    const size_t nx = size_t(B) * (D + A);
    for (int c = 0; c < n; ++c) {
        q_[c]->backward(dz_.data() + c * zsz, B, stream_, dx_.data());
        add_kernel<<<int((nx + 255) / 256), 256, 0, stream_>>>(nx, dx_sum_.data(), dx_.data(), c == 0);
    }
    actor_grad_kernel<<<nb(B), kT, 0, stream_>>>(B, D, A, out, eps_.data(), dx_sum_.data(), log_alpha_.data(),
                                                dout_.data());
    KERNEL_CHECK();
    actor_.backward(dout_.data(), B, stream_);
    if (capture_) {
        std::vector<Tensor*> p, gr;
        actor_.params(p, gr);
        for (size_t i = 0; i < gr.size(); ++i) grab("grad_pi." + std::to_string(i), *gr[i], gr[i]->size());
    }
    pi_opt_->step(stream_);

    // Alpha, stats, target critics.
    reduce_kernel<<<1, 512, 0, stream_>>>(B, part_.data(), log_alpha_.data(), -float(A), dlog_alpha_.data(),
                                          stats_.data());
    KERNEL_CHECK();
    if (capture_) grab("dlog_alpha", dlog_alpha_, 1);
    alpha_opt_->step(stream_);
    for (size_t i = 0; i < pq_.size(); ++i) {
        Tensor* t = pqt_[i];
        polyak_kernel<<<int((t->size() + 255) / 256), 256, 0, stream_>>>(t->size(), t->data(), pq_[i]->data(), cfg_.tau);
    }
    KERNEL_CHECK();
    if (capture_) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        capture_ = nullptr;
    }
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

namespace {
std::string critic_name(int c, bool target) { return "q" + std::to_string(c + 1) + (target ? "t" : ""); }
}  // namespace

Checkpoint Sac::checkpoint(const std::string& task) const {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto& self = const_cast<Sac&>(*this);
    Checkpoint c;
    c.algo = name();
    c.task = task;
    c.obs_dim = D_;
    c.act_dim = A_;
    c.layer_sizes["actor"] = self.actor_.dims();
    c.layer_sizes["critic"] = self.q_[0]->dims();
    c.iteration = iteration_;
    c.seed = cfg_.seed;
    c.optimizer_steps = grad_steps_;
    std::vector<std::pair<std::string, Mlp*>> nets = {{"actor", &self.actor_}};
    for (int k = 0; k < cfg_.n_critics; ++k) {
        nets.push_back({critic_name(k, false), self.q_[k].get()});
        nets.push_back({critic_name(k, true), self.qt_[k].get()});
    }
    for (auto& [name, net] : nets) {
        const auto lin = net->linears();
        for (size_t i = 0; i < lin.size(); ++i) {
            c.arrays[name + "." + std::to_string(i) + ".W"] = lin[i]->W.download();
            c.arrays[name + "." + std::to_string(i) + ".b"] = lin[i]->b.download();
        }
    }
    c.arrays["log_alpha"] = log_alpha_.download();
    c.arrays["critics"] = {float(cfg_.n_critics), float(cfg_.n_quantiles), float(cfg_.drop_per_critic)};
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
    if (c.algo != name()) throw std::runtime_error("checkpoint is for algorithm '" + c.algo + "', not " + name());
    if (c.obs_dim != D_ || c.act_dim != A_)
        throw std::runtime_error("checkpoint obs/act sizes " + std::to_string(c.obs_dim) + "/" +
                                 std::to_string(c.act_dim) + " do not match the task " + std::to_string(D_) + "/" +
                                 std::to_string(A_));
    if (c.layer_sizes.at("actor") != actor_.dims()) throw std::runtime_error("checkpoint actor sizes do not match");
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    std::vector<std::pair<std::string, Mlp*>> nets = {{"actor", &actor_}};
    if (with_optimizer) {  // play mode needs the actor only
        if (c.layer_sizes.at("critic") != q_[0]->dims() ||
            !c.arrays.count(critic_name(cfg_.n_critics - 1, false) + ".0.W") ||
            c.arrays.count(critic_name(cfg_.n_critics, false) + ".0.W"))
            throw std::runtime_error("checkpoint critics (number or sizes) do not match the config");
        for (int k = 0; k < cfg_.n_critics; ++k) {
            nets.push_back({critic_name(k, false), q_[k].get()});
            nets.push_back({critic_name(k, true), qt_[k].get()});
        }
    }
    for (auto& [name, net] : nets) {
        const auto lin = net->linears();
        for (size_t i = 0; i < lin.size(); ++i) {
            lin[i]->W.upload(c.array(name + "." + std::to_string(i) + ".W"));
            lin[i]->b.upload(c.array(name + "." + std::to_string(i) + ".b"));
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
