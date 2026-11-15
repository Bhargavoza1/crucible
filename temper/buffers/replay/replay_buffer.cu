#include "temper/buffers/replay/replay_buffer.h"

#include <stdexcept>

#include "forge/util/cuda_check.h"
#include "forge/util/rng.h"

namespace temper {

namespace {

__global__ void add_kernel(int n, int D, int A, int capacity, int pos, const float* obs, const float* act,
                           const float* reward, const float* next_obs, const float* final_obs, const int* term,
                           const int* trunc, float* b_obs, float* b_act, float* b_rew, float* b_next, float* b_done) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const int j = (pos + e) % capacity;
    const bool ended = term[e] || trunc[e];
    const float* nx = ended ? final_obs : next_obs;
    for (int k = 0; k < D; ++k) {
        b_obs[size_t(j) * D + k] = obs[size_t(e) * D + k];
        b_next[size_t(j) * D + k] = nx[size_t(e) * D + k];
    }
    for (int k = 0; k < A; ++k) b_act[size_t(j) * A + k] = act[size_t(e) * A + k];
    b_rew[j] = reward[e];
    b_done[j] = term[e] ? 1.0f : 0.0f;  // truncated: still bootstrap
}

__global__ void sample_kernel(int batch, int size, int D, int A, uint64_t seed, uint64_t stream_id,
                              const float* b_obs, const float* b_act, const float* b_rew, const float* b_next,
                              const float* b_done, float* obs, float* act, float* rew, float* next, float* done) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;
    forge::Rng rng(seed ^ 0x5EB1A7ull, stream_id, uint64_t(i));
    const int j = int(rng.next() % uint64_t(size));
    for (int k = 0; k < D; ++k) {
        obs[size_t(i) * D + k] = b_obs[size_t(j) * D + k];
        next[size_t(i) * D + k] = b_next[size_t(j) * D + k];
    }
    for (int k = 0; k < A; ++k) act[size_t(i) * A + k] = b_act[size_t(j) * A + k];
    rew[i] = b_rew[j];
    done[i] = b_done[j];
}

}  // namespace

ReplayBuffer::ReplayBuffer(int capacity, int obs_dim, int act_dim)
    : capacity_(capacity), obs_dim_(obs_dim), act_dim_(act_dim), obs_(capacity, obs_dim), act_(capacity, act_dim),
      reward_(1, capacity), next_obs_(capacity, obs_dim), done_(1, capacity) {
    if (capacity <= 0) throw std::runtime_error("ReplayBuffer: capacity must be > 0");
}

void ReplayBuffer::add(const float* obs, const float* act, const float* reward, const float* next_obs,
                       const float* final_obs, const int* terminated, const int* truncated, int n, cudaStream_t s) {
    if (n <= 0) return;
    if (n > capacity_) throw std::runtime_error("ReplayBuffer::add: more rows than the capacity");
    add_kernel<<<(n + 127) / 128, 128, 0, s>>>(n, obs_dim_, act_dim_, capacity_, pos_, obs, act, reward, next_obs,
                                               final_obs, terminated, truncated, obs_.data(), act_.data(),
                                               reward_.data(), next_obs_.data(), done_.data());
    KERNEL_CHECK();
    pos_ = (pos_ + n) % capacity_;
    size_ = std::min(size_ + n, capacity_);
}

void ReplayBuffer::sample(int batch, uint64_t seed, uint64_t stream_id, Tensor& obs, Tensor& act, Tensor& reward,
                          Tensor& next_obs, Tensor& done, cudaStream_t s) const {
    if (size_ == 0) throw std::runtime_error("ReplayBuffer::sample: empty");
    sample_kernel<<<(batch + 127) / 128, 128, 0, s>>>(batch, size_, obs_dim_, act_dim_, seed, stream_id, obs_.data(),
                                                     act_.data(), reward_.data(), next_obs_.data(), done_.data(),
                                                     obs.data(), act.data(), reward.data(), next_obs.data(),
                                                     done.data());
    KERNEL_CHECK();
}

}  // namespace temper
