#pragma once
// Replay ring buffer in GPU memory (doc 5.1) for off-policy algorithms.
// One transition = (obs, action, reward, next_obs, terminated). next_obs is the
// real last obs (final_obs) for envs that just ended (doc 3.6); terminated (not
// truncated) is what stops bootstrapping.
//
// add() writes num_envs transitions at once (the oldest are overwritten when full).
// sample() draws a batch uniformly with the counter RNG (same seed, same batches).
#include <cstdint>

#include <cuda_runtime.h>

#include "temper/nn/tensor.h"

namespace temper {

class ReplayBuffer {
public:
    ReplayBuffer(int capacity, int obs_dim, int act_dim);

    // All device arrays with n rows. next_obs = obs after the step, final_obs = the
    // real last obs where terminated or truncated: the buffer keeps final_obs there.
    void add(const float* obs, const float* act, const float* reward, const float* next_obs, const float* final_obs,
             const int* terminated, const int* truncated, int n, cudaStream_t stream);

    // Fills the batch tensors (batch rows) from random stored transitions.
    void sample(int batch, uint64_t seed, uint64_t stream_id, Tensor& obs, Tensor& act, Tensor& reward,
                Tensor& next_obs, Tensor& done, cudaStream_t stream) const;

    int size() const { return size_; }
    int capacity() const { return capacity_; }

private:
    int capacity_, obs_dim_, act_dim_;
    int pos_ = 0, size_ = 0;  // known on the host: add() always writes n rows
    Tensor obs_, act_, reward_, next_obs_, done_;
};

}  // namespace temper
