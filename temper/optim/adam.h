#pragma once
// Adam (PyTorch's formula, no weight decay) and global gradient-norm clipping.
// Both run fully on the GPU: no host sync in a training step.
//
//   GradClip clip(grads);            Adam adam(params, grads, {lr});
//   clip.compute(max_norm, stream);  // writes norm and scale to device memory
//   adam.step(stream, clip.scale()); // uses g * scale; scale = null means 1
#include <cuda_runtime.h>

#include <vector>

#include "forge/util/device_buffer.h"
#include "temper/nn/tensor.h"

namespace temper {

struct AdamOptions {
    float lr = 3e-4f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
};

class Adam {
public:
    Adam(std::vector<Tensor*> params, std::vector<Tensor*> grads, AdamOptions opt = {});
    //   m = b1 m + (1 - b1) g,  v = b2 v + (1 - b2) g^2
    //   p -= lr * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)
    void step(cudaStream_t stream, const float* grad_scale = nullptr);
    void set_lr(float lr) { opt_.lr = lr; }
    // Read the learning rate from GPU memory at each step instead (a kernel may change
    // it between steps without a sync: PPO's adaptive schedule). null = use lr().
    void set_lr_device(const float* lr) { lr_device_ = lr; }
    float lr() const { return opt_.lr; }
    long long steps() const { return t_; }

    // State for checkpoints (Phase 7c).
    std::vector<Tensor>& m() { return m_; }
    std::vector<Tensor>& v() { return v_; }
    void set_steps(long long t) { t_ = t; }

private:
    std::vector<Tensor*> params_, grads_;
    std::vector<Tensor> m_, v_;
    AdamOptions opt_;
    long long t_ = 0;
    const float* lr_device_ = nullptr;
};

// Global L2 norm over all gradients; scale = min(1, max_norm / (norm + 1e-6))
// (PyTorch clip_grad_norm_). Deterministic: fixed reduction order.
class GradClip {
public:
    explicit GradClip(std::vector<Tensor*> grads);
    void compute(float max_norm, cudaStream_t stream);
    const float* scale() const { return out_.data() + 1; }  // device pointer
    const float* norm() const { return out_.data(); }       // device pointer (for logs)

private:
    std::vector<Tensor*> grads_;
    forge::DeviceBuffer<float> partial_;  // one sum of squares per tensor
    forge::DeviceBuffer<float> out_;      // [0] norm, [1] scale
    forge::DeviceBuffer<const float*> ptrs_;
    forge::DeviceBuffer<int> sizes_;
};

}  // namespace temper
