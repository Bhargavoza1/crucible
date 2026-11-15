#pragma once
// MLP (doc 5.3): Linear, ELU, Linear, ELU, ..., Linear (no activation at the end).
//   Mlp net({obs_dim, 256, 256, 256, act_dim}, max_batch, seed);
//   const float* y = net.forward(x, batch, stream);
//   net.backward(dy, batch, stream);           // fills every parameter gradient
// Buffers for every layer's output are sized for max_batch once.
#include <cstdint>
#include <memory>
#include <vector>

#include "temper/nn/activations.h"
#include "temper/nn/linear.h"

namespace temper {

enum class Activation { kElu, kRelu };

class Mlp {
public:
    // dims = {in, hidden..., out}, at least {in, out}. Hidden activation: ELU (PPO,
    // doc 5.3) or ReLU (SAC, as CleanRL).
    Mlp(const std::vector<int>& dims, int max_batch, uint64_t seed, Activation act = Activation::kElu);

    // Copy all weights from another Mlp of the same sizes (target networks).
    void copy_from(Mlp& other, cudaStream_t stream);

    // CleanRL-style init: orthogonal, gain sqrt(2) on hidden layers, 'out_gain'
    // on the last one, biases 0.
    void init_orthogonal(float out_gain, uint64_t seed);

    // Returns the output buffer (batch x out), valid until the next forward.
    const float* forward(const float* x, int batch, cudaStream_t stream);
    // dy: dL/d(output), batch x out. dx (batch x in) may be null.
    void backward(const float* dy, int batch, cudaStream_t stream, float* dx = nullptr);

    void params(std::vector<Tensor*>& p, std::vector<Tensor*>& g);
    std::vector<Linear*> linears();
    int in_dim() const { return dims_.front(); }
    int out_dim() const { return dims_.back(); }
    int max_batch() const { return max_batch_; }
    const std::vector<int>& dims() const { return dims_; }

private:
    std::vector<int> dims_;
    int max_batch_;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::vector<Tensor> out_;   // out_[k]: output of layer k
    std::vector<Tensor> grad_;  // grad_[k]: dL/d(input of layer k), k >= 1
};

}  // namespace temper
