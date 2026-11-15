#pragma once
// Linear layer:  y = x W^T + b      x: batch x in,  W: out x in,  b: 1 x out
// Backward (doc 5.3):  dW = dy^T x,  db = sum over the batch of dy,  dx = dy W
// The three products are CUTLASS GEMMs (gemm.h).
#include <cstdint>

#include "temper/nn/layer.h"

namespace temper {

class Linear : public Layer {
public:
    // W and b uniform in [-1/sqrt(in), 1/sqrt(in)] (PyTorch's default), from 'seed'.
    Linear(int in, int out, uint64_t seed);

    // Orthogonal W (rows or columns orthonormal) times 'gain', b = 0. CleanRL's PPO
    // uses this (gain sqrt(2) hidden, 0.01 actor output, 1.0 critic output).
    void init_orthogonal(float gain, uint64_t seed);

    int in_dim() const override { return in_; }
    int out_dim() const override { return out_; }
    void forward(const float* x, float* y, int batch, cudaStream_t stream) override;
    void backward(const float* dy, float* dx, int batch, cudaStream_t stream) override;
    void params(std::vector<Tensor*>& p, std::vector<Tensor*>& g) override {
        p.push_back(&W);
        p.push_back(&b);
        g.push_back(&dW);
        g.push_back(&db);
    }

    Tensor W, b, dW, db;

private:
    int in_, out_;
    const float* x_ = nullptr;  // input of the last forward
};

}  // namespace temper
