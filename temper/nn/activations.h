#pragma once
// ELU (doc 5.3):  y = x if x > 0, else alpha (exp(x) - 1)
//                 dy/dx = 1 if x > 0, else alpha exp(x)
#include "temper/nn/layer.h"

namespace temper {

class Elu : public Layer {
public:
    Elu(int dim, float alpha = 1.0f) : dim_(dim), alpha_(alpha) {}
    int in_dim() const override { return dim_; }
    int out_dim() const override { return dim_; }
    void forward(const float* x, float* y, int batch, cudaStream_t stream) override;
    void backward(const float* dy, float* dx, int batch, cudaStream_t stream) override;
    void params(std::vector<Tensor*>&, std::vector<Tensor*>&) override {}

private:
    int dim_;
    float alpha_;
    const float* x_ = nullptr;
};

// ReLU (SAC, CleanRL):  y = max(0, x),  dy/dx = 1 if x > 0 else 0
class Relu : public Layer {
public:
    explicit Relu(int dim) : dim_(dim) {}
    int in_dim() const override { return dim_; }
    int out_dim() const override { return dim_; }
    void forward(const float* x, float* y, int batch, cudaStream_t stream) override;
    void backward(const float* dy, float* dx, int batch, cudaStream_t stream) override;
    void params(std::vector<Tensor*>&, std::vector<Tensor*>&) override {}

private:
    int dim_;
    const float* x_ = nullptr;
};

// Tanh (Phase 9d: the behavior-cloning policy's output):  y = tanh(x),
// dy/dx = 1 - y^2 (backward uses the output y of the last forward)
class Tanh : public Layer {
public:
    explicit Tanh(int dim) : dim_(dim) {}
    int in_dim() const override { return dim_; }
    int out_dim() const override { return dim_; }
    void forward(const float* x, float* y, int batch, cudaStream_t stream) override;
    void backward(const float* dy, float* dx, int batch, cudaStream_t stream) override;
    void params(std::vector<Tensor*>&, std::vector<Tensor*>&) override {}

private:
    int dim_;
    const float* y_ = nullptr;
};

}  // namespace temper
