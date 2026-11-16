#pragma once
// LayerNorm over each row (Phase 9d), like torch.nn.LayerNorm(dim):
//   y = (x - mean) / sqrt(var + eps) * gamma + beta   (biased variance, eps 1e-5)
// gamma starts at 1, beta at 0. One warp per row in forward and backward;
// dgamma, dbeta: one thread per column (sum over the batch). The CPU answer key is
// temper::ref::layernorm_forward / layernorm_backward.
#include "forge/util/device_buffer.h"
#include "temper/nn/layer.h"

namespace temper {

class LayerNorm : public Layer {
public:
    explicit LayerNorm(int dim, float eps = 1e-5f);
    int in_dim() const override { return dim_; }
    int out_dim() const override { return dim_; }
    void forward(const float* x, float* y, int batch, cudaStream_t stream) override;
    void backward(const float* dy, float* dx, int batch, cudaStream_t stream) override;
    void params(std::vector<Tensor*>& p, std::vector<Tensor*>& g) override {
        p.push_back(&gamma);
        p.push_back(&beta);
        g.push_back(&dgamma);
        g.push_back(&dbeta);
    }

    Tensor gamma, beta, dgamma, dbeta;

private:
    int dim_;
    float eps_;
    const float* x_ = nullptr;
    forge::DeviceBuffer<float> mean_, rstd_;  // per row, from the last forward
};

}  // namespace temper
