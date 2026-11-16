#pragma once
// 2D convolution layer on cuDNN (Phase 9d, doc 5.3), in the Layer interface: each
// sample is one row of C * H * W floats in PyTorch's order (channel, row, column), the
// output one row of K * Ho * Wo, so conv layers stack with Linear, ReLU and LayerNorm.
//   y = W * x + b  (cross-correlation, zero padding, like torch.nn.Conv2d)
//   W: K x (C k k) (PyTorch's [K][C][k][k]), b: 1 x K
//   Ho = (H + 2 pad - k) / stride + 1, Wo likewise
// Exact FP32 math (CUDNN_FMA_MATH: no TF32), deterministic algorithms (chosen by
// cuDNN's heuristics among the deterministic ones, per batch size). The CPU answer
// key is temper::ref::conv2d_forward / conv2d_backward.
#include <cstdint>

#include <cudnn.h>

#include "forge/util/device_buffer.h"
#include "temper/nn/layer.h"

namespace temper {

class Conv2d : public Layer {
public:
    // W and b uniform in [-1/sqrt(C k k), 1/sqrt(C k k)] (PyTorch's default), from 'seed'.
    Conv2d(int C, int H, int W, int K, int k, int stride, int pad, uint64_t seed);
    ~Conv2d() override;
    Conv2d(const Conv2d&) = delete;
    Conv2d& operator=(const Conv2d&) = delete;

    int in_dim() const override { return C_ * H_ * W_; }
    int out_dim() const override { return K_ * Ho_ * Wo_; }
    int out_channels() const { return K_; }
    int out_height() const { return Ho_; }
    int out_width() const { return Wo_; }
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
    void set_batch(int batch);  // descriptors and algorithms for this batch size

    int C_, H_, W_, K_, k_, stride_, pad_, Ho_, Wo_;
    int batch_ = 0;
    const float* x_ = nullptr;
    cudnnTensorDescriptor_t xdesc_ = nullptr, ydesc_ = nullptr, bdesc_ = nullptr;
    cudnnFilterDescriptor_t wdesc_ = nullptr;
    cudnnConvolutionDescriptor_t conv_ = nullptr;
    cudnnConvolutionFwdAlgo_t fwd_algo_{};
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo_{};
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_{};
    forge::DeviceBuffer<char> workspace_;
};

// One cuDNN handle per thread (made on first use), set to 'stream' on each call.
cudnnHandle_t cudnn_handle(cudaStream_t stream);

}  // namespace temper
