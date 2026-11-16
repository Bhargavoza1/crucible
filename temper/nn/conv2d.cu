#include "temper/nn/conv2d.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "forge/util/rng.h"

namespace temper {

namespace {

void cudnn_check(cudnnStatus_t s, const char* what) {
    if (s != CUDNN_STATUS_SUCCESS) throw std::runtime_error(std::string("cuDNN: ") + what + ": " + cudnnGetErrorString(s));
}
#define CUDNN_CHECK(x) cudnn_check((x), #x)

}  // namespace

cudnnHandle_t cudnn_handle(cudaStream_t stream) {
    thread_local cudnnHandle_t h = [] {
        cudnnHandle_t made = nullptr;
        CUDNN_CHECK(cudnnCreate(&made));
        return made;
    }();
    CUDNN_CHECK(cudnnSetStream(h, stream));
    return h;
}

Conv2d::Conv2d(int C, int H, int W_in, int K, int k, int stride, int pad, uint64_t seed)
    : C_(C), H_(H), W_(W_in), K_(K), k_(k), stride_(stride), pad_(pad) {
    if (C <= 0 || H <= 0 || W_in <= 0 || K <= 0 || k <= 0 || stride <= 0 || pad < 0)
        throw std::runtime_error("Conv2d: sizes must be > 0 (pad >= 0)");
    Ho_ = (H + 2 * pad - k) / stride + 1;
    Wo_ = (W_in + 2 * pad - k) / stride + 1;
    if (Ho_ <= 0 || Wo_ <= 0) throw std::runtime_error("Conv2d: kernel larger than the padded input");
    const int fan_in = C * k * k;
    W.resize(K, fan_in);
    dW.resize(K, fan_in);
    b.resize(1, K);
    db.resize(1, K);
    forge::Rng rng(seed, 0x434F4E, 0);
    const float bound = 1.0f / std::sqrt(float(fan_in));
    std::vector<float> w(size_t(K) * fan_in), bias(K);
    for (auto& x : w) x = rng.uniform(-bound, bound);
    for (auto& x : bias) x = rng.uniform(-bound, bound);
    W.upload(w);
    b.upload(bias);
    dW.zero();
    db.zero();

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xdesc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&ydesc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bdesc_));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&wdesc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bdesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, K, 1, 1));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(wdesc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, k, k));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_, CUDNN_FMA_MATH));  // exact FP32, no TF32
}

Conv2d::~Conv2d() {
    if (conv_) cudnnDestroyConvolutionDescriptor(conv_);
    if (wdesc_) cudnnDestroyFilterDescriptor(wdesc_);
    if (bdesc_) cudnnDestroyTensorDescriptor(bdesc_);
    if (ydesc_) cudnnDestroyTensorDescriptor(ydesc_);
    if (xdesc_) cudnnDestroyTensorDescriptor(xdesc_);
}

void Conv2d::set_batch(int batch) {
    if (batch == batch_) return;
    batch_ = batch;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xdesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, C_, H_, W_));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(ydesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, K_, Ho_, Wo_));
    int n = 0, on = 0;
    cudnnHandle_t h = cudnn_handle(nullptr);
    size_t ws = 0;
    // Heuristic order (fastest expected first); take the first one that works and is deterministic.
    {
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(h, &n));
        std::vector<cudnnConvolutionFwdAlgoPerf_t> perf(n);
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(h, xdesc_, wdesc_, conv_, ydesc_, n, &on, perf.data()));
        bool found = false;
        for (int i = 0; i < on && !found; ++i)
            if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].determinism == CUDNN_DETERMINISTIC) {
                fwd_algo_ = perf[i].algo;
                ws = std::max(ws, perf[i].memory);
                found = true;
            }
        if (!found) throw std::runtime_error("Conv2d: no deterministic cuDNN forward algorithm");
    }
    {
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(h, &n));
        std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf(n);
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(h, wdesc_, ydesc_, conv_, xdesc_, n, &on, perf.data()));
        bool found = false;
        for (int i = 0; i < on && !found; ++i)
            if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].determinism == CUDNN_DETERMINISTIC) {
                bwd_data_algo_ = perf[i].algo;
                ws = std::max(ws, perf[i].memory);
                found = true;
            }
        if (!found) throw std::runtime_error("Conv2d: no deterministic cuDNN backward-data algorithm");
    }
    {
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(h, &n));
        std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf(n);
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(h, xdesc_, ydesc_, conv_, wdesc_, n, &on, perf.data()));
        bool found = false;
        for (int i = 0; i < on && !found; ++i)
            if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].determinism == CUDNN_DETERMINISTIC) {
                bwd_filter_algo_ = perf[i].algo;
                ws = std::max(ws, perf[i].memory);
                found = true;
            }
        if (!found) throw std::runtime_error("Conv2d: no deterministic cuDNN backward-filter algorithm");
    }
    if (ws > workspace_.size()) workspace_.resize(ws);
}

void Conv2d::forward(const float* x, float* y, int batch, cudaStream_t stream) {
    set_batch(batch);
    x_ = x;
    cudnnHandle_t h = cudnn_handle(stream);
    const float one = 1.0f, zero = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(h, &one, xdesc_, x, wdesc_, W.data(), conv_, fwd_algo_, workspace_.data(),
                                        workspace_.size(), &zero, ydesc_, y));
    CUDNN_CHECK(cudnnAddTensor(h, &one, bdesc_, b.data(), &one, ydesc_, y));
}

void Conv2d::backward(const float* dy, float* dx, int batch, cudaStream_t stream) {
    set_batch(batch);
    cudnnHandle_t h = cudnn_handle(stream);
    const float one = 1.0f, zero = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardBias(h, &one, ydesc_, dy, &zero, bdesc_, db.data()));
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(h, &one, xdesc_, x_, ydesc_, dy, conv_, bwd_filter_algo_,
                                               workspace_.data(), workspace_.size(), &zero, wdesc_, dW.data()));
    if (dx)
        CUDNN_CHECK(cudnnConvolutionBackwardData(h, &one, wdesc_, W.data(), ydesc_, dy, conv_, bwd_data_algo_,
                                                 workspace_.data(), workspace_.size(), &zero, xdesc_, dx));
}

}  // namespace temper
