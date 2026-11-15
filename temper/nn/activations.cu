#include "temper/nn/activations.h"

#include <stdexcept>

#include "forge/util/cuda_check.h"

namespace temper {

namespace {

__global__ void elu_forward_kernel(const float* x, float* y, size_t n, float alpha) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    y[i] = v > 0.0f ? v : alpha * expm1f(v);
}

__global__ void elu_backward_kernel(const float* x, const float* dy, float* dx, size_t n, float alpha) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    dx[i] = dy[i] * (v > 0.0f ? 1.0f : alpha * expf(v));
}

int blocks(size_t n) { return int((n + 255) / 256); }

}  // namespace

void Elu::forward(const float* x, float* y, int batch, cudaStream_t stream) {
    x_ = x;
    const size_t n = size_t(batch) * dim_;
    if (n == 0) return;
    elu_forward_kernel<<<blocks(n), 256, 0, stream>>>(x, y, n, alpha_);
    KERNEL_CHECK();
}

void Elu::backward(const float* dy, float* dx, int batch, cudaStream_t stream) {
    if (!x_) throw std::runtime_error("Elu::backward before forward");
    const size_t n = size_t(batch) * dim_;
    if (n == 0 || !dx) return;
    elu_backward_kernel<<<blocks(n), 256, 0, stream>>>(x_, dy, dx, n, alpha_);
    KERNEL_CHECK();
}

}  // namespace temper
