#include "temper/optim/adam.h"

#include <cmath>
#include <stdexcept>

#include "forge/util/cuda_check.h"

namespace temper {

namespace {

__global__ void adam_kernel(float* p, const float* g, float* m, float* v, size_t n, float lr, float b1, float b2,
                            float eps, float bc1, float bc2, const float* scale) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gi = g[i] * (scale ? *scale : 1.0f);
    const float mi = b1 * m[i] + (1.0f - b1) * gi;
    const float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    p[i] -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
}

// One block per tensor: sum of squares in a fixed order (deterministic).
__global__ void sumsq_kernel(const float* const* ptrs, const int* sizes, float* partial) {
    __shared__ float s[256];
    const float* g = ptrs[blockIdx.x];
    const int n = sizes[blockIdx.x];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += g[i] * g[i];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

__global__ void clip_scale_kernel(const float* partial, int count, float max_norm, float* out) {
    float s = 0.0f;
    for (int k = 0; k < count; ++k) s += partial[k];
    const float norm = sqrtf(s);
    out[0] = norm;
    out[1] = fminf(1.0f, max_norm / (norm + 1e-6f));
}

}  // namespace

Adam::Adam(std::vector<Tensor*> params, std::vector<Tensor*> grads, AdamOptions opt)
    : params_(std::move(params)), grads_(std::move(grads)), opt_(opt) {
    if (params_.size() != grads_.size()) throw std::runtime_error("Adam: params and grads differ in count");
    for (size_t k = 0; k < params_.size(); ++k) {
        if (params_[k]->size() != grads_[k]->size()) throw std::runtime_error("Adam: param and grad sizes differ");
        m_.emplace_back(params_[k]->rows, params_[k]->cols);
        v_.emplace_back(params_[k]->rows, params_[k]->cols);
        m_.back().zero();
        v_.back().zero();
    }
}

void Adam::step(cudaStream_t stream, const float* grad_scale) {
    ++t_;
    const float bc1 = float(1.0 - std::pow(double(opt_.beta1), double(t_)));
    const float bc2 = float(1.0 - std::pow(double(opt_.beta2), double(t_)));
    for (size_t k = 0; k < params_.size(); ++k) {
        const size_t n = params_[k]->size();
        if (!n) continue;
        adam_kernel<<<int((n + 255) / 256), 256, 0, stream>>>(params_[k]->data(), grads_[k]->data(), m_[k].data(),
                                                             v_[k].data(), n, opt_.lr, opt_.beta1, opt_.beta2,
                                                             opt_.eps, bc1, bc2, grad_scale);
        KERNEL_CHECK();
    }
}

GradClip::GradClip(std::vector<Tensor*> grads) : grads_(std::move(grads)) {
    std::vector<const float*> p;
    std::vector<int> n;
    for (Tensor* t : grads_) {
        p.push_back(t->data());
        n.push_back(static_cast<int>(t->size()));
    }
    ptrs_.upload(p);
    sizes_.upload(n);
    partial_.resize(grads_.size());
    out_.resize(2);
}

void GradClip::compute(float max_norm, cudaStream_t stream) {
    const int count = static_cast<int>(grads_.size());
    if (count == 0) return;
    sumsq_kernel<<<count, 256, 0, stream>>>(ptrs_.data(), sizes_.data(), partial_.data());
    KERNEL_CHECK();
    clip_scale_kernel<<<1, 1, 0, stream>>>(partial_.data(), count, max_norm, out_.data());
    KERNEL_CHECK();
}

}  // namespace temper
