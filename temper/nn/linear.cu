#include "temper/nn/linear.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "forge/util/rng.h"
#include "temper/nn/gemm.h"

namespace temper {

namespace {

// db[j] = sum_i dy[i * out + j]: one thread per column (neighbor threads read
// neighbor addresses).
__global__ void column_sum_kernel(const float* dy, float* db, int batch, int out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out) return;
    float s = 0.0f;
    for (int i = 0; i < batch; ++i) s += dy[size_t(i) * out + j];
    db[j] = s;
}

float gaussian(forge::Rng& r) {  // Box-Muller
    const float u1 = fmaxf(r.uniform(), 1e-12f), u2 = r.uniform();
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
}

}  // namespace

Linear::Linear(int in, int out, uint64_t seed) : W(out, in), b(1, out), dW(out, in), db(1, out), in_(in), out_(out) {
    if (in <= 0 || out <= 0) throw std::runtime_error("Linear: sizes must be > 0");
    forge::Rng rng(seed, 0x4C494E, 0);
    const float bound = 1.0f / std::sqrt(float(in));
    std::vector<float> w(size_t(out) * in), bias(out);
    for (auto& x : w) x = rng.uniform(-bound, bound);
    for (auto& x : bias) x = rng.uniform(-bound, bound);
    W.upload(w);
    b.upload(bias);
    dW.zero();
    db.zero();
}

void Linear::init_orthogonal(float gain, uint64_t seed) {
    // Gram-Schmidt (in double) on the longer side of a Gaussian matrix, like
    // torch.nn.init.orthogonal_.
    forge::Rng rng(seed, 0x4F5254, 0);
    const int rows = std::min(out_, in_), cols = std::max(out_, in_);
    std::vector<double> q(size_t(rows) * cols);
    for (auto& x : q) x = gaussian(rng);
    for (int i = 0; i < rows; ++i) {
        double* qi = &q[size_t(i) * cols];
        for (int k = 0; k < i; ++k) {
            const double* qk = &q[size_t(k) * cols];
            double d = 0;
            for (int c = 0; c < cols; ++c) d += qi[c] * qk[c];
            for (int c = 0; c < cols; ++c) qi[c] -= d * qk[c];
        }
        double n = 0;
        for (int c = 0; c < cols; ++c) n += qi[c] * qi[c];
        n = std::sqrt(n);
        for (int c = 0; c < cols; ++c) qi[c] /= n;
    }
    std::vector<float> w(size_t(out_) * in_);
    for (int o = 0; o < out_; ++o)
        for (int i = 0; i < in_; ++i)  // W is out x in; q is rows x cols with rows = min side
            w[size_t(o) * in_ + i] = gain * float(out_ <= in_ ? q[size_t(o) * cols + i] : q[size_t(i) * cols + o]);
    W.upload(w);
    b.zero();
    CUDA_CHECK(cudaDeviceSynchronize());  // init only, never in a loop
}

void Linear::forward(const float* x, float* y, int batch, cudaStream_t stream) {
    x_ = x;
    // y (batch x out) = x (batch x in) * W^T  +  b repeated on every row (ldc = 0)
    gemm(Op::N, Op::T, batch, out_, in_, 1.0f, x, in_, W.data(), in_, 1.0f, b.data(), 0, y, out_, stream);
}

void Linear::backward(const float* dy, float* dx, int batch, cudaStream_t stream) {
    if (!x_) throw std::runtime_error("Linear::backward before forward");
    // dW (out x in) = dy^T (out x batch) * x (batch x in)
    gemm(Op::T, Op::N, out_, in_, batch, 1.0f, dy, out_, x_, in_, 0.0f, dW.data(), in_, dW.data(), in_, stream);
    column_sum_kernel<<<(out_ + 127) / 128, 128, 0, stream>>>(dy, db.data(), batch, out_);
    KERNEL_CHECK();
    // dx (batch x in) = dy (batch x out) * W (out x in)
    if (dx) gemm(Op::N, Op::N, batch, in_, out_, 1.0f, dy, out_, W.data(), in_, 0.0f, dx, in_, dx, in_, stream);
}

}  // namespace temper
