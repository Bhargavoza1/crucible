#include "temper/nn/gru.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "forge/util/rng.h"
#include "temper/nn/gemm.h"

namespace temper {

namespace {

__device__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

// hp[b][j] = (1 - reset[b]) h[b][j]  (h null: 0)
__global__ void masked_copy_kernel(const float* h, const float* reset, float* hp, int B, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H) return;
    const float m = reset ? 1.0f - reset[i / H] : 1.0f;
    hp[i] = h ? m * h[i] : 0.0f;
}

// h = (1 - z) n + z hp from the step's gx, gh (rows r, z, n)
__global__ void gates_forward_kernel(const float* gx, const float* gh, const float* hp, float* h, int B, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H) return;
    const int b = i / H, j = i % H;
    const float* x = gx + size_t(b) * 3 * H;
    const float* g = gh + size_t(b) * 3 * H;
    const float r = sigm(x[j] + g[j]), z = sigm(x[H + j] + g[H + j]);
    const float n = tanhf(x[2 * H + j] + r * g[2 * H + j]);
    h[i] = (1.0f - z) * n + z * hp[i];
}

// One step back: g = dh (from later steps) + dy; writes dgx, dgh [B][3H] and the
// direct part of dL/dhp (g z) into dhp.
__global__ void gates_backward_kernel(const float* gx, const float* gh, const float* hp, const float* dy,
                                      const float* dh, float* dgx, float* dgh, float* dhp, int B, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H) return;
    const int b = i / H, j = i % H;
    const float* x = gx + size_t(b) * 3 * H;
    const float* gg = gh + size_t(b) * 3 * H;
    const float r = sigm(x[j] + gg[j]), z = sigm(x[H + j] + gg[H + j]);
    const float n = tanhf(x[2 * H + j] + r * gg[2 * H + j]);
    const float g = dh[i] + dy[i];
    const float dn = g * (1.0f - z), dz = g * (hp[i] - n);
    const float dan = dn * (1.0f - n * n), dr = dan * gg[2 * H + j];
    const float dar = dr * r * (1.0f - r), daz = dz * z * (1.0f - z);
    float* ox = dgx + size_t(b) * 3 * H;
    float* oh = dgh + size_t(b) * 3 * H;
    ox[j] = dar, ox[H + j] = daz, ox[2 * H + j] = dan;
    oh[j] = dar, oh[H + j] = daz, oh[2 * H + j] = dan * r;
    dhp[i] = g * z;
}

// d[j] = sum_i a[i][j] (fixed order: deterministic)
__global__ void column_sum_kernel(const float* a, float* d, int rows, int cols) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float s = 0.0f;
    for (int i = 0; i < rows; ++i) s += a[size_t(i) * cols + j];
    d[j] = s;
}

int blocks(int n) { return (n + 255) / 256; }

}  // namespace

Gru::Gru(int in, int hidden, uint64_t seed)
    : W_ih(3 * hidden, in), W_hh(3 * hidden, hidden), b_ih(1, 3 * hidden), b_hh(1, 3 * hidden),
      dW_ih(3 * hidden, in), dW_hh(3 * hidden, hidden), db_ih(1, 3 * hidden), db_hh(1, 3 * hidden), in_(in),
      H_(hidden) {
    if (in <= 0 || hidden <= 0) throw std::runtime_error("Gru: sizes must be > 0");
    forge::Rng rng(seed, 0x475255, 0);
    const float bound = 1.0f / std::sqrt(float(hidden));
    for (Tensor* t : {&W_ih, &W_hh, &b_ih, &b_hh}) {
        std::vector<float> v(t->size());
        for (auto& x : v) x = rng.uniform(-bound, bound);
        t->upload(v);
    }
    for (Tensor* t : {&dW_ih, &dW_hh, &db_ih, &db_hh}) t->zero();
}

void Gru::allocate(int max_tb) {
    max_tb_ = max_tb;
    gx_.resize(max_tb, 3 * H_);
    gh_.resize(max_tb, 3 * H_);
    hp_.resize(max_tb, H_);
    dgx_.resize(max_tb, 3 * H_);
    dgh_.resize(max_tb, 3 * H_);
}

void Gru::forward(const float* x, const float* h0, const float* reset, int T, int B, float* y, cudaStream_t stream) {
    if (T * B > max_tb_) throw std::runtime_error("Gru: T * B = " + std::to_string(T * B) + " > allocated");
    x_ = x, reset_ = reset, T_ = T, B_ = B;
    const int H = H_, G = 3 * H, TB = T * B;
    // gx for all steps at once: [TB][3H] = x [TB][in] W_ih^T + b_ih
    gemm(Op::N, Op::T, TB, G, in_, 1.0f, x, in_, W_ih.data(), in_, 1.0f, b_ih.data(), 0, gx_.data(), G, stream);
    for (int t = 0; t < T; ++t) {
        const float* h_prev = t == 0 ? h0 : y + size_t(t - 1) * B * H;
        float* hp = hp_.data() + size_t(t) * B * H;
        masked_copy_kernel<<<blocks(B * H), 256, 0, stream>>>(h_prev, reset ? reset + size_t(t) * B : nullptr, hp,
                                                               B, H);
        KERNEL_CHECK();
        float* gh = gh_.data() + size_t(t) * B * G;
        gemm(Op::N, Op::T, B, G, H, 1.0f, hp, H, W_hh.data(), H, 1.0f, b_hh.data(), 0, gh, G, stream);
        gates_forward_kernel<<<blocks(B * H), 256, 0, stream>>>(gx_.data() + size_t(t) * B * G, gh, hp,
                                                                 y + size_t(t) * B * H, B, H);
        KERNEL_CHECK();
    }
}

void Gru::backward(const float* dy, const float* dhT, float* dx, float* dh0, cudaStream_t stream) {
    if (!x_) throw std::runtime_error("Gru::backward before forward");
    const int T = T_, B = B_, H = H_, G = 3 * H, TB = T * B;
    if (dh_.size() < size_t(B) * H) dh_.resize(B, H), dhp_.resize(B, H);
    if (dhT)
        CUDA_CHECK(cudaMemcpyAsync(dh_.data(), dhT, size_t(B) * H * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    else
        CUDA_CHECK(cudaMemsetAsync(dh_.data(), 0, size_t(B) * H * sizeof(float), stream));
    for (int t = T - 1; t >= 0; --t) {
        const size_t o = size_t(t) * B;
        float* dgh = dgh_.data() + o * G;
        gates_backward_kernel<<<blocks(B * H), 256, 0, stream>>>(gx_.data() + o * G, gh_.data() + o * G,
                                                                  hp_.data() + o * H, dy + o * H, dh_.data(),
                                                                  dgx_.data() + o * G, dgh, dhp_.data(), B, H);
        KERNEL_CHECK();
        // dL/dhp += dgh W_hh  ([B][3H] x [3H][H])
        gemm(Op::N, Op::N, B, H, G, 1.0f, dgh, G, W_hh.data(), H, 1.0f, dhp_.data(), H, dhp_.data(), H, stream);
        // into the step before: dh = (1 - reset) dhp
        masked_copy_kernel<<<blocks(B * H), 256, 0, stream>>>(dhp_.data(), reset_ ? reset_ + o : nullptr,
                                                               dh_.data(), B, H);
        KERNEL_CHECK();
    }
    if (dh0)
        CUDA_CHECK(cudaMemcpyAsync(dh0, dh_.data(), size_t(B) * H * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    // weights, once over all steps
    gemm(Op::T, Op::N, G, in_, TB, 1.0f, dgx_.data(), G, x_, in_, 0.0f, dW_ih.data(), in_, dW_ih.data(), in_, stream);
    gemm(Op::T, Op::N, G, H, TB, 1.0f, dgh_.data(), G, hp_.data(), H, 0.0f, dW_hh.data(), H, dW_hh.data(), H, stream);
    column_sum_kernel<<<blocks(G), 256, 0, stream>>>(dgx_.data(), db_ih.data(), TB, G);
    KERNEL_CHECK();
    column_sum_kernel<<<blocks(G), 256, 0, stream>>>(dgh_.data(), db_hh.data(), TB, G);
    KERNEL_CHECK();
    if (dx) gemm(Op::N, Op::N, TB, in_, G, 1.0f, dgx_.data(), G, W_ih.data(), in_, 0.0f, dx, in_, dx, in_, stream);
}

}  // namespace temper
