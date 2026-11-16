#include "temper/nn/layernorm.h"

#include <stdexcept>
#include <vector>

#include "forge/util/cuda_check.h"

namespace temper {

namespace {

__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One warp per row: mean, 1 / sqrt(var + eps), y.
__global__ void ln_forward_kernel(const float* x, const float* gamma, const float* beta, float* y, float* mean,
                                  float* rstd, int batch, int D, float eps) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= batch) return;
    const float* r = x + size_t(row) * D;
    float s = 0.0f;
    for (int d = lane; d < D; d += 32) s += r[d];
    const float m = warp_sum(s) / float(D);
    float v = 0.0f;
    for (int d = lane; d < D; d += 32) v += (r[d] - m) * (r[d] - m);
    const float inv = rsqrtf(warp_sum(v) / float(D) + eps);
    for (int d = lane; d < D; d += 32) y[size_t(row) * D + d] = (r[d] - m) * inv * gamma[d] + beta[d];
    if (lane == 0) {
        mean[row] = m;
        rstd[row] = inv;
    }
}

// dx = rstd / D (D g - sum(g) - xhat sum(g xhat)), g = dy gamma. One warp per row.
__global__ void ln_backward_dx_kernel(const float* x, const float* gamma, const float* dy, const float* mean,
                                      const float* rstd, float* dx, int batch, int D) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= batch) return;
    const float m = mean[row], inv = rstd[row];
    const float* r = x + size_t(row) * D;
    const float* g = dy + size_t(row) * D;
    float sg = 0.0f, sgx = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float gd = g[d] * gamma[d];
        sg += gd;
        sgx += gd * (r[d] - m) * inv;
    }
    sg = warp_sum(sg);
    sgx = warp_sum(sgx);
    for (int d = lane; d < D; d += 32) {
        const float xhat = (r[d] - m) * inv;
        dx[size_t(row) * D + d] = inv / float(D) * (float(D) * g[d] * gamma[d] - sg - xhat * sgx);
    }
}

// dgamma[d] = sum_n dy xhat, dbeta[d] = sum_n dy: one thread per column.
__global__ void ln_backward_params_kernel(const float* x, const float* dy, const float* mean, const float* rstd,
                                          float* dgamma, float* dbeta, int batch, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float sg = 0.0f, sb = 0.0f;
    for (int n = 0; n < batch; ++n) {
        const float g = dy[size_t(n) * D + d];
        sg += g * (x[size_t(n) * D + d] - mean[n]) * rstd[n];
        sb += g;
    }
    dgamma[d] = sg;
    dbeta[d] = sb;
}

}  // namespace

LayerNorm::LayerNorm(int dim, float eps) : gamma(1, dim), beta(1, dim), dgamma(1, dim), dbeta(1, dim), dim_(dim), eps_(eps) {
    if (dim <= 0) throw std::runtime_error("LayerNorm: dim must be > 0");
    gamma.upload(std::vector<float>(dim, 1.0f));
    beta.zero();
    dgamma.zero();
    dbeta.zero();
}

void LayerNorm::forward(const float* x, float* y, int batch, cudaStream_t stream) {
    if (size_t(batch) > mean_.size()) {
        mean_.resize(batch);
        rstd_.resize(batch);
    }
    x_ = x;
    const int rows_per_block = 4;
    ln_forward_kernel<<<(batch + rows_per_block - 1) / rows_per_block, 32 * rows_per_block, 0, stream>>>(
        x, gamma.data(), beta.data(), y, mean_.data(), rstd_.data(), batch, dim_, eps_);
    KERNEL_CHECK();
}

void LayerNorm::backward(const float* dy, float* dx, int batch, cudaStream_t stream) {
    ln_backward_params_kernel<<<(dim_ + 127) / 128, 128, 0, stream>>>(x_, dy, mean_.data(), rstd_.data(),
                                                                     dgamma.data(), dbeta.data(), batch, dim_);
    KERNEL_CHECK();
    if (dx) {
        const int rows_per_block = 4;
        ln_backward_dx_kernel<<<(batch + rows_per_block - 1) / rows_per_block, 32 * rows_per_block, 0, stream>>>(
            x_, gamma.data(), dy, mean_.data(), rstd_.data(), dx, batch, dim_);
        KERNEL_CHECK();
    }
}

}  // namespace temper
