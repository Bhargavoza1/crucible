#include "temper/common/obs_normalizer.h"

#include <vector>

#include "temper/algos/ppo/ppo_math.h"

namespace temper {

namespace {

// One thread per dimension: batch mean and variance (two passes, fixed order),
// then merge into the running statistics. Thread 0 also updates the count,
// after every thread has read the old count (one block, __syncthreads).
__global__ void update_kernel(const float* x, int n, int dim, float* mean, float* var, float* count) {
    const int k = threadIdx.x;
    const float c = *count;
    if (k < dim) {
        float m = 0.0f;
        for (int i = 0; i < n; ++i) m += x[size_t(i) * dim + k];
        m /= float(n);
        float v = 0.0f;
        for (int i = 0; i < n; ++i) {
            const float d = x[size_t(i) * dim + k] - m;
            v += d * d;
        }
        v /= float(n);
        float rm = mean[k], rv = var[k];
        running_merge(rm, rv, c, m, v, float(n));
        mean[k] = rm;
        var[k] = rv;
    }
    __syncthreads();
    if (k == 0) *count = c + float(n);
}

__global__ void normalize_kernel(const float* x, float* out, size_t total, int dim, const float* mean,
                                 const float* var, float clip) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int k = int(i % dim);
    out[i] = normalize_obs(x[i], mean[k], var[k], clip);
}

}  // namespace

ObsNormalizer::ObsNormalizer(int dim, float clip) : mean(1, dim), var(1, dim), count(1, 1), dim_(dim), clip_(clip) {
    if (dim <= 0 || dim > 1024) throw std::runtime_error("ObsNormalizer: dim must be 1..1024");
    mean.upload(std::vector<float>(dim, 0.0f));
    var.upload(std::vector<float>(dim, 1.0f));
    count.upload({1e-4f});  // gym's RunningMeanStd start
}

void ObsNormalizer::update(const float* x, int n, cudaStream_t stream) {
    if (n <= 0) return;
    update_kernel<<<1, ((dim_ + 31) / 32) * 32, 0, stream>>>(x, n, dim_, mean.data(), var.data(), count.data());
    KERNEL_CHECK();
}

void ObsNormalizer::normalize(const float* x, float* out, int n, cudaStream_t stream) const {
    const size_t total = size_t(n) * dim_;
    if (!total) return;
    normalize_kernel<<<int((total + 255) / 256), 256, 0, stream>>>(x, out, total, dim_, mean.data(), var.data(),
                                                                    clip_);
    KERNEL_CHECK();
}

}  // namespace temper
