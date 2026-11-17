#include "temper/algos/student/student_ops.h"

#include "forge/util/cuda_check.h"

namespace temper {
namespace student {

namespace {

__global__ void masked_mse_kernel(const float* pred, const float* target, const float* mask, const float* count,
                                  int rows, int cols, float* dy, float* loss_sum) {
    __shared__ float part[256];
    const float n = mask ? fmaxf(*count, 1.0f) : float(rows);
    const float inv = 1.0f / (n * float(cols));
    float s = 0.0f;
    for (int i = threadIdx.x; i < rows * cols; i += blockDim.x) {
        const float m = mask ? mask[i / cols] : 1.0f;
        const float e = pred[i] - target[i];
        dy[i] = 2.0f * m * e * inv;
        s += m * e * e;
    }
    part[threadIdx.x] = s;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w /= 2) {
        if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0 && loss_sum) *loss_sum += part[0] * inv;
}

}  // namespace

void masked_mse(const float* pred, const float* target, const float* mask, const float* count, int rows, int cols,
                float* dy, float* loss_sum, cudaStream_t stream) {
    masked_mse_kernel<<<1, 256, 0, stream>>>(pred, target, mask, count, rows, cols, dy, loss_sum);
    KERNEL_CHECK();
}

}  // namespace student
}  // namespace temper
