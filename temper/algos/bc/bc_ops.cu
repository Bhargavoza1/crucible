#include "temper/algos/bc/bc_ops.h"

#include "forge/util/cuda_check.h"

namespace temper {
namespace bc {

namespace {

__global__ void images_kernel(const uint8_t* in, int batch, int cams, int H, int W, const int* shifts, int pad,
                              float* out) {
    const size_t total = size_t(batch) * cams * 3 * H * W;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    // i walks out: [cam][n][ch][y][x]
    const int x = int(i % W);
    const int y = int((i / W) % H);
    const int ch = int((i / (size_t(W) * H)) % 3);
    const int n = int((i / (size_t(W) * H * 3)) % batch);
    const int c = int(i / (size_t(W) * H * 3 * batch));
    int sy = y, sx = x;
    if (shifts) {
        const int* s = shifts + (size_t(n) * cams + c) * 2;
        sx = min(max(x + s[0] - pad, 0), W - 1);
        sy = min(max(y + s[1] - pad, 0), H - 1);
    }
    out[i] = float(in[(((size_t(n) * cams + c) * H + sy) * W + sx) * 3 + ch]) * (1.0f / 255.0f);
}

__global__ void copy_columns_kernel(const float* src, int src_ld, int src_col, float* dst, int dst_ld, int dst_col,
                                    int rows, int cols) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * cols) return;
    const int r = int(i / cols), c = int(i % cols);
    dst[size_t(r) * dst_ld + dst_col + c] = src[size_t(r) * src_ld + src_col + c];
}

__global__ void mse_kernel(const float* pred, const float* target, int n, float* dy, float* loss_sum) {
    __shared__ float part[256];
    const float inv = 1.0f / float(n);
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float d = pred[i] - target[i];
        dy[i] = 2.0f * d * inv;
        s += d * d;
    }
    part[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (int(threadIdx.x) < o) part[threadIdx.x] += part[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0 && loss_sum) *loss_sum += part[0] * inv;
}

int blocks(size_t n) { return int((n + 255) / 256); }

}  // namespace

void images_to_input(const uint8_t* in, int batch, int cams, int H, int W, const int* shifts, int pad, float* out,
                     cudaStream_t stream) {
    const size_t total = size_t(batch) * cams * 3 * H * W;
    if (total == 0) return;
    images_kernel<<<blocks(total), 256, 0, stream>>>(in, batch, cams, H, W, shifts, pad, out);
    KERNEL_CHECK();
}

void copy_columns(const float* src, int src_ld, int src_col, float* dst, int dst_ld, int dst_col, int rows, int cols,
                  cudaStream_t stream) {
    const size_t total = size_t(rows) * cols;
    if (total == 0) return;
    copy_columns_kernel<<<blocks(total), 256, 0, stream>>>(src, src_ld, src_col, dst, dst_ld, dst_col, rows, cols);
    KERNEL_CHECK();
}

void mse_loss(const float* pred, const float* target, int rows, int cols, float* dy, float* loss_sum,
              cudaStream_t stream) {
    const int n = rows * cols;
    if (n == 0) return;
    mse_kernel<<<1, 256, 0, stream>>>(pred, target, n, dy, loss_sum);
    KERNEL_CHECK();
}

}  // namespace bc
}  // namespace temper
