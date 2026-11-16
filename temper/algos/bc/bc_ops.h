#pragma once
// GPU helpers for behavior cloning (Phase 9d). All asynchronous on 'stream'.
#include <cstdint>

#include <cuda_runtime.h>

namespace temper {
namespace bc {

// Camera images to network input, with DrQ random shifts:
//   in:  uint8 [batch][cams][H][W][3] (RGB, as in demo files)
//   out: float [cams][batch][3][H][W] in [0, 1] (each camera's block is the input
//        rows of its encoder, PyTorch's NCHW order)
//   shifts: int [batch][cams][2] (ox, oy in [0, 2 pad]) or null (no shift). With a
//   shift, pixel (y, x) = source (clamp(y + oy - pad), clamp(x + ox - pad)): the
//   same as replicate-padding by 'pad' and cropping at (oy, ox) (bc_train.py).
void images_to_input(const uint8_t* in, int batch, int cams, int H, int W, const int* shifts, int pad, float* out,
                     cudaStream_t stream);

// Copy a block of columns between row-major matrices:
//   dst[r][dst_col + c] = src[r][src_col + c],  r < rows, c < cols
// (concatenation in forward, splitting the gradient in backward).
void copy_columns(const float* src, int src_ld, int src_col, float* dst, int dst_ld, int dst_col, int rows, int cols,
                  cudaStream_t stream);

// Mean squared error over rows x cols (torch.nn.functional.mse_loss):
//   dy = 2 (pred - target) / (rows cols)
//   loss_sum (one float, device) += sum (pred - target)^2 / (rows cols)
// (adds, so a loop of steps can sum its losses on the GPU and read them once).
// Deterministic: one block, fixed order. loss_sum may be null.
void mse_loss(const float* pred, const float* target, int rows, int cols, float* dy, float* loss_sum,
              cudaStream_t stream);

}  // namespace bc
}  // namespace temper
