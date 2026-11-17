#pragma once
// GPU helpers for training the student (Phase 10). Asynchronous on 'stream'.
#include <cuda_runtime.h>

namespace temper {
namespace student {

// Mean squared error over the rows with mask[row] = 1 (mask null: all rows):
//   dy = 2 mask (pred - target) / (n cols),  n = number of masked rows (read from
//   'count', device: one float, the caller's sum of the mask)
//   loss_sum (device, one float) += sum mask (pred - target)^2 / (n cols)
// Deterministic (one block, fixed order). loss_sum may be null.
void masked_mse(const float* pred, const float* target, const float* mask, const float* count, int rows, int cols,
                float* dy, float* loss_sum, cudaStream_t stream);

}  // namespace student
}  // namespace temper
