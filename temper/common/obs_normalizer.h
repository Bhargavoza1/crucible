#pragma once
// Observation normalizer (doc 7.3, 11): running mean and variance per obs
// dimension, updated with every batch of raw observations during rollouts and
// saved in the checkpoint (play mode needs the same numbers).
//   z = clip((x - mean) / sqrt(var + 1e-8), -clip, clip)
#include <cuda_runtime.h>

#include "temper/nn/tensor.h"

namespace temper {

class ObsNormalizer {
public:
    ObsNormalizer(int dim, float clip);
    void update(const float* x, int n, cudaStream_t stream);                // x: n x dim, raw
    void normalize(const float* x, float* out, int n, cudaStream_t stream) const;
    int dim() const { return dim_; }
    float clip() const { return clip_; }

    Tensor mean, var;  // 1 x dim
    Tensor count;      // 1 x 1 (a float: exact up to 2^24 per batch merge, plenty here)

private:
    int dim_;
    float clip_;
};

}  // namespace temper
