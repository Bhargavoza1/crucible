#pragma once
// Tiny layer interface (doc 5.3). No autograd: each layer has its own
// forward() and backward(). All calls are asynchronous on 'stream'.
//
//   forward(x, y, batch):   y = f(x). The layer keeps the pointer x (not a copy)
//                           for backward, so x must stay valid until then.
//   backward(dy, dx, batch): fills the parameter gradients (overwrites, no
//                           accumulation) and dx = dL/dx. dx may be null (the
//                           first layer of a network needs no dx).
#include <cuda_runtime.h>

#include <vector>

#include "temper/nn/tensor.h"

namespace temper {

class Layer {
public:
    virtual ~Layer() = default;
    virtual int in_dim() const = 0;
    virtual int out_dim() const = 0;
    virtual void forward(const float* x, float* y, int batch, cudaStream_t stream) = 0;
    virtual void backward(const float* dy, float* dx, int batch, cudaStream_t stream) = 0;
    // Parameter tensors and their gradients, in the same order. Empty for ELU.
    virtual void params(std::vector<Tensor*>& p, std::vector<Tensor*>& g) = 0;
};

}  // namespace temper
