#pragma once
// A row-major float matrix in GPU memory (rows x cols). Vectors are 1 x n.
#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"

namespace temper {

struct Tensor {
    int rows = 0, cols = 0;
    forge::DeviceBuffer<float> buf;

    Tensor() = default;
    Tensor(int r, int c) { resize(r, c); }

    void resize(int r, int c) {
        rows = r;
        cols = c;
        buf.resize(size_t(r) * c);
    }
    float* data() { return buf.data(); }
    const float* data() const { return buf.data(); }
    size_t size() const { return buf.size(); }

    void upload(const std::vector<float>& h) {
        if (h.size() != size()) throw std::runtime_error("Tensor::upload: size mismatch");
        buf.upload(h);
    }
    std::vector<float> download() const { return buf.download(); }  // syncs
    void zero(cudaStream_t s = nullptr) {
        if (size()) CUDA_CHECK(cudaMemsetAsync(data(), 0, size() * sizeof(float), s));
    }
};

}  // namespace temper
