#pragma once
// RAII owner of one GPU allocation (doc Appendix A: no loose cudaMalloc/cudaFree).
#include <cstddef>
#include <utility>
#include <vector>
#include "forge/util/cuda_check.h"

namespace forge {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { resize(count); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr;
        o.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = std::exchange(o.ptr_, nullptr);
            count_ = std::exchange(o.count_, 0);
        }
        return *this;
    }

    // Drops old content.
    void resize(size_t count) {
        release();
        if (count > 0) CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        count_ = count;
    }

    void upload(const std::vector<T>& host) {
        if (host.size() != count_) resize(host.size());
        if (count_ > 0)
            CUDA_CHECK(cudaMemcpy(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    std::vector<T> download() const {
        std::vector<T> host(count_);
        if (count_ > 0)
            CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return count_; }

private:
    void release() {
        if (ptr_) cudaFree(ptr_);  // no CUDA_CHECK: never abort inside a destructor
        ptr_ = nullptr;
        count_ = 0;
    }
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

}  // namespace forge
