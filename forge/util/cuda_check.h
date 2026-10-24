#pragma once
// CUDA_CHECK wraps every CUDA runtime call. On error it prints file and line
// and aborts. KERNEL_CHECK goes right after a kernel launch (doc Appendix A).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                         cudaGetErrorName(err_), __FILE__, __LINE__,            \
                         cudaGetErrorString(err_));                             \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

#define KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())
