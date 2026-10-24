#pragma once
// FORGE_HD marks a function that runs on both CPU and GPU.
// With nvcc it becomes __host__ __device__. With a plain C++ compiler it is just
// inline, so the same math headers build and run in CPU-only tests.

#if defined(__CUDACC__)
#define FORGE_HD __host__ __device__ __forceinline__
#else
#define FORGE_HD inline
#endif

#include <math.h>  // sqrtf, sinf, cosf: plain C names work on host and device
