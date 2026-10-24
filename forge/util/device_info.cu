#include "forge/util/device_info.h"

#include <cstdio>
#include <cuda_runtime.h>

namespace forge {

int cuda_device_count() {
    int n = 0;
    // No CUDA_CHECK here: "no device" or "no driver" is a normal answer, not a crash.
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();  // clear the sticky error
        return 0;
    }
    return n;
}

std::string cuda_device_description(int device) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, device) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    char buf[512];  // name is up to 256 chars, plus the rest
    std::snprintf(buf, sizeof(buf), "%s (sm_%d%d, %.1f GB)", p.name, p.major, p.minor,
                  double(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

bool built_with_cuda() { return true; }

}  // namespace forge
