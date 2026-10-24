#pragma once
// Small host-side helpers to ask about CUDA devices.
// These always link: in a build without CUDA they report 0 devices.
#include <string>

namespace forge {

// Number of usable CUDA devices. 0 when there is no GPU, no driver, or no CUDA build.
int cuda_device_count();

// One-line description, e.g. "NVIDIA GeForce RTX 4070 Ti (sm_89, 12.0 GB)".
// Empty string if the device does not exist.
std::string cuda_device_description(int device);

// True if forge was compiled with CUDA.
bool built_with_cuda();

}  // namespace forge
