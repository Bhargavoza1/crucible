#pragma once
// Named ranges for the Nsight Systems timeline:
//
//   {
//       FORGE_NVTX_RANGE("physics");
//       integrate_gpu(...);
//   }   // range ends here
//
// The range covers the CPU time of the block (launching kernels). In nsys,
// kernels launched inside it are linked to it, so the timeline shows
// "physics" over the right kernels.
// Header-only NVTX3 ships with the CUDA Toolkit. Without it this is a no-op.

#if defined(CRUCIBLE_HAS_CUDA) && __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>

namespace forge {
struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};
}  // namespace forge

#define FORGE_NVTX_CONCAT2(a, b) a##b
#define FORGE_NVTX_CONCAT(a, b) FORGE_NVTX_CONCAT2(a, b)
#define FORGE_NVTX_RANGE(name) ::forge::NvtxRange FORGE_NVTX_CONCAT(nvtx_range_, __LINE__)(name)

#else
#define FORGE_NVTX_RANGE(name) ((void)0)
#endif
