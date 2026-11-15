// Used only when forge is built without a CUDA compiler (CPU-only machines, CI).
#include "forge/util/device_info.h"

namespace forge {

int cuda_device_count() { return 0; }
std::string cuda_device_description(int) { return {}; }
std::optional<std::array<uint8_t, 16>> cuda_device_uuid(int) { return std::nullopt; }
bool built_with_cuda() { return false; }

}  // namespace forge
