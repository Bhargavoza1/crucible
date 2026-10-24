#pragma once
// VK_CHECK wraps every Vulkan call that returns VkResult (doc Appendix A).
#include <cstdio>
#include <cstdlib>
#include <vulkan/vulkan.h>

namespace flux {
const char* vk_result_name(VkResult r);
}

#define VK_CHECK(call)                                                          \
    do {                                                                        \
        VkResult res_ = (call);                                                 \
        if (res_ != VK_SUCCESS) {                                               \
            std::fprintf(stderr, "Vulkan error %s at %s:%d: %s\n",              \
                         flux::vk_result_name(res_), __FILE__, __LINE__, #call); \
            std::abort();                                                       \
        }                                                                       \
    } while (0)
