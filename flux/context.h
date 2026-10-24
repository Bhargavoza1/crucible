#pragma once
// Vulkan instance, physical device, logical device and queues.
// One Context per app. Later (Phase 3) it also checks that the Vulkan device
// is the same GPU as the CUDA device, by comparing device UUIDs (doc 4.2).
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

namespace flux {

struct ContextOptions {
    bool validation = true;  // use VK_LAYER_KHRONOS_validation if it is installed
    std::vector<const char*> instance_extensions;  // e.g. from Window
};

class Context {
public:
    // 'surface_for' is called right after the instance exists, to make the
    // window surface. The chosen queue family must be able to present to it.
    template <typename MakeSurface>
    Context(const ContextOptions& opt, MakeSurface&& surface_for) {
        create_instance(opt);
        surface = surface_for(instance);
        pick_physical_device();
        create_device();
    }
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = 0;  // graphics + present
    VkQueue queue = VK_NULL_HANDLE;
    uint8_t device_uuid[VK_UUID_SIZE] = {};
    bool validation_on = false;

private:
    void create_instance(const ContextOptions& opt);
    void pick_physical_device();
    void create_device();
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
};

}  // namespace flux
