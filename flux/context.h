#pragma once
// Vulkan instance, physical device, logical device and queue.
// One Context per app.
//
// Two ways to make one:
//   Context(opt, make_surface)  window mode: the queue must present to the surface
//   Context(opt)                no window (tests, later offscreen rendering, doc 4.3)
//
// With opt.interop = true it also turns on what CUDA-Vulkan sharing needs
// (doc 4.2): external memory fd, external semaphore fd, timeline semaphores.
// With opt.required_uuid set, it only accepts the GPU with that UUID, so
// Vulkan and CUDA are sure to use the same card.
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace flux {

using DeviceUuid = std::array<uint8_t, VK_UUID_SIZE>;

struct ContextOptions {
    bool validation = true;  // use VK_LAYER_KHRONOS_validation if it is installed
    std::vector<const char*> instance_extensions;  // e.g. from Window
    bool interop = false;                          // require CUDA sharing support
    std::optional<DeviceUuid> required_uuid;       // e.g. the CUDA device UUID
};

class Context {
public:
    template <typename MakeSurface>
    Context(const ContextOptions& opt, MakeSurface&& surface_for) {
        create_instance(opt);
        surface = surface_for(instance);
        pick_physical_device(opt);
        create_device(opt);
    }
    explicit Context(const ContextOptions& opt) {
        create_instance(opt);
        pick_physical_device(opt);
        create_device(opt);
    }
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;  // null without a window
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = 0;  // graphics (+ present with a window)
    VkQueue queue = VK_NULL_HANDLE;
    DeviceUuid device_uuid{};
    bool validation_on = false;
    bool interop_on = false;

    // Loaded only when interop_on (extension functions are not exported by the loader).
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = nullptr;
    PFN_vkGetSemaphoreFdKHR vkGetSemaphoreFdKHR = nullptr;

private:
    void create_instance(const ContextOptions& opt);
    void pick_physical_device(const ContextOptions& opt);
    void create_device(const ContextOptions& opt);
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
};

// Hex string of a UUID, for logs.
std::string uuid_hex(const DeviceUuid& u);

}  // namespace flux
