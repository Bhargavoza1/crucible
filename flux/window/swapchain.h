#pragma once
// Swapchain and its image views. Recreated when the window size changes.
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

namespace flux {

class Context;

class Swapchain {
public:
    // vsync = false: MAILBOX if the driver has it, else IMMEDIATE, else FIFO.
    Swapchain(const Context& ctx, uint32_t width, uint32_t height, bool vsync = true);
    ~Swapchain();
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    // Destroys and rebuilds with the new size. Caller must wait for the GPU first.
    void recreate(uint32_t width, uint32_t height);

    VkSwapchainKHR handle = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    std::vector<VkImage> images;
    std::vector<VkImageView> views;

private:
    void create(uint32_t width, uint32_t height, VkSwapchainKHR old);
    void destroy_views();
    VkPresentModeKHR pick_present_mode() const;
    const Context& ctx_;
    bool vsync_ = true;
};

}  // namespace flux
