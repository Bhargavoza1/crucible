#include "flux/window/swapchain.h"

#include <algorithm>

#include "flux/context.h"
#include "flux/vk_check.h"

namespace flux {

Swapchain::Swapchain(const Context& ctx, uint32_t width, uint32_t height, bool vsync)
    : ctx_(ctx), vsync_(vsync) {
    create(width, height, VK_NULL_HANDLE);
}

Swapchain::~Swapchain() {
    destroy_views();
    if (handle) vkDestroySwapchainKHR(ctx_.device, handle, nullptr);
}

void Swapchain::recreate(uint32_t width, uint32_t height) {
    destroy_views();
    VkSwapchainKHR old = handle;
    create(width, height, old);
    vkDestroySwapchainKHR(ctx_.device, old, nullptr);
}

void Swapchain::destroy_views() {
    for (VkImageView v : views) vkDestroyImageView(ctx_.device, v, nullptr);
    views.clear();
}

VkPresentModeKHR Swapchain::pick_present_mode() const {
    if (vsync_) return VK_PRESENT_MODE_FIFO_KHR;  // vsync, always supported
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx_.physical, ctx_.surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx_.physical, ctx_.surface, &n, modes.data());
    for (VkPresentModeKHR want : {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR})
        for (VkPresentModeKHR m : modes)
            if (m == want) return m;
    return VK_PRESENT_MODE_FIFO_KHR;
}

void Swapchain::create(uint32_t width, uint32_t height, VkSwapchainKHR old) {
    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx_.physical, ctx_.surface, &caps));

    // Format: prefer 8-bit BGRA sRGB, else take the first one offered.
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.physical, ctx_.surface, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.physical, ctx_.surface, &n, formats.data());
    VkSurfaceFormatKHR fmt = formats[0];
    for (const auto& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) fmt = f;
    format = fmt.format;

    // Size: the surface decides, unless it says "any" (0xFFFFFFFF).
    if (caps.currentExtent.width != UINT32_MAX) {
        extent = caps.currentExtent;
    } else {
        extent.width = std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    uint32_t count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) count = std::min(count, caps.maxImageCount);

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = ctx_.surface;
    ci.minImageCount = count;
    ci.imageFormat = fmt.format;
    ci.imageColorSpace = fmt.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    // TRANSFER_SRC lets us read back frames later (recording, doc 7.1).
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    readable = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;  // one queue does draw + present
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = pick_present_mode();
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = old;
    VK_CHECK(vkCreateSwapchainKHR(ctx_.device, &ci, nullptr, &handle));

    vkGetSwapchainImagesKHR(ctx_.device, handle, &n, nullptr);
    images.resize(n);
    vkGetSwapchainImagesKHR(ctx_.device, handle, &n, images.data());

    views.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = images[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(ctx_.device, &vi, nullptr, &views[i]));
    }
}

}  // namespace flux
