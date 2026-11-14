#include "flux/memory.h"

#include <stdexcept>

#include "flux/context.h"
#include "flux/vk_check.h"

namespace flux {

uint32_t find_memory_type(const Context& ctx, uint32_t type_bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(ctx.physical, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    throw std::runtime_error("flux: no suitable Vulkan memory type");
}

HostBuffer create_host_buffer(const Context& ctx, VkDeviceSize size, VkBufferUsageFlags usage) {
    HostBuffer b;
    b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(ctx.device, &bi, nullptr, &b.buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx.device, b.buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(
        ctx, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(ctx.device, &ai, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(ctx.device, b.buffer, b.memory, 0));
    VK_CHECK(vkMapMemory(ctx.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    return b;
}

void destroy(const Context& ctx, HostBuffer& b) {
    if (b.memory) {
        vkUnmapMemory(ctx.device, b.memory);
        vkFreeMemory(ctx.device, b.memory, nullptr);
    }
    if (b.buffer) vkDestroyBuffer(ctx.device, b.buffer, nullptr);
    b = HostBuffer{};
}

Image create_image(const Context& ctx, uint32_t width, uint32_t height, VkFormat format,
                   VkImageUsageFlags usage, VkImageAspectFlags aspect) {
    Image img;
    img.format = format;
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = format;
    ii.extent = {width, height, 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = usage;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(ctx.device, &ii, nullptr, &img.image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(ctx.device, img.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(ctx, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(ctx.device, &ai, nullptr, &img.memory));
    VK_CHECK(vkBindImageMemory(ctx.device, img.image, img.memory, 0));

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = {aspect, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(ctx.device, &vi, nullptr, &img.view));
    return img;
}

void destroy(const Context& ctx, Image& img) {
    if (img.view) vkDestroyImageView(ctx.device, img.view, nullptr);
    if (img.image) vkDestroyImage(ctx.device, img.image, nullptr);
    if (img.memory) vkFreeMemory(ctx.device, img.memory, nullptr);
    img = Image{};
}

VkFormat pick_depth_format(const Context& ctx) {
    const VkFormat candidates[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                                   VK_FORMAT_D24_UNORM_S8_UINT};
    for (VkFormat f : candidates) {
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(ctx.physical, f, &p);
        if (p.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) return f;
    }
    throw std::runtime_error("flux: no depth format supported");
}

}  // namespace flux
