#pragma once
// Small helpers for Vulkan memory. No allocator library yet: we only have a
// handful of buffers and images. Revisit (VMA) if the count grows.
#include <cstdint>

#include <vulkan/vulkan.h>

namespace flux {

class Context;

// Index of a memory type that allows 'type_bits' and has all 'props'. Throws if none.
uint32_t find_memory_type(const Context& ctx, uint32_t type_bits, VkMemoryPropertyFlags props);

// A buffer in host-visible, host-coherent memory, mapped for its whole life.
// The CPU writes with memcpy and the GPU sees it without a flush.
struct HostBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
    bool coherent = true;  // false: call invalidate() before the CPU reads what the GPU wrote
};

// for_reading = false: memory the CPU writes and the GPU reads (often write-combined:
// fast to write, very slow for the CPU to read). for_reading = true: memory the GPU
// writes and the CPU reads (readback): HOST_CACHED, so CPU reads are normal speed.
HostBuffer create_host_buffer(const Context& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                              bool for_reading = false);
// Makes GPU writes visible to the CPU for a non-coherent buffer (no-op if coherent).
void invalidate(const Context& ctx, const HostBuffer& b);
void destroy(const Context& ctx, HostBuffer& b);

// A 2D image in device-local memory, with one view. Used for depth buffers,
// multisampled color targets and the shadow map.
struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

Image create_image(const Context& ctx, uint32_t width, uint32_t height, VkFormat format,
                   VkImageUsageFlags usage, VkImageAspectFlags aspect,
                   VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);
void destroy(const Context& ctx, Image& img);

// First depth format the device supports as a depth attachment.
VkFormat pick_depth_format(const Context& ctx);

}  // namespace flux
