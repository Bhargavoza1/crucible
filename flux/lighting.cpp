#include "flux/lighting.h"

#include <cmath>
#include <cstring>

#include "flux/context.h"
#include "flux/vk_check.h"

namespace flux {

namespace {

// Puts a 1 x 1 depth image cleared to 1 (far: "lit") into SHADER_READ_ONLY layout.
void clear_dummy_map(const Context& ctx, VkImage image) {
    VkCommandPool pool;
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = ctx.queue_family;
    VK_CHECK(vkCreateCommandPool(ctx.device, &pci, nullptr, &pool));
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &ai, &cmd));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    VkClearDepthStencilValue one{1.0f, 0};
    vkCmdClearDepthStencilImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &one, 1, &b.subresourceRange);
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(ctx.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(ctx.queue));
    vkDestroyCommandPool(ctx.device, pool, nullptr);
}

}  // namespace

Lighting::Lighting(const Context& ctx, int slots, uint32_t shadow_size, bool encode_srgb)
    : ctx_(ctx), encode_srgb_(encode_srgb), size_(shadow_size > 0 ? shadow_size : 1) {
    VkDevice dev = ctx.device;
    // A depth format we can draw into and sample with linear filtering: D32 float
    // (NVIDIA, AMD, lavapipe), else D16 (always a depth attachment; sampled almost everywhere).
    VkFormat fmt = VK_FORMAT_D16_UNORM;
    {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(ctx.physical, VK_FORMAT_D32_SFLOAT, &fp);
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                          VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        if ((fp.optimalTilingFeatures & need) == need) fmt = VK_FORMAT_D32_SFLOAT;
    }
    map_ = create_image(ctx, size_, size_, fmt, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;  // with compare: 2 x 2 filtered shadow lookups
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sci.compareEnable = VK_TRUE;
    sci.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    sci.maxLod = 0.0f;
    VK_CHECK(vkCreateSampler(dev, &sci, nullptr, &sampler_));

    if (shadow_size > 0) {
        VkAttachmentDescription att{};
        att.format = fmt;
        att.samples = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.pDepthStencilAttachment = &ref;
        // Two frames in flight share the one map: the previous frame's reads must end
        // before this pass writes; this pass's writes must end before the main pass reads.
        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        VkRenderPassCreateInfo rci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rci.attachmentCount = 1;
        rci.pAttachments = &att;
        rci.subpassCount = 1;
        rci.pSubpasses = &sub;
        rci.dependencyCount = 2;
        rci.pDependencies = deps;
        VK_CHECK(vkCreateRenderPass(dev, &rci, nullptr, &shadow_pass_));
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = shadow_pass_;
        fci.attachmentCount = 1;
        fci.pAttachments = &map_.view;
        fci.width = fci.height = size_;
        fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(dev, &fci, nullptr, &framebuffer_));
    }
    // Before the first shadow pass (and always without one) the map must be readable.
    clear_dummy_map(ctx, map_.image);

    VkDescriptorSetLayoutBinding b[2] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 2;
    dl.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dl, nullptr, &set_layout_));
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uint32_t(slots)},
                                  {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, uint32_t(slots)}};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = uint32_t(slots);
    pci.poolSizeCount = 2;
    pci.pPoolSizes = ps;
    VK_CHECK(vkCreateDescriptorPool(dev, &pci, nullptr, &pool_));
    sets_.resize(size_t(slots));
    std::vector<VkDescriptorSetLayout> layouts(size_t(slots), set_layout_);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = uint32_t(slots);
    ai.pSetLayouts = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(dev, &ai, sets_.data()));
    ubos_.resize(size_t(slots));
    for (int s = 0; s < slots; ++s) {
        ubos_[size_t(s)] = create_host_buffer(ctx, sizeof(LightingUbo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        VkDescriptorBufferInfo bi{ubos_[size_t(s)].buffer, 0, sizeof(LightingUbo)};
        VkDescriptorImageInfo ii{sampler_, map_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w[2]{};
        for (int k = 0; k < 2; ++k) {
            w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[k].dstSet = sets_[size_t(s)];
            w[k].dstBinding = uint32_t(k);
            w[k].descriptorCount = 1;
        }
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[0].pBufferInfo = &bi;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[1].pImageInfo = &ii;
        vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);
        update(s, Mat4{}, Vec3{0, 0, 0}, false);
    }
}

Lighting::~Lighting() {
    VkDevice dev = ctx_.device;
    for (auto& u : ubos_) destroy(ctx_, u);
    if (pool_) vkDestroyDescriptorPool(dev, pool_, nullptr);
    if (set_layout_) vkDestroyDescriptorSetLayout(dev, set_layout_, nullptr);
    if (framebuffer_) vkDestroyFramebuffer(dev, framebuffer_, nullptr);
    if (shadow_pass_) vkDestroyRenderPass(dev, shadow_pass_, nullptr);
    if (sampler_) vkDestroySampler(dev, sampler_, nullptr);
    destroy(ctx_, map_);
}

Mat4 Lighting::sun_view_proj(Vec3 center, Vec3 half) const {
    const Vec3 d = forge::normalize(settings.sun_dir);
    const float r = forge::length(half);  // a sphere around the box: any sun direction fits
    const Vec3 up = std::fabs(d.z) > 0.99f ? Vec3{0, 1, 0} : Vec3{0, 0, 1};
    const Mat4 view = look_at(center + d * (2.0f * r), center, up);
    // orthographic, Vulkan clip space (y down, depth 0..1), as perspective()
    const float n = 0.5f * r, f = 3.5f * r;
    Mat4 p = mat4_zero();
    p.at(0, 0) = 1.0f / r;
    p.at(1, 1) = -1.0f / r;
    p.at(2, 2) = -1.0f / (f - n);
    p.at(2, 3) = -n / (f - n);
    p.at(3, 3) = 1.0f;
    return p * view;
}

void Lighting::update(int slot, const Mat4& light_vp, Vec3 eye, bool shadows) {
    const LightingSettings& s = settings;
    const Vec3 d = forge::normalize(s.sun_dir);
    LightingUbo u{};
    u.light_vp = light_vp;
    const float sd[4] = {d.x, d.y, d.z, s.sun_intensity};
    const float sc[4] = {s.sun_color.x, s.sun_color.y, s.sun_color.z, 1.0f / float(size_)};
    const float e[4] = {eye.x, eye.y, eye.z, shadows && has_shadows() ? 1.0f : 0.0f};
    const float sk[4] = {s.sky.x, s.sky.y, s.sky.z, s.exposure};
    const float gr[4] = {s.ground.x, s.ground.y, s.ground.z, encode_srgb_ ? 1.0f : 0.0f};
    std::memcpy(u.sun_dir, sd, sizeof(sd));
    std::memcpy(u.sun_color, sc, sizeof(sc));
    std::memcpy(u.eye, e, sizeof(e));
    std::memcpy(u.sky, sk, sizeof(sk));
    std::memcpy(u.ground, gr, sizeof(gr));
    std::memcpy(ubos_[size_t(slot)].mapped, &u, sizeof(u));
}

void Lighting::begin_shadow(VkCommandBuffer cmd) const {
    VkClearValue clear{};
    clear.depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = shadow_pass_;
    rp.framebuffer = framebuffer_;
    rp.renderArea = {{0, 0}, {size_, size_}};
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport vp{0.0f, 0.0f, float(size_), float(size_), 0.0f, 1.0f};
    const VkRect2D sc{{0, 0}, {size_, size_}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

void Lighting::end_shadow(VkCommandBuffer cmd) const { vkCmdEndRenderPass(cmd); }

}  // namespace flux
