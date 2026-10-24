#include "flux/renderer.h"

#include <stdexcept>

#include "flux/context.h"
#include "flux/pipeline/triangle_pipeline.h"
#include "flux/vk_check.h"
#include "flux/window/swapchain.h"
#include "flux/window/window.h"

namespace flux {

Renderer::Renderer(Window& window, const RendererOptions& opt) : window_(window), opt_(opt) {
    ContextOptions co;
    co.validation = opt.validation;
    co.instance_extensions = Window::required_instance_extensions();
    ctx_ = std::make_unique<Context>(co, [&](VkInstance inst) { return window_.create_surface(inst); });

    int w = 0, h = 0;
    window_.framebuffer_size(w, h);
    swapchain_ = std::make_unique<Swapchain>(*ctx_, static_cast<uint32_t>(w), static_cast<uint32_t>(h));
    create_render_pass();
    pipeline_ = std::make_unique<TrianglePipeline>(ctx_->device, render_pass_, opt_.shader_dir);
    create_framebuffers();
    create_commands_and_sync();
}

Renderer::~Renderer() {
    if (!ctx_) return;
    wait_idle();
    VkDevice d = ctx_->device;
    for (int i = 0; i < kFramesInFlight; ++i) {
        vkDestroySemaphore(d, image_ready_[i], nullptr);
        vkDestroyFence(d, in_flight_[i], nullptr);
    }
    for (VkSemaphore s : render_done_) vkDestroySemaphore(d, s, nullptr);
    vkDestroyCommandPool(d, pool_, nullptr);
    destroy_framebuffers();
    pipeline_.reset();
    vkDestroyRenderPass(d, render_pass_, nullptr);
    swapchain_.reset();
    ctx_.reset();
}

void Renderer::wait_idle() { vkDeviceWaitIdle(ctx_->device); }

void Renderer::create_render_pass() {
    VkAttachmentDescription color{};
    color.format = swapchain_->format;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;

    // Wait for the acquired image before writing color to it.
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = 1;
    ci.pAttachments = &color;
    ci.subpassCount = 1;
    ci.pSubpasses = &sub;
    ci.dependencyCount = 1;
    ci.pDependencies = &dep;
    VK_CHECK(vkCreateRenderPass(ctx_->device, &ci, nullptr, &render_pass_));
}

void Renderer::create_framebuffers() {
    framebuffers_.resize(swapchain_->views.size());
    for (size_t i = 0; i < framebuffers_.size(); ++i) {
        VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass = render_pass_;
        ci.attachmentCount = 1;
        ci.pAttachments = &swapchain_->views[i];
        ci.width = swapchain_->extent.width;
        ci.height = swapchain_->extent.height;
        ci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(ctx_->device, &ci, nullptr, &framebuffers_[i]));
    }
}

void Renderer::destroy_framebuffers() {
    for (VkFramebuffer f : framebuffers_) vkDestroyFramebuffer(ctx_->device, f, nullptr);
    framebuffers_.clear();
}

void Renderer::create_commands_and_sync() {
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = ctx_->queue_family;
    VK_CHECK(vkCreateCommandPool(ctx_->device, &pci, nullptr, &pool_));

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = kFramesInFlight;
    VK_CHECK(vkAllocateCommandBuffers(ctx_->device, &ai, cmd_));

    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // first wait returns at once
    for (int i = 0; i < kFramesInFlight; ++i) {
        VK_CHECK(vkCreateSemaphore(ctx_->device, &si, nullptr, &image_ready_[i]));
        VK_CHECK(vkCreateFence(ctx_->device, &fi, nullptr, &in_flight_[i]));
    }
    render_done_.resize(swapchain_->images.size());
    for (VkSemaphore& s : render_done_) VK_CHECK(vkCreateSemaphore(ctx_->device, &si, nullptr, &s));
}

bool Renderer::rebuild_swapchain() {
    int w = 0, h = 0;
    window_.framebuffer_size(w, h);
    if (w == 0 || h == 0) return false;  // minimized: try again next frame
    wait_idle();
    destroy_framebuffers();
    const VkFormat old_format = swapchain_->format;
    swapchain_->recreate(static_cast<uint32_t>(w), static_cast<uint32_t>(h));
    if (swapchain_->format != old_format)
        throw std::runtime_error("flux: swapchain format changed; render pass rebuild not supported yet");
    create_framebuffers();
    // Image count can change: rebuild the per-image semaphores.
    for (VkSemaphore s : render_done_) vkDestroySemaphore(ctx_->device, s, nullptr);
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    render_done_.resize(swapchain_->images.size());
    for (VkSemaphore& s : render_done_) VK_CHECK(vkCreateSemaphore(ctx_->device, &si, nullptr, &s));
    return true;
}

void Renderer::record(VkCommandBuffer cmd, uint32_t image_index) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    VkClearValue clear{};
    clear.color = {{opt_.clear_color[0], opt_.clear_color[1], opt_.clear_color[2], 1.0f}};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = render_pass_;
    rp.framebuffer = framebuffers_[image_index];
    rp.renderArea = {{0, 0}, swapchain_->extent};
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->pipeline);
    VkViewport vp{0.0f, 0.0f, float(swapchain_->extent.width), float(swapchain_->extent.height), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, swapchain_->extent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));
}

bool Renderer::draw_frame() {
    VkDevice d = ctx_->device;
    VK_CHECK(vkWaitForFences(d, 1, &in_flight_[frame_], VK_TRUE, UINT64_MAX));

    uint32_t image = 0;
    VkResult r = vkAcquireNextImageKHR(d, swapchain_->handle, UINT64_MAX, image_ready_[frame_],
                                       VK_NULL_HANDLE, &image);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        rebuild_swapchain();
        return false;
    }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) VK_CHECK(r);

    // Reset the fence only when we know we will submit, or the next wait hangs.
    VK_CHECK(vkResetFences(d, 1, &in_flight_[frame_]));
    VK_CHECK(vkResetCommandBuffer(cmd_[frame_], 0));
    record(cmd_[frame_], image);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &image_ready_[frame_];
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_[frame_];
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &render_done_[image];
    VK_CHECK(vkQueueSubmit(ctx_->queue, 1, &si, in_flight_[frame_]));

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &render_done_[image];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_->handle;
    pi.pImageIndices = &image;
    r = vkQueuePresentKHR(ctx_->queue, &pi);

    frame_ = (frame_ + 1) % kFramesInFlight;
    ++frames_drawn_;

    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || window_.take_resized()) {
        rebuild_swapchain();
    } else if (r != VK_SUCCESS) {
        VK_CHECK(r);
    }
    return true;
}

}  // namespace flux
