#include "flux/batch_camera.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "flux/context.h"
#include "flux/pipeline/particle_pipeline.h"
#include "flux/vk_check.h"

namespace flux {

BatchCamera::BatchCamera(const BatchCameraOptions& opt) : opt_(opt) {
    if (opt.num_envs <= 0 || opt.num_cams <= 0 || opt.width <= 0 || opt.height <= 0)
        throw std::invalid_argument("BatchCamera: envs, cameras, width and height must be > 0");
    if (opt.points_per_env < 0 || opt.quads_per_env < 0) throw std::invalid_argument("BatchCamera: negative counts");
    ContextOptions co;
    co.validation = opt.validation;
    ctx_ = std::make_unique<Context>(co);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx_->physical, &props);
    device_name_ = props.deviceName;

    // Tile layout: about square, within the device's largest image size.
    const int max_dim = static_cast<int>(props.limits.maxImageDimension2D);
    tiles_ = opt.num_envs * opt.num_cams;
    tiles_per_row_ = std::max(1, int(std::ceil(std::sqrt(double(tiles_)))));
    tiles_per_row_ = std::min(tiles_per_row_, std::max(1, max_dim / opt.width));
    tile_rows_ = (tiles_ + tiles_per_row_ - 1) / tiles_per_row_;
    if (tile_rows_ * opt.height > max_dim || tiles_per_row_ * opt.width > max_dim)
        throw std::runtime_error("BatchCamera: too many images for one atlas (" + std::to_string(tiles_) + " of " +
                                 std::to_string(opt.width) + "x" + std::to_string(opt.height) + ")");

    depth_format_ = pick_depth_format(*ctx_);
    create_render_pass();
    create_targets();
    points_pipeline_ = std::make_unique<ParticlePipeline>(ctx_->device, render_pass_, opt.shader_dir);
    floor_pipeline_ = std::make_unique<QuadPipeline>(ctx_->device, render_pass_, opt.shader_dir, false);
    solid_pipeline_ = std::make_unique<QuadPipeline>(ctx_->device, render_pass_, opt.shader_dir, true);

    // Point buffer: 5 regions (px, py, pz, radius, env_id) like the viewer. env_id stays 0:
    // the env grid offset is off (spacing 0), every tile shows its env at its true place.
    const size_t npts = std::max<size_t>(1, size_t(opt.num_envs) * opt.points_per_env);
    point_buf_ = create_host_buffer(*ctx_, npts * 4 * ParticlePipeline::kNumBindings, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    std::memset(point_buf_.mapped, 0, npts * 4 * ParticlePipeline::kNumBindings);
    const size_t nq = std::max<size_t>(1, size_t(opt.num_envs) * opt.quads_per_env);
    quad_buf_ = create_host_buffer(*ctx_, nq * sizeof(Quad), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    readback_ = create_host_buffer(*ctx_, VkDeviceSize(atlas_width()) * atlas_height() * 4,
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = ctx_->queue_family;
    VK_CHECK(vkCreateCommandPool(ctx_->device, &pci, nullptr, &pool_));
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(ctx_->device, &ai, &cmd_));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(ctx_->device, &fi, nullptr, &fence_));
}

BatchCamera::~BatchCamera() {
    if (!ctx_) return;
    VkDevice d = ctx_->device;
    vkDeviceWaitIdle(d);
    vkDestroyFence(d, fence_, nullptr);
    vkDestroyCommandPool(d, pool_, nullptr);
    destroy(*ctx_, point_buf_);
    destroy(*ctx_, quad_buf_);
    destroy(*ctx_, static_buf_);
    destroy(*ctx_, readback_);
    points_pipeline_.reset();
    floor_pipeline_.reset();
    solid_pipeline_.reset();
    vkDestroyFramebuffer(d, framebuffer_, nullptr);
    destroy(*ctx_, color_);
    destroy(*ctx_, depth_);
    vkDestroyRenderPass(d, render_pass_, nullptr);
    ctx_.reset();
}

void BatchCamera::create_render_pass() {
    VkAttachmentDescription att[2]{};
    att[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;  // copied to the host after the pass
    att[1].format = depth_format_;
    att[1].samples = VK_SAMPLE_COUNT_1_BIT;
    att[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depth_ref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &color_ref;
    sub.pDepthStencilAttachment = &depth_ref;
    VkSubpassDependency dep[2]{};
    // the previous frame's copy (transfer read) and depth writes before this frame writes
    dep[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dep[0].dstSubpass = 0;
    dep[0].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dep[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dep[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    // color writes done before the copy reads the image
    dep[1].srcSubpass = 0;
    dep[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dep[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dep[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = 2;
    ci.pAttachments = att;
    ci.subpassCount = 1;
    ci.pSubpasses = &sub;
    ci.dependencyCount = 2;
    ci.pDependencies = dep;
    VK_CHECK(vkCreateRenderPass(ctx_->device, &ci, nullptr, &render_pass_));
}

void BatchCamera::create_targets() {
    const uint32_t w = static_cast<uint32_t>(atlas_width()), h = static_cast<uint32_t>(atlas_height());
    color_ = create_image(*ctx_, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    depth_ = create_image(*ctx_, w, h, depth_format_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    VkImageView views[2] = {color_.view, depth_.view};
    VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    ci.renderPass = render_pass_;
    ci.attachmentCount = 2;
    ci.pAttachments = views;
    ci.width = w;
    ci.height = h;
    ci.layers = 1;
    VK_CHECK(vkCreateFramebuffer(ctx_->device, &ci, nullptr, &framebuffer_));
}

void BatchCamera::set_static_quads(const std::vector<Quad>& quads) {
    vkDeviceWaitIdle(ctx_->device);
    destroy(*ctx_, static_buf_);
    static_count_ = static_cast<uint32_t>(quads.size());
    if (quads.empty()) return;
    static_buf_ = create_host_buffer(*ctx_, quads.size() * sizeof(Quad), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    std::copy(quads.begin(), quads.end(), static_cast<Quad*>(static_buf_.mapped));
}

Mat4 BatchCamera::view_proj(const CameraView& v) const {
    return perspective(v.fovy, float(opt_.width) / float(opt_.height), opt_.z_near, opt_.z_far) *
           look_at(v.eye, v.target, v.up);
}

bool BatchCamera::project(const CameraView& view, Vec3 p, float& x, float& y) const {
    const Mat4 m = view_proj(view);
    const float in[4] = {p.x, p.y, p.z, 1.0f};
    float c[4] = {0, 0, 0, 0};
    for (int r = 0; r < 4; ++r)
        for (int k = 0; k < 4; ++k) c[r] += m.at(r, k) * in[k];
    if (c[3] <= 0.0f) return false;
    x = (0.5f * c[0] / c[3] + 0.5f) * float(opt_.width);
    y = (0.5f * c[1] / c[3] + 0.5f) * float(opt_.height);  // Vulkan clip y points down
    return true;
}

void BatchCamera::render(const float* px, const float* py, const float* pz, const float* radius, const Quad* quads,
                         const CameraView* views) {
    const int E = opt_.num_envs, P = opt_.points_per_env, Q = opt_.quads_per_env;
    const size_t npts = size_t(E) * P, region = std::max<size_t>(1, npts) * 4;
    char* base = static_cast<char*>(point_buf_.mapped);
    if (npts) {
        std::memcpy(base + 0 * region, px, npts * 4);
        std::memcpy(base + 1 * region, py, npts * 4);
        std::memcpy(base + 2 * region, pz, npts * 4);
        std::memcpy(base + 3 * region, radius, npts * 4);
    }
    if (Q > 0) std::memcpy(quad_buf_.mapped, quads, size_t(E) * Q * sizeof(Quad));

    VK_CHECK(vkResetCommandBuffer(cmd_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));
    VkClearValue clear[2]{};
    clear[0].color = {{opt_.clear_color[0], opt_.clear_color[1], opt_.clear_color[2], 1.0f}};
    clear[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = render_pass_;
    rp.framebuffer = framebuffer_;
    rp.renderArea = {{0, 0}, {uint32_t(atlas_width()), uint32_t(atlas_height())}};
    rp.clearValueCount = 2;
    rp.pClearValues = clear;
    vkCmdBeginRenderPass(cmd_, &rp, VK_SUBPASS_CONTENTS_INLINE);

    VkBuffer pbufs[ParticlePipeline::kNumBindings];
    VkDeviceSize poffs[ParticlePipeline::kNumBindings];
    for (uint32_t i = 0; i < ParticlePipeline::kNumBindings; ++i) {
        pbufs[i] = point_buf_.buffer;
        poffs[i] = VkDeviceSize(i) * region;
    }
    const Vec3 up_world{0, 0, 1};
    for (int t = 0; t < tiles_; ++t) {
        const int e = t / opt_.num_cams;
        const CameraView& cv = views[t];
        const int tx = (t % tiles_per_row_) * opt_.width, ty = (t / tiles_per_row_) * opt_.height;
        const VkViewport vp{float(tx), float(ty), float(opt_.width), float(opt_.height), 0.0f, 1.0f};
        const VkRect2D sc{{tx, ty}, {uint32_t(opt_.width), uint32_t(opt_.height)}};
        vkCmdSetViewport(cmd_, 0, 1, &vp);
        vkCmdSetScissor(cmd_, 0, 1, &sc);
        const Mat4 m = view_proj(cv);
        VkDeviceSize zero = 0;
        if (static_count_ > 0) {
            vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, floor_pipeline_->pipeline);
            vkCmdPushConstants(cmd_, floor_pipeline_->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4), &m);
            vkCmdBindVertexBuffers(cmd_, 0, 1, &static_buf_.buffer, &zero);
            vkCmdDraw(cmd_, 6, static_count_, 0, 0);
        }
        if (Q > 0) {
            vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, solid_pipeline_->pipeline);
            vkCmdPushConstants(cmd_, solid_pipeline_->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4), &m);
            vkCmdBindVertexBuffers(cmd_, 0, 1, &quad_buf_.buffer, &zero);
            vkCmdDraw(cmd_, 6, uint32_t(Q), 0, uint32_t(e * Q));
        }
        if (P > 0) {
            ParticlePushConstants pc{};
            pc.view_proj = m;
            const Vec3 fwd = forge::normalize(cv.target - cv.eye);
            Vec3 right = forge::cross(fwd, cv.up);
            if (forge::length(right) < 1e-6f) right = forge::cross(fwd, up_world);
            right = forge::normalize(right);
            const Vec3 cup = forge::cross(right, fwd);
            const float r4[4] = {right.x, right.y, right.z, 0.0f}, u4[4] = {cup.x, cup.y, cup.z, 0.0f};
            const float g4[4] = {0.0f, 1.0f, 1.0f, std::max(opt_.color_height, 1e-3f)};  // no env grid offset
            std::memcpy(pc.cam_right, r4, sizeof(r4));
            std::memcpy(pc.cam_up, u4, sizeof(u4));
            std::memcpy(pc.grid, g4, sizeof(g4));
            vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, points_pipeline_->pipeline);
            vkCmdPushConstants(cmd_, points_pipeline_->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            vkCmdBindVertexBuffers(cmd_, 0, ParticlePipeline::kNumBindings, pbufs, poffs);
            vkCmdDraw(cmd_, 6, uint32_t(P), 0, uint32_t(e * P));
        }
    }
    vkCmdEndRenderPass(cmd_);

    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {uint32_t(atlas_width()), uint32_t(atlas_height()), 1};
    vkCmdCopyImageToBuffer(cmd_, color_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_.buffer, 1, &copy);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0,
                         nullptr);
    VK_CHECK(vkEndCommandBuffer(cmd_));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    VK_CHECK(vkResetFences(ctx_->device, 1, &fence_));
    VK_CHECK(vkQueueSubmit(ctx_->queue, 1, &si, fence_));
    VK_CHECK(vkWaitForFences(ctx_->device, 1, &fence_, VK_TRUE, UINT64_MAX));
}

const uint8_t* BatchCamera::pixels() const { return static_cast<const uint8_t*>(readback_.mapped); }

void BatchCamera::copy_rgb(int env, int cam, uint8_t* out) const {
    if (env < 0 || env >= opt_.num_envs || cam < 0 || cam >= opt_.num_cams)
        throw std::out_of_range("BatchCamera::copy_rgb: no such image");
    const int t = env * opt_.num_cams + cam;
    const int tx = (t % tiles_per_row_) * opt_.width, ty = (t / tiles_per_row_) * opt_.height;
    const uint8_t* src = pixels();
    const size_t stride = size_t(atlas_width()) * 4;
    for (int y = 0; y < opt_.height; ++y) {
        const uint8_t* row = src + size_t(ty + y) * stride + size_t(tx) * 4;
        for (int x = 0; x < opt_.width; ++x) {
            uint8_t* o = out + (size_t(y) * opt_.width + x) * 3;
            o[0] = row[4 * x], o[1] = row[4 * x + 1], o[2] = row[4 * x + 2];
        }
    }
}

}  // namespace flux
