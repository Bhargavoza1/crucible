#include "flux/renderer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "flux/context.h"
#include "flux/gizmo.h"
#include "flux/lighting.h"
#include "flux/pipeline/mesh_pipeline.h"
#include "flux/pipeline/particle_pipeline.h"
#include "flux/pipeline/sky_pipeline.h"
#include "flux/vk_check.h"
#include "flux/window/swapchain.h"
#include "flux/window/window.h"

namespace flux {

Renderer::Renderer(Window& window, const RendererOptions& opt) : window_(window), opt_(opt) {
    ContextOptions co;
    co.validation = opt.validation;
    co.instance_extensions = Window::required_instance_extensions();
    co.interop = opt.interop;
    co.required_uuid = opt.required_uuid;
    ctx_ = std::make_unique<Context>(co, [&](VkInstance inst) { return window_.create_surface(inst); });

    int w = 0, h = 0;
    window_.framebuffer_size(w, h);
    swapchain_ = std::make_unique<Swapchain>(*ctx_, static_cast<uint32_t>(w), static_cast<uint32_t>(h), opt.vsync);
    depth_format_ = pick_depth_format(*ctx_);
    {  // multisampling: the most samples <= opt.msaa that color and depth both support
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(ctx_->physical, &props);
        const VkSampleCountFlags ok =
            props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
        for (int n = 8; n >= 1; n /= 2)
            if (n <= opt_.msaa && (ok & VkSampleCountFlags(n))) {
                samples_ = VkSampleCountFlagBits(n);
                break;
            }
    }
    create_render_pass();
    lighting_ = std::make_unique<Lighting>(*ctx_, kFramesInFlight, opt_.shadows ? opt_.shadow_size : 0u, false);
    pipeline_ = std::make_unique<ParticlePipeline>(ctx_->device, render_pass_, opt_.shader_dir, samples_);
    quad_pipeline_ = std::make_unique<QuadPipeline>(ctx_->device, render_pass_, opt_.shader_dir, false, samples_);
    create_depth_and_framebuffers();
    create_commands_and_sync();
    solid_pipeline_ = std::make_unique<QuadPipeline>(ctx_->device, render_pass_, opt_.shader_dir, true, samples_);
    sky_ = std::make_unique<SkyPipeline>(ctx_->device, render_pass_, opt_.shader_dir, samples_);
    lit_floor_ = std::make_unique<QuadPipeline>(ctx_->device, render_pass_, opt_.shader_dir, false, samples_,
                                                lighting_->set_layout());
    lit_solid_ = std::make_unique<QuadPipeline>(ctx_->device, render_pass_, opt_.shader_dir, true, samples_,
                                                lighting_->set_layout());
    if (lighting_->has_shadows())
        shadow_quads_ = std::make_unique<QuadPipeline>(ctx_->device, lighting_->shadow_pass(), opt_.shader_dir, true,
                                                       VK_SAMPLE_COUNT_1_BIT, VK_NULL_HANDLE, true);
    if (opt_.corner_axes) {
        const std::vector<Quad> g = axes_gizmo_3d(1.0f);  // unit axes around the origin; drawn in its own viewport
        gizmo_quads_ = static_cast<uint32_t>(g.size());
        gizmo_buf_ = create_host_buffer(*ctx_, g.size() * sizeof(Quad), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        std::copy(g.begin(), g.end(), static_cast<Quad*>(gizmo_buf_.mapped));
    }
}

Renderer::~Renderer() {
    if (!ctx_) return;
    wait_idle();
    VkDevice d = ctx_->device;
    for (int i = 0; i < kFramesInFlight; ++i) {
        vkDestroySemaphore(d, image_ready_[i], nullptr);
        vkDestroyFence(d, in_flight_[i], nullptr);
        destroy(*ctx_, particle_buf_[i]);
    }
    for (VkSemaphore s : render_done_) vkDestroySemaphore(d, s, nullptr);
    vkDestroyCommandPool(d, pool_, nullptr);
    destroy_depth_and_framebuffers();
    destroy(*ctx_, quad_buf_);
    destroy(*ctx_, gizmo_buf_);
    for (int i = 0; i < kFramesInFlight; ++i) destroy(*ctx_, solid_buf_[i]);
    destroy(*ctx_, capture_buf_);
    mesh_.reset();
    shadow_quads_.reset();
    lit_solid_.reset();
    lit_floor_.reset();
    sky_.reset();
    lighting_.reset();
    solid_pipeline_.reset();
    quad_pipeline_.reset();
    pipeline_.reset();
    vkDestroyRenderPass(d, render_pass_, nullptr);
    swapchain_.reset();
    ctx_.reset();
}

void Renderer::wait_idle() { vkDeviceWaitIdle(ctx_->device); }

void Renderer::set_robot_mesh(const RobotMesh& mesh, int lod) {
    if (!mesh_)
        mesh_ = std::make_unique<MeshDrawer>(*ctx_, render_pass_, opt_.shader_dir, kFramesInFlight,
                                             lighting_->set_layout(), samples_, lighting_->shadow_pass());
    mesh_->set_mesh(mesh, lod);
}

void Renderer::set_scene(const SceneQuads& scene) {
    wait_idle();  // the old buffer may still be in use
    destroy(*ctx_, quad_buf_);
    quads_before_ = static_cast<uint32_t>(scene.before.size());
    quads_after_ = static_cast<uint32_t>(scene.after.size());
    const size_t n = scene.before.size() + scene.after.size();
    if (n == 0) return;
    quad_buf_ = create_host_buffer(*ctx_, n * sizeof(Quad), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    Quad* dst = static_cast<Quad*>(quad_buf_.mapped);
    std::copy(scene.before.begin(), scene.before.end(), dst);
    std::copy(scene.after.begin(), scene.after.end(), dst + scene.before.size());
}

void Renderer::create_render_pass() {
    // Attachments: 0 color, 1 depth, and with multisampling 2 = the swapchain image the
    // multisampled color is resolved into (then 0 is a transient multisampled image).
    const bool ms = samples_ != VK_SAMPLE_COUNT_1_BIT;
    VkAttachmentDescription att[3]{};
    att[0].format = swapchain_->format;
    att[0].samples = samples_;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp = ms ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout = ms ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    // 1: depth. Cleared each frame, not kept after the pass.
    att[1].format = depth_format_;
    att[1].samples = samples_;
    att[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    // 2: the swapchain image (resolve target)
    att[2].format = swapchain_->format;
    att[2].samples = VK_SAMPLE_COUNT_1_BIT;
    att[2].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[2].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depth_ref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkAttachmentReference resolve_ref{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &color_ref;
    sub.pResolveAttachments = ms ? &resolve_ref : nullptr;
    sub.pDepthStencilAttachment = &depth_ref;

    // Two frames in flight share ONE depth image (and one multisampled color image).
    // This dependency makes frame N+1 wait until frame N has finished its attachment
    // writes (write-after-write), and waits for the acquired swapchain image before
    // writing color.
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = ms ? 3 : 2;
    ci.pAttachments = att;
    ci.subpassCount = 1;
    ci.pSubpasses = &sub;
    ci.dependencyCount = 1;
    ci.pDependencies = &dep;
    VK_CHECK(vkCreateRenderPass(ctx_->device, &ci, nullptr, &render_pass_));
}

void Renderer::create_depth_and_framebuffers() {
    const bool ms = samples_ != VK_SAMPLE_COUNT_1_BIT;
    depth_ = create_image(*ctx_, swapchain_->extent.width, swapchain_->extent.height, depth_format_,
                          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, samples_);
    if (ms)
        color_msaa_ = create_image(*ctx_, swapchain_->extent.width, swapchain_->extent.height, swapchain_->format,
                                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                                   VK_IMAGE_ASPECT_COLOR_BIT, samples_);
    framebuffers_.resize(swapchain_->views.size());
    for (size_t i = 0; i < framebuffers_.size(); ++i) {
        VkImageView views[3] = {ms ? color_msaa_.view : swapchain_->views[i], depth_.view, swapchain_->views[i]};
        VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass = render_pass_;
        ci.attachmentCount = ms ? 3 : 2;
        ci.pAttachments = views;
        ci.width = swapchain_->extent.width;
        ci.height = swapchain_->extent.height;
        ci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(ctx_->device, &ci, nullptr, &framebuffers_[i]));
    }
}

void Renderer::destroy_depth_and_framebuffers() {
    for (VkFramebuffer f : framebuffers_) vkDestroyFramebuffer(ctx_->device, f, nullptr);
    framebuffers_.clear();
    destroy(*ctx_, depth_);
    destroy(*ctx_, color_msaa_);
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

void Renderer::ensure_particle_capacity(int count) {
    if (count <= particle_capacity_) return;
    // Grow by 1.5x so a slowly growing count does not reallocate every frame.
    const int cap = std::max(count, particle_capacity_ + particle_capacity_ / 2);
    wait_idle();  // the old buffers may still be in use
    for (auto& b : particle_buf_) {
        destroy(*ctx_, b);
        b = create_host_buffer(*ctx_, VkDeviceSize(cap) * 4 * ParticlePipeline::kNumBindings,
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    }
    particle_capacity_ = cap;
}

void Renderer::upload_particles(int frame, const ParticleDrawData& d) {
    if (d.count <= 0) return;
    char* base = static_cast<char*>(particle_buf_[frame].mapped);
    const size_t region = size_t(particle_capacity_) * 4;
    const size_t bytes = size_t(d.count) * 4;
    const void* src[ParticlePipeline::kNumBindings] = {d.px, d.py, d.pz, d.radius, d.env_id};
    for (uint32_t i = 0; i < ParticlePipeline::kNumBindings; ++i) std::memcpy(base + i * region, src[i], bytes);
}

void Renderer::set_capture(bool on) {
    if (on && !swapchain_->readable)
        throw std::runtime_error("flux: this swapchain's images can not be read back (no TRANSFER_SRC): no capture");
    capture_ = on;
}

bool Renderer::captured_bgra() const {
    return swapchain_->format == VK_FORMAT_B8G8R8A8_SRGB || swapchain_->format == VK_FORMAT_B8G8R8A8_UNORM;
}

// After the render pass: the swapchain image (PRESENT_SRC) -> TRANSFER_SRC, copied
// into capture_buf_, back to PRESENT_SRC; the copy made visible to the host.
void Renderer::record_capture(VkCommandBuffer cmd, uint32_t image_index) {
    const VkImage img = swapchain_->images[image_index];
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {swapchain_->extent.width, swapchain_->extent.height, 1};
    vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, capture_buf_.buffer, 1, &r);
    b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.dstAccessMask = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    VkBufferMemoryBarrier hb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    hb.srcQueueFamilyIndex = hb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hb.buffer = capture_buf_.buffer;
    hb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &hb, 0,
                         nullptr);
}

bool Renderer::rebuild_swapchain() {
    int w = 0, h = 0;
    window_.framebuffer_size(w, h);
    if (w == 0 || h == 0) return false;  // minimized: try again next frame
    wait_idle();
    destroy_depth_and_framebuffers();
    const VkFormat old_format = swapchain_->format;
    swapchain_->recreate(static_cast<uint32_t>(w), static_cast<uint32_t>(h));
    if (swapchain_->format != old_format)
        throw std::runtime_error("flux: swapchain format changed; render pass rebuild not supported yet");
    create_depth_and_framebuffers();
    // Image count can change: rebuild the per-image semaphores.
    for (VkSemaphore s : render_done_) vkDestroySemaphore(ctx_->device, s, nullptr);
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    render_done_.resize(swapchain_->images.size());
    for (VkSemaphore& s : render_done_) VK_CHECK(vkCreateSemaphore(ctx_->device, &si, nullptr, &s));
    return true;
}

void Renderer::record(VkCommandBuffer cmd, uint32_t image_index, const FrameInput& in) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    // Shadow map (Phase 10): the meshes and the opaque quads, seen from the sun.
    const bool mesh_on = mesh_ && mesh_envs_ > 0;
    const int mesh_per_row = std::max(1, int(std::ceil(std::sqrt(double(std::max(mesh_envs_, 1))))));
    const int mesh_rows = (mesh_envs_ + mesh_per_row - 1) / mesh_per_row;
    if (shadow_quads_ && (mesh_on || solid_count_ > 0)) {
        lighting_->begin_shadow(cmd);
        if (mesh_on)
            mesh_->draw_shadow(cmd, frame_, light_vp_, true, in.env_spacing, mesh_per_row, mesh_rows, 0, mesh_envs_);
        if (solid_count_ > 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_quads_->pipeline);
            vkCmdPushConstants(cmd, shadow_quads_->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4), &light_vp_);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &solid_buf_[frame_].buffer, &off);
            vkCmdDraw(cmd, 6, solid_count_, 0, 0);
        }
        lighting_->end_shadow(cmd);
    }

    VkClearValue clear[2]{};
    clear[0].color = {{opt_.clear_color[0], opt_.clear_color[1], opt_.clear_color[2], 1.0f}};
    clear[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = render_pass_;
    rp.framebuffer = framebuffers_[image_index];
    rp.renderArea = {{0, 0}, swapchain_->extent};
    rp.clearValueCount = 2;
    rp.pClearValues = clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    const VkExtent2D ext = swapchain_->extent;
    VkViewport vp{0.0f, 0.0f, float(ext.width), float(ext.height), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, ext};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    // Camera, shared by all pipelines.
    const OrbitCamera& cam = in.camera;
    const Vec3 eye = cam.eye();
    const Vec3 up{0.0f, 0.0f, 1.0f};
    const float aspect = float(ext.width) / float(ext.height);
    const Mat4 view_proj = perspective(cam.fovy, aspect, 0.05f, 500.0f) * look_at(eye, cam.target, up);

    auto draw_quads = [&](uint32_t first, uint32_t count) {
        if (count == 0) return;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, quad_pipeline_->pipeline);
        vkCmdPushConstants(cmd, quad_pipeline_->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4), &view_proj);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &quad_buf_.buffer, &off);
        vkCmdDraw(cmd, 6, count, 0, first);
    };

    sky_->draw(cmd, eye, cam.target, cam.fovy, aspect);  // background (no depth)
    const VkDescriptorSet light_set = lighting_->set(frame_);
    if (quads_before_ > 0) {  // floor: lit and shadowed, no depth write (as before)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_floor_->pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_floor_->layout, 0, 1, &light_set, 0, nullptr);
        vkCmdPushConstants(cmd, lit_floor_->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4), &view_proj);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &quad_buf_.buffer, &off);
        vkCmdDraw(cmd, 6, quads_before_, 0, 0);
    }
    if (solid_count_ > 0) {  // terrain: opaque, lit, writes depth
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_solid_->pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_solid_->layout, 0, 1, &light_set, 0, nullptr);
        vkCmdPushConstants(cmd, lit_solid_->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4), &view_proj);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &solid_buf_[frame_].buffer, &off);
        vkCmdDraw(cmd, 6, solid_count_, 0, 0);
    }
    if (mesh_on)  // robot meshes: opaque, lit, write depth
        mesh_->draw(cmd, frame_, view_proj, eye, true, in.env_spacing, mesh_per_row, mesh_rows, 0, mesh_envs_,
                    light_set);

    const ParticleDrawData& d = in.particles;
    if (d.count > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->pipeline);

        // Push constants: camera and env grid.
        ParticlePushConstants pc{};
        pc.view_proj = view_proj;
        const Vec3 fwd = forge::normalize(cam.target - eye);
        const Vec3 right = forge::normalize(forge::cross(fwd, up));
        const Vec3 cam_up = forge::cross(right, fwd);
        const int per_row = std::max(1, int(std::ceil(std::sqrt(double(d.num_envs)))));
        const int rows = (d.num_envs + per_row - 1) / per_row;
        float r4[4] = {right.x, right.y, right.z, 0.0f};
        float u4[4] = {cam_up.x, cam_up.y, cam_up.z, 0.0f};
        float g4[4] = {in.env_spacing, float(per_row), float(rows), std::max(in.color_height, 1e-3f)};
        std::memcpy(pc.cam_right, r4, sizeof(r4));
        std::memcpy(pc.cam_up, u4, sizeof(u4));
        std::memcpy(pc.grid, g4, sizeof(g4));
        vkCmdPushConstants(cmd, pipeline_->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

        // Bind the 5 regions (px, py, pz, radius, env_id) as 5 vertex buffers.
        // Stage B: the buffer CUDA wrote into. Stage A: this frame's host-visible copy.
        VkBuffer bufs[ParticlePipeline::kNumBindings];
        VkDeviceSize offs[ParticlePipeline::kNumBindings];
        for (uint32_t i = 0; i < ParticlePipeline::kNumBindings; ++i) {
            if (in.shared) {
                bufs[i] = in.shared->buffer;
                offs[i] = in.shared->offsets[i];
            } else {
                bufs[i] = particle_buf_[frame_].buffer;
                offs[i] = VkDeviceSize(i) * VkDeviceSize(particle_capacity_) * 4;
            }
        }
        vkCmdBindVertexBuffers(cmd, 0, ParticlePipeline::kNumBindings, bufs, offs);
        vkCmdDraw(cmd, 6, static_cast<uint32_t>(d.count), 0, 0);  // ONE draw for all particles
    }

    draw_quads(quads_before_, quads_after_);  // see-through walls, over the particles

    // Corner axes (like Unreal / Blender): 3D arrows in a small square in the bottom-left corner,
    // its own camera with the same yaw and pitch but no position, so the axes show
    // the world directions as seen from the main camera. Depth is cleared there first,
    // so the scene never hides the gizmo.
    if (gizmo_quads_ > 0) {
        const uint32_t size = std::max(60u, ext.height / 7), margin = 12;
        if (size + margin < ext.width && size + margin < ext.height) {
            const VkRect2D r{{int32_t(margin), int32_t(ext.height - size - margin)}, {size, size}};
            VkClearAttachment ca{};
            ca.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            ca.clearValue.depthStencil = {1.0f, 0};
            const VkClearRect cr{r, 0, 1};
            vkCmdClearAttachments(cmd, 1, &ca, 1, &cr);
            const VkViewport gvp{float(r.offset.x), float(r.offset.y), float(size), float(size), 0.0f, 1.0f};
            vkCmdSetViewport(cmd, 0, 1, &gvp);
            vkCmdSetScissor(cmd, 0, 1, &r);
            OrbitCamera gc = cam;
            gc.target = {0, 0, 0};
            gc.distance = 3.6f;
            const Mat4 g_view_proj = perspective(0.75f, 1.0f, 0.1f, 20.0f) * look_at(gc.eye(), gc.target, up);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, solid_pipeline_->pipeline);
            vkCmdPushConstants(cmd, solid_pipeline_->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4), &g_view_proj);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &gizmo_buf_.buffer, &off);
            vkCmdDraw(cmd, 6, gizmo_quads_, 0, 0);
        }
    }

    vkCmdEndRenderPass(cmd);
    if (capture_) record_capture(cmd, image_index);
    VK_CHECK(vkEndCommandBuffer(cmd));
}

void Renderer::signal_without_drawing(const SharedDrawSource& s) {
    // An empty submit: wait for CUDA's value, signal ours. Keeps the timeline
    // moving when a frame is skipped, so CUDA's next wait does not hang.
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkTimelineSemaphoreSubmitInfo ti{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ti.waitSemaphoreValueCount = 1;
    ti.pWaitSemaphoreValues = &s.wait_value;
    ti.signalSemaphoreValueCount = 1;
    ti.pSignalSemaphoreValues = &s.signal_value;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext = &ti;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &s.timeline;
    si.pWaitDstStageMask = &stage;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &s.timeline;
    VK_CHECK(vkQueueSubmit(ctx_->queue, 1, &si, VK_NULL_HANDLE));
}

bool Renderer::draw_frame(const FrameInput& in) {
    VkDevice d = ctx_->device;
    if (!in.shared) ensure_particle_capacity(in.particles.count);
    VK_CHECK(vkWaitForFences(d, 1, &in_flight_[frame_], VK_TRUE, UINT64_MAX));

    uint32_t image = 0;
    VkResult r = vkAcquireNextImageKHR(d, swapchain_->handle, UINT64_MAX, image_ready_[frame_],
                                       VK_NULL_HANDLE, &image);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        if (in.shared) signal_without_drawing(*in.shared);
        rebuild_swapchain();
        return false;
    }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) VK_CHECK(r);

    // Stage A: the fence said the GPU is done with this frame slot, so its
    // buffer is free to overwrite. Stage B: nothing to copy.
    if (!in.shared) upload_particles(frame_, in.particles);
    // Same for the per-frame opaque quads: this slot's buffer is free now.
    solid_count_ = 0;
    if (in.solid_quads && !in.solid_quads->empty()) {
        const size_t n = in.solid_quads->size();
        if (n > solid_cap_[frame_]) {
            destroy(*ctx_, solid_buf_[frame_]);
            solid_cap_[frame_] = n + n / 2;
            solid_buf_[frame_] =
                create_host_buffer(*ctx_, solid_cap_[frame_] * sizeof(Quad), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        }
        std::copy(in.solid_quads->begin(), in.solid_quads->end(), static_cast<Quad*>(solid_buf_[frame_].mapped));
        solid_count_ = static_cast<uint32_t>(n);
    }

    mesh_envs_ = 0;
    if (mesh_ && in.mesh_poses && in.mesh_envs > 0) {
        mesh_->set_poses(frame_, in.mesh_poses, in.mesh_envs);
        mesh_envs_ = in.mesh_envs;
    }
    {  // lighting: the sun's camera covers the env grid (as the env grid offsets place it)
        const int envs = std::max({1, in.particles.num_envs, mesh_envs_});
        const int per_row = std::max(1, int(std::ceil(std::sqrt(double(envs)))));
        const int rows = (envs + per_row - 1) / per_row;
        const float hz = 0.5f * std::max(in.scene_height, 0.1f);
        const Vec3 half{0.5f * per_row * in.env_spacing, 0.5f * rows * in.env_spacing, hz};
        light_vp_ = lighting_->sun_view_proj({0.0f, 0.0f, hz}, half);
        lighting_->update(frame_, light_vp_, in.camera.eye(), opt_.shadows);
    }

    if (capture_) {  // the read-back buffer fits this swapchain (it changes on a resize)
        const VkDeviceSize need = VkDeviceSize(swapchain_->extent.width) * swapchain_->extent.height * 4;
        if (capture_buf_.size != need) {
            wait_idle();  // the old buffer may still be in use
            destroy(*ctx_, capture_buf_);
            capture_buf_ = create_host_buffer(*ctx_, need, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
        }
    }

    // Reset the fence only when we know we will submit, or the next wait hangs.
    VK_CHECK(vkResetFences(d, 1, &in_flight_[frame_]));
    VK_CHECK(vkResetCommandBuffer(cmd_[frame_], 0));
    record(cmd_[frame_], image, in);

    // Wait list: the swapchain image (binary). Stage B adds CUDA's timeline value,
    // waited for at VERTEX_INPUT: the earliest stage that reads the particle buffer.
    // Signal list: render done for present (binary). Stage B adds our timeline value.
    // Values for binary semaphores are ignored, but the arrays must line up.
    VkSemaphore waits[2] = {image_ready_[frame_], VK_NULL_HANDLE};
    VkPipelineStageFlags wait_stages[2] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                           VK_PIPELINE_STAGE_VERTEX_INPUT_BIT};
    uint64_t wait_values[2] = {0, 0};
    VkSemaphore signals[2] = {render_done_[image], VK_NULL_HANDLE};
    uint64_t signal_values[2] = {0, 0};
    uint32_t n_wait = 1, n_signal = 1;
    if (in.shared) {
        waits[1] = in.shared->timeline;
        wait_values[1] = in.shared->wait_value;
        signals[1] = in.shared->timeline;
        signal_values[1] = in.shared->signal_value;
        n_wait = n_signal = 2;
    }
    VkTimelineSemaphoreSubmitInfo ti{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ti.waitSemaphoreValueCount = n_wait;
    ti.pWaitSemaphoreValues = wait_values;
    ti.signalSemaphoreValueCount = n_signal;
    ti.pSignalSemaphoreValues = signal_values;

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    if (in.shared) si.pNext = &ti;
    si.waitSemaphoreCount = n_wait;
    si.pWaitSemaphores = waits;
    si.pWaitDstStageMask = wait_stages;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_[frame_];
    si.signalSemaphoreCount = n_signal;
    si.pSignalSemaphores = signals;
    VK_CHECK(vkQueueSubmit(ctx_->queue, 1, &si, in_flight_[frame_]));
    if (capture_) {  // wait for this frame, then take its pixels
        VK_CHECK(vkWaitForFences(d, 1, &in_flight_[frame_], VK_TRUE, UINT64_MAX));
        if (!capture_buf_.coherent) invalidate(*ctx_, capture_buf_);
        cap_w_ = int(swapchain_->extent.width), cap_h_ = int(swapchain_->extent.height);
        const auto* src = static_cast<const uint8_t*>(capture_buf_.mapped);
        captured_.assign(src, src + size_t(cap_w_) * cap_h_ * 4);
    }

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
