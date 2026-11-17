#include "flux/renderer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "flux/context.h"
#include "flux/gizmo.h"
#include "flux/pipeline/mesh_pipeline.h"
#include "flux/pipeline/particle_pipeline.h"
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
    create_render_pass();
    pipeline_ = std::make_unique<ParticlePipeline>(ctx_->device, render_pass_, opt_.shader_dir);
    quad_pipeline_ = std::make_unique<QuadPipeline>(ctx_->device, render_pass_, opt_.shader_dir);
    create_depth_and_framebuffers();
    create_commands_and_sync();
    solid_pipeline_ = std::make_unique<QuadPipeline>(ctx_->device, render_pass_, opt_.shader_dir, true);
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
    mesh_.reset();
    solid_pipeline_.reset();
    quad_pipeline_.reset();
    pipeline_.reset();
    vkDestroyRenderPass(d, render_pass_, nullptr);
    swapchain_.reset();
    ctx_.reset();
}

void Renderer::wait_idle() { vkDeviceWaitIdle(ctx_->device); }

void Renderer::set_robot_mesh(const RobotMesh& mesh, int lod) {
    if (!mesh_) mesh_ = std::make_unique<MeshDrawer>(*ctx_, render_pass_, opt_.shader_dir, kFramesInFlight);
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
    VkAttachmentDescription att[2]{};
    // 0: color (swapchain image)
    att[0].format = swapchain_->format;
    att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    // 1: depth. Cleared each frame, not kept after the pass.
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

    // Two frames in flight share ONE depth image. This dependency makes frame N+1
    // wait until frame N has finished its depth writes (write-after-write), and
    // waits for the acquired swapchain image before writing color.
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = 2;
    ci.pAttachments = att;
    ci.subpassCount = 1;
    ci.pSubpasses = &sub;
    ci.dependencyCount = 1;
    ci.pDependencies = &dep;
    VK_CHECK(vkCreateRenderPass(ctx_->device, &ci, nullptr, &render_pass_));
}

void Renderer::create_depth_and_framebuffers() {
    depth_ = create_image(*ctx_, swapchain_->extent.width, swapchain_->extent.height, depth_format_,
                          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    framebuffers_.resize(swapchain_->views.size());
    for (size_t i = 0; i < framebuffers_.size(); ++i) {
        VkImageView views[2] = {swapchain_->views[i], depth_.view};
        VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass = render_pass_;
        ci.attachmentCount = 2;
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

    draw_quads(0, quads_before_);  // floor
    if (solid_count_ > 0) {        // terrain: opaque, writes depth
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, solid_pipeline_->pipeline);
        vkCmdPushConstants(cmd, solid_pipeline_->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4), &view_proj);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &solid_buf_[frame_].buffer, &off);
        vkCmdDraw(cmd, 6, solid_count_, 0, 0);
    }
    if (mesh_ && mesh_envs_ > 0) {  // robot meshes: opaque, write depth
        const int per_row = std::max(1, int(std::ceil(std::sqrt(double(mesh_envs_)))));
        const int rows = (mesh_envs_ + per_row - 1) / per_row;
        mesh_->draw(cmd, frame_, view_proj, true, in.env_spacing, per_row, rows, 0, mesh_envs_);
    }

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
