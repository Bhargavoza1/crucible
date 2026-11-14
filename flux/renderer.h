#pragma once
// Window renderer. Draws all particles with one instanced draw (doc 4.1).
//
// Stage A of CUDA-Vulkan sharing (doc 4.2): the app gives host arrays every
// frame and draw_frame() copies them into a host-visible Vulkan buffer.
// Phase 3 (Stage B) replaces the copy with buffers shared with CUDA.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "flux/camera.h"
#include "flux/memory.h"

namespace flux {

class Window;
class Context;
class Swapchain;
class ParticlePipeline;

struct RendererOptions {
    bool validation = true;
    std::string shader_dir;  // folder with the .spv files
    float clear_color[3] = {0.08f, 0.09f, 0.11f};
};

// Host pointers to SoA arrays, all of length 'count'. Not owned.
struct ParticleDrawData {
    const float* px = nullptr;
    const float* py = nullptr;
    const float* pz = nullptr;
    const float* radius = nullptr;
    const int* env_id = nullptr;
    int count = 0;
    int num_envs = 1;
};

struct FrameInput {
    ParticleDrawData particles;
    OrbitCamera camera;
    float env_spacing = 4.0f;  // m between env centers in the grid
};

class Renderer {
public:
    Renderer(Window& window, const RendererOptions& opt);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Copies the particle arrays, draws and presents one frame. Returns false
    // if the frame was skipped (window minimized or swapchain just rebuilt).
    bool draw_frame(const FrameInput& in);

    // Wait until the GPU has finished all submitted work.
    void wait_idle();

    uint64_t frames_drawn() const { return frames_drawn_; }

private:
    static constexpr int kFramesInFlight = 2;

    void create_render_pass();
    void create_depth_and_framebuffers();
    void destroy_depth_and_framebuffers();
    void create_commands_and_sync();
    void ensure_particle_capacity(int count);
    void upload_particles(int frame, const ParticleDrawData& d);
    void record(VkCommandBuffer cmd, uint32_t image_index, const FrameInput& in);
    bool rebuild_swapchain();

    Window& window_;
    RendererOptions opt_;
    std::unique_ptr<Context> ctx_;
    std::unique_ptr<Swapchain> swapchain_;
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    Image depth_;
    VkRenderPass render_pass_ = VK_NULL_HANDLE;
    std::unique_ptr<ParticlePipeline> pipeline_;
    std::vector<VkFramebuffer> framebuffers_;

    // One buffer per frame in flight, so the CPU never writes a buffer the GPU
    // is still reading. Inside: 5 regions of 'capacity' 4-byte values
    // (px, py, pz, radius, env_id).
    HostBuffer particle_buf_[kFramesInFlight];
    int particle_capacity_ = 0;

    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_[kFramesInFlight] = {};
    VkSemaphore image_ready_[kFramesInFlight] = {};
    VkFence in_flight_[kFramesInFlight] = {};
    // One "render done" semaphore per swapchain image: present may still use it
    // after the frame fence signals, so it cannot be shared per frame slot.
    std::vector<VkSemaphore> render_done_;

    int frame_ = 0;
    uint64_t frames_drawn_ = 0;
};

}  // namespace flux
