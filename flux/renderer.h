#pragma once
// Window renderer. Phase 1 draws one triangle. Later phases replace the
// triangle with one instanced draw of all bodies (doc 4.1).
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace flux {

class Window;
class Context;
class Swapchain;
class TrianglePipeline;

struct RendererOptions {
    bool validation = true;
    std::string shader_dir;  // folder with the .spv files
    float clear_color[3] = {0.08f, 0.09f, 0.11f};
};

class Renderer {
public:
    Renderer(Window& window, const RendererOptions& opt);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Draws and presents one frame. Returns false if the frame was skipped
    // (window minimized or swapchain just rebuilt).
    bool draw_frame();

    // Wait until the GPU has finished all submitted work.
    void wait_idle();

    uint64_t frames_drawn() const { return frames_drawn_; }

private:
    static constexpr int kFramesInFlight = 2;

    void create_render_pass();
    void create_framebuffers();
    void destroy_framebuffers();
    void create_commands_and_sync();
    void record(VkCommandBuffer cmd, uint32_t image_index);
    bool rebuild_swapchain();

    Window& window_;
    RendererOptions opt_;
    std::unique_ptr<Context> ctx_;
    std::unique_ptr<Swapchain> swapchain_;
    VkRenderPass render_pass_ = VK_NULL_HANDLE;
    std::unique_ptr<TrianglePipeline> pipeline_;
    std::vector<VkFramebuffer> framebuffers_;

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
