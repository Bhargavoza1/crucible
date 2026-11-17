#pragma once
// Window renderer. Draws all particles with one instanced draw (doc 4.1).
//
// Two ways to give it particles (doc 4.2):
//   Stage A  host arrays (ParticleDrawData). draw_frame copies them into a
//            host-visible Vulkan buffer. Used by the CPU backend and as a fallback.
//   Stage B  a SharedDrawSource: a Vulkan buffer CUDA already wrote into, and a
//            timeline semaphore. No copy. The renderer stays pure Vulkan; the
//            CUDA side lives in flux/interop.
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "flux/camera.h"
#include "flux/context.h"
#include "flux/memory.h"
#include "flux/pipeline/quad_pipeline.h"
#include "flux/robot_mesh.h"

namespace flux {

class Window;
class Context;
class Swapchain;
class ParticlePipeline;
class MeshDrawer;

// Static scene geometry. 'before' is drawn before the particles (floor),
// 'after' is drawn after them (see-through walls blend over the particles).
struct SceneQuads {
    std::vector<Quad> before;
    std::vector<Quad> after;
};

// Offset of env 'env' in the viewer grid (XY plane, centered on the origin).
// MUST match the env_offset formula in particle.vert.
inline Vec3 env_grid_offset(int env, int num_envs, float spacing) {
    int per_row = 1;
    while (per_row * per_row < num_envs) ++per_row;  // ceil(sqrt(num_envs))
    const int rows = (num_envs + per_row - 1) / per_row;
    const float col = float(env % per_row), row = float(env / per_row);
    return {(col - 0.5f * (per_row - 1)) * spacing, (row - 0.5f * (rows - 1)) * spacing, 0.0f};
}

struct RendererOptions {
    bool validation = true;
    bool vsync = true;                        // false: draw as fast as possible
    bool interop = false;                     // enable CUDA sharing in the Context
    std::optional<DeviceUuid> required_uuid;  // pick the GPU CUDA uses
    std::string shader_dir;  // folder with the .spv files
    float clear_color[3] = {0.08f, 0.09f, 0.11f};
    bool corner_axes = true;  // 3D world-axes gizmo in the bottom-left corner (turns with the camera)
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

// Stage B: particle data already in GPU memory, written by CUDA.
// draw_frame waits for 'wait_value' on 'timeline' before reading the buffer, and
// ALWAYS signals 'signal_value' when it is done, even if it skips the frame,
// so the CUDA side never waits forever.
struct SharedDrawSource {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offsets[5] = {};  // px, py, pz, radius, env_id
    VkSemaphore timeline = VK_NULL_HANDLE;
    uint64_t wait_value = 0;
    uint64_t signal_value = 0;
};

struct FrameInput {
    // Stage A uses the pointers in 'particles'. Stage B sets 'shared' and uses
    // only particles.count and particles.num_envs.
    ParticleDrawData particles;
    std::optional<SharedDrawSource> shared;
    OrbitCamera camera;
    float env_spacing = 4.0f;  // m between env centers in the grid
    float color_height = 5.0f; // m: particles at this height are fully orange
    // Opaque quads drawn this frame only (terrain under moving robots), with depth
    // write, before the particles. null = none.
    const std::vector<Quad>* solid_quads = nullptr;
    // Robot meshes (after set_robot_mesh): world link poses, mesh_envs * links * 12
    // floats (3 rows of R | t per link), drawn on the same env grid. null = none.
    const float* mesh_poses = nullptr;
    int mesh_envs = 0;
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

    // Replaces the static scene quads (waits for the GPU; call at setup, not per frame).
    void set_scene(const SceneQuads& scene);
    // Robot mesh drawn in every env (waits for the GPU; call at setup).
    void set_robot_mesh(const RobotMesh& mesh, int lod = 0);

    uint64_t frames_drawn() const { return frames_drawn_; }
    const Context& context() const { return *ctx_; }

private:
    static constexpr int kFramesInFlight = 2;

    void create_render_pass();
    void create_depth_and_framebuffers();
    void destroy_depth_and_framebuffers();
    void create_commands_and_sync();
    void ensure_particle_capacity(int count);
    void upload_particles(int frame, const ParticleDrawData& d);
    void record(VkCommandBuffer cmd, uint32_t image_index, const FrameInput& in);
    void signal_without_drawing(const SharedDrawSource& s);
    bool rebuild_swapchain();

    Window& window_;
    RendererOptions opt_;
    std::unique_ptr<Context> ctx_;
    std::unique_ptr<Swapchain> swapchain_;
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    Image depth_;
    VkRenderPass render_pass_ = VK_NULL_HANDLE;
    std::unique_ptr<ParticlePipeline> pipeline_;
    std::unique_ptr<QuadPipeline> quad_pipeline_;
    std::unique_ptr<QuadPipeline> solid_pipeline_;
    std::unique_ptr<MeshDrawer> mesh_;
    int mesh_envs_ = 0;  // envs whose poses this frame's slot holds  // same shaders, depth write on (terrain, corner axes)
    HostBuffer quad_buf_;
    uint32_t quads_before_ = 0, quads_after_ = 0;
    HostBuffer gizmo_buf_;  // corner axes, built once
    HostBuffer solid_buf_[kFramesInFlight];  // FrameInput::solid_quads, one buffer per frame slot
    size_t solid_cap_[kFramesInFlight] = {};
    uint32_t solid_count_ = 0;
    uint32_t gizmo_quads_ = 0;
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
