#pragma once
// Offscreen camera images for many envs at once (Phase 9c, doc 4.3).
//
// Every env has num_cams cameras. All their images are drawn into ONE offscreen
// color image (the atlas), one tile of width x height pixels per (env, camera):
// tile t = env * num_cams + cam, at column t % tiles_per_row, row t / tiles_per_row.
// Each tile shows that env alone: the static quads (floor, same for all envs),
// the env's own opaque quads (a cube: quads_per_env of them) and its points
// (robot skeleton spheres: points_per_env of them), seen from that camera.
// Drawn with the viewer's pipelines (sphere sprites, flat quads), so a tile looks
// like the sandbox at low resolution.
//
// Host path: the points and quads are copied from the host each frame, and
// render() waits for the GPU and copies the atlas back to the host (pixels(), in
// HOST_CACHED memory: the CPU reads it at normal speed; write-combined memory, the
// usual upload kind, made reading 33 MB per frame take seconds on an NVIDIA GPU).
// No window: works on any Vulkan device (also CPU drivers such as lavapipe).
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "flux/camera.h"
#include "flux/memory.h"
#include "flux/pipeline/quad_pipeline.h"
#include "flux/robot_mesh.h"

namespace flux {

class Context;
class ParticlePipeline;
class MeshDrawer;

// The 6 faces of a box (center c, unit axes ax ay az, half sizes h) as opaque quads,
// shaded by face (top brightest), into out[0..5].
inline void box_quads(Vec3 c, Vec3 ax, Vec3 ay, Vec3 az, Vec3 h, const float rgb[3], Quad* out) {
    const Vec3 axes[3] = {ax, ay, az};
    const float hs[3] = {h.x, h.y, h.z};
    int k = 0;
    for (int a = 0; a < 3; ++a)
        for (int s = -1; s <= 1; s += 2) {
            const Vec3 n = axes[a] * (hs[a] * float(s));
            const Vec3 u = axes[(a + 1) % 3] * hs[(a + 1) % 3], v = axes[(a + 2) % 3] * hs[(a + 2) % 3];
            const float shade = a == 2 ? 1.0f : (a == 0 ? 0.85f : 0.7f);
            out[k++] = Quad{{c.x + n.x, c.y + n.y, c.z + n.z}, {u.x, u.y, u.z}, {v.x, v.y, v.z},
                            {rgb[0] * shade, rgb[1] * shade, rgb[2] * shade, 1.0f}, 0.0f};
        }
}

struct CameraView {
    Vec3 eye{0, 0, 1}, target{0, 0, 0}, up{0, 0, 1};
    float fovy = 0.8f;  // rad, vertical
};

struct BatchCameraOptions {
    int num_envs = 1, num_cams = 1;
    int width = 64, height = 64;  // pixels per image
    int points_per_env = 0, quads_per_env = 0;
    float color_height = 1.0f;    // sphere color: blue at z = 0, orange at this height (m)
    float z_near = 0.02f, z_far = 10.0f;
    float clear_color[3] = {0.55f, 0.62f, 0.70f};  // background (sky)
    bool validation = false;
    std::string shader_dir;
};

class BatchCamera {
public:
    explicit BatchCamera(const BatchCameraOptions& opt);
    ~BatchCamera();
    BatchCamera(const BatchCamera&) = delete;
    BatchCamera& operator=(const BatchCamera&) = delete;

    // Quads drawn in every tile before everything else (the floor). Call at setup.
    void set_static_quads(const std::vector<Quad>& quads);
    // Robot mesh drawn in every tile (Phase 10). Call at setup. lod 1 = the small one.
    void set_robot_mesh(const RobotMesh& mesh, int lod = 1);

    // Draws all tiles and copies the atlas to the host. Waits for the GPU.
    //   px, py, pz, radius: num_envs * points_per_env values, env after env
    //   quads: num_envs * quads_per_env opaque quads, env after env (may be null if 0)
    //   views: num_envs * num_cams cameras, env after env
    //   mesh_poses: after set_robot_mesh, num_envs * links * 12 floats (world link
    //   poses, 3 rows of R | t); null = no mesh this frame
    void render(const float* px, const float* py, const float* pz, const float* radius, const Quad* quads,
                const CameraView* views, const float* mesh_poses = nullptr);

    // The atlas after render(): RGBA8, atlas_width() * atlas_height() * 4 bytes, row 0 = top.
    const uint8_t* pixels() const;
    int atlas_width() const { return tiles_per_row_ * opt_.width; }
    int atlas_height() const { return tile_rows_ * opt_.height; }
    // One image as RGB8 (width * height * 3 bytes, row 0 = top) into out.
    void copy_rgb(int env, int cam, uint8_t* out) const;
    // Where a world point lands in the image of (env, cam): pixel x, y (x right, y
    // down, pixel centers at +0.5). Returns false if it is behind the camera.
    bool project(const CameraView& view, Vec3 p, float& x, float& y) const;

    const BatchCameraOptions& options() const { return opt_; }
    const char* device_name() const { return device_name_.c_str(); }

private:
    void create_targets();
    void create_render_pass();
    Mat4 view_proj(const CameraView& v) const;

    BatchCameraOptions opt_;
    int tiles_ = 0, tiles_per_row_ = 0, tile_rows_ = 0;
    std::unique_ptr<Context> ctx_;
    std::string device_name_;
    VkRenderPass render_pass_ = VK_NULL_HANDLE;
    VkFramebuffer framebuffer_ = VK_NULL_HANDLE;
    Image color_, depth_;
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    std::unique_ptr<ParticlePipeline> points_pipeline_;
    std::unique_ptr<QuadPipeline> floor_pipeline_, solid_pipeline_;
    std::unique_ptr<MeshDrawer> mesh_;
    HostBuffer point_buf_, quad_buf_, static_buf_, readback_;
    uint32_t static_count_ = 0;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

}  // namespace flux
