#pragma once
// Robot meshes (Phase 10): pipeline + GPU data for drawing a RobotMesh in many envs.
//
//   MeshDrawer d(ctx, render_pass, shader_dir, slots, lighting.set_layout(), samples, shadow_pass);
//   d.set_mesh(mesh, lod);                      // once: vertices to device memory
//   d.set_poses(slot, poses, num_envs);         // per frame: num_envs * links * 12 floats
//   d.draw_shadow(cmd, slot, light_vp, ...)     // in Lighting's shadow pass (if given)
//   d.draw(cmd, slot, view_proj, ..., lighting.set(slot))   // in the main render pass
// 'slots' pose buffers (one per frame in flight), so the CPU never overwrites poses
// the GPU still reads. Depth test and write on, no blending, both sides drawn, lit by
// flux/shaders/lighting.glsl (the lighting set is set 1, the poses set 0).
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "flux/camera.h"
#include "flux/memory.h"
#include "flux/robot_mesh.h"

namespace flux {

class Context;

class MeshDrawer {
public:
    MeshDrawer(const Context& ctx, VkRenderPass render_pass, const std::string& shader_dir, int slots,
               VkDescriptorSetLayout lighting, VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
               VkRenderPass shadow_pass = VK_NULL_HANDLE);
    ~MeshDrawer();
    MeshDrawer(const MeshDrawer&) = delete;
    MeshDrawer& operator=(const MeshDrawer&) = delete;

    // Uploads one level of detail (waits for the GPU). lod past the last = the last.
    void set_mesh(const RobotMesh& mesh, int lod);
    bool has_mesh() const { return index_count_ > 0; }
    int num_links() const { return links_; }
    // poses: num_envs * num_links() * 12 floats (per link: 3 rows of R | t, world).
    void set_poses(int slot, const float* poses, int num_envs);
    // Viewer: all envs as instances, on the env grid (spacing, per_row, rows as in
    // env_grid_offset). Tiles: grid = false, only env 'first_env'.
    // eye: the camera position (specular highlights).
    void draw(VkCommandBuffer cmd, int slot, const Mat4& view_proj, Vec3 eye, bool grid, float spacing, int per_row,
              int rows, int first_env, int count, VkDescriptorSet lighting) const;
    // Depth only, from the sun (needs the shadow pass given at construction).
    void draw_shadow(VkCommandBuffer cmd, int slot, const Mat4& light_vp, bool grid, float spacing, int per_row,
                     int rows, int first_env, int count) const;

private:
    void ensure_pose_capacity(int slot, size_t bytes);
    VkPipeline make_pipeline(VkRenderPass pass, const std::string& shader_dir, VkSampleCountFlagBits samples,
                             bool shadow) const;
    void push_and_draw(VkCommandBuffer cmd, VkPipeline pipe, int slot, const Mat4& m, Vec3 eye, bool grid,
                       float spacing, int per_row, int rows, int first_env, int count) const;

    const Context& ctx_;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipeline shadow_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets_;
    std::vector<HostBuffer> poses_;
    VkBuffer vbuf_ = VK_NULL_HANDLE;
    VkDeviceMemory vmem_ = VK_NULL_HANDLE;
    uint32_t index_count_ = 0;
    VkDeviceSize index_offset_ = 0;  // indices follow the vertices in vbuf_
    int links_ = 0;
};

}  // namespace flux
