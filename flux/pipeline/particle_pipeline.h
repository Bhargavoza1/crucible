#pragma once
// Graphics pipeline that draws all particles with ONE instanced draw call
// (doc 4.1). Per-instance data comes from 5 vertex buffers, one per SoA array.
#include <string>

#include <vulkan/vulkan.h>

#include "flux/camera.h"

namespace flux {

// Reads a .spv file and makes a shader module. Throws if the file is missing.
VkShaderModule load_shader(VkDevice device, const std::string& path);

// Must match the push_constant block in particle.vert (112 bytes, limit is 128).
struct ParticlePushConstants {
    Mat4 view_proj;
    float cam_right[4];
    float cam_up[4];
    float grid[4];  // spacing, envs per row, rows, color height
};
static_assert(sizeof(ParticlePushConstants) == 112, "must match particle.vert");

class ParticlePipeline {
public:
    // Vertex buffer bindings, in this order: px, py, pz, radius, env_id.
    static constexpr uint32_t kNumBindings = 5;

    ParticlePipeline(VkDevice device, VkRenderPass render_pass, const std::string& shader_dir);
    ~ParticlePipeline();
    ParticlePipeline(const ParticlePipeline&) = delete;
    ParticlePipeline& operator=(const ParticlePipeline&) = delete;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;

private:
    VkDevice device_;
};

}  // namespace flux
