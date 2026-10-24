#pragma once
// Graphics pipeline for the Phase 1 triangle. Vertices live in the shader
// (gl_VertexIndex), so there is no vertex buffer yet. Viewport and scissor are
// dynamic, so a window resize does not rebuild the pipeline.
#include <string>

#include <vulkan/vulkan.h>

namespace flux {

// Reads a .spv file and makes a shader module. Throws if the file is missing.
VkShaderModule load_shader(VkDevice device, const std::string& path);

class TrianglePipeline {
public:
    TrianglePipeline(VkDevice device, VkRenderPass render_pass, const std::string& shader_dir);
    ~TrianglePipeline();
    TrianglePipeline(const TrianglePipeline&) = delete;
    TrianglePipeline& operator=(const TrianglePipeline&) = delete;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;

private:
    VkDevice device_;
};

}  // namespace flux
