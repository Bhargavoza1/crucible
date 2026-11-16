#pragma once
// Pipeline for flat scene rectangles (floor, walls), one instanced draw.
// Blending on (walls are see-through), depth test on, depth WRITE off, so
// quads never hide particles drawn after them in the depth buffer.
#include <string>

#include <vulkan/vulkan.h>

namespace flux {

// One rectangle. Must match the vertex inputs of quad.vert (14 floats).
struct Quad {
    float center[3];
    float u[3];      // half edge 1
    float v[3];      // half edge 2
    float color[4];  // rgba, a < 1 is see-through
    float grid;      // grid line spacing in m, 0 = none
};
static_assert(sizeof(Quad) == 56, "must match quad.vert");

class QuadPipeline {
public:
    // depth_write = true: opaque 3D shapes made of quads (the corner gizmo).
    QuadPipeline(VkDevice device, VkRenderPass render_pass, const std::string& shader_dir, bool depth_write = false);
    ~QuadPipeline();
    QuadPipeline(const QuadPipeline&) = delete;
    QuadPipeline& operator=(const QuadPipeline&) = delete;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;

private:
    VkDevice device_;
};

}  // namespace flux
