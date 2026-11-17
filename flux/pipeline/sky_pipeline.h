#pragma once
// Sky behind the scene (Phase 10): a gradient by view direction, drawn first, one
// triangle over the whole screen, no depth test or write.
#include <string>

#include <vulkan/vulkan.h>

#include "flux/camera.h"

namespace flux {

// Must match sky.frag.
struct SkyPush {
    float right[4], up[4], forward[4];  // camera basis, right and up scaled by the view's half size
    float zenith[4], horizon[4], below[4];
};
static_assert(sizeof(SkyPush) == 96, "must match sky.frag");

class SkyPipeline {
public:
    SkyPipeline(VkDevice device, VkRenderPass render_pass, const std::string& shader_dir,
                VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);
    ~SkyPipeline();
    SkyPipeline(const SkyPipeline&) = delete;
    SkyPipeline& operator=(const SkyPipeline&) = delete;

    // Camera at 'eye' looking at 'target' (z up), vertical field of view fovy, aspect w / h.
    void draw(VkCommandBuffer cmd, Vec3 eye, Vec3 target, float fovy, float aspect) const;

private:
    VkDevice device_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
};

}  // namespace flux
