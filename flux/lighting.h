#pragma once
// Scene lighting (Phase 10): what the lit shaders (flux/shaders/lighting.glsl) read.
//
// One descriptor set per frame slot: binding 0 = LightingUbo (sun, sky, camera,
// shadow matrix), binding 1 = the shadow map (depth, compare sampler).
// The shadow map is drawn in its own depth-only render pass (shadow_pass()) from
// the sun, with an orthographic camera that covers the scene (sun_view_proj()).
// shadow_size 0: no shadow pass; a 1 x 1 map that says "lit" (offscreen cameras,
// where all envs share one place in the world and one map would mix them).
#include <vector>

#include <vulkan/vulkan.h>

#include "flux/camera.h"
#include "flux/memory.h"

namespace flux {

class Context;

// Must match LightingUbo in lighting.glsl (std140).
struct LightingUbo {
    Mat4 light_vp;
    float sun_dir[4];    // xyz: toward the sun (unit), w: intensity
    float sun_color[4];  // rgb, w: 1 / shadow map size
    float eye[4];        // xyz: camera, w: 1 = shadows on
    float sky[4];        // rgb, w: exposure
    float ground[4];     // rgb, w: 1 = encode sRGB in the shader (UNORM targets)
};
static_assert(sizeof(LightingUbo) == 144, "must match lighting.glsl");

struct LightingSettings {
    Vec3 sun_dir = forge::normalize(Vec3{0.45f, -0.35f, 0.82f});  // toward the sun: high, from the front right
    float sun_intensity = 3.2f;
    Vec3 sun_color{1.0f, 0.95f, 0.88f};  // a little warm
    Vec3 sky{0.40f, 0.46f, 0.56f};       // blue sky light from above
    Vec3 ground{0.20f, 0.19f, 0.18f};    // bounce from the floor
    float exposure = 1.0f;
};

class Lighting {
public:
    Lighting(const Context& ctx, int slots, uint32_t shadow_size, bool encode_srgb);
    ~Lighting();
    Lighting(const Lighting&) = delete;
    Lighting& operator=(const Lighting&) = delete;

    VkDescriptorSetLayout set_layout() const { return set_layout_; }
    VkDescriptorSet set(int slot) const { return sets_[size_t(slot)]; }
    bool has_shadows() const { return shadow_pass_ != VK_NULL_HANDLE; }
    VkRenderPass shadow_pass() const { return shadow_pass_; }

    // Orthographic camera from the sun that sees the box 'center' +- 'half' (m).
    Mat4 sun_view_proj(Vec3 center, Vec3 half) const;
    // Writes this slot's UBO. light_vp: from sun_view_proj (ignored without shadows).
    void update(int slot, const Mat4& light_vp, Vec3 eye, bool shadows);
    // Begin / end the shadow pass (clears the map; set viewport and scissor inside).
    void begin_shadow(VkCommandBuffer cmd) const;
    void end_shadow(VkCommandBuffer cmd) const;
    uint32_t shadow_size() const { return size_; }

    LightingSettings settings;

private:
    const Context& ctx_;
    bool encode_srgb_;
    uint32_t size_ = 1;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets_;
    std::vector<HostBuffer> ubos_;
    Image map_;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkRenderPass shadow_pass_ = VK_NULL_HANDLE;
    VkFramebuffer framebuffer_ = VK_NULL_HANDLE;
};

}  // namespace flux
