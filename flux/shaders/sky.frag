#version 450
// Sky color by view direction: zenith -> horizon above, a darker haze below
// (linear colors: the window's swapchain encodes sRGB).
layout(location = 0) in vec2 v_ndc;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    vec4 right;    // camera right * tan(fovy / 2) * aspect
    vec4 up;       // camera up * tan(fovy / 2)
    vec4 forward;  // camera forward
    vec4 zenith;
    vec4 horizon;
    vec4 below;
} pc;

void main() {
    vec3 d = normalize(pc.forward.xyz + v_ndc.x * pc.right.xyz - v_ndc.y * pc.up.xyz);  // NDC y points down
    float z = d.z;
    vec3 c = z >= 0.0 ? mix(pc.horizon.rgb, pc.zenith.rgb, pow(z, 0.6))
                      : mix(pc.horizon.rgb, pc.below.rgb, pow(-z, 0.4));
    out_color = vec4(c, 1.0);
}
