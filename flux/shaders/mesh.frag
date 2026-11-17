#version 450
// Robot meshes (Phase 10): lit by the sun, sky and ground (lighting.glsl).
#extension GL_GOOGLE_include_directive : require
#define LIGHTING_SET 1
#include "lighting.glsl"

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec3 v_world;
layout(location = 3) flat in vec3 v_eye;
layout(location = 0) out vec4 out_color;

void main() {
    // degenerate (zero-area) triangles have a zero normal: normalize() would give NaN
    float len = length(v_normal);
    vec3 n = len > 1e-6 ? v_normal / len : vec3(0.0, 0.0, 1.0);
    vec3 base = srgb_to_linear(v_color.rgb);  // mesh colors are sRGB, as in the model files
    out_color = vec4(finish(shade(base, n, v_world, 0.5, v_eye)), 1.0);
}
