#version 450
// Opaque scene quads in the window (floor, terrain, boxes), lit like the meshes
// (lighting.glsl) and shadowed by them. Grid lines as in quad.frag.
#extension GL_GOOGLE_include_directive : require
#define LIGHTING_SET 0
#include "lighting.glsl"

layout(location = 0) in vec3 v_world;
layout(location = 1) in vec4 v_color;
layout(location = 2) in float v_grid;
layout(location = 3) in vec3 v_normal;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 base = srgb_to_linear(v_color.rgb);
    if (v_grid > 0.0) {
        vec2 g = v_world.xy / v_grid;
        vec2 dist = abs(fract(g - 0.5) - 0.5) / fwidth(g);
        float line = 1.0 - min(min(dist.x, dist.y), 1.0);
        base = mix(base, base * 1.6 + 0.03, line * 0.5);
    }
    float len = length(v_normal);
    vec3 n = len > 1e-9 ? v_normal / len : vec3(0.0, 0.0, 1.0);
    out_color = vec4(finish(shade(base, n, v_world, 0.85, L.eye.xyz)), v_color.a);
}
