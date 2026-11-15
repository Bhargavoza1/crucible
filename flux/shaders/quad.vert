#version 450
// Flat rectangles for the scene (floor, walls). One instance per quad.
// corner = center + c.x * u + c.y * v, with c in {-1, 1}^2. u and v are half edges.

layout(location = 0) in vec3 in_center;
layout(location = 1) in vec3 in_u;
layout(location = 2) in vec3 in_v;
layout(location = 3) in vec4 in_color;
layout(location = 4) in float in_grid;  // grid line spacing in m, 0 = no grid

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
} pc;

layout(location = 0) out vec3 v_world;
layout(location = 1) out vec4 v_color;
layout(location = 2) out float v_grid;

const vec2 kCorner[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1),
                               vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

void main() {
    vec2 c = kCorner[gl_VertexIndex];
    vec3 world = in_center + c.x * in_u + c.y * in_v;
    gl_Position = pc.view_proj * vec4(world, 1.0);
    v_world = world;
    v_color = in_color;
    v_grid = in_grid;
}
