#version 450
// Phase 1 triangle. Positions and colors are constants; no vertex buffer.
// Vulkan clip space: y points down, so the first vertex (y = -0.6) is at the top.

layout(location = 0) out vec3 v_color;

const vec2 kPos[3] = vec2[](vec2(0.0, -0.6), vec2(0.6, 0.6), vec2(-0.6, 0.6));
const vec3 kColor[3] = vec3[](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));

void main() {
    gl_Position = vec4(kPos[gl_VertexIndex], 0.0, 1.0);
    v_color = kColor[gl_VertexIndex];
}
