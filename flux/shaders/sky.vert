#version 450
// Sky behind everything (Phase 10): one triangle that covers the screen.
layout(location = 0) out vec2 v_ndc;
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
    gl_Position = vec4(p, 1.0, 1.0);
    v_ndc = p;
}
