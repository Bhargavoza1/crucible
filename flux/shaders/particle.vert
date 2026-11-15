#version 450
// One instance per particle, 6 vertices per instance (a quad made of 2 triangles).
// Each SoA array (px, py, pz, radius, env_id) is its own vertex buffer, read
// per instance. This is the same layout forge uses, so in Phase 3 these
// buffers can be the shared CUDA memory without changing this shader.

layout(location = 0) in float in_px;
layout(location = 1) in float in_py;
layout(location = 2) in float in_pz;
layout(location = 3) in float in_radius;
layout(location = 4) in int in_env_id;

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    vec4 cam_right;  // xyz: camera right axis in world space
    vec4 cam_up;     // xyz: camera up axis in world space
    vec4 grid;       // x: env spacing (m), y: envs per row, z: number of rows, w: color height (m)
} pc;

layout(location = 0) out vec2 v_uv;      // -1..1 across the quad
layout(location = 1) out float v_height; // 0 at the floor, 1 at the color height

const vec2 kCorner[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1),
                               vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

void main() {
    // Env grid offset on the XY plane, centered on the origin (doc 4.1).
    int per_row = int(pc.grid.y);
    float col = float(in_env_id % per_row);
    float row = float(in_env_id / per_row);
    vec2 center_shift = 0.5 * vec2(float(per_row) - 1.0, pc.grid.z - 1.0);
    vec3 env_offset = vec3((vec2(col, row) - center_shift) * pc.grid.x, 0.0);

    vec2 c = kCorner[gl_VertexIndex];
    vec3 world = vec3(in_px, in_py, in_pz) + env_offset
               + (c.x * pc.cam_right.xyz + c.y * pc.cam_up.xyz) * in_radius;

    gl_Position = pc.view_proj * vec4(world, 1.0);
    v_uv = c;
    v_height = in_pz / pc.grid.w;
}
