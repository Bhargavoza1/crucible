#version 450
// Robot meshes (Phase 10). Each vertex is in its link's frame; the link's world
// pose comes from a storage buffer: 3 rows (vec4: R row | t) per (env, link).
// Viewer: one instance per env, placed on the env grid (like particle.vert).
// Camera tiles: one env per draw (pc.info.x), no grid offset.

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_color;
layout(location = 3) in uint in_link;

layout(std430, set = 0, binding = 0) readonly buffer Poses { vec4 rows[]; } poses;

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    vec4 grid;   // x: env spacing (m), y: envs per row, z: rows, w: 1 = place instances on the grid
    ivec4 info;  // x: first env, y: links per env
} pc;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec4 v_color;

void main() {
    int env = pc.info.x + gl_InstanceIndex;
    int base = (env * pc.info.y + int(in_link)) * 3;
    vec4 r0 = poses.rows[base], r1 = poses.rows[base + 1], r2 = poses.rows[base + 2];
    vec3 world = vec3(dot(r0.xyz, in_pos) + r0.w, dot(r1.xyz, in_pos) + r1.w, dot(r2.xyz, in_pos) + r2.w);
    vec3 n = vec3(dot(r0.xyz, in_normal), dot(r1.xyz, in_normal), dot(r2.xyz, in_normal));
    if (pc.grid.w > 0.5) {
        int per_row = int(pc.grid.y);
        vec2 cell = vec2(float(env % per_row), float(env / per_row));
        vec2 center_shift = 0.5 * vec2(float(per_row) - 1.0, pc.grid.z - 1.0);
        world += vec3((cell - center_shift) * pc.grid.x, 0.0);
    }
    gl_Position = pc.view_proj * vec4(world, 1.0);
    v_normal = n;
    v_color = in_color;
}
