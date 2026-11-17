#version 450
// Two lights from above (key and fill) plus ambient; both sides of a triangle lit
// the same (abs of the dot products), since some meshes are not consistently wound.

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec4 v_color;
layout(location = 0) out vec4 out_color;

void main() {
    // degenerate (zero-area) triangles have a zero normal: normalize() would give NaN
    float len = length(v_normal);
    vec3 n = len > 1e-6 ? v_normal / len : vec3(0.0, 0.0, 1.0);
    const vec3 key = normalize(vec3(0.4, -0.3, 0.85));
    const vec3 fill = normalize(vec3(-0.5, 0.6, 0.4));
    float light = 0.35 + 0.55 * abs(dot(n, key)) + 0.2 * abs(dot(n, fill));
    out_color = vec4(v_color.rgb * light, 1.0);
}
