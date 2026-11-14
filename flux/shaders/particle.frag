#version 450
// Draws a round dot that looks like a small shaded sphere (fake normal from uv).

layout(location = 0) in vec2 v_uv;
layout(location = 1) in float v_height;
layout(location = 0) out vec4 out_color;

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 > 1.0) discard;  // corners of the quad: not part of the dot

    vec3 n = vec3(v_uv.x, v_uv.y, sqrt(1.0 - r2));   // normal in camera space
    vec3 light = normalize(vec3(0.4, 0.6, 1.0));
    float diffuse = max(dot(n, light), 0.0);

    // Color by height: low = blue, high = orange. Shows the fall at a glance.
    float t = clamp(v_height / 5.0, 0.0, 1.0);
    vec3 base = mix(vec3(0.15, 0.45, 0.95), vec3(0.98, 0.55, 0.15), t);
    out_color = vec4(base * (0.25 + 0.75 * diffuse), 1.0);
}
