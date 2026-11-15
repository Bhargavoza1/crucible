#version 450

layout(location = 0) in vec3 v_world;
layout(location = 1) in vec4 v_color;
layout(location = 2) in float v_grid;
layout(location = 0) out vec4 out_color;

void main() {
    vec4 c = v_color;
    if (v_grid > 0.0) {
        // Thin grid lines on the XY plane, about 1 pixel wide at any distance
        // (fwidth = how much the coordinate changes from one pixel to the next).
        vec2 g = v_world.xy / v_grid;
        vec2 dist = abs(fract(g - 0.5) - 0.5) / fwidth(g);
        float line = 1.0 - min(min(dist.x, dist.y), 1.0);
        c.rgb = mix(c.rgb, vec3(0.55), line * 0.6);
    }
    out_color = c;
}
