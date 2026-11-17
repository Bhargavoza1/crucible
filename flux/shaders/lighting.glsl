// Shared lighting (Phase 10), included by mesh.frag and quad_lit.frag after
// "#define LIGHTING_SET n". Physically based, in linear color:
//   - the sun: Lambert diffuse + GGX specular (dielectric, F0 = 0.04), with a
//     shadow map (3 x 3 hardware-filtered samples)
//   - sky and ground light: a hemisphere ambient (sky color from above, ground
//     bounce from below)
//   - ACES tone mapping; sRGB encoding in the shader only for UNORM targets
//     (the window's swapchain is sRGB: the hardware encodes there)
// Must match flux/lighting.h (LightingUbo).

layout(std140, set = LIGHTING_SET, binding = 0) uniform LightingUbo {
    mat4 light_vp;   // world -> shadow map clip space
    vec4 sun_dir;    // xyz: toward the sun (unit), w: sun intensity
    vec4 sun_color;  // rgb, w: 1 / shadow map size (texels)
    vec4 eye;        // xyz: camera position (the window's; meshes get theirs per draw), w: 1 = shadows on
    vec4 sky;        // rgb: sky light (from above), w: exposure
    vec4 ground;     // rgb: ground bounce (from below), w: 1 = write sRGB-encoded values
} L;
layout(set = LIGHTING_SET, binding = 1) uniform sampler2DShadow shadow_map;

const float kPi = 3.14159265;

vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

// 1 = lit by the sun, 0 = in shadow. The point is pushed a little along its
// normal first (against "shadow acne": a surface shadowing itself).
float sun_visibility(vec3 world, vec3 n) {
    if (L.eye.w < 0.5) return 1.0;
    vec4 c = L.light_vp * vec4(world + n * 0.004, 1.0);
    vec3 p = c.xyz / c.w;
    vec2 uv = p.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || p.z > 1.0) return 1.0;
    float s = 0.0;
    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j) s += texture(shadow_map, vec3(uv + vec2(i, j) * L.sun_color.w, p.z - 0.0005));
    return s / 9.0;
}

// Light leaving a surface point toward the camera at 'eye' (linear, before tone mapping).
vec3 shade(vec3 base, vec3 n, vec3 world, float roughness, vec3 eye) {
    vec3 v = normalize(eye - world);
    if (dot(n, v) < 0.0) n = -n;  // seen from behind (thin parts, mixed winding): light the side we see
    vec3 l = L.sun_dir.xyz;
    vec3 h = normalize(l + v);
    float ndl = max(dot(n, l), 0.0), ndv = max(dot(n, v), 1e-4), ndh = max(dot(n, h), 0.0);
    float a = roughness * roughness, a2 = a * a;
    float dd = ndh * ndh * (a2 - 1.0) + 1.0;
    float D = a2 / (kPi * dd * dd);
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G = ndv / (ndv * (1.0 - k) + k) * ndl / (ndl * (1.0 - k) + k);
    vec3 F = vec3(0.04) + vec3(0.96) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
    vec3 spec = D * G * F / max(4.0 * ndv * ndl, 1e-4);
    vec3 diffuse = (vec3(1.0) - F) * base / kPi;
    vec3 sun = L.sun_color.rgb * L.sun_dir.w * ndl * sun_visibility(world, n);
    vec3 ambient = base * mix(L.ground.rgb, L.sky.rgb, 0.5 + 0.5 * n.z);
    return (diffuse + spec) * sun + ambient;
}

// Exposure, ACES tone curve (Narkowicz fit), and sRGB encoding for UNORM targets.
vec3 finish(vec3 c) {
    c *= L.sky.w;
    c = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
    if (L.ground.w > 0.5) c = mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
    return c;
}
