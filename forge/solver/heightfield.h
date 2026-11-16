#pragma once
// Heightfield terrain (Phase 8d), FORGE_HD: the solver and the tasks use the same code.
//
// A grid of nx * ny heights, vertex (i, j) at (x0 + i * cell, y0 + j * cell, h[j * nx + i]).
// Each cell is two triangles split on the diagonal (i, j)-(i + 1, j + 1), as MuJoCo's
// hfield (checked: with u, v the position inside the cell along x and y, u >= v is the
// triangle (00, 10, 11), else (00, 01, 11)). Outside the grid the ground is flat at the
// height of the nearest edge point.
//
// A sphere touches the plane of the triangle under its center. For small spheres (the
// G1's feet: 5 mm) that is the triangle it rests on. Near an edge, MuJoCo may make one
// contact per touching triangle; this makes one.
#include "forge/math/hd.h"
#include "forge/math/vec3.h"

namespace forge {

struct HeightfieldView {
    int nx = 0, ny = 0;
    float cell = 0.0f, x0 = 0.0f, y0 = 0.0f;
    const float* h = nullptr;  // [ny * nx], null = no heightfield
};

// A point on, and the upward unit normal of, the ground plane under (x, y).
FORGE_HD void hfield_plane(const HeightfieldView& f, float x, float y, Vec3& point, Vec3& normal) {
    float gx = (x - f.x0) / f.cell, gy = (y - f.y0) / f.cell;
    const bool outside = gx < 0.0f || gy < 0.0f || gx > float(f.nx - 1) || gy > float(f.ny - 1);
    gx = fminf(fmaxf(gx, 0.0f), float(f.nx - 1));
    gy = fminf(fmaxf(gy, 0.0f), float(f.ny - 1));
    int i = int(gx), j = int(gy);
    if (i > f.nx - 2) i = f.nx - 2;
    if (j > f.ny - 2) j = f.ny - 2;
    const float u = gx - float(i), v = gy - float(j), c = f.cell;
    const float h00 = f.h[j * f.nx + i], h10 = f.h[j * f.nx + i + 1];
    const float h01 = f.h[(j + 1) * f.nx + i], h11 = f.h[(j + 1) * f.nx + i + 1];
    const Vec3 p00{f.x0 + float(i) * c, f.y0 + float(j) * c, h00};
    Vec3 n;
    if (u >= v) n = cross(Vec3{c, 0.0f, h10 - h00}, Vec3{c, c, h11 - h00});  // triangle (00, 10, 11)
    else n = cross(Vec3{c, c, h11 - h00}, Vec3{0.0f, c, h01 - h00});         // triangle (00, 11, 01)
    n = normalize(n);                                                        // z = c^2 / |.| > 0: up
    if (outside) {  // flat at the edge point's height
        const float z = p00.z - (n.x * (f.x0 + gx * c - p00.x) + n.y * (f.y0 + gy * c - p00.y)) / n.z;
        point = Vec3{x, y, z};
        normal = Vec3{0.0f, 0.0f, 1.0f};
        return;
    }
    point = p00;
    normal = n;
}

// Ground height under (x, y).
FORGE_HD float hfield_height(const HeightfieldView& f, float x, float y) {
    Vec3 p, n;
    hfield_plane(f, x, y, p, n);
    return p.z - (n.x * (x - p.x) + n.y * (y - p.y)) / n.z;
}

}  // namespace forge
