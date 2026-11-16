#pragma once
// Terrain for heightfield tasks (Phase 8d), built on the host once per task.
//
// A square field of tiles (default 32 m, 4 x 4 tiles of 8 m, 10 cm cells). Each
// column of tiles has one kind, each row is harder than the one before
// (legged_gym's terrain layout, without the automatic curriculum yet):
//   column 0: smooth, a little noise (0.5 to 2 cm)
//   column 1: rough, uniform noise (1 to 7 cm), made on a 20 cm grid and
//             interpolated (no 10 cm spikes), heights in 5 mm steps
//   column 2: pyramid hill, slope 0.05 to 0.29 (3 to 16 degrees), 1 m flat top
//   column 3: pyramid valley, same slopes, plus 1 cm noise
// Tiles start and end near height 0, so tile borders have small steps only.
#include <cstdint>
#include <vector>

namespace forge {

struct TerrainOptions {
    float size_m = 32.0f, tile_m = 8.0f, cell = 0.1f;
    uint64_t seed = 1;
};

struct Terrain {
    int nx = 0, ny = 0;
    float cell = 0.0f, x0 = 0.0f, y0 = 0.0f;  // vertex (i, j) at (x0 + i cell, y0 + j cell), centered on 0
    std::vector<float> h;                     // [ny * nx]
};

Terrain make_terrain(const TerrainOptions& opt = {});

}  // namespace forge
