#include "forge/env/terrain.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "forge/util/rng.h"

namespace forge {

Terrain make_terrain(const TerrainOptions& o) {
    if (!(o.cell > 0.0f) || !(o.tile_m >= 2.0f * o.cell) || !(o.size_m >= o.tile_m))
        throw std::runtime_error("make_terrain: needs cell > 0, tile >= 2 cells, size >= tile");
    Terrain t;
    t.cell = o.cell;
    const int tiles = int(std::lround(o.size_m / o.tile_m));
    const int per_tile = int(std::lround(o.tile_m / o.cell));  // cells per tile
    t.nx = t.ny = tiles * per_tile + 1;
    t.x0 = t.y0 = -0.5f * float(tiles * per_tile) * o.cell;
    t.h.assign(size_t(t.nx) * t.ny, 0.0f);

    for (int ty = 0; ty < tiles; ++ty)
        for (int tx = 0; tx < tiles; ++tx) {
            const int kind = tx % 4;
            const float level = tiles > 1 ? float(ty) / float(tiles - 1) : 0.0f;  // 0 (easy) .. 1 (hard)
            Rng rng(o.seed, uint64_t(tx), uint64_t(ty));
            // Noise on a 20 cm grid (2 cells), interpolated to the cells.
            const int nc = per_tile / 2 + 2;
            std::vector<float> coarse(size_t(nc) * nc);
            float amp = 0.0f;
            if (kind == 0) amp = 0.005f + 0.015f * level;
            if (kind == 1) amp = 0.01f + 0.06f * level;
            if (kind == 3) amp = 0.01f;
            for (float& c : coarse) c = std::round(rng.uniform(-amp, amp) / 0.005f) * 0.005f;
            const float slope = 0.05f + 0.24f * level;
            const float half = 0.5f * o.tile_m, top = 0.5f;  // pyramid: flat top 1 m wide
            for (int j = 0; j <= per_tile; ++j)
                for (int i = 0; i <= per_tile; ++i) {
                    const float cx = 0.5f * float(i), cy = 0.5f * float(j);
                    const int ci = std::min(int(cx), nc - 2), cj = std::min(int(cy), nc - 2);
                    const float u = cx - float(ci), v = cy - float(cj);
                    float z = (1 - u) * (1 - v) * coarse[size_t(cj) * nc + ci] + u * (1 - v) * coarse[size_t(cj) * nc + ci + 1] +
                              (1 - u) * v * coarse[size_t(cj + 1) * nc + ci] + u * v * coarse[size_t(cj + 1) * nc + ci + 1];
                    if (kind >= 2) {  // distance to the tile edge -> height
                        const float x = float(i) * o.cell - half, y = float(j) * o.cell - half;
                        const float d = std::min(half - std::fabs(x), half - std::fabs(y));  // 0 at the edge
                        const float rise = slope * std::min(d, half - top);
                        z += kind == 2 ? rise : -rise;
                    }
                    // Fade the noise to 0 on the outer 2 cells: tile borders stay small steps.
                    const int edge = std::min({i, j, per_tile - i, per_tile - j});
                    if (edge < 2 && kind != 2 && kind != 3) z *= float(edge) / 2.0f;
                    const int gi = tx * per_tile + i, gj = ty * per_tile + j;
                    t.h[size_t(gj) * t.nx + gi] = z;  // shared border vertices: the later tile wins
                }
        }
    return t;
}

}  // namespace forge
