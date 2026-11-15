#pragma once
// Neighbor search through the uniform grid. FORGE_HD, so the GPU kernels and a
// CPU test run the SAME traversal code (the test builds the sorted arrays on the
// CPU and compares the result with brute force).
#include <cstdint>

#include "forge/collision/particle_grid.h"
#include "forge/math/hd.h"
#include "forge/particles/particle_step.h"

namespace forge {

// The grid after it is built: particles sorted by cell key, and for each key
// the range [cell_start[key], cell_end[key]) in the sorted list.
struct GridView {
    const uint32_t* cell_start;
    const uint32_t* cell_end;
    const int* idx_sorted;  // particle index at each sorted position
    float inv_cell;         // 1 / cell_size
    uint32_t mask;          // table_size - 1
};

// Calls fn(j) for every particle j != i in the same env whose cell is one of the
// 27 cells around i's cell. Each j is visited exactly once, even when two of the
// 27 cells share a hash, because j must really be in the cell being visited.
template <typename Fn>
FORGE_HD void for_each_neighbor(const ParticleView& s, int i, const GridView& g, Fn&& fn) {
    const int env = s.env_id[i];
    const CellCoord c = cell_of(particle_position(s, i), g.inv_cell);
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const CellCoord nc{c.x + dx, c.y + dy, c.z + dz};
                const uint32_t key = cell_hash(env, nc, g.mask);
                const uint32_t end = g.cell_end[key];
                for (uint32_t k = g.cell_start[key]; k < end; ++k) {
                    const int j = g.idx_sorted[k];
                    if (j == i || s.env_id[j] != env) continue;
                    if (!(cell_of(particle_position(s, j), g.inv_cell) == nc)) continue;
                    fn(j);
                }
            }
}

}  // namespace forge
