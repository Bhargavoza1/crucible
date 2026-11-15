#pragma once
// Uniform grid helpers for the particle broad phase (doc 3.3).
//
// Space is cut into cubes of edge 'cell_size'. With cell_size >= 2 * largest
// radius, two touching spheres are always in the same cell or in neighbor
// cells, so each particle only checks the 3 x 3 x 3 = 27 cells around it.
//
// Cells are not stored in a 3D array (space has no fixed size). Instead each
// (env, cell) gets a hash in [0, table_size). Two different cells can share a
// hash: that only costs extra checks, because every candidate is checked for
// "really in this cell" and "same env" before its force is used.
#include <cstdint>

#include "forge/math/hd.h"
#include "forge/math/vec3.h"

namespace forge {

struct CellCoord {
    int x, y, z;
};

FORGE_HD bool operator==(CellCoord a, CellCoord b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

FORGE_HD CellCoord cell_of(Vec3 p, float inv_cell_size) {
    return {static_cast<int>(floorf(p.x * inv_cell_size)), static_cast<int>(floorf(p.y * inv_cell_size)),
            static_cast<int>(floorf(p.z * inv_cell_size))};
}

// 'mask' = table_size - 1, with table_size a power of two.
FORGE_HD uint32_t cell_hash(int env, CellCoord c, uint32_t mask) {
    const uint32_t h = (static_cast<uint32_t>(c.x) * 73856093u) ^ (static_cast<uint32_t>(c.y) * 19349663u) ^
                       (static_cast<uint32_t>(c.z) * 83492791u) ^ (static_cast<uint32_t>(env) * 2654435761u);
    return h & mask;
}

}  // namespace forge
