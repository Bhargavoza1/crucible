// Tests the grid neighbor search (the same for_each_neighbor the GPU runs).
// The grid is built here on the CPU the same way the GPU builds it:
// keys -> stable sort -> start/end per key. Then every touching pair must be
// found exactly once, compared with brute force.
#include <algorithm>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "forge/particles/grid_traversal.h"
#include "forge/particles/spawn.h"
#include "forge/particles/step_cpu.h"

using namespace forge;

namespace {

struct CpuGrid {
    std::vector<uint32_t> start, end;
    std::vector<int> sorted;
    GridView view;
};

CpuGrid build_grid(const ParticleView& s, float cell_size, uint32_t table_size) {
    const int n = s.count();
    const float inv = 1.0f / cell_size;
    const uint32_t mask = table_size - 1;
    std::vector<uint32_t> keys(n);
    for (int i = 0; i < n; ++i) keys[i] = cell_hash(s.env_id[i], cell_of(particle_position(s, i), inv), mask);
    CpuGrid g;
    g.sorted.resize(n);
    std::iota(g.sorted.begin(), g.sorted.end(), 0);
    std::stable_sort(g.sorted.begin(), g.sorted.end(), [&](int a, int b) { return keys[a] < keys[b]; });
    g.start.assign(table_size, 0);
    g.end.assign(table_size, 0);
    for (int k = 0; k < n; ++k) {
        const uint32_t key = keys[g.sorted[k]];
        if (k == 0 || keys[g.sorted[k - 1]] != key) g.start[key] = k;
        if (k == n - 1 || keys[g.sorted[k + 1]] != key) g.end[key] = k + 1;
    }
    g.view = GridView{g.start.data(), g.end.data(), g.sorted.data(), inv, mask};
    return g;
}

// Overlapping random particles, several envs, centered on the origin so cells
// with negative coordinates are used too.
ParticlesHost dense_scene() {
    SpawnBox box;
    box.center = {0, 0, 0};
    box.half_extent = {0.4f, 0.4f, 0.4f};
    box.radius = 0.05f;
    ParticlesHost h;
    spawn_box(h, 4, 400, box, 77);
    return h;
}

void check_traversal(uint32_t table_size) {
    ParticlesHost h = dense_scene();
    ParticleView s = h.view();
    const CpuGrid g = build_grid(s, 0.1f, table_size);  // cell = 2 * radius (the minimum)

    long long touching_pairs = 0;
    for (int i = 0; i < s.count(); ++i) {
        std::vector<int> seen;
        for_each_neighbor(s, i, g.view, [&](int j) { seen.push_back(j); });
        std::sort(seen.begin(), seen.end());
        ASSERT_TRUE(std::adjacent_find(seen.begin(), seen.end()) == seen.end()) << "duplicate neighbor for i=" << i;
        // Every touching particle must be among the visited ones.
        const int first = (i / s.particles_per_env) * s.particles_per_env;
        for (int j = first; j < first + s.particles_per_env; ++j) {
            if (!particles_touch(s, i, j)) continue;
            ++touching_pairs;
            ASSERT_TRUE(std::binary_search(seen.begin(), seen.end(), j)) << "missed pair " << i << " " << j;
        }
        for (int j : seen) ASSERT_EQ(s.env_id[j], s.env_id[i]) << "crossed envs";
    }
    EXPECT_GT(touching_pairs, 1000);  // the scene really is dense
}

}  // namespace

TEST(GridTraversal, FindsEveryPairOnceNormalTable) { check_traversal(4096); }

// Tiny table: many cells share a hash. Pairs must still be found exactly once.
TEST(GridTraversal, FindsEveryPairOnceTinyTable) { check_traversal(16); }

// Worst case: one bucket for everything. Slow, but still correct.
TEST(GridTraversal, FindsEveryPairOnceSingleBucket) { check_traversal(1); }

TEST(GridTraversal, CellOfHandlesNegativeCoordinates) {
    const CellCoord c = cell_of({-0.01f, 0.0f, 0.15f}, 10.0f);  // cell size 0.1
    EXPECT_EQ(c.x, -1);  // floor, not truncation toward zero
    EXPECT_EQ(c.y, 0);
    EXPECT_EQ(c.z, 1);
}

TEST(GridTraversal, HashStaysInTable) {
    for (int e = 0; e < 50; ++e)
        for (int x = -20; x < 20; x += 3) EXPECT_LT(cell_hash(e, {x, -x, 2 * x}, 1023u), 1024u);
}
