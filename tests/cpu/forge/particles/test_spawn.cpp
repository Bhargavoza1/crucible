#include <cmath>

#include <gtest/gtest.h>

#include "forge/particles/spawn.h"

using namespace forge;

TEST(Spawn, InsideBoxAndSpeedLimit) {
    SpawnBox box;
    box.center = {1.0f, -2.0f, 3.0f};
    box.half_extent = {0.5f, 1.0f, 2.0f};
    box.max_speed = 0.7f;
    ParticlesHost h;
    spawn_box(h, 4, 500, box, 7);
    ASSERT_EQ(h.count(), 2000);
    for (int i = 0; i < h.count(); ++i) {
        EXPECT_LE(std::fabs(h.px[i] - box.center.x), box.half_extent.x);
        EXPECT_LE(std::fabs(h.py[i] - box.center.y), box.half_extent.y);
        EXPECT_LE(std::fabs(h.pz[i] - box.center.z), box.half_extent.z);
        EXPECT_LE(std::fabs(h.vx[i]), box.max_speed);
        EXPECT_LE(std::fabs(h.vz[i]), box.max_speed);
        EXPECT_FLOAT_EQ(h.radius[i], box.radius);
        EXPECT_FLOAT_EQ(h.inv_mass[i], 1.0f);
    }
}

TEST(Spawn, SameSeedSameParticles) {
    ParticlesHost a, b;
    spawn_box(a, 3, 100, SpawnBox{}, 42);
    spawn_box(b, 3, 100, SpawnBox{}, 42);
    EXPECT_EQ(a.px, b.px);
    EXPECT_EQ(a.vz, b.vz);
}

TEST(Spawn, DifferentSeedDifferentParticles) {
    ParticlesHost a, b;
    spawn_box(a, 1, 100, SpawnBox{}, 1);
    spawn_box(b, 1, 100, SpawnBox{}, 2);
    EXPECT_NE(a.px, b.px);
}

// Env e looks the same no matter how many envs are spawned, and envs differ.
TEST(Spawn, EnvsAreIndependent) {
    ParticlesHost small, big;
    spawn_box(small, 2, 50, SpawnBox{}, 9);
    spawn_box(big, 10, 50, SpawnBox{}, 9);
    for (int b = 0; b < 50; ++b) {
        EXPECT_EQ(small.px[env_index(1, b, 50)], big.px[env_index(1, b, 50)]);
    }
    EXPECT_NE(big.px[env_index(0, 0, 50)], big.px[env_index(1, 0, 50)]);
    EXPECT_EQ(big.env_id[env_index(7, 3, 50)], 7);
}

// Values come from mt19937 bits only, so they are the same on every compiler.
// If this number changes, saved scenes and test data change too.
TEST(Spawn, FixedValueAcrossPlatforms) {
    ParticlesHost h;
    SpawnBox box;
    box.center = {0, 0, 0};
    spawn_box(h, 1, 1, box, 1234);
    EXPECT_FLOAT_EQ(h.px[0], -0.7061137f);  // checked with numpy RandomState (MT19937)
}

TEST(SpawnLattice, NoOverlapAtStart) {
    SpawnLattice lat;
    lat.nx = lat.ny = 6;
    ParticlesHost h;
    spawn_lattice(h, 2, 300, lat, 11);
    ASSERT_EQ(h.count(), 600);
    for (int e = 0; e < 2; ++e)
        for (int a = 0; a < 300; ++a)
            for (int b = a + 1; b < 300; ++b) {
                const int i = env_index(e, a, 300), j = env_index(e, b, 300);
                const float dx = h.px[i] - h.px[j], dy = h.py[i] - h.py[j], dz = h.pz[i] - h.pz[j];
                ASSERT_GT(dx * dx + dy * dy + dz * dz, 4 * lat.radius * lat.radius) << i << " " << j;
            }
    EXPECT_FLOAT_EQ(h.vz[0], 0.0f);
    EXPECT_FLOAT_EQ(h.radius[0], lat.radius);
}

TEST(SpawnLattice, LayersGoUp) {
    SpawnLattice lat;
    lat.nx = 4;
    lat.ny = 3;
    lat.jitter = 0.0f;
    ParticlesHost h;
    spawn_lattice(h, 1, 25, lat, 1);
    // Particle 12 starts layer 1 (4 * 3 = 12 per layer).
    EXPECT_FLOAT_EQ(h.pz[12] - h.pz[0], lat.spacing);
    EXPECT_FLOAT_EQ(h.px[1] - h.px[0], lat.spacing);
}

TEST(SpawnLattice, BadSpacingThrows) {
    SpawnLattice lat;
    lat.spacing = 0.09f;  // < 2 * 0.05
    ParticlesHost h;
    EXPECT_THROW(spawn_lattice(h, 1, 10, lat, 1), std::invalid_argument);
}
