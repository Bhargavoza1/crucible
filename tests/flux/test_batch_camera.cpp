// Phase 9c: offscreen batch camera (flux/batch_camera.h). Runs on any Vulkan device
// (also lavapipe, a CPU driver). Checks geometry, not just "an image came out":
// pixels where the projection says a box is have the box's color, pixels just
// outside do not, every tile shows only its own env, and the image is not mirrored.
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "flux/batch_camera.h"

using namespace flux;

namespace {

const float kRed[3] = {0.9f, 0.1f, 0.1f};

BatchCameraOptions opts(int envs, int cams, int quads, int points) {
    BatchCameraOptions o;
    o.num_envs = envs;
    o.num_cams = cams;
    o.width = 64;
    o.height = 64;
    o.quads_per_env = quads;
    o.points_per_env = points;
    o.shader_dir = FLUX_SHADER_DIR;
    return o;
}

// RGB of pixel (x, y) of image (env, cam).
struct Rgb {
    int r, g, b;
};
Rgb pixel(const BatchCamera& cam, int env, int c, int x, int y) {
    std::vector<uint8_t> img(64 * 64 * 3);
    cam.copy_rgb(env, c, img.data());
    const uint8_t* p = img.data() + (size_t(y) * 64 + x) * 3;
    return {p[0], p[1], p[2]};
}
bool is_red(Rgb c) { return c.r > 150 && c.g < 60 && c.b < 60; }

// A camera 1 m above the origin looking straight down, x to the right in the image.
CameraView top_view() {
    CameraView v;
    v.eye = {0.0f, 0.0f, 1.0f};
    v.target = {0.0f, 0.0f, 0.0f};
    v.up = {0.0f, 1.0f, 0.0f};  // +y is up in the image
    v.fovy = 0.8f;
    return v;
}

TEST(BatchCamera, EmptySceneIsBackground) {
    BatchCamera cam(opts(2, 1, 0, 0));
    std::printf("BatchCamera on %s, atlas %d x %d\n", cam.device_name(), cam.atlas_width(), cam.atlas_height());
    const CameraView views[2] = {top_view(), top_view()};
    cam.render(nullptr, nullptr, nullptr, nullptr, nullptr, views);
    const Rgb c = pixel(cam, 1, 0, 10, 50);
    EXPECT_NEAR(c.r, 0.55 * 255, 2);
    EXPECT_NEAR(c.g, 0.62 * 255, 2);
    EXPECT_NEAR(c.b, 0.70 * 255, 2);
}

// A 10 cm red box seen from above: its projected center and the points just inside
// its corners are red, points 3 pixels outside are not. Off-center in +x and +y, so
// a mirrored image fails (x must go right, y up = toward the image top).
TEST(BatchCamera, BoxLandsWhereTheProjectionSays) {
    BatchCamera cam(opts(1, 1, 6, 0));
    Quad q[6];
    const Vec3 c{0.15f, 0.10f, 0.05f}, h{0.05f, 0.05f, 0.05f};
    box_quads(c, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, h, kRed, q);
    const CameraView v = top_view();
    cam.render(nullptr, nullptr, nullptr, nullptr, q, &v);
    float x, y;
    ASSERT_TRUE(cam.project(v, c + Vec3{0, 0, h.z}, x, y));  // top face center
    EXPECT_GT(x, 32.0f) << "+x is to the right";
    EXPECT_LT(y, 32.0f) << "+y is up in the image (y pixel smaller)";
    EXPECT_TRUE(is_red(pixel(cam, 0, 0, int(x), int(y))));
    int inside = 0, outside = 0;
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sy = -1; sy <= 1; sy += 2) {
            float cx, cy;
            ASSERT_TRUE(cam.project(v, c + Vec3{sx * h.x, sy * h.y, h.z}, cx, cy));  // top corners
            const float dx = cx > x ? 1.0f : -1.0f, dy = cy > y ? 1.0f : -1.0f;
            inside += is_red(pixel(cam, 0, 0, int(cx - 2 * dx), int(cy - 2 * dy)));
            outside += is_red(pixel(cam, 0, 0, int(cx + 3 * dx), int(cy + 3 * dy)));
        }
    EXPECT_EQ(inside, 4) << "2 pixels inside each top corner: red";
    EXPECT_EQ(outside, 0) << "3 pixels outside each top corner: not red";
}

// Several envs and cameras: each tile shows only its env's box, from its camera.
TEST(BatchCamera, EachTileShowsItsOwnEnv) {
    const int E = 5, C = 2;
    BatchCamera cam(opts(E, C, 6, 0));
    std::vector<Quad> q(E * 6);
    std::vector<CameraView> views(E * C);
    for (int e = 0; e < E; ++e) {
        // env e: its box at x = -0.2 + 0.1 e; camera 0 from above, camera 1 from the side
        box_quads({-0.2f + 0.1f * e, 0.0f, 0.05f}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0.03f, 0.03f, 0.03f}, kRed,
                  &q[e * 6]);
        views[e * C + 0] = top_view();
        views[e * C + 1].eye = {0.0f, -1.0f, 0.05f};
        views[e * C + 1].target = {0.0f, 0.0f, 0.05f};
        views[e * C + 1].up = {0.0f, 0.0f, 1.0f};
    }
    cam.render(nullptr, nullptr, nullptr, nullptr, q.data(), views.data());
    for (int e = 0; e < E; ++e)
        for (int c = 0; c < C; ++c) {
            const Vec3 center{-0.2f + 0.1f * e, 0.0f, 0.05f};
            float x, y;
            ASSERT_TRUE(cam.project(views[e * C + c], center, x, y));
            EXPECT_TRUE(is_red(pixel(cam, e, c, int(x), int(y)))) << "env " << e << " camera " << c;
            // where the NEXT env's box would be: not red in this tile
            const Vec3 other{-0.2f + 0.1f * ((e + 2) % E), 0.0f, 0.05f};
            ASSERT_TRUE(cam.project(views[e * C + c], other, x, y));
            EXPECT_FALSE(is_red(pixel(cam, e, c, int(x), int(y)))) << "env " << e << " camera " << c << " sees env "
                                                                      << (e + 2) % E;
        }
}

// Points (robot spheres) are drawn: a big sphere at the center covers the center
// pixel and not a pixel far away; a box in front of it hides it (depth works).
TEST(BatchCamera, SpheresAndDepth) {
    BatchCamera cam(opts(2, 1, 6, 1));
    const float px[2] = {0.0f, 0.0f}, py[2] = {0.0f, 0.0f}, pz[2] = {0.2f, 0.2f}, r[2] = {0.05f, 0.05f};
    std::vector<Quad> q(12);
    box_quads({0.3f, 0.0f, 0.05f}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0.02f, 0.02f, 0.02f}, kRed, &q[0]);  // env 0: aside
    box_quads({0.0f, 0.0f, 0.5f}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0.1f, 0.1f, 0.02f}, kRed, &q[6]);    // env 1: above the sphere
    const CameraView views[2] = {top_view(), top_view()};
    cam.render(px, py, pz, r, q.data(), views);
    const Rgb bg{140, 158, 178};
    const Rgb center0 = pixel(cam, 0, 0, 32, 32);
    EXPECT_FALSE(is_red(center0));
    EXPECT_GT(std::abs(center0.r - bg.r) + std::abs(center0.g - bg.g) + std::abs(center0.b - bg.b), 30)
        << "env 0: the sphere covers the center";
    const Rgb far0 = pixel(cam, 0, 0, 2, 2);
    EXPECT_LT(std::abs(far0.r - bg.r) + std::abs(far0.g - bg.g) + std::abs(far0.b - bg.b), 6) << "a corner: background";
    EXPECT_TRUE(is_red(pixel(cam, 1, 0, 32, 32))) << "env 1: the box above the sphere hides it";
}

}  // namespace
