// Mouse camera control (flux/camera_control.h): drag orbits, right drag pans,
// wheel zooms, limits hold. Header only, so it runs in tests_cpu.
#include <cmath>

#include <gtest/gtest.h>

#include "flux/camera_control.h"

using namespace flux;

TEST(OrbitControl, FirstFrameDoesNotJump) {
    OrbitCamera cam;
    const OrbitCamera before = cam;
    OrbitControl c;
    MouseInput m;
    m.x = 500;
    m.y = 300;
    m.left = true;  // button already down on the first frame: no delta yet
    c.update(cam, m);
    EXPECT_EQ(cam.yaw, before.yaw);
    EXPECT_EQ(cam.pitch, before.pitch);
}

TEST(OrbitControl, LeftDragOrbitsAndPitchIsLimited) {
    OrbitCamera cam;
    OrbitControl c;
    MouseInput m;
    c.update(cam, m);
    const float yaw0 = cam.yaw, dist0 = cam.distance;
    const Vec3 target0 = cam.target;
    m.left = true;
    m.x = 100;  // drag right
    c.update(cam, m);
    EXPECT_LT(cam.yaw, yaw0);
    EXPECT_EQ(cam.distance, dist0) << "orbit does not zoom";
    EXPECT_EQ(cam.target.x, target0.x) << "orbit does not pan";
    m.y = 100000;  // drag far down
    c.update(cam, m);
    EXPECT_LE(cam.pitch, 1.5f);
}

TEST(OrbitControl, RightDragPansWheelZooms) {
    OrbitCamera cam;
    cam.target = {0, 0, 1};
    OrbitControl c;
    MouseInput m;
    c.update(cam, m);
    m.right = true;
    m.x = 50;
    c.update(cam, m);
    const float moved = std::hypot(cam.target.x, cam.target.y);
    EXPECT_GT(moved, 0.0f);
    m.right = false;
    m.scroll = 2;  // two steps up: closer
    const float d0 = cam.distance;
    c.update(cam, m);
    EXPECT_NEAR(cam.distance, d0 * 0.81f, 1e-4f);
    m.scroll = -1000;  // far out: clamped
    c.update(cam, m);
    EXPECT_LE(cam.distance, 500.0f);
}
