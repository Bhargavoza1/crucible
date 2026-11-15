#pragma once
// Mouse control for the orbit camera (doc 4.1: "Orbit camera with mouse").
// Host only, no GLFW here (so it is easy to test): the window gives a MouseInput
// each frame, update() changes the camera.
//
//   left drag           orbit: turn around the target (yaw) and tilt (pitch)
//   right / middle drag pan: move the target in the camera's view plane
//   scroll wheel        zoom: change the distance to the target
#include <algorithm>
#include <cmath>

#include "flux/camera.h"

namespace flux {

struct MouseInput {
    double x = 0, y = 0;   // cursor position, pixels (y down)
    bool left = false, right = false, middle = false;
    double scroll = 0;     // wheel steps since the last frame (up > 0)
};

struct OrbitControl {
    float orbit_speed = 0.005f;  // rad per pixel
    float pan_speed = 0.0015f;   // fraction of the distance per pixel
    float zoom_step = 0.9f;      // distance factor per wheel step up
    float min_distance = 0.2f, max_distance = 500.0f;

    void update(OrbitCamera& cam, const MouseInput& in) {
        const double dx = have_last_ ? in.x - last_x_ : 0.0, dy = have_last_ ? in.y - last_y_ : 0.0;
        last_x_ = in.x;
        last_y_ = in.y;
        have_last_ = true;
        if (in.left) {
            cam.yaw -= orbit_speed * float(dx);
            cam.pitch = std::clamp(cam.pitch + orbit_speed * float(dy), -1.5f, 1.5f);
        } else if (in.right || in.middle) {
            // Camera right and up vectors (world Z up), from the current yaw and pitch.
            const Vec3 right{-std::sin(cam.yaw), std::cos(cam.yaw), 0.0f};
            const Vec3 up{-std::sin(cam.pitch) * std::cos(cam.yaw), -std::sin(cam.pitch) * std::sin(cam.yaw),
                          std::cos(cam.pitch)};
            const float s = pan_speed * cam.distance;
            cam.target = cam.target - right * (s * float(dx)) + up * (s * float(dy));
        }
        if (in.scroll != 0.0)
            cam.distance = std::clamp(cam.distance * std::pow(zoom_step, float(in.scroll)), min_distance, max_distance);
    }

private:
    double last_x_ = 0, last_y_ = 0;
    bool have_last_ = false;
};

}  // namespace flux
