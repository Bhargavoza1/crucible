#pragma once
// Shared parts of the sandbox scenes: arguments, real-time step clock, and the
// viewer (renderer + zero-copy buffer when the GPU backend can share memory).
#include <algorithm>
#include <chrono>
#include <memory>
#include <string>

#include "flux/renderer.h"
#include "flux/window/window.h"
#include "forge/util/config.h"
#ifdef FLUX_HAS_INTEROP
#include "flux/interop/shared_particles.h"
#endif

namespace sandbox {

struct Args {
    forge::Config* cfg = nullptr;
    int width = 1280, height = 720;
    int max_frames = 0;  // 0 = until the window is closed
    std::string backend;  // "cpu" or "gpu" (already resolved from "auto")
    std::string interop;  // "auto", "on" or "off"
    flux::RendererOptions ro;
    float orbit_speed = 0.15f;  // rad/s
    int max_steps_per_frame = 8;
};

using Clock = std::chrono::steady_clock;

inline double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// Fixed physics step at real-time speed: each frame runs as many dt steps as
// the real time that passed, capped so one slow frame does not spiral.
struct StepClock {
    double dt;
    int max_steps;
    double accumulator = 0.0;
    int steps_for(double frame_dt) {
        accumulator += frame_dt;
        int n = std::min(int(accumulator / dt), max_steps);
        accumulator -= n * dt;
        if (accumulator > dt * max_steps) accumulator = 0.0;  // drop backlog
        return n;
    }
};

// Renderer, plus the shared CUDA-Vulkan point buffer when Stage B works.
struct Viewer {
    std::unique_ptr<flux::Renderer> renderer;
#ifdef FLUX_HAS_INTEROP
    std::unique_ptr<flux::SharedParticles> shared;
#endif
    bool stage_b = false;

    ~Viewer();  // destroys 'shared' first: it uses the renderer's Vulkan device
    // Stage B: point the frame at the shared buffer and the timeline values.
    void set_shared_source(flux::FrameInput& frame);
    // Stage B: call after draw_frame.
    void next_frame();
};

// Tries Stage B (zero-copy, 'capacity' points) when the backend is gpu and interop
// is not off; falls back to Stage A with a message. Throws if interop is "on" and
// Stage B fails.
void make_viewer(Viewer& v, flux::Window& window, const Args& a, int capacity);

// Prints unknown config keys (typos). Call after a scene has read all its keys.
void warn_unused_keys(const forge::Config& cfg);

int run_particle_scene(Args& a, const std::string& scene);  // "bin" or "rain"
int run_robot_scene(Args& a);
int run_task_scene(Args& a);  // --task NAME [--load checkpoint]: play mode

}  // namespace sandbox
