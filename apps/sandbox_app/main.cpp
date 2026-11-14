// sandbox_app: window mode. Phase 2 drops particles and draws them (Stage A).
//
//   ./sandbox_app                              1000 particles, GPU if there is one
//   ./sandbox_app --envs 16 --particles 2000   16 envs in a grid
//   ./sandbox_app --backend cpu                use the CPU reference integrator
//   ./sandbox_app --frames 300                 exit after 300 frames (for tests)
//
// No collisions yet (Phase 4), so particles fall forever. The scene restarts
// every --reset_every seconds.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>

#include "flux/renderer.h"
#include "flux/window/window.h"
#include "forge/particles/integrate_cpu.h"
#include "forge/particles/spawn.h"
#include "forge/util/config.h"
#include "forge/util/device_info.h"
#ifdef CRUCIBLE_HAS_CUDA
#include "forge/particles/integrate_gpu.h"
#include "forge/particles/particles_device.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        forge::Config cfg = forge::Config::from_cli(argc, argv);
        const int width = cfg.get_int("width", 1280);
        const int height = cfg.get_int("height", 720);
        const int max_frames = cfg.get_int("frames", 0);  // 0 = until closed
        const int num_envs = cfg.get_int("envs", 1);
        const int per_env = cfg.get_int("particles", 1000);
        const int seed = cfg.get_int("seed", 1);
        const float reset_every = cfg.get_float("reset_every", 3.0f);  // s of sim time
        const int max_steps_per_frame = cfg.get_int("max_steps_per_frame", 8);
        const float orbit_speed = cfg.get_float("orbit_speed", 0.15f);  // rad/s
        std::string backend = cfg.get_string("backend", "auto");

        forge::ParticleParams params;
        params.dt = cfg.get_float("physics_dt", 0.005f);
        params.gravity.z = cfg.get_float("gravity", -9.81f);

        flux::RendererOptions ro;
        ro.validation = cfg.get_bool("validation", true);
        ro.shader_dir = cfg.get_string("shader_dir", FLUX_SHADER_DIR);

        for (const auto& k : cfg.unused_keys())
            std::fprintf(stderr, "warning: unknown config key '%s' (typo?)\n", k.c_str());
        if (num_envs <= 0 || per_env <= 0) throw std::runtime_error("envs and particles must be > 0");
        if (params.dt <= 0.0f) throw std::runtime_error("physics_dt must be > 0");

        // ---- pick backend ------------------------------------------------------
        const bool have_gpu = forge::cuda_device_count() > 0;
        if (backend == "auto") backend = have_gpu ? "gpu" : "cpu";
        if (backend != "cpu" && backend != "gpu") throw std::runtime_error("backend must be auto, cpu or gpu");
        if (backend == "gpu" && !have_gpu)
            throw std::runtime_error(forge::built_with_cuda() ? "backend gpu: no CUDA device found"
                                                              : "backend gpu: built without CUDA");
        if (have_gpu) std::printf("cuda: %s\n", forge::cuda_device_description(0).c_str());
        std::printf("sandbox: %d envs x %d particles = %d, backend %s, dt %.4f s\n", num_envs, per_env,
                    num_envs * per_env, backend.c_str(), params.dt);

        // ---- scene -------------------------------------------------------------
        forge::SpawnBox box;  // centered at z = 3 m, 2 m cube, speeds up to 1 m/s
        forge::ParticlesHost initial;
        forge::spawn_box(initial, num_envs, per_env, box, static_cast<uint32_t>(seed));
        forge::ParticlesHost host = initial;  // what we draw (CPU backend also steps it)
#ifdef CRUCIBLE_HAS_CUDA
        forge::ParticlesDevice dev;
        if (backend == "gpu") dev.upload(initial);
#endif

        flux::Window window(width, height, "crucible sandbox");
        flux::Renderer renderer(window, ro);

        flux::FrameInput frame;
        frame.camera.target = {0.0f, 0.0f, 1.5f};
        frame.camera.distance = 9.0f + 1.5f * float(num_envs > 1 ? std::sqrt(float(num_envs)) * 4.0f : 0.0f);
        frame.env_spacing = 4.0f;

        // ---- loop --------------------------------------------------------------
        // Fixed physics step, real-time speed: each frame runs as many dt steps as
        // the real time that passed (capped, so a slow frame does not spiral).
        const auto t_start = Clock::now();
        auto t_last = t_start;
        double accumulator = 0.0, sim_time_since_reset = 0.0;
        long long total_steps = 0;
        double physics_seconds = 0.0;  // wall time in step + copy, for the report

        while (!window.should_close()) {
            window.poll_events();
            const double frame_dt = std::min(seconds_since(t_last), 0.1);
            t_last = Clock::now();
            accumulator += frame_dt;
            int steps = std::min(int(accumulator / params.dt), max_steps_per_frame);
            accumulator -= steps * params.dt;
            if (accumulator > params.dt * max_steps_per_frame) accumulator = 0.0;  // drop backlog

            const auto t_phys = Clock::now();
            if (sim_time_since_reset >= reset_every) {
                sim_time_since_reset = 0.0;
                host = initial;
#ifdef CRUCIBLE_HAS_CUDA
                if (backend == "gpu") dev.upload(initial);
#endif
            }
            if (backend == "gpu") {
#ifdef CRUCIBLE_HAS_CUDA
                forge::integrate_gpu(dev.view(), params, steps);
                dev.download_positions(host);  // Stage A: GPU -> host (waits for the kernels)
#endif
            } else {
                forge::integrate_cpu(host.view(), params, steps);
            }
            physics_seconds += seconds_since(t_phys);
            total_steps += steps;
            sim_time_since_reset += steps * params.dt;

            frame.camera.yaw += orbit_speed * float(frame_dt);
            frame.particles = {host.px.data(), host.py.data(), host.pz.data(), host.radius.data(),
                               host.env_id.data(), host.count(), host.num_envs()};
            renderer.draw_frame(frame);  // Stage A: host -> Vulkan buffer
            if (max_frames > 0 && renderer.frames_drawn() >= static_cast<uint64_t>(max_frames)) break;
        }
        renderer.wait_idle();

        const double wall = seconds_since(t_start);
        const double frames = double(std::max<uint64_t>(renderer.frames_drawn(), 1));
        std::printf("sandbox: %llu frames in %.2f s (%.1f FPS), %lld physics steps, "
                    "physics + copy %.3f ms/frame\n",
                    static_cast<unsigned long long>(renderer.frames_drawn()), wall, frames / wall,
                    total_steps, 1000.0 * physics_seconds / frames);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
