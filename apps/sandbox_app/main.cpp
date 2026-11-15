// sandbox_app: window mode. Drops particles and draws them.
//
//   ./sandbox_app                              10,000 particles fall into a bin (Phase 4)
//   ./sandbox_app --scene rain                 free fall, no collisions (Phase 2)
//   ./sandbox_app --envs 16 --particles 2000   16 envs in a grid, one bin each
//   ./sandbox_app --backend cpu                CPU reference integrator (Stage A)
//   ./sandbox_app --interop off                GPU physics, but copy through the host (Stage A)
//   ./sandbox_app --vsync false                no 60 FPS limit, to compare speed
//   ./sandbox_app --frames 300                 exit after 300 frames (for tests)
//
// Drawing paths (doc 4.2):
//   Stage A  GPU -> host -> Vulkan buffer every frame (cudaMemcpy). CPU backend, fallback.
//   Stage B  CUDA writes into a Vulkan buffer directly; a timeline semaphore syncs
//            both sides. No copy, no cudaDeviceSynchronize in the loop.
//
// Scenes:
//   bin   floor + 4 walls, sphere-sphere contacts on a uniform grid. Contacts
//         are stiff springs, so each physics step is split into --substeps
//         small steps (default 10 x 0.5 ms) to stay stable in a dense pile.
//   rain  gravity only, particles fall forever.
// The scene restarts every --reset_every seconds of sim time.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>

#include "flux/renderer.h"
#include "flux/window/window.h"
#include "forge/particles/integrate_cpu.h"
#include "forge/particles/spawn.h"
#include "forge/particles/stepper_cpu.h"
#include "forge/util/config.h"
#include "forge/util/device_info.h"
#include "forge/util/nvtx.h"
#ifdef CRUCIBLE_HAS_CUDA
#include "forge/particles/integrate_gpu.h"
#include "forge/particles/particles_device.h"
#include "forge/particles/stepper_gpu.h"
#endif
#ifdef FLUX_HAS_INTEROP
#include "flux/interop/shared_particles.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

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

}  // namespace

int main(int argc, char** argv) {
    try {
        forge::Config cfg = forge::Config::from_cli(argc, argv);
        const int width = cfg.get_int("width", 1280);
        const int height = cfg.get_int("height", 720);
        const int max_frames = cfg.get_int("frames", 0);  // 0 = until closed
        const std::string scene = cfg.get_string("scene", "bin");
        if (scene != "bin" && scene != "rain") throw std::runtime_error("scene must be bin or rain");
        const bool bin = scene == "bin";
        const int num_envs = cfg.get_int("envs", 1);
        const int per_env = cfg.get_int("particles", bin ? 10000 : 1000);
        const int seed = cfg.get_int("seed", 1);
        const float reset_every = cfg.get_float("reset_every", bin ? 10.0f : 3.0f);  // s of sim time
        const int substeps = bin ? cfg.get_int("substeps", 10) : 1;
        const float bin_half = cfg.get_float("bin_half", 1.0f);  // m, half width of the bin
        const float radius = cfg.get_float("radius", 0.05f);     // m
        const int max_steps_per_frame = cfg.get_int("max_steps_per_frame", 8);
        const float orbit_speed = cfg.get_float("orbit_speed", 0.15f);  // rad/s
        std::string backend = cfg.get_string("backend", "auto");
        const std::string interop = cfg.get_string("interop", "auto");  // auto | on | off

        const float physics_dt = cfg.get_float("physics_dt", 0.005f);
        forge::ParticleParams params;  // params.dt is the SUBstep
        params.gravity.z = cfg.get_float("gravity", -9.81f);
        params.contact.timeconst = cfg.get_float("timeconst", params.contact.timeconst);
        params.contact.dampratio = cfg.get_float("dampratio", params.contact.dampratio);
        params.contact.friction = cfg.get_float("friction", params.contact.friction);
        params.cell_size = 2.0f * radius;  // smallest cell that still finds every contact
        params.particle_collisions = bin;
        if (bin) forge::add_bin_planes(params, bin_half);

        flux::RendererOptions ro;
        ro.validation = cfg.get_bool("validation", true);
        ro.vsync = cfg.get_bool("vsync", true);
        ro.shader_dir = cfg.get_string("shader_dir", FLUX_SHADER_DIR);

        for (const auto& k : cfg.unused_keys())
            std::fprintf(stderr, "warning: unknown config key '%s' (typo?)\n", k.c_str());
        if (num_envs <= 0 || per_env <= 0) throw std::runtime_error("envs and particles must be > 0");
        if (physics_dt <= 0.0f || substeps <= 0) throw std::runtime_error("physics_dt and substeps must be > 0");
        params.dt = physics_dt / float(substeps);
        if (bin && params.dt > 0.5f * params.contact.timeconst)
            std::fprintf(stderr, "warning: substep %.5f s is more than half of timeconst %.4f s: contacts may be "
                                 "unstable. Use more --substeps.\n",
                         params.dt, params.contact.timeconst);
        if (interop != "auto" && interop != "on" && interop != "off")
            throw std::runtime_error("interop must be auto, on or off");

        // ---- pick backend ------------------------------------------------------
        const bool have_gpu = forge::cuda_device_count() > 0;
        if (backend == "auto") backend = have_gpu ? "gpu" : "cpu";
        if (backend != "cpu" && backend != "gpu") throw std::runtime_error("backend must be auto, cpu or gpu");
        if (backend == "gpu" && !have_gpu)
            throw std::runtime_error(forge::built_with_cuda() ? "backend gpu: no CUDA device found"
                                                              : "backend gpu: built without CUDA");
        if (interop == "on" && backend != "gpu") throw std::runtime_error("interop on needs backend gpu");
        if (have_gpu) std::printf("cuda: %s\n", forge::cuda_device_description(0).c_str());

        // ---- scene -------------------------------------------------------------
        forge::ParticlesHost initial;
        if (bin) {
            // Column of non-overlapping particles above the bin.
            forge::SpawnLattice lat;
            lat.radius = radius;
            lat.spacing = 2.3f * radius;
            lat.nx = lat.ny = std::max(1, int((2.0f * bin_half - 2.0f * radius) / lat.spacing));
            const float w = lat.nx * lat.spacing;
            lat.origin = {-0.5f * w, -0.5f * w, 0.3f};
            forge::spawn_lattice(initial, num_envs, per_env, lat, static_cast<uint32_t>(seed));
        } else {
            forge::SpawnBox box;  // centered at z = 3 m, 2 m cube, speeds up to 1 m/s
            box.radius = radius;
            forge::spawn_box(initial, num_envs, per_env, box, static_cast<uint32_t>(seed));
        }
        forge::ParticlesHost host = initial;  // Stage A: what we draw

        flux::Window window(width, height, "crucible sandbox");

        // ---- renderer, and Stage B if we can ------------------------------------
        bool stage_b = false;
        std::unique_ptr<flux::Renderer> renderer;
#ifdef FLUX_HAS_INTEROP
        std::unique_ptr<flux::SharedParticles> shared;
        if (backend == "gpu" && interop != "off") {
            try {
                flux::RendererOptions ro_b = ro;
                ro_b.interop = true;
                auto uuid = forge::cuda_device_uuid(0);
                if (uuid) ro_b.required_uuid = *uuid;  // Vulkan must pick the CUDA GPU
                renderer = std::make_unique<flux::Renderer>(window, ro_b);
                shared = std::make_unique<flux::SharedParticles>(renderer->context(), initial.count());
                stage_b = true;
            } catch (const std::exception& e) {
                if (interop == "on") throw;
                std::fprintf(stderr, "sandbox: zero-copy not available, using Stage A copy.\n  reason: %s\n",
                             e.what());
                shared.reset();
                renderer.reset();
            }
        }
#else
        if (interop == "on") throw std::runtime_error("interop on: this build has no CUDA-Vulkan interop");
#endif
        if (!renderer) renderer = std::make_unique<flux::Renderer>(window, ro);

        std::printf("sandbox: scene %s, %d envs x %d particles = %d, backend %s, draw %s\n", scene.c_str(),
                    num_envs, per_env, initial.count(), backend.c_str(),
                    stage_b ? "stage B (zero-copy)" : "stage A (copy via host)");
        std::printf("sandbox: physics dt %.4f s = %d x %.5f s substeps, vsync %s\n", physics_dt, substeps,
                    params.dt, ro.vsync ? "on" : "off");

        // ---- static scene geometry (per env: floor, and see-through walls) ----------
        const float env_spacing = bin ? 2.0f * bin_half + 1.0f : 4.0f;
        if (bin) {
            flux::SceneQuads quads;
            const float h = bin_half, wall_h = 1.5f;  // walls are drawn 3 m tall
            for (int e = 0; e < num_envs; ++e) {
                const flux::Vec3 o = flux::env_grid_offset(e, num_envs, env_spacing);
                quads.before.push_back({{o.x, o.y, 0.0f}, {h + 0.4f, 0, 0}, {0, h + 0.4f, 0},
                                        {0.16f, 0.17f, 0.19f, 1.0f}, 0.25f});
                const flux::Vec3 wall_color{0.45f, 0.65f, 0.95f};
                const float sides[4][2] = {{-h, 0}, {h, 0}, {0, -h}, {0, h}};
                for (const auto& sd : sides) {
                    const bool x_wall = sd[0] != 0.0f;
                    flux::Quad q{{o.x + sd[0], o.y + sd[1], wall_h},
                                 {x_wall ? 0.0f : h, x_wall ? h : 0.0f, 0.0f},
                                 {0.0f, 0.0f, wall_h},
                                 {wall_color.x, wall_color.y, wall_color.z, 0.10f},
                                 0.0f};
                    quads.after.push_back(q);
                }
            }
            renderer->set_scene(quads);
        }

        // ---- GPU state -----------------------------------------------------------
#ifdef CRUCIBLE_HAS_CUDA
        cudaStream_t stream = nullptr;
        forge::ParticleStepperGpu stepper_gpu;
        forge::ParticlesDevice dev;          // working state (Stage B: velocities, inv_mass)
        forge::ParticlesDevice initial_dev;  // start state kept on the GPU, for resets
        forge::ParticleView gpu_view{};
        if (backend == "gpu") {
            CUDA_CHECK(cudaStreamCreate(&stream));
            dev.upload(initial);
            gpu_view = dev.view();
        }
#endif
#ifdef FLUX_HAS_INTEROP
        if (stage_b) {
            // Positions, radius and env_id live in the shared buffer. One upload at
            // start; after that the data never leaves the GPU.
            initial_dev.upload(initial);
            const size_t bytes = size_t(initial.count()) * 4;
            CUDA_CHECK(cudaMemcpy(shared->px(), initial.px.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(shared->py(), initial.py.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(shared->pz(), initial.pz.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(shared->radius(), initial.radius.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(shared->env_id(), initial.env_id.data(), bytes, cudaMemcpyHostToDevice));
            gpu_view.px = shared->px();
            gpu_view.py = shared->py();
            gpu_view.pz = shared->pz();
            gpu_view.radius = shared->radius();
            gpu_view.env_id = shared->env_id();
        }
        // GPU time of the physics kernels, measured with events (no CPU wait).
        cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
        bool ev_pending = false;
        double gpu_physics_ms = 0.0;
        long long gpu_physics_samples = 0;
        if (stage_b) {
            CUDA_CHECK(cudaEventCreate(&ev_start));
            CUDA_CHECK(cudaEventCreate(&ev_stop));
        }
#endif

        flux::FrameInput frame;
        frame.camera.target = {0.0f, 0.0f, bin ? 0.9f : 1.5f};
        frame.camera.distance = (bin ? 6.0f : 9.0f) + (num_envs > 1 ? 0.8f * env_spacing * std::sqrt(float(num_envs)) : 0.0f);
        frame.env_spacing = env_spacing;
        frame.color_height = bin ? 2.5f : 5.0f;
        forge::ParticleStepperCpu stepper_cpu;
        frame.particles.count = initial.count();
        frame.particles.num_envs = initial.num_envs();

        // ---- loop --------------------------------------------------------------
        StepClock clock{physics_dt, max_steps_per_frame};
        const auto t_start = Clock::now();
        auto t_last = t_start;
        double sim_time_since_reset = 0.0;
        long long total_steps = 0;
        double cpu_physics_seconds = 0.0;  // Stage A: wall time of step + copy

        while (!window.should_close()) {
            window.poll_events();
            const double frame_dt = std::min(seconds_since(t_last), 0.1);
            t_last = Clock::now();
            const int steps = clock.steps_for(frame_dt);
            const bool reset = sim_time_since_reset >= reset_every;
            if (reset) sim_time_since_reset = 0.0;

            if (stage_b) {
#ifdef FLUX_HAS_INTEROP
                FORGE_NVTX_RANGE("physics (stage B)");
                if (ev_pending && cudaEventQuery(ev_stop) == cudaSuccess) {  // last frame's timing
                    float ms = 0.0f;
                    cudaEventElapsedTime(&ms, ev_start, ev_stop);
                    gpu_physics_ms += ms;
                    ++gpu_physics_samples;
                    ev_pending = false;
                }
                shared->cuda_wait_for_vulkan(stream);  // Vulkan finished reading last frame
                if (reset) {
                    // Rare (every few seconds), device to device: start state back in place.
                    const size_t bytes = size_t(initial.count()) * 4;
                    const forge::ParticleView s = initial_dev.view();
                    const std::pair<const float*, float*> copies[] = {
                        {s.px, gpu_view.px}, {s.py, gpu_view.py}, {s.pz, gpu_view.pz},
                        {s.vx, gpu_view.vx}, {s.vy, gpu_view.vy}, {s.vz, gpu_view.vz}};
                    for (const auto& [src, dst] : copies)
                        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
                }
                if (!ev_pending) CUDA_CHECK(cudaEventRecord(ev_start, stream));
                if (bin) stepper_gpu.step(gpu_view, params, steps * substeps, stream);
                else forge::integrate_gpu(gpu_view, params, steps, stream);
                if (!ev_pending) {
                    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
                    ev_pending = true;
                }
                shared->cuda_signal_done(stream);  // Vulkan may read now

                flux::SharedDrawSource src;
                src.buffer = shared->buffer();
                for (int f = 0; f < flux::SharedParticles::kNumFields; ++f)
                    src.offsets[f] = shared->offset(static_cast<flux::SharedParticles::Field>(f));
                src.timeline = shared->timeline();
                src.wait_value = shared->vk_wait_value();
                src.signal_value = shared->vk_signal_value();
                frame.shared = src;
#endif
            } else {
                FORGE_NVTX_RANGE("physics (stage A)");
                const auto t_phys = Clock::now();
                if (reset) {
                    host = initial;
#ifdef CRUCIBLE_HAS_CUDA
                    if (backend == "gpu") dev.upload(initial);
#endif
                }
                if (backend == "gpu") {
#ifdef CRUCIBLE_HAS_CUDA
                    if (bin) stepper_gpu.step(gpu_view, params, steps * substeps, stream);
                    else forge::integrate_gpu(gpu_view, params, steps, stream);
                    dev.download_positions(host);  // Stage A: GPU -> host (waits for the kernels)
#endif
                } else {
                    if (bin) stepper_cpu.step(host.view(), params, steps * substeps);
                    else forge::integrate_cpu(host.view(), params, steps);
                }
                cpu_physics_seconds += seconds_since(t_phys);
                frame.particles.px = host.px.data();
                frame.particles.py = host.py.data();
                frame.particles.pz = host.pz.data();
                frame.particles.radius = host.radius.data();
                frame.particles.env_id = host.env_id.data();
            }
            total_steps += steps;
            sim_time_since_reset += steps * physics_dt;
            frame.camera.yaw += orbit_speed * float(frame_dt);

            {
                FORGE_NVTX_RANGE("draw");
                renderer->draw_frame(frame);
            }
#ifdef FLUX_HAS_INTEROP
            if (stage_b) shared->next_frame();
#endif
            if (max_frames > 0 && renderer->frames_drawn() >= static_cast<uint64_t>(max_frames)) break;
        }
        renderer->wait_idle();

        const double wall = seconds_since(t_start);
        const double frames = double(std::max<uint64_t>(renderer->frames_drawn(), 1));
        std::printf("sandbox: %llu frames in %.2f s (%.1f FPS), %lld physics steps\n",
                    static_cast<unsigned long long>(renderer->frames_drawn()), wall, frames / wall, total_steps);
        if (stage_b) {
#ifdef FLUX_HAS_INTEROP
            std::printf("sandbox: GPU physics %.3f ms/frame (avg of %lld frames, cuda events)\n",
                        gpu_physics_samples ? gpu_physics_ms / double(gpu_physics_samples) : 0.0,
                        gpu_physics_samples);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
            shared.reset();  // before the renderer: it uses the renderer's Vulkan device
#endif
        } else {
            std::printf("sandbox: physics + copy %.3f ms/frame (CPU wall time)\n",
                        1000.0 * cpu_physics_seconds / frames);
        }
#ifdef CRUCIBLE_HAS_CUDA
        if (stream) cudaStreamDestroy(stream);
#endif
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
