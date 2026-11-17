// Particle scenes (Phases 2-4):
//   bin   floor + 4 walls, sphere-sphere contacts on a uniform grid. Contacts
//         are stiff springs, so each physics step is split into --substeps
//         small steps (default 10 x 0.5 ms) to stay stable in a dense pile.
//   rain  gravity only, particles fall forever.
// The scene restarts every --reset_every seconds of sim time.
#include <cmath>
#include <cstdio>

#include "sandbox.h"
#include "forge/particles/integrate_cpu.h"
#include "forge/particles/spawn.h"
#include "forge/particles/stepper_cpu.h"
#include "forge/util/nvtx.h"
#ifdef CRUCIBLE_HAS_CUDA
#include "forge/particles/integrate_gpu.h"
#include "forge/particles/particles_device.h"
#include "forge/particles/stepper_gpu.h"
#endif

namespace sandbox {

int run_particle_scene(Args& a, const std::string& scene) {
    forge::Config& cfg = *a.cfg;
    const bool bin = scene == "bin";
    const int num_envs = cfg.get_int("envs", 1);
    const int per_env = cfg.get_int("particles", bin ? 10000 : 1000);
    const int seed = cfg.get_int("seed", 1);
    const float reset_every = cfg.get_float("reset_every", bin ? 10.0f : 3.0f);  // s of sim time
    const int substeps = bin ? cfg.get_int("substeps", 10) : 1;
    const float bin_half = cfg.get_float("bin_half", 1.0f);  // m, half width of the bin
    const float radius = cfg.get_float("radius", 0.05f);     // m
    const float physics_dt = cfg.get_float("physics_dt", 0.005f);
    forge::ParticleParams params;  // params.dt is the SUBstep
    params.gravity.z = cfg.get_float("gravity", -9.81f);
    params.contact.timeconst = cfg.get_float("timeconst", params.contact.timeconst);
    params.contact.dampratio = cfg.get_float("dampratio", params.contact.dampratio);
    params.contact.friction = cfg.get_float("friction", params.contact.friction);
    params.cell_size = 2.0f * radius;  // smallest cell that still finds every contact
    params.particle_collisions = bin;
    if (bin) forge::add_bin_planes(params, bin_half);
    warn_unused_keys(cfg);

    if (num_envs <= 0 || per_env <= 0) throw std::runtime_error("envs and particles must be > 0");
    if (physics_dt <= 0.0f || substeps <= 0) throw std::runtime_error("physics_dt and substeps must be > 0");
    params.dt = physics_dt / float(substeps);
    if (bin && params.dt > 0.5f * params.contact.timeconst)
        std::fprintf(stderr, "warning: substep %.5f s is more than half of timeconst %.4f s: contacts may be "
                             "unstable. Use more --substeps.\n",
                     params.dt, params.contact.timeconst);

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

    flux::Window window(a.width, a.height, "crucible sandbox");
    Viewer viewer;
    make_viewer(viewer, window, a, initial.count());
    flux::Renderer& renderer = *viewer.renderer;
    const bool stage_b = viewer.stage_b;

    std::printf("sandbox: scene %s, %d envs x %d particles = %d, backend %s, draw %s\n", scene.c_str(), num_envs,
                per_env, initial.count(), a.backend.c_str(), stage_b ? "stage B (zero-copy)" : "stage A (copy via host)");
    std::printf("sandbox: physics dt %.4f s = %d x %.5f s substeps, vsync %s\n", physics_dt, substeps, params.dt,
                a.ro.vsync ? "on" : "off");

    // ---- static scene geometry (per env: floor, and see-through walls) ----------
    const float env_spacing = bin ? 2.0f * bin_half + 1.0f : 4.0f;
    if (bin) {
        flux::SceneQuads quads;
        const float h = bin_half, wall_h = 1.5f;  // walls are drawn 3 m tall
        for (int e = 0; e < num_envs; ++e) {
            const flux::Vec3 o = flux::env_grid_offset(e, num_envs, env_spacing);
            quads.before.push_back({{o.x, o.y, 0.0f}, {h + 0.4f, 0, 0}, {0, h + 0.4f, 0}, {0.42f, 0.43f, 0.45f, 1.0f}, 0.25f});
            const float sides[4][2] = {{-h, 0}, {h, 0}, {0, -h}, {0, h}};
            for (const auto& sd : sides) {
                const bool x_wall = sd[0] != 0.0f;
                quads.after.push_back({{o.x + sd[0], o.y + sd[1], wall_h},
                                       {x_wall ? 0.0f : h, x_wall ? h : 0.0f, 0.0f},
                                       {0.0f, 0.0f, wall_h},
                                       {0.45f, 0.65f, 0.95f, 0.10f},
                                       0.0f});
            }
        }
        renderer.set_scene(quads);
    }

    // ---- GPU state -----------------------------------------------------------
#ifdef CRUCIBLE_HAS_CUDA
    cudaStream_t stream = nullptr;
    forge::ParticleStepperGpu stepper_gpu;
    forge::ParticlesDevice dev;          // working state (Stage B: velocities, inv_mass)
    forge::ParticlesDevice initial_dev;  // start state kept on the GPU, for resets
    forge::ParticleView gpu_view{};
    if (a.backend == "gpu") {
        CUDA_CHECK(cudaStreamCreate(&stream));
        dev.upload(initial);
        gpu_view = dev.view();
    }
#endif
#ifdef FLUX_HAS_INTEROP
    if (stage_b) {
        // Positions, radius and env_id live in the shared buffer. One upload at
        // start; after that the data never leaves the GPU.
        flux::SharedParticles& sh = *viewer.shared;
        initial_dev.upload(initial);
        const size_t bytes = size_t(initial.count()) * 4;
        CUDA_CHECK(cudaMemcpy(sh.px(), initial.px.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sh.py(), initial.py.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sh.pz(), initial.pz.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sh.radius(), initial.radius.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sh.env_id(), initial.env_id.data(), bytes, cudaMemcpyHostToDevice));
        gpu_view.px = sh.px();
        gpu_view.py = sh.py();
        gpu_view.pz = sh.pz();
        gpu_view.radius = sh.radius();
        gpu_view.env_id = sh.env_id();
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
    StepClock clock{physics_dt, a.max_steps_per_frame};
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
            viewer.shared->cuda_wait_for_vulkan(stream);  // Vulkan finished reading last frame
            if (reset) {
                // Rare (every few seconds), device to device: start state back in place.
                const size_t bytes = size_t(initial.count()) * 4;
                const forge::ParticleView s = initial_dev.view();
                const std::pair<const float*, float*> copies[] = {{s.px, gpu_view.px}, {s.py, gpu_view.py},
                                                                  {s.pz, gpu_view.pz}, {s.vx, gpu_view.vx},
                                                                  {s.vy, gpu_view.vy}, {s.vz, gpu_view.vz}};
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
            viewer.shared->cuda_signal_done(stream);  // Vulkan may read now
            viewer.set_shared_source(frame);
#endif
        } else {
            FORGE_NVTX_RANGE("physics (stage A)");
            const auto t_phys = Clock::now();
            if (reset) {
                host = initial;
#ifdef CRUCIBLE_HAS_CUDA
                if (a.backend == "gpu") dev.upload(initial);
#endif
            }
            if (a.backend == "gpu") {
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
        viewer.update_camera(window, frame.camera, a.orbit_speed, frame_dt);

        {
            FORGE_NVTX_RANGE("draw");
            renderer.draw_frame(frame);
        }
        viewer.next_frame();
        if (a.max_frames > 0 && renderer.frames_drawn() >= static_cast<uint64_t>(a.max_frames)) break;
    }
    renderer.wait_idle();

    const double wall = seconds_since(t_start);
    const double frames = double(std::max<uint64_t>(renderer.frames_drawn(), 1));
    std::printf("sandbox: %llu frames in %.2f s (%.1f FPS), %lld physics steps\n",
                static_cast<unsigned long long>(renderer.frames_drawn()), wall, frames / wall, total_steps);
    if (stage_b) {
#ifdef FLUX_HAS_INTEROP
        std::printf("sandbox: GPU physics %.3f ms/frame (avg of %lld frames, cuda events)\n",
                    gpu_physics_samples ? gpu_physics_ms / double(gpu_physics_samples) : 0.0, gpu_physics_samples);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
#endif
    } else {
        std::printf("sandbox: physics + copy %.3f ms/frame (CPU wall time)\n", 1000.0 * cpu_physics_seconds / frames);
    }
#ifdef CRUCIBLE_HAS_CUDA
    if (stream) cudaStreamDestroy(stream);
#endif
    return 0;
}

}  // namespace sandbox
