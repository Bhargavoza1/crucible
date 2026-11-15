// Task scene / play mode (doc 7.1): a task from forge/env/tasks/, run through the
// same Env as training, drawn as skeletons. With --load, a trained policy acts
// (the mean action, no noise); without it, the action is 0.
//
//   ./sandbox_app --task cartpole --load runs/cartpole_.../best.bin     watch the policy
//   ./sandbox_app --task cartpole                                       no policy: the poles fall
//   ./sandbox_app --task ant --load runs/ant_.../best.bin --envs 16    walking ants
//   ./sandbox_app --task cartpole --load ... --envs 4
//
// Options: --envs (9), --seed, --print_every (s), --axes (also draw the world axes
// at each env's origin; default off), --physics_dt and --decimation
// (use the same values as in training).
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "sandbox.h"
#include "flux/gizmo.h"
#include "forge/dynamics/robot_batch_host.h"
#include "forge/env/env.h"
#include "forge/robots/skeleton.h"
#include "forge/util/nvtx.h"
#ifdef CRUCIBLE_HAS_TEMPER
#include "forge/util/device_buffer.h"
#include "temper/algos/algorithm.h"
#endif

namespace sandbox {

int run_task_scene(Args& a) {
    forge::Config& cfg = *a.cfg;
    cfg.set("backend", a.backend);  // already resolved from "auto" in main
    forge::EnvOptions eo = forge::EnvOptions::from_config(cfg, CRUCIBLE_ASSET_DIR);
    eo.num_envs = cfg.get_int("envs", 9);
    eo.backend = a.backend == "gpu" ? forge::EnvBackend::kGpu : forge::EnvBackend::kCpu;
    const std::string load = cfg.get_string("load", "");
    const bool show_axes = cfg.get_bool("axes", false);  // world axes at each env origin: +X red, +Y green, +Z blue
    const float print_every = cfg.get_float("print_every", 5.0f);
    warn_unused_keys(cfg);
    if (!load.empty() && eo.backend != forge::EnvBackend::kGpu)
        throw std::runtime_error("--load needs the gpu backend (the policy runs on the GPU)");

    auto env = forge::make_env(eo);
    const forge::RobotModel& model = env->model();
    const int N = env->num_envs(), A = env->act_dim();

#ifdef CRUCIBLE_HAS_TEMPER
    std::unique_ptr<temper::Algorithm> policy;
    forge::DeviceBuffer<float> dact;
    if (!load.empty()) {
        const temper::Checkpoint ck = temper::Checkpoint::load(load);
        if (ck.task != eo.task)
            throw std::runtime_error("checkpoint is for task '" + ck.task + "', not '" + eo.task + "'");
        policy = temper::make_algorithm(ck.algo, env->obs_dim(), A, N, nullptr,
                                        static_cast<cudaStream_t>(env->stream()));
        policy->load(ck, false);
        std::printf("sandbox: policy %s from %s (trained %lld iterations)\n", ck.algo.c_str(), load.c_str(),
                    static_cast<long long>(ck.iteration));
    }
    if (eo.backend == forge::EnvBackend::kGpu) {
        dact.resize(size_t(N) * A);
        CUDA_CHECK(cudaMemset(dact.data(), 0, dact.size() * sizeof(float)));
    }
#else
    if (!load.empty()) throw std::runtime_error("--load needs a build with CUDA (temper)");
#endif
    std::vector<float> hact(size_t(N) * A, 0.0f);

    // Skeleton: an unlimited slider's rail covers +-2.4 m (cartpole's range).
    const std::vector<forge::SkeletonPoint> skel = forge::make_skeleton(model, 6, 2.4f);
    const int npts = static_cast<int>(skel.size()), total = N * npts;
    const float extent = forge::skeleton_extent(model, skel);
    forge::RobotBatchHost draw(model, N);  // qpos copied here each frame; FK on the CPU (Stage A)
    std::vector<float> px(total), py(total), pz(total), radius(total);
    std::vector<int> env_id(total);
    for (int e = 0; e < N; ++e)
        for (int k = 0; k < npts; ++k) {
            radius[size_t(e) * npts + k] = skel[k].radius;
            env_id[size_t(e) * npts + k] = e;
        }

    flux::Window window(a.width, a.height, "crucible sandbox: task " + eo.task);
    Args a_copy = a;
    a_copy.interop = "off";  // few envs: copy through the host (Stage A) is simple and cheap here
    Viewer viewer;
    make_viewer(viewer, window, a_copy, total);
    flux::Renderer& renderer = *viewer.renderer;

    const float env_spacing = 1.3f * extent + 0.3f;
    {
        flux::SceneQuads quads;
        for (int e = 0; e < N; ++e) {
            const flux::Vec3 o = flux::env_grid_offset(e, N, env_spacing);
            const float h = 0.5f * env_spacing - 0.1f;
            quads.before.push_back({{o.x, o.y, 0.0f}, {h, 0, 0}, {0, h, 0}, {0.16f, 0.17f, 0.19f, 1.0f}, 0.25f});
            if (show_axes) flux::add_axes_gizmo(quads, {o.x, o.y, 0.0f}, 0.3f * env_spacing);  // at each env's world origin
        }
        renderer.set_scene(quads);
    }
    std::printf("sandbox: task %s, %d envs, obs %d, act %d, control dt %.3f s (physics %.4f x %d), %s\n",
                eo.task.c_str(), N, env->obs_dim(), A, eo.physics_dt * eo.decimation, eo.physics_dt, eo.decimation,
#ifdef CRUCIBLE_HAS_TEMPER
                policy ? "policy acts (mean action)" : "no policy: action 0");
#else
                "no policy: action 0");
#endif

    flux::FrameInput frame;
    frame.camera.target = {0.0f, 0.0f, 0.6f * extent};
    frame.camera.distance = 1.6f * extent + (N > 1 ? 0.6f * env_spacing * std::sqrt(float(N)) : 0.0f);
    frame.env_spacing = env_spacing;
    frame.color_height = 2.0f * extent;
    frame.particles.count = total;
    frame.particles.num_envs = N;

    StepClock clock{double(eo.physics_dt) * eo.decimation, std::max(1, a.max_steps_per_frame)};
    const auto t_start = Clock::now();
    auto t_last = t_start, t_print = t_start;
    long long steps_total = 0;
    forge::Env::HostCopy h;
    std::vector<float> last_return(N);
    std::vector<int> last_length(N);

    while (!window.should_close()) {
        window.poll_events();
        const double frame_dt = std::min(seconds_since(t_last), 0.1);
        t_last = Clock::now();
        const int steps = clock.steps_for(frame_dt);
        {
            FORGE_NVTX_RANGE("task step");
            for (int s = 0; s < steps; ++s) {
#ifdef CRUCIBLE_HAS_TEMPER
                if (eo.backend == forge::EnvBackend::kGpu) {
                    if (policy) policy->act_deterministic(env->obs(), dact.data(), N);
                    env->step(dact.data());
                    continue;
                }
#endif
                env->step(hact.data());
            }
        }
        steps_total += steps;
        env->download(h);  // syncs; qpos for drawing
        draw.qpos = h.qpos;
        forge::skeleton_points_cpu(model.view(), draw.view(), draw.work(), skel.data(), npts, px.data(), py.data(),
                                   pz.data());
        frame.particles.px = px.data();
        frame.particles.py = py.data();
        frame.particles.pz = pz.data();
        frame.particles.radius = radius.data();
        frame.particles.env_id = env_id.data();
        viewer.update_camera(window, frame.camera, a.orbit_speed, frame_dt);
        renderer.draw_frame(frame);
        viewer.next_frame();

        if (seconds_since(t_print) >= print_every) {
            t_print = Clock::now();
#ifdef CRUCIBLE_HAS_CUDA
            if (eo.backend == forge::EnvBackend::kGpu) {
                CUDA_CHECK(cudaMemcpy(last_return.data(), env->last_return(), N * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(last_length.data(), env->last_length(), N * sizeof(int), cudaMemcpyDeviceToHost));
            } else
#endif
            {
                std::copy(env->last_return(), env->last_return() + N, last_return.begin());
                std::copy(env->last_length(), env->last_length() + N, last_length.begin());
            }
            double r = 0, l = 0;
            int done = 0;
            for (int e = 0; e < N; ++e)
                if (last_length[e] > 0) {
                    r += last_return[e];
                    l += last_length[e];
                    ++done;
                }
            // A floating robot: how far it is from its start (+x is forward), now.
            char dist[96] = "";
            if (model.nbody() > 0 && model.jtype[0] == forge::kJointFree) {
                double x = 0;
                for (int e = 0; e < N; ++e) x += h.qpos[size_t(e) * model.nq + model.qadr[0]];
                std::snprintf(dist, sizeof(dist), "; base x now: mean %.2f m", x / N);
            }
            if (done)
                std::printf("sandbox: %.0f s: last episodes of %d envs: mean return %.1f, mean length %.0f steps%s\n",
                            seconds_since(t_start), done, r / done, l / done, dist);
            else
                std::printf("sandbox: %.0f s: no episode has ended yet%s\n", seconds_since(t_start), dist);
        }
        if (a.max_frames > 0 && renderer.frames_drawn() >= static_cast<uint64_t>(a.max_frames)) break;
    }
    renderer.wait_idle();
    const double wall = seconds_since(t_start);
    std::printf("sandbox: %llu frames in %.2f s (%.1f FPS), %lld control steps (%.1f%% of real time)\n",
                static_cast<unsigned long long>(renderer.frames_drawn()), wall,
                double(std::max<uint64_t>(renderer.frames_drawn(), 1)) / wall, steps_total,
                100.0 * steps_total * eo.physics_dt * eo.decimation / wall);
    return 0;
}

}  // namespace sandbox
