// Robot scene (Phase 5): many copies of a URDF robot moving under gravity,
// drawn as skeletons. No contacts yet (Phase 6): a floating robot falls
// through the floor, so use a fixed-base robot or --gravity 0.
//
//   ./sandbox_app --scene robot                                    16 arms swinging
//   ./sandbox_app --scene robot --robot ../assets/robots/pendulum.urdf --envs 64
//   ./sandbox_app --scene robot --robot ../assets/robots/quadruped.urdf --floating true --gravity 0
#include <cmath>
#include <cstdio>
#include <random>

#include "sandbox.h"
#include "forge/dynamics/robot_batch_host.h"
#include "forge/robots/skeleton.h"
#include "forge/robots/urdf.h"
#include "forge/util/nvtx.h"
#ifdef CRUCIBLE_HAS_CUDA
#include "forge/dynamics/robot_gpu.h"
#endif

namespace sandbox {

namespace {

// Different random start pose and velocity for every env.
void random_start(const forge::RobotModel& m, forge::RobotBatchHost& b, float spread, float speed, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (int e = 0; e < b.num_envs(); ++e) {
        float* q = b.env_qpos(e);
        for (int k = 0; k < m.nbody(); ++k) {
            const int a = m.qadr[k];
            if (m.jtype[k] == forge::kJointFree) {
                q[a + 2] = 1.0f;
                const forge::Quat r = forge::normalize(forge::Quat{0.3f * n(g), 0.3f * n(g), 0.3f * n(g), 1.0f});
                q[a + 3] = r.w;
                q[a + 4] = r.x;
                q[a + 5] = r.y;
                q[a + 6] = r.z;
            } else if (m.jtype[k] == forge::kJointRevolute) {
                q[a] = spread * u(g);
            } else if (m.jtype[k] == forge::kJointPrismatic) {
                q[a] = 0.2f * spread * u(g);
            }
        }
        for (int i = 0; i < m.nv; ++i) b.env_qvel(e)[i] = speed * u(g);
        // A free base gets a random spin but no drift, so it stays in its own cell.
        for (int k = 0; k < m.nbody(); ++k)
            if (m.jtype[k] == forge::kJointFree)
                for (int i = 0; i < 3; ++i) b.env_qvel(e)[m.dadr[k] + i] = 0.0f;
    }
}

}  // namespace

int run_robot_scene(Args& a) {
    forge::Config& cfg = *a.cfg;
    const std::string urdf = cfg.get_string("robot", std::string(CRUCIBLE_ASSET_DIR) + "/robots/arm3.urdf");
    const bool floating = cfg.get_bool("floating", false);
    const int num_envs = cfg.get_int("envs", 16);
    const int seed = cfg.get_int("seed", 1);
    const float physics_dt = cfg.get_float("physics_dt", 0.002f);  // MuJoCo's default step
    const float reset_every = cfg.get_float("reset_every", 10.0f);
    const float spread = cfg.get_float("init_spread", 1.0f);  // rad, random start angles
    const float speed = cfg.get_float("init_speed", 0.5f);    // random start velocities
    const float gravity = cfg.get_float("gravity", -9.81f);
    warn_unused_keys(cfg);
    if (num_envs <= 0 || physics_dt <= 0.0f) throw std::runtime_error("envs and physics_dt must be > 0");

    forge::UrdfOptions opt;
    opt.floating_base = floating;
    forge::RobotModel model = forge::load_urdf(urdf, opt);
    model.gravity = {0.0f, 0.0f, gravity};
    const std::vector<forge::SkeletonPoint> skel = forge::make_skeleton(model);
    const int npts = static_cast<int>(skel.size());
    const int total_pts = num_envs * npts;
    const float extent = forge::skeleton_extent(model, skel);

    forge::RobotBatchHost initial(model, num_envs);
    random_start(model, initial, spread, speed, static_cast<unsigned>(seed));
    forge::RobotBatchHost host = initial;

    // Static per-point data: radius and env id (positions change every frame).
    std::vector<float> radius(total_pts), px(total_pts), py(total_pts), pz(total_pts);
    std::vector<int> env_id(total_pts);
    for (int e = 0; e < num_envs; ++e)
        for (int k = 0; k < npts; ++k) {
            radius[size_t(e) * npts + k] = skel[k].radius;
            env_id[size_t(e) * npts + k] = e;
        }

    flux::Window window(a.width, a.height, "crucible sandbox");
    Viewer viewer;
    make_viewer(viewer, window, a, total_pts);
    flux::Renderer& renderer = *viewer.renderer;
    const bool stage_b = viewer.stage_b;

    std::printf("sandbox: scene robot '%s' (%d bodies, nq %d, nv %d, %s), %d envs, backend %s, draw %s\n",
                model.name.c_str(), model.nbody(), model.nq, model.nv, floating ? "floating base" : "fixed base",
                num_envs, a.backend.c_str(), stage_b ? "stage B (zero-copy)" : "stage A (copy via host)");
    std::printf("sandbox: physics dt %.4f s, gravity %.2f, %d skeleton points per robot\n", physics_dt, gravity, npts);
    if (floating && gravity != 0.0f)
        std::printf("sandbox: note: no contacts until Phase 6, so a floating robot falls through the floor\n");
    bool has_limits = false;
    for (bool l : model.joint_limited) has_limits = has_limits || l;
    if (has_limits)
        std::printf("sandbox: note: joint limits are not enforced until Phase 6 (a slider can slide away)\n");

    // Floor quad per env (visual only).
    const float env_spacing = 1.3f * extent + 0.3f;  // envs may overlap a little in the picture, never in physics
    {
        flux::SceneQuads quads;
        for (int e = 0; e < num_envs; ++e) {
            const flux::Vec3 o = flux::env_grid_offset(e, num_envs, env_spacing);
            const float h = 0.5f * env_spacing - 0.1f;
            quads.before.push_back({{o.x, o.y, 0.0f}, {h, 0, 0}, {0, h, 0}, {0.16f, 0.17f, 0.19f, 1.0f}, 0.25f});
        }
        renderer.set_scene(quads);
    }

#ifdef CRUCIBLE_HAS_CUDA
    cudaStream_t stream = nullptr;
    std::unique_ptr<forge::RobotModelDevice> dmodel;
    std::unique_ptr<forge::RobotBatchDevice> dbatch;
    forge::DeviceBuffer<forge::SkeletonPoint> dskel;
    forge::DeviceBuffer<float> dpx, dpy, dpz;  // Stage A on the GPU: points before the copy
    float *out_x = nullptr, *out_y = nullptr, *out_z = nullptr;
    if (a.backend == "gpu") {
        CUDA_CHECK(cudaStreamCreate(&stream));
        dmodel = std::make_unique<forge::RobotModelDevice>(model);
        dbatch = std::make_unique<forge::RobotBatchDevice>(model, num_envs);
        dbatch->upload_state(initial);
        dskel.upload(skel);
#ifdef FLUX_HAS_INTEROP
        if (stage_b) {
            flux::SharedParticles& sh = *viewer.shared;
            CUDA_CHECK(cudaMemcpy(sh.radius(), radius.data(), total_pts * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(sh.env_id(), env_id.data(), total_pts * 4, cudaMemcpyHostToDevice));
            out_x = sh.px();
            out_y = sh.py();
            out_z = sh.pz();
        }
#endif
        if (!stage_b) {
            dpx.resize(total_pts);
            dpy.resize(total_pts);
            dpz.resize(total_pts);
            out_x = dpx.data();
            out_y = dpy.data();
            out_z = dpz.data();
        }
    }
#endif

    flux::FrameInput frame;
    frame.camera.target = {0.0f, 0.0f, 0.6f * extent};
    frame.camera.distance = 2.5f * extent + (num_envs > 1 ? 0.6f * env_spacing * std::sqrt(float(num_envs)) : 0.0f);
    frame.env_spacing = env_spacing;
    frame.color_height = 2.0f * extent;
    frame.particles.count = total_pts;
    frame.particles.num_envs = num_envs;

    StepClock clock{physics_dt, std::max(a.max_steps_per_frame, int(0.05f / physics_dt))};
    const auto t_start = Clock::now();
    auto t_last = t_start;
    double sim_time_since_reset = 0.0;
    long long total_steps = 0;
    int bad_envs_reported = 0;

    while (!window.should_close()) {
        window.poll_events();
        const double frame_dt = std::min(seconds_since(t_last), 0.1);
        t_last = Clock::now();
        const int steps = clock.steps_for(frame_dt);
        const bool reset = sim_time_since_reset >= reset_every;
        if (reset) sim_time_since_reset = 0.0;

        if (a.backend == "gpu") {
#ifdef CRUCIBLE_HAS_CUDA
            FORGE_NVTX_RANGE("robot physics");
#ifdef FLUX_HAS_INTEROP
            if (stage_b) viewer.shared->cuda_wait_for_vulkan(stream);
#endif
            if (reset) dbatch->upload_state(initial);  // rare: every few seconds
            forge::robot_step_gpu(dmodel->view(), dbatch->view(), dbatch->work(), physics_dt, steps, stream);
            forge::skeleton_points_gpu(dmodel->view(), dbatch->view(), dbatch->work(), dskel.data(), npts, out_x,
                                       out_y, out_z, stream);
            if (stage_b) {
#ifdef FLUX_HAS_INTEROP
                viewer.shared->cuda_signal_done(stream);
                viewer.set_shared_source(frame);
#endif
            } else {
                CUDA_CHECK(cudaMemcpyAsync(px.data(), out_x, total_pts * 4, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(py.data(), out_y, total_pts * 4, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(pz.data(), out_z, total_pts * 4, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
#endif
        } else {
            FORGE_NVTX_RANGE("robot physics");
            if (reset) host = initial;
            forge::robot_step_cpu(model.view(), host.view(), host.work(), physics_dt, steps);
            forge::skeleton_points_cpu(model.view(), host.view(), host.work(), skel.data(), npts, px.data(),
                                       py.data(), pz.data());
            if (!bad_envs_reported)
                for (int s : host.status)
                    if (s) {
                        std::fprintf(stderr, "sandbox: some envs report a singular mass matrix (bad model)\n");
                        bad_envs_reported = 1;
                        break;
                    }
        }
        if (!stage_b) {
            frame.particles.px = px.data();
            frame.particles.py = py.data();
            frame.particles.pz = pz.data();
            frame.particles.radius = radius.data();
            frame.particles.env_id = env_id.data();
        }
        total_steps += steps;
        sim_time_since_reset += steps * physics_dt;
        frame.camera.yaw += a.orbit_speed * float(frame_dt);
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
    std::printf("sandbox: %llu frames in %.2f s (%.1f FPS), %lld physics steps (%.1f%% of real time)\n",
                static_cast<unsigned long long>(renderer.frames_drawn()), wall, frames / wall, total_steps,
                100.0 * total_steps * physics_dt / wall);
#ifdef CRUCIBLE_HAS_CUDA
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaStreamDestroy(stream);
    }
#endif
    return 0;
}

}  // namespace sandbox
