// Robot scene (Phase 6): many copies of a URDF robot, drawn as skeletons, with
// the constraint solver: joint limits always, a floor for floating robots, and
// PD actuators that hold each env's start pose (on by default for floating robots).
//
//   ./sandbox_app --scene robot                                 16 hinge arms swinging into their limits
//   ./sandbox_app --scene robot --robot ../assets/robots/quadruped.urdf --floating true    quadrupeds standing
//   ./sandbox_app --scene robot --robot ../assets/robots/arm3.urdf     slider stops at its limits
//   ./sandbox_app --scene robot --kp 30                         arms held by PD control
//
// Options: --floor (default: on for floating robots), --kp --kv --force (PD
// gains and force limit; --kp 0 = no actuators), --init_spread (rad),
// --init_speed, --gravity, --physics_dt, --reset_every (s), --envs, --seed.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "sandbox.h"
#include "forge/dynamics/robot_batch_host.h"
#include "forge/dynamics/robot_work.h"
#include "forge/robots/actuators.h"
#include "forge/robots/skeleton.h"
#include "forge/robots/urdf.h"
#include "forge/solver/constraint_batch_host.h"
#include "forge/util/nvtx.h"
#ifdef CRUCIBLE_HAS_CUDA
#include "forge/dynamics/robot_gpu.h"
#include "forge/solver/solver_gpu.h"
#endif

namespace sandbox {

namespace {

// Different random start pose and velocity for every env. Joints stay inside
// their limits. A floating robot gets a small tilt and is lifted so its lowest
// collision sphere is 5 cm above the floor. ctrl (the PD targets) = the start pose.
void random_start(const forge::RobotModel& m, forge::RobotBatchHost& b, std::vector<float>& ctrl,
                  const forge::Actuators& act, float spread, float speed, bool on_floor, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    forge::RobotWork work(m);
    forge::DynamicsWork w = work.view();
    for (int e = 0; e < b.num_envs(); ++e) {
        float* q = b.env_qpos(e);
        int free_qadr = -1;
        for (int k = 0; k < m.nbody(); ++k) {
            const int a = m.qadr[k];
            if (m.jtype[k] == forge::kJointFree) {
                free_qadr = a;
                q[a + 2] = 1.0f;
                const forge::Quat r = forge::normalize(forge::Quat{0.1f * u(g), 0.1f * u(g), 0.1f * u(g), 1.0f});
                q[a + 3] = r.w;
                q[a + 4] = r.x;
                q[a + 5] = r.y;
                q[a + 6] = r.z;
            } else if (m.jtype[k] == forge::kJointRevolute || m.jtype[k] == forge::kJointPrismatic) {
                float v = (m.jtype[k] == forge::kJointRevolute ? 1.0f : 0.2f) * spread * u(g);
                if (m.joint_limited[k]) v = std::clamp(v, m.joint_lower[k] + 0.05f, m.joint_upper[k] - 0.05f);
                q[a] = v;
            }
        }
        if (on_floor && free_qadr >= 0 && !m.spheres.empty()) {
            forge::forward_kinematics(m.view(), q, w);
            float lowest = 1e9f;
            for (const auto& sp : m.spheres) {
                const forge::Vec3 c = forge::load3(w.xpos + 3 * sp.body) +
                                      forge::rotate(forge::load_quat(w.xquat + 4 * sp.body), sp.pos);
                lowest = std::min(lowest, c.z - sp.radius);
            }
            q[free_qadr + 2] += 0.05f - lowest;
        }
        for (int i = 0; i < m.nv; ++i) b.env_qvel(e)[i] = speed * u(g);
        // A free base gets a random spin but no drift, so it stays in its own cell.
        for (int k = 0; k < m.nbody(); ++k)
            if (m.jtype[k] == forge::kJointFree)
                for (int i = 0; i < 3; ++i) b.env_qvel(e)[m.dadr[k] + i] = 0.0f;
        for (int k = 0; k < act.nu(); ++k) ctrl[size_t(e) * act.nu() + k] = q[act.qadr[k]];
    }
}

}  // namespace

int run_robot_scene(Args& a) {
    forge::Config& cfg = *a.cfg;
    const std::string urdf = cfg.get_string("robot", std::string(CRUCIBLE_ASSET_DIR) + "/robots/arm.urdf");
    const bool floating = cfg.get_bool("floating", false);
    const bool floor = cfg.get_bool("floor", floating);
    const float kp = cfg.get_float("kp", floating ? 80.0f : 0.0f);
    const float kv = cfg.get_float("kv", 2.0f);
    const float force = cfg.get_float("force", 40.0f);
    const int num_envs = cfg.get_int("envs", 16);
    const int seed = cfg.get_int("seed", 1);
    const float physics_dt = cfg.get_float("physics_dt", 0.002f);  // MuJoCo's default step
    const float reset_every = cfg.get_float("reset_every", 10.0f);
    const float spread = cfg.get_float("init_spread", floating ? 0.3f : 1.0f);  // rad, random start angles
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

    forge::ConstraintOptions copt;
    copt.timestep = physics_dt;
    forge::ConstraintModel cmodel(model, copt);
    if (floor) cmodel.add_ground();
    const forge::Actuators act =
        kp > 0.0f ? forge::Actuators::position_all(model, kp, kv, -force, force) : forge::Actuators{};

    forge::RobotBatchHost initial(model, num_envs);
    forge::ConstraintBatchHost chost(cmodel, act.nu(), num_envs);
    random_start(model, initial, chost.ctrl, act, spread, speed, floor, static_cast<unsigned>(seed));
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
    std::printf("sandbox: constraints: %d joint limits, %s, %d PD actuators (kp %.0f, kv %.1f, force limit %.0f)\n",
                int(cmodel.limit_qadr.size()), floor ? "floor contacts" : "no floor", act.nu(), kp, kv, force);
    if (floor && cmodel.sphere_radius.empty())
        std::printf("sandbox: note: this robot has no collision spheres, so nothing touches the floor\n");
    if (floating && !floor && gravity != 0.0f)
        std::printf("sandbox: note: --floor false, so the floating robot falls forever\n");

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
    std::unique_ptr<forge::ConstraintModelDevice> dcmodel;
    std::unique_ptr<forge::ActuatorsDevice> dact;
    std::unique_ptr<forge::ConstraintBatchDevice> dcbatch;
    std::vector<int> status_host(num_envs);
    forge::DeviceBuffer<forge::SkeletonPoint> dskel;
    forge::DeviceBuffer<float> dpx, dpy, dpz;  // Stage A on the GPU: points before the copy
    float *out_x = nullptr, *out_y = nullptr, *out_z = nullptr;
    if (a.backend == "gpu") {
        CUDA_CHECK(cudaStreamCreate(&stream));
        dmodel = std::make_unique<forge::RobotModelDevice>(model);
        dbatch = std::make_unique<forge::RobotBatchDevice>(model, num_envs);
        dbatch->upload_state(initial);
        dcmodel = std::make_unique<forge::ConstraintModelDevice>(cmodel);
        dact = std::make_unique<forge::ActuatorsDevice>(act);
        dcbatch = std::make_unique<forge::ConstraintBatchDevice>(cmodel, act.nu(), num_envs);
        dcbatch->upload_ctrl(chost);
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
            forge::robot_step_constrained_gpu(dmodel->view(), dcmodel->view(), dact->view(), dbatch->view(),
                                              dbatch->work(), dcbatch->view(), physics_dt, steps, stream);
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
            // Now and then: did any env report a dynamics error? (a small copy)
            if (!bad_envs_reported && renderer.frames_drawn() % 120 == 60) {
                CUDA_CHECK(cudaMemcpyAsync(status_host.data(), dbatch->status.data(), num_envs * sizeof(int),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                for (int st : status_host)
                    if (st) {
                        std::fprintf(stderr, "sandbox: some envs report a dynamics error (status %d)\n", st);
                        bad_envs_reported = 1;
                        break;
                    }
            }
#endif
        } else {
            FORGE_NVTX_RANGE("robot physics");
            if (reset) host = initial;
            forge::robot_step_constrained_cpu(model.view(), cmodel.view(), act.view(), host.view(), host.work(),
                                              chost.view(), physics_dt, steps);
            forge::skeleton_points_cpu(model.view(), host.view(), host.work(), skel.data(), npts, px.data(),
                                       py.data(), pz.data());
            if (!bad_envs_reported)
                for (int s : host.status)
                    if (s) {
                        std::fprintf(stderr, "sandbox: some envs report a dynamics error (status %d)\n", s);
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
