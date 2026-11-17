// Task scene / play mode (doc 7.1): a task from forge/env/tasks/, run through the
// same Env as training, drawn with the robot's meshes (Phase 10; skeletons for robots
// without a .cmesh). With --load, a trained policy acts
// (the mean action, no noise); without it, the action is 0.
//
//   ./sandbox_app --task cartpole --load runs/cartpole_.../best.bin     watch the policy
//   ./sandbox_app --task cartpole                                       no policy: the poles fall
//   ./sandbox_app --task ant --load runs/ant_.../best.bin --envs 16    walking ants
//   ./sandbox_app --task cartpole --load ... --envs 4
//   ./sandbox_app --task g1_walk --load runs/g1_walk_.../best.bin --envs 1   drive it
//   ./sandbox_app --task panda_pick --expert true --envs 4             the scripted expert picks cubes
//
// Driving (tasks with a velocity command, g1_walk): hold W / up arrow to walk
// forward, S / down back, A / D sideways, Q / E or left / right arrows to turn,
// space for a zero command. The first key switches keyboard control on (command
// 0 while no key is held) and the camera follows env 0; R goes back to the
// task's random commands. With --envs 1 the floor is 50 x 50 m.
//
// Terrain tasks (g1_walk_rough): a treadmill view. Each robot stays in the middle
// of its cell and the terrain around it (8 x 8 m with --envs 1) slides under it.
//
// Options: --envs (9), --seed, --print_every (s), --axes (also draw the world axes
// at each env's origin; default off), --physics_dt and --decimation
// (use the same values as in training), --drive_vx (0.8 m/s forward; back is half),
// --drive_vy (0.4 m/s), --drive_wz (0.8 rad/s), --expert (true: the task's scripted
// expert acts, for tasks that have one), --mesh (true: draw <robot>.cmesh if there
// is one), --skeleton (draw the skeleton spheres; default: only without a mesh).
// Collision boxes (panda_pick's cube) are drawn as solid boxes, and a task's
// visual-only boxes (openarm_pick_place's table, tray and post) in place of the floor tile.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include <GLFW/glfw3.h>

#include "sandbox.h"
#include "flux/gizmo.h"
#include "forge/dynamics/articulated.h"
#include "forge/dynamics/robot_batch_host.h"
#include "forge/dynamics/robot_work.h"
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
    const float drive_vx = cfg.get_float("drive_vx", 0.8f), drive_vy = cfg.get_float("drive_vy", 0.4f);
    const float drive_wz = cfg.get_float("drive_wz", 0.8f);
    const bool use_expert = cfg.get_bool("expert", false);
    const bool want_mesh = cfg.get_bool("mesh", true);
    const bool skeleton_set = cfg.has("skeleton");
    const bool skeleton_val = cfg.get_bool("skeleton", false);
    warn_unused_keys(cfg);
    if (use_expert && !load.empty()) throw std::runtime_error("use --expert or --load, not both");
    if (!load.empty() && eo.backend != forge::EnvBackend::kGpu)
        throw std::runtime_error("--load needs the gpu backend (the policy runs on the GPU)");

    auto env = forge::make_env(eo);
    const forge::RobotModel& model = env->model();
    const int N = env->num_envs(), A = env->act_dim();
    if (use_expert && !env->has_expert()) throw std::runtime_error("task '" + eo.task + "' has no scripted expert");

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
    // Free bodies drawn as boxes (the cube) get no skeleton markers (they would stick out).
    std::vector<char> box_root(model.nbody(), 0);
    for (const auto& bx : model.boxes)
        if (model.parent[bx.body] < 0 && model.jtype[bx.body] == forge::kJointFree) box_root[bx.body] = 1;
    std::vector<forge::SkeletonPoint> skel;
    for (const auto& pt : forge::make_skeleton(model, 6, 2.4f))
        if (!box_root[pt.body]) skel.push_back(pt);
    const int npts = static_cast<int>(skel.size()), total = N * npts;
    const float extent = forge::skeleton_extent(model, skel);
    forge::RobotBatchHost draw(model, N);  // qpos copied here each frame; FK on the CPU (Stage A)
    forge::RobotWork box_work(model);         // FK for drawing collision boxes
    forge::DynamicsWork box_w = box_work.view();
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

    float env_spacing = 1.3f * extent + 0.3f;
    for (const auto& vb : model.visuals)  // a task's drawn scene (openarm_pick_place's cell floor) fits in a cell
        env_spacing = std::max(env_spacing, 2.0f * std::max(std::fabs(vb.center.x) + vb.half.x,
                                                            std::fabs(vb.center.y) + vb.half.y) + 0.1f);
    MeshView meshes;
    if (want_mesh) meshes.load(find_cmesh(eo.asset_dir, model.name), model, renderer, N);
    const bool show_skeleton = skeleton_set ? skeleton_val : !meshes.active();
    {
        flux::SceneQuads quads;
        for (int e = 0; e < N; ++e) {
            const flux::Vec3 o = flux::env_grid_offset(e, N, env_spacing);
            const float h = N == 1 ? 25.0f : 0.5f * env_spacing - 0.1f;  // one env: room to walk
            if (env->terrain().h.empty() && model.visuals.empty())  // terrain tasks draw their ground every
                                                                     // frame; tasks with visuals (a table) their own
                quads.before.push_back({{o.x, o.y, 0.0f}, {h, 0, 0}, {0, h, 0}, {0.42f, 0.43f, 0.45f, 1.0f}, 0.25f});
            if (show_axes) flux::add_axes_gizmo(quads, {o.x, o.y, 0.0f}, 0.3f * env_spacing);  // at each env's world origin
        }
        renderer.set_scene(quads);
    }
    // the task's visual-only boxes (openarm_pick_place: table, tray, post), every env, lit
    std::vector<flux::Quad> visual_quads;
    for (int e = 0; e < N; ++e) {
        const flux::Vec3 o = flux::env_grid_offset(e, N, env_spacing);
        for (const auto& vb : model.visuals) {
            const float hs[3] = {vb.half.x, vb.half.y, vb.half.z};
            for (int ax = 0; ax < 3; ++ax)
                for (int sg = -1; sg <= 1; sg += 2) {
                    float n[3] = {0, 0, 0}, u[3] = {0, 0, 0}, v[3] = {0, 0, 0};
                    n[ax] = hs[ax] * float(sg);
                    u[(ax + 1) % 3] = hs[(ax + 1) % 3];
                    v[(ax + 2) % 3] = hs[(ax + 2) % 3];
                    visual_quads.push_back({{o.x + vb.center.x + n[0], o.y + vb.center.y + n[1], vb.center.z + n[2]},
                                            {u[0], u[1], u[2]},
                                            {v[0], v[1], v[2]},
                                            {vb.rgb[0], vb.rgb[1], vb.rgb[2], 1.0f},
                                            0.0f});
                }
        }
    }
    std::printf("sandbox: task %s, %d envs, obs %d, act %d, control dt %.3f s (physics %.4f x %d), %s\n",
                eo.task.c_str(), N, env->obs_dim(), A, eo.physics_dt * eo.decimation, eo.physics_dt, eo.decimation,
#ifdef CRUCIBLE_HAS_TEMPER
                policy ? "policy acts (mean action)" : "no policy: action 0");
#else
                "no policy: action 0");
#endif
    if (use_expert) std::printf("sandbox: the scripted expert acts\n");

    flux::FrameInput frame;
    frame.camera.target = {0.0f, 0.0f, 0.6f * extent};
    frame.camera.distance = 1.6f * extent + (N > 1 ? 0.6f * env_spacing * std::sqrt(float(N)) : 0.0f);
    if (model.view_distance > 0.0f && N == 1) {  // the task's suggested view
        frame.camera.target = {model.view_target.x, model.view_target.y, model.view_target.z};
        frame.camera.distance = model.view_distance;
        frame.camera.yaw = model.view_yaw;
        frame.camera.pitch = model.view_pitch;
    }
    frame.env_spacing = env_spacing;
    frame.color_height = 2.0f * extent;
    frame.particles.count = show_skeleton ? total : 0;
    frame.particles.num_envs = N;
    frame.scene_height = 1.5f * extent;
    meshes.attach(frame);

    StepClock clock{double(eo.physics_dt) * eo.decimation, std::max(1, a.max_steps_per_frame)};
    const auto t_start = Clock::now();
    auto t_last = t_start, t_print = t_start;
    long long steps_total = 0;
    forge::Env::HostCopy h;
    std::vector<float> last_return(N);
    std::vector<int> last_length(N);

    // Keyboard driving (see the top of this file).
    const bool can_drive = env->task_input_dim() == 3;
    bool driving = false, drive_sent = false;
    float drive_cmd[3] = {0.0f, 0.0f, 0.0f};
    int base_q = -1;
    for (int b = 0; b < model.nbody(); ++b)
        if (model.jtype[b] == forge::kJointFree) base_q = model.qadr[b];
    // Terrain: the ground patch under each robot, rebuilt every frame (treadmill view).
    const forge::Terrain& terr = env->terrain();
    const bool has_terrain = !terr.h.empty() && base_q >= 0;
    std::vector<flux::Quad> ground_quads;
    const float patch = N == 1 ? 4.0f : std::max(0.3f, 0.5f * env_spacing - 0.05f);  // half size, m
    auto build_ground = [&](const std::vector<float>& qpos) {
        ground_quads.clear();
        const int nq = model.nq, half = int(patch / terr.cell);
        const forge::Vec3 light = forge::normalize(forge::Vec3{0.35f, -0.45f, 0.82f});
        for (int e = 0; e < N; ++e) {
            const flux::Vec3 o = flux::env_grid_offset(e, N, env_spacing);
            const float bx = qpos[size_t(e) * nq + base_q], by = qpos[size_t(e) * nq + base_q + 1];
            const int ci = int(std::floor((bx - terr.x0) / terr.cell)), cj = int(std::floor((by - terr.y0) / terr.cell));
            for (int j = cj - half; j < cj + half; ++j)
                for (int i = ci - half; i < ci + half; ++i) {
                    if (i < 0 || j < 0 || i + 1 >= terr.nx || j + 1 >= terr.ny) continue;
                    const float h00 = terr.h[size_t(j) * terr.nx + i], h10 = terr.h[size_t(j) * terr.nx + i + 1];
                    const float h01 = terr.h[size_t(j + 1) * terr.nx + i], h11 = terr.h[size_t(j + 1) * terr.nx + i + 1];
                    const float c = terr.cell;
                    const forge::Vec3 u{0.5f * c, 0.0f, 0.25f * (h10 + h11 - h00 - h01)};
                    const forge::Vec3 v{0.0f, 0.5f * c, 0.25f * (h01 + h11 - h00 - h10)};
                    const forge::Vec3 n = forge::normalize(forge::cross(u, v));
                    const float lit = (0.55f + 0.45f * std::fmax(0.0f, forge::dot(n, light))) * (((i + j) & 1) ? 0.92f : 1.0f);
                    const float x = terr.x0 + (float(i) + 0.5f) * c - bx + o.x, y = terr.y0 + (float(j) + 0.5f) * c - by + o.y;
                    const float z = 0.25f * (h00 + h10 + h01 + h11);
                    ground_quads.push_back({{x, y, z}, {u.x, u.y, u.z}, {v.x, v.y, v.z},
                                            {0.30f * lit, 0.33f * lit, 0.36f * lit, 1.0f}, 0.0f});
                }
        }
    };
    if (has_terrain) std::printf("sandbox: terrain %d x %d (%.2f m cells): treadmill view, the ground slides under each robot\n",
                                 terr.nx, terr.ny, terr.cell);

    if (can_drive)
        std::printf("sandbox: drive with W/S (or up/down), A/D, Q/E (or left/right), space = stop, R = random commands\n");

    while (!window.should_close()) {
        window.poll_events();
        if (can_drive) {
            GLFWwindow* gw = window.handle();
            auto down = [gw](int k) { return glfwGetKey(gw, k) == GLFW_PRESS; };
            const bool fwd = down(GLFW_KEY_W) || down(GLFW_KEY_UP), back = down(GLFW_KEY_S) || down(GLFW_KEY_DOWN);
            const bool left = down(GLFW_KEY_A), right = down(GLFW_KEY_D);
            const bool turn_l = down(GLFW_KEY_Q) || down(GLFW_KEY_LEFT), turn_r = down(GLFW_KEY_E) || down(GLFW_KEY_RIGHT);
            if (!driving && (fwd || back || left || right || turn_l || turn_r || down(GLFW_KEY_SPACE))) {
                driving = true;
                std::printf("sandbox: keyboard control on\n");
            }
            if (driving && down(GLFW_KEY_R)) {
                driving = false;
                drive_sent = false;
                env->set_task_input(nullptr);
                std::printf("sandbox: random commands again\n");
            }
            if (driving) {
                const float c[3] = {(fwd ? drive_vx : 0.0f) - (back ? 0.5f * drive_vx : 0.0f),
                                    (left ? drive_vy : 0.0f) - (right ? drive_vy : 0.0f),
                                    (turn_l ? drive_wz : 0.0f) - (turn_r ? drive_wz : 0.0f)};
                if (!drive_sent || c[0] != drive_cmd[0] || c[1] != drive_cmd[1] || c[2] != drive_cmd[2]) {
                    std::copy(c, c + 3, drive_cmd);
                    env->set_task_input(drive_cmd);  // only when it changes (the gpu copy syncs)
                    drive_sent = true;
                    std::printf("sandbox: command vx %.2f m/s, vy %.2f m/s, wz %.2f rad/s\n", c[0], c[1], c[2]);
                }
            }
        }
        const double frame_dt = std::min(seconds_since(t_last), 0.1);
        t_last = Clock::now();
        const int steps = clock.steps_for(frame_dt);
        {
            FORGE_NVTX_RANGE("task step");
            for (int s = 0; s < steps; ++s) {
#ifdef CRUCIBLE_HAS_TEMPER
                if (eo.backend == forge::EnvBackend::kGpu) {
                    if (use_expert) env->expert_actions(dact.data());
                    else if (policy) policy->act_deterministic(env->obs(), dact.data(), N);
                    env->step(dact.data());
                    continue;
                }
#endif
                if (use_expert) env->expert_actions(hact.data());
                env->step(hact.data());
            }
        }
        steps_total += steps;
        env->download(h);  // syncs; qpos for drawing
        draw.qpos = h.qpos;
        forge::skeleton_points_cpu(model.view(), draw.view(), draw.work(), skel.data(), npts, px.data(), py.data(),
                                   pz.data());
        if (meshes.active())
            forge::body_poses_cpu(model.view(), draw.view(), draw.work(), meshes.bodies.data(),
                                  int(meshes.bodies.size()), meshes.poses.data());
        if (has_terrain) {  // treadmill: each robot drawn in the middle of its cell
            const size_t nl = meshes.bodies.size();
            for (int e = 0; e < N; ++e) {
                const float bx = h.qpos[size_t(e) * model.nq + base_q], by = h.qpos[size_t(e) * model.nq + base_q + 1];
                for (int k = 0; k < npts; ++k) {
                    px[size_t(e) * npts + k] -= bx;
                    py[size_t(e) * npts + k] -= by;
                }
                for (size_t k = 0; k < nl; ++k) {  // pose rows: [r00 r01 r02 tx | ... ty | ... tz]
                    meshes.poses[(size_t(e) * nl + k) * 12 + 3] -= bx;
                    meshes.poses[(size_t(e) * nl + k) * 12 + 7] -= by;
                }
            }
            build_ground(h.qpos);
            frame.solid_quads = &ground_quads;
        }
        if (!model.boxes.empty()) {  // collision boxes as solid boxes (CPU FK per env, few envs)
            if (!has_terrain) ground_quads.clear();
            const int nq = model.nq;
            for (int e = 0; e < N; ++e) {
                forward_kinematics(model.view(), h.qpos.data() + size_t(e) * nq, box_w);
                const flux::Vec3 o = flux::env_grid_offset(e, N, env_spacing);
                for (const auto& bx : model.boxes) {
                    const forge::Quat qb = forge::load_quat(box_w.xquat + 4 * bx.body);
                    forge::Vec3 c = forge::load3(box_w.xpos + 3 * bx.body) + forge::rotate(qb, bx.pos);
                    if (has_terrain) {
                        c.x -= h.qpos[size_t(e) * nq + base_q];
                        c.y -= h.qpos[size_t(e) * nq + base_q + 1];
                    }
                    const forge::Mat3 R = forge::quat_to_mat3(qb * bx.rot);
                    const float hs[3] = {bx.half.x, bx.half.y, bx.half.z};
                    for (int ax = 0; ax < 3; ++ax)
                        for (int sg = -1; sg <= 1; sg += 2) {
                            const forge::Vec3 n = forge::col(R, ax) * (hs[ax] * float(sg));
                            const forge::Vec3 u = forge::col(R, (ax + 1) % 3) * hs[(ax + 1) % 3];
                            const forge::Vec3 v = forge::col(R, (ax + 2) % 3) * hs[(ax + 2) % 3];
                            const float shade = ax == 2 ? 1.0f : (ax == 0 ? 0.85f : 0.7f);
                            ground_quads.push_back({{o.x + c.x + n.x, o.y + c.y + n.y, c.z + n.z}, {u.x, u.y, u.z},
                                                    {v.x, v.y, v.z}, {0.95f * shade, 0.45f * shade, 0.1f * shade, 1.0f},
                                                    0.0f});
                        }
                }
            }
            frame.solid_quads = &ground_quads;
        }
        if (!visual_quads.empty()) {
            if (model.boxes.empty() && !has_terrain) ground_quads.clear();
            ground_quads.insert(ground_quads.end(), visual_quads.begin(), visual_quads.end());
            frame.solid_quads = &ground_quads;
        }
        frame.particles.px = px.data();
        frame.particles.py = py.data();
        frame.particles.pz = pz.data();
        frame.particles.radius = radius.data();
        frame.particles.env_id = env_id.data();
        if (driving && base_q >= 0 && !has_terrain) {  // follow env 0 (terrain: the treadmill keeps it centered)
            const flux::Vec3 o = flux::env_grid_offset(0, N, env_spacing);
            frame.camera.target.x = o.x + h.qpos[size_t(base_q)];
            frame.camera.target.y = o.y + h.qpos[size_t(base_q) + 1];
        }
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
