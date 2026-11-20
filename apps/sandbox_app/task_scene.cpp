// Task scene / play mode (doc 7.1): a task from forge/env/tasks/, run through the
// same Env as training, drawn with the robot's meshes (Phase 10; skeletons for robots
// without a .cmesh). With --load, a trained policy acts
// (the mean action, no noise); without it, the action is 0.
// openarm_pick_place, policies that see the cell's cameras (vision_policy.h):
//   ./sandbox_app --config ../configs/openarm_pick_place.txt --student runs/student.bin
//   ./sandbox_app --config ../configs/openarm_pick_place.txt --vla runs/oa_vla12_d1.onnx --exec 3
// openarm_stack (both arms):
//   ./sandbox_app --config ../configs/openarm_stack.txt --vla runs/st_vla.onnx --exec 4
//   ./sandbox_app --config ../configs/openarm_stack.txt --student runs/st_student.bin
// --noise (the VLA's flow noise, as vla_app): fresh (default) | zero
//
// Video (needs ffmpeg): --video out.mp4 records every drawn frame (the window's
// size: --width / --height; H.264). While recording, each frame moves the sim by
// exactly --video_speed / --video_fps seconds (default 1 / 50: one control step a
// frame, real time in the video, however slow the drawing is); --video_speed 0.5 is
// slow motion. --video_seconds N stops after N seconds of video (default 0: when the
// window closes). Without a mouse, --orbit_speed turns the camera.
//   ./sandbox_app --config ../configs/openarm_stack.txt --student runs/st_final.bin --envs 1
//       --video stack.mp4 --video_seconds 30 --width 1920 --height 1080
// and the status line counts the episodes that ended done (pick-place: cube in the
// tray, arm home; stack: the tower built, both hands up; vla_app's rules) out of all
// that ended.
//
//   ./sandbox_app --task cartpole --load runs/cartpole_.../best.bin     watch the policy
//   ./sandbox_app --task cartpole                                       no policy: the poles fall
//   ./sandbox_app --task ant --load runs/ant_.../best.bin --envs 16    walking ants
//   ./sandbox_app --task cartpole --load ... --envs 4
//   ./sandbox_app --task g1_walk --load runs/g1_walk_.../best.bin --envs 1   drive it
//   ./sandbox_app --task panda_pick --expert true --envs 4             the scripted expert picks cubes
//
// Driving (tasks with a velocity command, g1_walk), like a car. The arrow keys (or
// W / S for up / down) combine: up = walk forward, down = walk back (half speed),
// left / right = steer: alone it turns on the spot; with up it curves left / right;
// with down it backs up like a car in reverse (left: the back swings left, the face
// turns right). A / D step sideways, Q / E turn without the reverse flip, space =
// stop. The command eases to the keys (about half a second to full speed), so the
// robot starts and stops smoothly. The first key switches keyboard control on
// (command 0 while no key is held) and the camera follows env 0; R goes back to the
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
// visual-only boxes (openarm_pick_place's room floor, and its tray at each env's spot) in place of the floor tile.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <memory>
#include <vector>

#include <GLFW/glfw3.h>

#include "sandbox.h"
#include "vision_policy.h"
#include "flux/gizmo.h"
#include "flux/video_writer.h"
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

namespace {
// openarm_stack's final obs (vla_app's and collect_app's rule): red (obs 48-50) on the
// table within 2.5 cm of the spot (0.41, 0), green (55-57) on red and blue (62-64) on
// green (1.8 cm in x y, 4 cm +- 5 mm up), both TCPs up (obs 38, 44)
bool stack_built(const float* fo) {
    const float *r = fo + 48, *g = fo + 55, *b = fo + 62;
    auto on = [](const float* x, const float* u) {
        return std::hypot(x[0] - u[0], x[1] - u[1]) < 0.018f && std::fabs(x[2] - u[2] - 0.04f) < 0.005f;
    };
    return std::hypot(r[0] - 0.41f, r[1]) < 0.025f && r[2] < 0.025f && on(g, r) && on(b, g) && fo[38] > 0.12f &&
           fo[44] > 0.12f;
}
}  // namespace

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
    const std::string student_path = cfg.get_string("student", ""), vla_path = cfg.get_string("vla", "");
    const int vla_exec = cfg.get_int("exec", 1);
    const std::string vla_noise = cfg.get_string("noise", "fresh");  // --vla: fresh | zero
    const bool want_mesh = cfg.get_bool("mesh", true);
    const std::string video_path = cfg.get_string("video", "");
    const int video_fps = cfg.get_int("video_fps", 50);
    const float video_speed = cfg.get_float("video_speed", 1.0f), video_seconds = cfg.get_float("video_seconds", 0.0f);
    if (!video_path.empty() && (video_fps <= 0 || !(video_speed > 0.0f) || video_seconds < 0.0f))
        throw std::runtime_error("video: video_fps > 0, video_speed > 0, video_seconds >= 0");
    if (!video_path.empty() && !flux::VideoWriter::ffmpeg_available())
        throw std::runtime_error("--video needs the ffmpeg program (sudo apt install ffmpeg)");
    const bool skeleton_set = cfg.has("skeleton");
    const bool skeleton_val = cfg.get_bool("skeleton", false);
    warn_unused_keys(cfg);
    if (use_expert && !load.empty()) throw std::runtime_error("use --expert or --load, not both");
    const int drivers = int(use_expert) + int(!load.empty()) + int(!student_path.empty()) + int(!vla_path.empty());
    if (drivers > 1) throw std::runtime_error("use one of --expert, --load, --student, --vla");
    if (!student_path.empty() && eo.task != "openarm_pick_place" && eo.task != "openarm_stack")
        throw std::runtime_error("--student is for openarm_pick_place and openarm_stack (their cameras)");
    if (!vla_path.empty() && eo.task != "openarm_pick_place" && eo.task != "openarm_stack")
        throw std::runtime_error("--vla is for openarm_pick_place and openarm_stack (their cameras)");
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
    std::unique_ptr<VisionPolicy> vision;
    if (!student_path.empty()) vision = make_student_policy(model, N, student_path, eo.task);
    if (!vla_path.empty()) vision = make_vla_policy(model, N, vla_path, vla_exec, eo.task, vla_noise);
    if (vision) std::printf("sandbox: %s acts\n", vision->describe().c_str());
    std::vector<char> new_episode(N, 1);  // vision policies: which envs just started an episode
    long long vision_ended = 0, vision_done = 0;

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
    flux::VideoWriter video;  // opened at the first captured frame (its size is the window's)
    long video_skipped = 0;   // frames of another size (the window was resized), not written
    if (!video_path.empty()) renderer.set_capture(true);

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
    // the task's visual-only boxes (openarm_pick_place: room floor, tray), every env, lit;
    // boxes that move per env (the tray: obs_xy) at their spot from the env's obs
    std::vector<flux::Quad> visual_quads;
    bool moving_visuals = false;
    for (const auto& vb : model.visuals) moving_visuals = moving_visuals || vb.obs_xy >= 0;
    auto build_visuals = [&](const float* obs, int obs_dim) {
        visual_quads.clear();
        for (int e = 0; e < N; ++e) {
            const flux::Vec3 o = flux::env_grid_offset(e, N, env_spacing);
            for (const auto& vb0 : model.visuals) {
                forge::RobotModel::VisualBox vb = vb0;
                if (vb.obs_xy >= 0) {
                    vb.center.x += obs[size_t(e) * obs_dim + vb.obs_xy];
                    vb.center.y += obs[size_t(e) * obs_dim + vb.obs_xy + 1];
                }
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
    };
    {
        forge::Env::HostCopy h0;
        env->download(h0);
        build_visuals(h0.obs.data(), env->obs_dim());
    }
    std::printf("sandbox: task %s, %d envs, obs %d, act %d, control dt %.3f s (physics %.4f x %d), %s\n",
                eo.task.c_str(), N, env->obs_dim(), A, eo.physics_dt * eo.decimation, eo.physics_dt, eo.decimation,
#ifdef CRUCIBLE_HAS_TEMPER
                vision ? "camera policy" : policy ? "policy acts (mean action)" : "no policy: action 0");
#else
                vision ? "camera policy" : "no policy: action 0");
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
    float drive_cmd[3] = {0.0f, 0.0f, 0.0f};    // the command sent (eases toward the keys' target)
    float drive_target[3] = {0.0f, 0.0f, 0.0f}; // what the keys ask for now
    double drive_dt = 0.02;                     // the last frame's sim time (the easing's step)
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
        std::printf("sandbox: drive like a car: arrows (or W/S) combine: up/down = forward/back, left/right = steer "
                    "(with down: like a car in reverse); A/D sideways, Q/E turn, space = stop, R = random commands\n");

    while (!window.should_close()) {
        window.poll_events();
        if (can_drive) {
            GLFWwindow* gw = window.handle();
            auto down = [gw](int k) { return glfwGetKey(gw, k) == GLFW_PRESS; };
            const bool fwd = down(GLFW_KEY_W) || down(GLFW_KEY_UP), back = down(GLFW_KEY_S) || down(GLFW_KEY_DOWN);
            const bool left = down(GLFW_KEY_A), right = down(GLFW_KEY_D);
            const bool steer_l = down(GLFW_KEY_LEFT), steer_r = down(GLFW_KEY_RIGHT);  // car steering
            const bool turn_l = down(GLFW_KEY_Q), turn_r = down(GLFW_KEY_E);              // plain turn
            if (!driving &&
                (fwd || back || left || right || steer_l || steer_r || turn_l || turn_r || down(GLFW_KEY_SPACE))) {
                driving = true;
                std::printf("sandbox: keyboard control on\n");
            }
            if (driving && down(GLFW_KEY_R)) {
                driving = false;
                drive_sent = false;
                for (int k = 0; k < 3; ++k) drive_cmd[k] = drive_target[k] = 0.0f;  // the next drive eases from 0
                env->set_task_input(nullptr);
                std::printf("sandbox: random commands again\n");
            }
            if (driving) {
                // the keys' target; reversing (down without up) flips the steering, as a car's
                const bool reverse = back && !fwd;
                const float steer = (steer_l ? 1.0f : 0.0f) - (steer_r ? 1.0f : 0.0f);
                const float t[3] = {(fwd ? drive_vx : 0.0f) - (back ? 0.5f * drive_vx : 0.0f),
                                    (left ? drive_vy : 0.0f) - (right ? drive_vy : 0.0f),
                                    (reverse ? -steer : steer) * drive_wz +
                                        ((turn_l ? drive_wz : 0.0f) - (turn_r ? drive_wz : 0.0f))};
                const float tw = std::fmax(-drive_wz, std::fmin(drive_wz, t[2]));
                const float target[3] = {t[0], t[1], tw};
                if (target[0] != drive_target[0] || target[1] != drive_target[1] || target[2] != drive_target[2]) {
                    std::copy(target, target + 3, drive_target);
                    std::printf("sandbox: keys: vx %.2f m/s, vy %.2f m/s, wz %.2f rad/s\n", target[0], target[1],
                                target[2]);
                }
                // ease toward it: full speed in about half a second (a car's throttle, not a switch)
                const float rate[3] = {2.0f * drive_vx, 2.0f * drive_vy, 2.0f * drive_wz};  // per second
                float c[3];
                for (int k = 0; k < 3; ++k) {
                    const float step = rate[k] * float(drive_dt);
                    c[k] = drive_cmd[k] + std::fmax(-step, std::fmin(step, drive_target[k] - drive_cmd[k]));
                }
                if (!drive_sent || c[0] != drive_cmd[0] || c[1] != drive_cmd[1] || c[2] != drive_cmd[2]) {
                    std::copy(c, c + 3, drive_cmd);
                    env->set_task_input(drive_cmd);  // only when it changes (the gpu copy syncs)
                    drive_sent = true;
                }
            }
        }
        // recording: a fixed sim time per frame (the video's clock), not the wall clock
        const double frame_dt =
            video_path.empty() ? std::min(seconds_since(t_last), 0.1) : double(video_speed) / double(video_fps);
        t_last = Clock::now();
        drive_dt = frame_dt;
        const int steps = clock.steps_for(frame_dt);
        {
            FORGE_NVTX_RANGE("task step");
            for (int s = 0; s < steps; ++s) {
                if (vision) {  // cameras every decision: the state after every step, on the host
                    if (steps_total == 0 && s == 0) env->download(h);
                    vision->act(h, new_episode, hact.data());
#ifdef CRUCIBLE_HAS_TEMPER
                    if (eo.backend == forge::EnvBackend::kGpu) {
                        CUDA_CHECK(cudaMemcpy(dact.data(), hact.data(), hact.size() * sizeof(float),
                                              cudaMemcpyHostToDevice));
                        env->step(dact.data());
                    } else
#endif
                        env->step(hact.data());
                    env->download(h);
                    for (int e = 0; e < N; ++e) {
                        new_episode[e] = h.terminated[e] || h.truncated[e];
                        if (!new_episode[e]) continue;
                        ++vision_ended;
                        const float* fo = h.final_obs.data() + size_t(e) * env->obs_dim();  // vla_app's rules
                        vision_done += h.terminated[e] && (eo.task == "openarm_stack"
                                                               ? stack_built(fo)
                                                               : std::fabs(fo[36]) < 0.06f && std::fabs(fo[37]) < 0.06f &&
                                                                     fo[26] < 0.031f && fo[20] > 0.08f);
                    }
                    continue;
                }
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
                    if (!bx.drawn) continue;
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
                                                    {v.x, v.y, v.z}, {bx.rgb[0] * shade, bx.rgb[1] * shade, bx.rgb[2] * shade, 1.0f},
                                                    0.0f});
                        }
                }
            }
            frame.solid_quads = &ground_quads;
        }
        if (moving_visuals) build_visuals(h.obs.data(), env->obs_dim());
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
        const bool drawn = renderer.draw_frame(frame);
        viewer.next_frame();
        if (drawn && !video_path.empty()) {
            if (!video.is_open()) {
                video.open(video_path, renderer.captured_width(), renderer.captured_height(), video_fps,
                           renderer.captured_bgra());
                std::printf("sandbox: recording %s: %d x %d, %d fps, sim speed x%.2f\n", video_path.c_str(),
                            video.width(), video.height(), video_fps, double(video_speed));
            }
            if (renderer.captured_width() == video.width() && renderer.captured_height() == video.height())
                video.write(renderer.captured().data());
            else if (video_skipped++ == 0)
                std::fprintf(stderr, "sandbox: the window size changed: its frames are not recorded\n");
            if (video_seconds > 0.0f && video.frames() >= long(std::lround(double(video_seconds) * video_fps))) break;
        }

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
            if (vision && vision_ended > 0)
                std::snprintf(dist, sizeof(dist), "; done %lld of %lld episodes (%.1f%%)", vision_done, vision_ended,
                              100.0 * double(vision_done) / double(vision_ended));
            if (done)
                std::printf("sandbox: %.0f s: last episodes of %d envs: mean return %.1f, mean length %.0f steps%s\n",
                            seconds_since(t_start), done, r / done, l / done, dist);
            else
                std::printf("sandbox: %.0f s: no episode has ended yet%s\n", seconds_since(t_start), dist);
        }
        if (a.max_frames > 0 && renderer.frames_drawn() >= static_cast<uint64_t>(a.max_frames)) break;
    }
    renderer.wait_idle();
    if (video.is_open()) {
        const long n = video.frames();
        video.close();
        std::printf("sandbox: wrote %s: %ld frames (%.1f s of video)%s\n", video_path.c_str(), n,
                    double(n) / video_fps, video_skipped ? " (some frames skipped: the window was resized)" : "");
    }
    const double wall = seconds_since(t_start);
    std::printf("sandbox: %llu frames in %.2f s (%.1f FPS), %lld control steps (%.1f%% of real time)\n",
                static_cast<unsigned long long>(renderer.frames_drawn()), wall,
                double(std::max<uint64_t>(renderer.frames_drawn(), 1)) / wall, steps_total,
                100.0 * steps_total * eo.physics_dt * eo.decimation / wall);
    return 0;
}

}  // namespace sandbox
