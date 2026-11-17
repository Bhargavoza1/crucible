// Grasp scene (Phase 9a): the Franka Panda and a free cube on the floor, driven by
// hand. The hand target moves with the keyboard, inverse kinematics (forge/robots/
// ik.h) turns it into arm joint targets every frame, PD actuators follow them,
// and gravity is compensated (as the real Panda's controller does). One env, CPU
// physics (the answer-key code), drawn as a skeleton plus the cube.
//
//   ./sandbox_app --scene grasp
//
// Keys: arrows = move the hand in x / y, W / S = up / down, Q / E = turn the hand,
//       G = close / open the gripper, R = start again.
// Options: --cube (half size, m, 0.025), --cube_mass (kg, 0.1), --grip_kp (finger PD
// stiffness, N/m, 200: about 5 N per finger on a 5 cm cube), --friction (1),
// --impratio (10: MuJoCo's usual value for grasping; 1 lets a held cube creep down
// ~3 mm/s, as in MuJoCo), --physics_dt (0.002).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <GLFW/glfw3.h>

#include "sandbox.h"
#include "forge/dynamics/articulated.h"
#include "forge/dynamics/robot_batch_host.h"
#include "forge/dynamics/robot_work.h"
#include "forge/robots/actuators.h"
#include "forge/robots/ik.h"
#include "forge/robots/skeleton.h"
#include "forge/robots/urdf.h"
#include "forge/solver/constraint_batch_host.h"

namespace sandbox {

int run_grasp_scene(Args& args) {
    forge::Config& cfg = *args.cfg;
    const float half = cfg.get_float("cube", 0.025f);
    const float cube_mass = cfg.get_float("cube_mass", 0.1f);
    const float grip_kp = cfg.get_float("grip_kp", 200.0f);
    const float friction = cfg.get_float("friction", 1.0f);
    const float impratio = cfg.get_float("impratio", 10.0f);  // MuJoCo's usual grasping value: less friction creep
    const float physics_dt = cfg.get_float("physics_dt", 0.002f);
    warn_unused_keys(cfg);
    if (!(half > 0.0f && cube_mass > 0.0f && physics_dt > 0.0f)) throw std::runtime_error("cube, cube_mass, physics_dt must be > 0");

    // Panda (Menagerie's armature and damping) + the cube, its own free body.
    forge::UrdfOptions uopt;
    uopt.armature = 0.1f;
    uopt.damping = 1.0f;
    forge::RobotModel model = forge::load_urdf(std::string(CRUCIBLE_ASSET_DIR) + "/robots/panda/panda.urdf", uopt);
    const int cube = forge::add_free_box(model, "cube", {half, half, half}, cube_mass);
    const int hand = model.body_index("hand");
    std::vector<int> arm;
    for (int k = 1; k <= 7; ++k) arm.push_back(model.joint_index("joint" + std::to_string(k)));
    const int finger[2] = {model.joint_index("finger_joint1"), model.joint_index("finger_joint2")};
    if (hand < 0 || finger[0] < 0 || finger[1] < 0) throw std::runtime_error("grasp scene: unexpected Panda URDF");

    forge::ConstraintOptions copt;
    copt.timestep = physics_dt;
    copt.friction = friction;
    copt.impratio = impratio;
    forge::ConstraintModel cmodel(model, copt);
    cmodel.add_ground();
    // Arm PD: kp 1000, kv 30 (stable with explicit Euler at 2 ms; Menagerie's 4500 /
    // 450 needs its implicit integrator); forces capped by the URDF effort.
    forge::Actuators act;
    for (int j : arm) act.add_position(model, model.joint_name[j], 1000.0f, 30.0f, 0.0f, 0.0f);
    for (int j : finger) act.add_position(model, model.joint_name[j], grip_kp, 10.0f, 0.0f, 0.0f);

    forge::RobotBatchHost host(model, 1);
    forge::ConstraintBatchHost chost(cmodel, act.nu(), 1);
    forge::RobotWork work(model);
    forge::DynamicsWork w = work.view();

    // Hand orientation of Menagerie's home pose (pointing down), turned by 'yaw'.
    const float home[7] = {0.0f, 0.0f, 0.0f, -1.57079f, 0.0f, 1.57079f, -0.7853f};
    std::vector<float> plan(model.nq, 0.0f);  // IK state: the joint targets (not the sagging physics state)
    for (int k = 0; k < 7; ++k) plan[model.qadr[arm[k]]] = home[k];
    forward_kinematics(model.view(), plan.data(), w);
    const forge::Quat down = forge::load_quat(w.xquat + 4 * hand);
    const float pad_z = 0.0584f + 0.0445f;  // finger pad center below the hand origin
    forge::Vec3 target{0.5f, 0.0f, 0.25f};
    float yaw = 0.0f;
    bool closed = false;
    auto target_rot = [&]() { return forge::quat_from_axis_angle({0, 0, 1}, yaw) * down; };

    auto reset = [&]() {
        target = {0.5f, 0.0f, 0.25f};
        yaw = 0.0f;
        closed = false;
        for (int k = 0; k < 7; ++k) plan[model.qadr[arm[k]]] = home[k];
        forge::solve_ik(model, hand, target, target_rot(), arm, plan);
        float* q = host.env_qpos(0);
        std::fill(q, q + model.nq, 0.0f);
        std::fill(host.env_qvel(0), host.env_qvel(0) + model.nv, 0.0f);
        for (int j : arm) q[model.qadr[j]] = plan[model.qadr[j]];
        for (int j : finger) q[model.qadr[j]] = 0.04f;
        const int c = model.qadr[cube];  // on the floor under the hand, faces lined up with the fingers
        forward_kinematics(model.view(), q, w);
        const forge::Vec3 hp = forge::load3(w.xpos + 3 * hand);
        const forge::Mat3 R = forge::quat_to_mat3(forge::load_quat(w.xquat + 4 * hand));
        const float cy = std::atan2(R(1, 0), R(0, 0));
        q[c + 0] = hp.x;
        q[c + 1] = hp.y;
        q[c + 2] = half;
        q[c + 3] = std::cos(0.5f * cy);
        q[c + 6] = std::sin(0.5f * cy);
    };
    reset();

    // Drawing: the skeleton of the Panda (not of the cube) and the cube as 6 quads.
    std::vector<forge::SkeletonPoint> skel;
    for (const auto& p : forge::make_skeleton(model))
        if (p.body != cube) skel.push_back(p);
    const int npts = static_cast<int>(skel.size());
    std::vector<float> px(npts), py(npts), pz(npts), radius(npts);
    std::vector<int> env_id(npts, 0);
    for (int k = 0; k < npts; ++k) radius[k] = skel[k].radius;

    Args a = args;
    a.backend = "cpu";  // CPU physics: draw through the host (Stage A)
    flux::Window window(a.width, a.height, "crucible sandbox: grasp");
    Viewer viewer;
    make_viewer(viewer, window, a, npts);
    flux::Renderer& renderer = *viewer.renderer;
    {
        flux::SceneQuads quads;
        quads.before.push_back({{0.3f, 0.0f, 0.0f}, {1.0f, 0, 0}, {0, 1.0f, 0}, {0.42f, 0.43f, 0.45f, 1.0f}, 0.1f});
        renderer.set_scene(quads);
    }
    std::vector<flux::Quad> cube_quads;

    flux::FrameInput frame;
    frame.camera.target = {0.45f, 0.0f, 0.15f};
    frame.camera.distance = 1.1f;
    frame.env_spacing = 4.0f;
    frame.color_height = 1.2f;
    frame.particles.count = npts;
    frame.particles.num_envs = 1;

    std::printf("sandbox: scene grasp: Panda (nq %d with the cube, nv %d), cube %.0f mm, %.2f kg, friction %.2f, "
                "impratio %.0f\n", model.nq, model.nv, 2000.0f * half, cube_mass, friction, impratio);
    std::printf("sandbox: %d constraint rows at most, physics dt %.4f s (CPU)\n", cmodel.max_rows(), physics_dt);
    std::printf("sandbox: keys: arrows = move hand x / y, W / S = up / down, Q / E = turn, G = grip, R = reset\n");

    StepClock clock{physics_dt, std::max(a.max_steps_per_frame, int(0.05f / physics_dt))};
    const auto t_start = Clock::now();
    auto t_last = t_start, t_print = t_start;
    long long total_steps = 0;
    bool g_was_down = false, r_was_down = false;
    int bad = 0;
    while (!window.should_close()) {
        window.poll_events();
        const double frame_dt = std::min(seconds_since(t_last), 0.1);
        t_last = Clock::now();
        GLFWwindow* gw = window.handle();
        auto down_key = [gw](int k) { return glfwGetKey(gw, k) == GLFW_PRESS; };
        const float v = 0.2f * float(frame_dt), turn = 1.0f * float(frame_dt);
        if (down_key(GLFW_KEY_UP)) target.x += v;
        if (down_key(GLFW_KEY_DOWN)) target.x -= v;
        if (down_key(GLFW_KEY_LEFT)) target.y += v;
        if (down_key(GLFW_KEY_RIGHT)) target.y -= v;
        if (down_key(GLFW_KEY_W)) target.z += v;
        if (down_key(GLFW_KEY_S)) target.z -= v;
        if (down_key(GLFW_KEY_Q)) yaw += turn;
        if (down_key(GLFW_KEY_E)) yaw -= turn;
        target.z = std::max(target.z, pad_z + 0.01f);  // lowest pad sphere (8.5 mm below the pad center) clear of the floor
        const bool g = down_key(GLFW_KEY_G), r = down_key(GLFW_KEY_R);
        if (g && !g_was_down) {
            closed = !closed;
            std::printf("sandbox: gripper %s\n", closed ? "closing" : "opening");
        }
        if (r && !r_was_down) {
            reset();
            std::printf("sandbox: reset\n");
        }
        g_was_down = g;
        r_was_down = r;

        // IK from the last plan (a few iterations per frame are enough while moving).
        forge::IkOptions ik;
        ik.max_iterations = 30;
        forge::solve_ik(model, hand, target, target_rot(), arm, plan, ik);
        for (int k = 0; k < 7; ++k) chost.ctrl[k] = plan[model.qadr[arm[k]]];
        chost.ctrl[7] = chost.ctrl[8] = closed ? 0.0f : 0.04f;

        // Gravity compensation for the arm and fingers: bias forces at zero velocity.
        {
            std::vector<float> zero(model.nv, 0.0f);
            forward_kinematics(model.view(), host.env_qpos(0), w);
            body_inertias(model.view(), w);
            bias_forces(model.view(), zero.data(), w);
            for (int i = 0; i < model.nv; ++i) host.tau[i] = model.dof_body[i] == cube ? 0.0f : w.bias[i];
        }
        const int steps = clock.steps_for(frame_dt);
        forge::robot_step_constrained_cpu(model.view(), cmodel.view(), act.view(), host.view(), host.work(),
                                          chost.view(), physics_dt, steps);
        total_steps += steps;
        if (host.status[0] && !bad) {
            std::fprintf(stderr, "sandbox: dynamics error (status %d); press R\n", host.status[0]);
            bad = 1;
        }

        // Draw.
        forward_kinematics(model.view(), host.env_qpos(0), w);
        forge::skeleton_world(skel.data(), npts, w.xpos, w.xquat, px.data(), py.data(), pz.data());
        {
            const forge::Vec3 c = forge::load3(w.xpos + 3 * cube);
            const forge::Mat3 R = forge::quat_to_mat3(forge::load_quat(w.xquat + 4 * cube));
            cube_quads.clear();
            for (int ax = 0; ax < 3; ++ax)
                for (int s = -1; s <= 1; s += 2) {
                    const forge::Vec3 n = forge::col(R, ax) * (half * float(s));
                    const forge::Vec3 u = forge::col(R, (ax + 1) % 3) * half, vv = forge::col(R, (ax + 2) % 3) * half;
                    const float shade = ax == 2 ? 1.0f : (ax == 0 ? 0.85f : 0.7f);
                    cube_quads.push_back({{c.x + n.x, c.y + n.y, c.z + n.z}, {u.x, u.y, u.z}, {vv.x, vv.y, vv.z},
                                          {0.95f * shade, 0.45f * shade, 0.1f * shade, 1.0f}, 0.0f});
                }
        }
        frame.solid_quads = &cube_quads;
        frame.particles.px = px.data();
        frame.particles.py = py.data();
        frame.particles.pz = pz.data();
        frame.particles.radius = radius.data();
        frame.particles.env_id = env_id.data();
        viewer.update_camera(window, frame.camera, a.orbit_speed, frame_dt);
        renderer.draw_frame(frame);
        viewer.next_frame();

        if (seconds_since(t_print) >= 1.0) {
            t_print = Clock::now();
            const float* q = host.env_qpos(0);
            std::printf("sandbox: hand target (%.2f %.2f %.2f), gripper %s, fingers %.4f %.4f m, cube height %.3f m\n",
                        target.x, target.y, target.z, closed ? "closed" : "open", q[model.qadr[finger[0]]],
                        q[model.qadr[finger[1]]], q[model.qadr[cube] + 2]);
        }
        if (a.max_frames > 0 && renderer.frames_drawn() >= static_cast<uint64_t>(a.max_frames)) break;
    }
    renderer.wait_idle();
    const double wall = seconds_since(t_start);
    std::printf("sandbox: %llu frames in %.2f s, %lld physics steps (%.1f%% of real time)\n",
                static_cast<unsigned long long>(renderer.frames_drawn()), wall, total_steps,
                100.0 * total_steps * physics_dt / wall);
    return 0;
}

}  // namespace sandbox
