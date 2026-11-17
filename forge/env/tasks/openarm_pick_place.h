#pragma once
// OpenArm pick and place (Phase 10): the OpenArm v2 Cell (enactic/openarm_mujoco
// cell.xml, as enactic/openarm_mjlab's pick_place task): both arms hang from the
// lifter at the back of the table; the LEFT arm picks a 4 cm cube from the table and
// puts it into the tray, the right arm and the lifter stay at the cell's home pose
// (frozen in the URDF: assets/robots/openarm/openarm_cell.urdf, tools/openarm_cell_to_urdf.py).
// The floor plane is the table top (z = 0; the world is moved down 1.005 m). The cell
// is drawn from its meshes; the tray and the room floor are drawn only
// (RobotModel::visuals). The tray has no walls yet (they need box-box contacts,
// Phase 10b): "in the tray" = resting inside its inner square.
//
// Reach (scratch analysis, then tests/cpu/forge/env OpenArmPickPlace.WorkspaceReachable):
// the left arm cannot swing its upper arm inward (joint2 <= 0.17 rad) and the wrist
// bends +-45 degrees, so it reaches its own side of the table, with the gripper
// tilted forward: ik_step_axis sets the TCP and the closing axis (level, turned by
// 'turn'), and pulls the pointing direction toward 'forward, 30 degrees down'; the
// tilt about the closing axis follows the joints. Work area: a disc (ws_c, ws_r),
// TCP height z_lo-z_hi, turns +-max_turn.
//
//   action (5), a in [-1, 1]: TCP move dx dy dz = 0.02 m * a[0..2] per step, turn of
//      the closing axis about z = 0.1 rad * a[3], gripper a[4]: -1 open (fingers at
//      0.5 rad: a 9.5 cm gap), +1 closed (target 0). The TCP is the finger-pad center
//      at closed fingers (hand frame (-0.00143, 0, -0.1635)). The target is FK(the
//      joint targets) + the move, kept in the work area; 3 axis-IK steps.
//   control: the cell's position servos (kp 230 / 230 / 190 / 190 / 30 / 30 / 30, kv
//      2.7 / 2.7 / 2.2 / 2.2 / 1.5 / 1.5 / 1.5; fingers kp 10 / kv 0.2), forces capped
//      by the joint force ranges; gravity compensation. Armature and damping per joint
//      from the cell (joints 1-2 0.016 / 1.0, 3-4 0.030 / 0.9, 5-7 0.002 / 0.9,
//      fingers 0.002 / 0.6).
//   obs (43): joint positions (9: arm, finger 1, finger 2), joint velocities (9), TCP
//      position (3), sin and cos of the turn (2), finger opening (1: mean of the two
//      finger angles, 0 closed .. 0.79 open), cube position (3), cube quaternion
//      w x y z (4, w >= 0), cube minus TCP (3), tray center x y (2), cube minus tray
//      center x y (2), last action (5).
//   reward terms: reach 1 - tanh(10 |TCP - cube|), grasp 0.5 (both fingers touch the
//      cube), lift clamp((cube height - rest) / 0.05, 0, 1) while grasped, carry
//      2 (1 - tanh(5 |cube - tray| in x y)) while lifted, placed 5 (success),
//      action_rate -0.01 sum (a - a_prev)^2.
//   success: cube in the tray (center within the inner square minus the cube's half
//      size plus 5 mm), resting (center below 2.5 cm), not grasped.
//   episode: 400 steps (8 s at 50 Hz), no early end.
//   reset: TCP at a random spot near the work-area center, fingers open; cube anywhere
//      in its spawn disc, turned anywhere.
//   randomization: table friction 0.6-1.2, cube mass x0.5-2 (0.05-0.2 kg).
//   expert (scripted, true state, no clock; Phase 9d lessons), in order:
//     cube in the tray, not held                  -> open, up
//     holding (opening 0.14-0.30 rad, cube between the pads):
//         not above the tray: lift to carry_z first, then carry over the tray
//         above the tray: down until the cube is 4 mm above the table, then open
//     closed on nothing (opening < 0.12)          -> open, back up
//     at the grasp pose (or closing there)        -> close, hold still
//     above the cube and turned to a face         -> down, open
//     else -> 10 cm above the cube, turned to the face nearest the current turn
//   At most 0.4 of full speed.
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "forge/dynamics/articulated.h"
#include "forge/dynamics/robot_work.h"
#include "forge/env/env_view.h"
#include "forge/math/quat.h"
#include "forge/robots/actuators.h"
#include "forge/robots/ik.h"
#include "forge/robots/urdf.h"
#include "forge/solver/constraint_model.h"
#include "forge/solver/constraints.h"
#include "forge/util/rng.h"

namespace forge {

struct OpenArmPickPlaceTask {
    static constexpr const char* kName = "openarm_pick_place";
    static constexpr const char* kUrdf = "robots/openarm/openarm_cell.urdf";
    static constexpr bool kFloating = false;
    static constexpr int kArm = 7, kJoints = 9, kPads = 6, kMaxQ = 32;
    static constexpr int kObsDim = 43, kActDim = 5, kMaxEpisodeSteps = 400;
    static constexpr int kControlHz = 50;
    static constexpr float kHalf = 0.02f;  // cube half size

    struct Params {
        int q[kJoints] = {}, d[kJoints] = {};  // arm joints 1-7, finger 1, finger 2
        IkJoints ik;                           // the 7 arm joints, posture = start posture
        float start[kJoints] = {};
        int hand = 0, cube = 0, cube_q = 0;
        float tcp[3] = {-0.00143f, 0.0f, -0.1635f};  // finger-pad center (closed), hand frame
        float turn0 = -0.2f;  // closing-axis yaw at the start posture (about along y; pointing forward, +x)
        float ws_c[2] = {0.43f, 0.32f}, ws_r = 0.11f, z_lo = 0.012f, z_hi = 0.13f, max_turn = 0.4f;
        float approach_down = 0.5236f;  // preferred pointing direction: forward, 30 degrees down
        float move = 0.02f, turn = 0.1f, finger_open = 0.5f;
        int ik_steps = 3;
        float posture_gain = 0.2f, upright_gain = 0.5f;
        float tray_c[2] = {0.49f, 0.26f}, tray_half = 0.045f;  // inner square half size
        float w_reach = 1.0f, w_grasp = 0.5f, w_lift = 1.0f, w_carry = 2.0f, w_placed = 5.0f, w_action = 0.01f;
        float touch = 0.002f;
        float cube_c[2] = {0.39f, 0.37f}, cube_r = 0.04f, cube_tray_gap = 0.09f, cube_turn = 0.35f;
        float hand_r = 0.04f, hand_z_lo = 0.09f, hand_z_hi = 0.13f, hand_turn0 = 0.3f;
        int reset_ik_steps = 40;
        int pad_body[kPads] = {}, pad_side[kPads] = {};
        float pad_pos[kPads][3] = {}, pad_r[kPads] = {};
        // expert
        float exp_hold_lo = 0.14f, exp_hold_hi = 0.30f, exp_empty = 0.12f;
        float exp_xy = 0.008f, exp_turn = 0.08f, exp_z = 0.006f, carry_z = 0.11f;
        float expert_speed = 0.4f;
        bool randomize = true;
        float friction_lo = 0.6f, friction_hi = 1.2f, cube_mass_lo = 0.5f, cube_mass_hi = 2.0f;
    };

    // ---- host ----
    static void urdf_options(UrdfOptions&) {}
    static std::vector<int> joint_bodies(const RobotModel& m) {
        std::vector<int> j;
        for (int k = 1; k <= kArm; ++k) j.push_back(m.joint_index("openarm_left_joint" + std::to_string(k)));
        j.push_back(m.joint_index("openarm_left_finger_joint1"));
        j.push_back(m.joint_index("openarm_left_finger_joint2"));
        for (int b : j)
            if (b < 0)
                throw std::runtime_error("openarm_pick_place: needs openarm_left_joint1-7 and openarm_left_finger_joint1-2");
        return j;
    }
    static void edit_model(RobotModel& m) {
        add_free_box(m, "cube", {kHalf, kHalf, kHalf}, 0.1f);
        // armature / damping per joint (the cell's values)
        m.dof_armature.assign(size_t(m.nv), 0.0f);
        m.dof_damping.assign(size_t(m.nv), 0.0f);
        const float arm[kJoints] = {0.016f, 0.016f, 0.030f, 0.030f, 0.002f, 0.002f, 0.002f, 0.002f, 0.002f};
        const float damp[kJoints] = {1.0f, 1.0f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.6f, 0.6f};
        const std::vector<int> j = joint_bodies(m);
        for (int k = 0; k < kJoints; ++k) {
            m.dof_armature[size_t(m.dadr[j[k]])] = arm[k];
            m.dof_damping[size_t(m.dadr[j[k]])] = damp[k];
        }
        // drawn only: the tray (base and low walls, black as the cell's) and the room
        // floor under the cell (its cabinet stands on it, 1.005 m below the table top)
        const Params p;
        auto vis = [&](Vec3 c, Vec3 h, float r, float g, float b) { m.visuals.push_back({c, h, {r, g, b}}); };
        vis({0.4f, 0.0f, -1.015f}, {2.0f, 2.0f, 0.01f}, 0.42f, 0.43f, 0.45f);
        const float wall = 0.005f, wh = 0.02f, in = p.tray_half, k = 0.1f;
        const float cx = p.tray_c[0], cy = p.tray_c[1];
        vis({cx, cy, 0.0025f}, {in + wall, in + wall, 0.0025f}, k, k, k);
        vis({cx + in + 0.5f * wall, cy, wh}, {0.5f * wall, in + wall, wh}, k, k, k);
        vis({cx - in - 0.5f * wall, cy, wh}, {0.5f * wall, in + wall, wh}, k, k, k);
        vis({cx, cy + in + 0.5f * wall, wh}, {in, 0.5f * wall, wh}, k, k, k);
        vis({cx, cy - in - 0.5f * wall, wh}, {in, 0.5f * wall, wh}, k, k, k);
        // viewer: from the front left of the table, looking at the left arm's work area
        m.view_target = {0.40f, 0.20f, 0.10f};
        m.view_distance = 0.95f, m.view_yaw = 0.55f, m.view_pitch = 0.45f;
    }
    static Params make_params(const RobotModel& m) {
        Params p;
        if (m.nq > kMaxQ) throw std::runtime_error("openarm_pick_place: nq too large");
        const std::vector<int> j = joint_bodies(m);
        for (int k = 0; k < kJoints; ++k) {
            p.q[k] = m.qadr[j[k]];
            p.d[k] = m.dadr[j[k]];
        }
        p.hand = m.body_index("openarm_left_ee_base_link");
        p.cube = m.body_index("cube");
        if (p.hand < 0 || p.cube < 0) throw std::runtime_error("openarm_pick_place: needs openarm_left_ee_base_link and cube");
        p.cube_q = m.qadr[p.cube];
        // Start posture: from a seed found by the reach analysis, axis IK to the work
        // area center, 0.12 m up, closing axis at turn0.
        std::vector<float> q(m.nq, 0.0f);
        const float seed[kArm] = {0.61f, -0.175f, -0.544f, 2.064f, -0.341f, -0.242f, -0.387f};
        for (int k = 0; k < kArm; ++k) q[p.q[k]] = seed[k];
        for (int b = 0; b < m.nbody(); ++b)
            if (m.jtype[b] == kJointFree) q[m.qadr[b] + 3] = 1.0f;
        const std::vector<int> armj(j.begin(), j.begin() + kArm);
        p.ik = make_ik_joints(m, armj, std::vector<float>(seed, seed + kArm));
        RobotWork work(m);
        DynamicsWork w = work.view();
        IkOptions o;
        o.posture_gain = p.posture_gain;
        IkResult r;
        const Vec3 target{p.ws_c[0], p.ws_c[1], 0.12f};
        for (int it = 0; it < 200; ++it) {
            forward_kinematics(m.view(), q.data(), w);
            ik_step_axis(m.view(), w, p.hand, tcp_vec(p), target, Vec3{0, 1, 0}, axis_for_turn(p.turn0),
                         Vec3{0, 0, -1}, approach_for_turn(p, p.turn0), p.upright_gain, p.ik, q.data(), o, r);
        }
        if (r.pos_error > 1e-3f || r.rot_error > 1e-2f) throw std::runtime_error("openarm_pick_place: start IK failed");
        for (int k = 0; k < kArm; ++k) p.start[k] = q[p.q[k]];
        p.start[7] = p.start[8] = p.finger_open;
        p.ik = make_ik_joints(m, armj, std::vector<float>(p.start, p.start + kArm));
        if (static_cast<int>(m.spheres.size()) != kPads)
            throw std::runtime_error("openarm_pick_place: expected 6 finger pad spheres");
        for (int s = 0; s < kPads; ++s) {
            const auto& sp = m.spheres[size_t(s)];
            p.pad_body[s] = sp.body;
            p.pad_side[s] = m.body_name[size_t(sp.body)].find("inner") != std::string::npos ? 0 : 1;
            p.pad_pos[s][0] = sp.pos.x, p.pad_pos[s][1] = sp.pos.y, p.pad_pos[s][2] = sp.pos.z;
            p.pad_r[s] = sp.radius;
        }
        return p;
    }
    static Actuators make_actuators(const RobotModel& m) {
        Actuators a;
        const std::vector<int> j = joint_bodies(m);
        const float kp[kJoints] = {230, 230, 190, 190, 30, 30, 30, 10, 10};
        const float kv[kJoints] = {2.7f, 2.7f, 2.2f, 2.2f, 1.5f, 1.5f, 1.5f, 0.2f, 0.2f};
        for (int k = 0; k < kJoints; ++k) a.add_position(m, m.joint_name[size_t(j[k])], kp[k], kv[k], 0.0f, 0.0f);
        return a;
    }
    static void setup_constraints(ConstraintModel& c) {
        c.opt.impratio = 10.0f;  // as panda_pick: little friction creep in the grasp
        c.add_ground();
    }

    // ---- per env ----
    struct Fk {
        float xpos[3 * kMaxBodies], xquat[4 * kMaxBodies];
        SpatialVec S[kMaxDofs];
        DynamicsWork w{xpos, xquat, S, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    };
    FORGE_HD static Vec3 tcp_vec(const Params& p) { return Vec3{p.tcp[0], p.tcp[1], p.tcp[2]}; }
    // closing axis (hand y) for a turn: (-sin, cos, 0)
    FORGE_HD static Vec3 axis_for_turn(float t) { return Vec3{-sinf(t), cosf(t), 0.0f}; }
    // preferred pointing direction: across the closing axis, forward, approach_down below level
    FORGE_HD static Vec3 approach_for_turn(const Params& p, float t) {
        const float c = cosf(p.approach_down);
        return Vec3{c * cosf(t), c * sinf(t), -sinf(p.approach_down)};
    }
    FORGE_HD static float wrap_pi(float a) {
        while (a > 3.14159265f) a -= 6.2831853f;
        while (a < -3.14159265f) a += 6.2831853f;
        return a;
    }
    FORGE_HD static Vec3 tcp_pos(const Params& p, const Fk& f) {
        return load3(f.xpos + 3 * p.hand) + rotate(load_quat(f.xquat + 4 * p.hand), tcp_vec(p));
    }
    // yaw of the closing axis (the axis kept near level by the IK), near turn0
    FORGE_HD static float hand_turn(const Params& p, const Fk& f) {
        const Vec3 a = rotate(load_quat(f.xquat + 4 * p.hand), Vec3{0, 1, 0});
        const float t = atan2f(-a.x, a.y);
        return p.turn0 + wrap_pi(t - p.turn0);
    }
    FORGE_HD static void plan_qpos(const Params& p, const EnvView& v, int e, float* qp) {
        const float* q = v.qpos(e);
        for (int i = 0; i < v.model.nq; ++i) qp[i] = q[i];
        for (int k = 0; k < kArm; ++k) qp[p.q[k]] = v.ctrl[size_t(e) * v.nu + k];
    }
    FORGE_HD static Vec3 cube_pos(const Params& p, const EnvView& v, int e) { return load3(v.qpos(e) + p.cube_q); }
    FORGE_HD static float cube_yaw(const Params& p, const EnvView& v, int e) {
        const Quat c = normalize(load_quat_wxyz(v.qpos(e) + p.cube_q + 3));
        return atan2f(2.0f * (c.w * c.z + c.x * c.y), 1.0f - 2.0f * (c.y * c.y + c.z * c.z));
    }
    FORGE_HD static float opening(const Params& p, const EnvView& v, int e) {
        return 0.5f * (v.qpos(e)[p.q[7]] + v.qpos(e)[p.q[8]]);
    }
    FORGE_HD static Vec3 clamp_ws(const Params& p, Vec3 t) {
        const float dx = t.x - p.ws_c[0], dy = t.y - p.ws_c[1], r = sqrtf(dx * dx + dy * dy);
        if (r > p.ws_r) t.x = p.ws_c[0] + dx * (p.ws_r / r), t.y = p.ws_c[1] + dy * (p.ws_r / r);
        t.z = fminf(fmaxf(t.z, p.z_lo), p.z_hi);
        return t;
    }
    // 'steps' axis-IK steps toward TCP target t and turn. The posture and upright pulls
    // only in the first step: the damped null-space projection leaks a little into the
    // TCP, and the action's target is FK(the joint targets) + the move, so a leak in
    // every step made the arm creep with zero action (2 cm in 5 s); the later steps
    // remove it.
    FORGE_HD static void ik_to(const Params& p, const ModelView& m, Vec3 t, float turn, int steps, float* qp, Fk& f) {
        IkOptions o;
        IkResult r;
        for (int k = 0; k < steps; ++k) {
            const bool pull = k == 0 || k < steps - 2;
            o.posture_gain = pull ? p.posture_gain : 0.0f;
            forward_kinematics(m, qp, f.w);
            ik_step_axis(m, f.w, p.hand, tcp_vec(p), t, Vec3{0, 1, 0}, axis_for_turn(turn), Vec3{0, 0, -1},
                         approach_for_turn(p, turn), pull ? p.upright_gain : 0.0f, p.ik, qp, o, r);
        }
    }

    FORGE_HD static void apply_action(const Params& p, const EnvView& v, int e, const float* a) {
        float qp[kMaxQ];
        plan_qpos(p, v, e, qp);
        Fk f;
        forward_kinematics(v.model, qp, f.w);
        const Vec3 h = tcp_pos(p, f);
        const Vec3 t = clamp_ws(p, Vec3{h.x + p.move * a[0], h.y + p.move * a[1], h.z + p.move * a[2]});
        const float turn = fminf(fmaxf(hand_turn(p, f) + p.turn * a[3], p.turn0 - p.max_turn), p.turn0 + p.max_turn);
        ik_to(p, v.model, t, turn, p.ik_steps, qp, f);
        float* ctrl = v.ctrl + size_t(e) * v.nu;
        for (int k = 0; k < kArm; ++k) ctrl[k] = qp[p.q[k]];
        const float g = fminf(fmaxf(a[4], -1.0f), 1.0f);
        ctrl[7] = ctrl[8] = p.finger_open * 0.5f * (1.0f - g);
        forward_kinematics(v.model, v.qpos(e), f.w);
        gravity_torques(v.model, f.w, 0, v.tau(e));
    }

    FORGE_HD static void observe(const Params& p, const EnvView& v, int e, float* o) {
        const float* q = v.qpos(e);
        const float* qd = v.qvel(e);
        Fk f;
        forward_kinematics(v.model, q, f.w);
        for (int k = 0; k < kJoints; ++k) {
            o[k] = q[p.q[k]];
            o[kJoints + k] = qd[p.d[k]];
        }
        const Vec3 h = tcp_pos(p, f);
        const float turn = hand_turn(p, f);
        o[18] = h.x, o[19] = h.y, o[20] = h.z;
        o[21] = sinf(turn), o[22] = cosf(turn);
        o[23] = opening(p, v, e);
        const Vec3 c = cube_pos(p, v, e);
        Quat cq = normalize(load_quat_wxyz(q + p.cube_q + 3));
        if (cq.w < 0.0f) cq = Quat{-cq.x, -cq.y, -cq.z, -cq.w};
        o[24] = c.x, o[25] = c.y, o[26] = c.z;
        o[27] = cq.w, o[28] = cq.x, o[29] = cq.y, o[30] = cq.z;
        o[31] = c.x - h.x, o[32] = c.y - h.y, o[33] = c.z - h.z;
        o[34] = p.tray_c[0], o[35] = p.tray_c[1];
        o[36] = c.x - p.tray_c[0], o[37] = c.y - p.tray_c[1];
        for (int k = 0; k < kActDim; ++k) o[38 + k] = v.last_action[size_t(e) * kActDim + k];
    }
    FORGE_HD static bool failed(const Params&, const EnvView&, int) { return false; }

    FORGE_HD static bool grasped(const Params& p, const EnvView& v, int e, const Fk& f) {
        const Vec3 c = cube_pos(p, v, e);
        const Mat3 R = quat_to_mat3(normalize(load_quat_wxyz(v.qpos(e) + p.cube_q + 3)));
        bool side[2] = {false, false};
        for (int s = 0; s < kPads; ++s) {
            const int b = p.pad_body[s];
            const Vec3 sc = load3(f.xpos + 3 * b) +
                            rotate(load_quat(f.xquat + 4 * b), Vec3{p.pad_pos[s][0], p.pad_pos[s][1], p.pad_pos[s][2]});
            float dist;
            Vec3 n;
            if (sphere_box(sc, p.pad_r[s] + p.touch, c, R, Vec3{kHalf, kHalf, kHalf}, dist, n)) side[p.pad_side[s]] = true;
        }
        return side[0] && side[1];
    }
    FORGE_HD static bool in_tray(const Params& p, Vec3 c) {
        const float lim = p.tray_half - kHalf + 0.005f;
        return fabsf(c.x - p.tray_c[0]) < lim && fabsf(c.y - p.tray_c[1]) < lim && c.z < kHalf + 0.005f;
    }
    FORGE_HD static bool placed(const Params& p, const EnvView& v, int e, const Fk& f) {
        return in_tray(p, cube_pos(p, v, e)) && !grasped(p, v, e, f);
    }
    FORGE_HD static bool success(const Params& p, const EnvView& v, int e) {
        Fk f;
        forward_kinematics(v.model, v.qpos(e), f.w);
        return placed(p, v, e, f);
    }

    static constexpr int kRewardTerms = 6;
    static const char* reward_term_name(int k) {
        static const char* const n[kRewardTerms] = {"reach", "grasp", "lift", "carry", "placed", "action_rate"};
        return n[k];
    }
    FORGE_HD static float combine_terms(const Params&, const float* t) {
        return t[0] + t[1] + t[2] + t[3] + t[4] + t[5];
    }
    FORGE_HD static void reward_terms(const Params& p, const EnvView& v, int e, const float* a, float* t) {
        Fk f;
        forward_kinematics(v.model, v.qpos(e), f.w);
        const Vec3 c = cube_pos(p, v, e);
        const bool g = grasped(p, v, e, f);
        const float lift = fminf(fmaxf((c.z - kHalf) / 0.05f, 0.0f), 1.0f);
        const float dx = c.x - p.tray_c[0], dy = c.y - p.tray_c[1];
        t[0] = p.w_reach * (1.0f - tanhf(10.0f * length(c - tcp_pos(p, f))));
        t[1] = g ? p.w_grasp : 0.0f;
        t[2] = g ? p.w_lift * lift : 0.0f;
        t[3] = g && lift > 0.2f ? p.w_carry * (1.0f - tanhf(5.0f * sqrtf(dx * dx + dy * dy))) : 0.0f;
        t[4] = placed(p, v, e, f) ? p.w_placed : 0.0f;
        const float* prev = v.prev_action + size_t(e) * kActDim;
        float act2 = 0.0f;
        for (int k = 0; k < kActDim; ++k) act2 += (a[k] - prev[k]) * (a[k] - prev[k]);
        t[5] = -p.w_action * act2;
    }
    FORGE_HD static float reward(const Params& p, const EnvView& v, int e, const float* a) {
        float t[kRewardTerms];
        reward_terms(p, v, e, a, t);
        return combine_terms(p, t);
    }

    FORGE_HD static void reset(const Params& p, const EnvView& v, int e, Rng& rng) {
        float* q = v.qpos(e);
        float* ctrl = v.ctrl + size_t(e) * v.nu;
        // cube: in its spawn disc, away from the tray (up to 64 tries)
        float x = p.cube_c[0], y = p.cube_c[1];
        for (int tries = 0; tries < 64; ++tries) {
            const float r = p.cube_r * sqrtf(rng.uniform()), th = rng.uniform(-3.14159265f, 3.14159265f);
            const float cx = p.cube_c[0] + r * cosf(th), cy = p.cube_c[1] + r * sinf(th);
            if (fabsf(cx - p.tray_c[0]) < p.cube_tray_gap && fabsf(cy - p.tray_c[1]) < p.cube_tray_gap) continue;
            x = cx, y = cy;
            break;
        }
        // turned within turn0 +- cube_turn (plus any quarter turn: it looks the same), so
        // a pair of faces lies inside the turns the arm reaches (+-max_turn)
        const float yaw = p.turn0 + rng.uniform(-p.cube_turn, p.cube_turn) + 1.5707963f * float(int(rng.uniform() * 4.0f));
        float* c = q + p.cube_q;
        c[0] = x, c[1] = y, c[2] = kHalf;
        c[3] = cosf(0.5f * yaw), c[4] = 0.0f, c[5] = 0.0f, c[6] = sinf(0.5f * yaw);
        // the TCP at a random start spot: axis IK from the start posture
        for (int k = 0; k < kJoints; ++k) q[p.q[k]] = p.start[k];
        const float r = p.hand_r * sqrtf(rng.uniform()), th = rng.uniform(-3.14159265f, 3.14159265f);
        const Vec3 t{p.ws_c[0] + r * cosf(th), p.ws_c[1] + r * sinf(th), rng.uniform(p.hand_z_lo, p.hand_z_hi)};
        Fk f;
        ik_to(p, v.model, t, p.turn0 + rng.uniform(-p.hand_turn0, p.hand_turn0), p.reset_ik_steps, q, f);
        for (int k = 0; k < kJoints; ++k) ctrl[k] = q[p.q[k]];
        if (!p.randomize) return;
        if (v.rand_friction) v.rand_friction[e] = rng.uniform(p.friction_lo, p.friction_hi);
        if (v.rand_body_scale) {
            float* s = v.rand_body_scale + size_t(v.model.nbody) * e;
            for (int b = 0; b < v.model.nbody; ++b) s[b] = 1.0f;
            s[p.cube] = rng.uniform(p.cube_mass_lo, p.cube_mass_hi);
        }
    }

    // Scripted expert (see the top of this file): proportional moves toward the
    // phase's TCP goal, from the planned TCP (the joint targets); the phase from the
    // measured state.
    FORGE_HD static void expert(const Params& p, const EnvView& v, int e, float* a) {
        float qp[kMaxQ];
        plan_qpos(p, v, e, qp);
        Fk f;
        forward_kinematics(v.model, qp, f.w);
        const Vec3 h = tcp_pos(p, f);
        const float plan_turn = hand_turn(p, f);
        Fk fm;
        forward_kinematics(v.model, v.qpos(e), fm.w);
        const Vec3 hm = tcp_pos(p, fm);
        const float turn_m = hand_turn(p, fm);
        const float open = opening(p, v, e);
        const Vec3 c = cube_pos(p, v, e);
        const Vec3 tray{p.tray_c[0], p.tray_c[1], 0.0f};
        // closing axis across a pair of faces: the face nearest the current turn
        float d = wrap_pi(cube_yaw(p, v, e) - turn_m);
        d -= 1.5707963f * rintf(d / 1.5707963f);
        float want = turn_m + d;
        if (want > p.turn0 + p.max_turn) want -= 1.5707963f;
        if (want < p.turn0 - p.max_turn) want += 1.5707963f;
        const float grasp_z = c.z;
        const float dxy = sqrtf((c.x - hm.x) * (c.x - hm.x) + (c.y - hm.y) * (c.y - hm.y));
        const float dturn = fabsf(wrap_pi(want - turn_m));
        const float dz = hm.z - grasp_z;
        const Vec3 dpad = c - hm;
        const bool between = sqrtf(dpad.x * dpad.x + dpad.y * dpad.y) < 0.02f && dpad.z > -0.035f && dpad.z < 0.02f;
        const bool holding = open > p.exp_hold_lo && open < p.exp_hold_hi && between;
        Vec3 goal{c.x, c.y, grasp_z + 0.10f};
        float grip = -1.0f, turn_goal = want;
        if (in_tray(p, c) && !holding) {  // placed: open, up above the tray
            goal = Vec3{h.x, h.y, p.carry_z};
            turn_goal = plan_turn;
        } else if (holding) {
            grip = 1.0f;
            turn_goal = plan_turn;
            // TCP goals from where the cube sits in the hand (measured): the goal does
            // not move with the arm's lag behind its plan (a goal of plan + (tray - cube)
            // circled around the tray).
            const Vec3 off = hm - c;
            const float tx = tray.x - c.x, ty = tray.y - c.y, dtray = sqrtf(tx * tx + ty * ty);
            if (dtray > 0.01f) {
                if (hm.z < p.carry_z - 0.02f && dtray > 0.03f)
                    goal = Vec3{h.x, h.y, p.carry_z};  // lift first
                else
                    goal = Vec3{tray.x + off.x, tray.y + off.y, p.carry_z};  // carry over the tray
            } else if (c.z > kHalf + 0.006f) {
                goal = Vec3{tray.x + off.x, tray.y + off.y, kHalf + 0.004f + off.z};  // down
            } else {
                goal = h;  // low over the tray: let go
                grip = -1.0f;
            }
        } else if (open <= p.exp_empty) {  // closed on nothing: open, back up
            goal = Vec3{h.x, h.y, fmaxf(h.z, grasp_z + 0.10f)};
        } else if ((dxy < p.exp_xy && dturn < p.exp_turn && fabsf(dz) < p.exp_z) ||
                   (open < p.finger_open - 0.05f && fabsf(dz) < 2.0f * p.exp_z && dxy < 2.0f * p.exp_xy)) {
            goal = h;  // at the grasp pose (or closing there): close, hold still
            grip = 1.0f;
            turn_goal = plan_turn;
        } else if ((dxy < p.exp_xy && dturn < p.exp_turn) ||
                   (dz < 0.09f && dxy < 2.0f * p.exp_xy && dturn < 2.0f * p.exp_turn)) {
            goal = Vec3{c.x, c.y, grasp_z};  // above and turned (or on the way down): go down
        }
        const float s = p.expert_speed;
        a[0] = fminf(fmaxf((goal.x - h.x) / p.move, -s), s);
        a[1] = fminf(fmaxf((goal.y - h.y) / p.move, -s), s);
        a[2] = fminf(fmaxf((goal.z - h.z) / p.move, -s), s);
        a[3] = fminf(fmaxf(wrap_pi(turn_goal - plan_turn) / p.turn, -s), s);
        a[4] = grip;
    }
};

}  // namespace forge
