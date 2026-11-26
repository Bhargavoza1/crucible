#pragma once
// openarm_handover (Phase 12): OpenArm v2 cell, a two-hand handover. A block lies on the
// left half of the table, where only the left arm reaches it. The left hand picks it
// at one end and carries it to the meeting point in the middle (M, 10 cm above the
// table); the right hand takes the block's other end; the left hand lets go; the right
// hand puts the block down on the place spot (a gray square on the right half, out of
// the left arm's reach); both hands go home.
//
// The block is 10 x 4 x 4 cm (not a 4 cm cube): at the meeting point both hands can
// only close along x (the IK's natural pose there, checked), so they hold the same two
// faces. On a 4 cm cube their pads would touch; on the block each hand holds one end,
// 2.5 cm from the center (grip_off), and the pads stay 2.4 cm apart.
//
// The arms, actions, control, start posture, cameras and IK are openarm_stack's (the
// right arm is the left's mirror image across y = 0; see openarm_stack.h).
//
//   action (10): left arm 5, right arm 5, as openarm_stack's (TCP move dx dy dz = 0.02 m
//      * a[0..2], turn of the closing axis about z = 0.1 rad * a[3], both cut to
//      +-0.35; gripper a[4]: -1 open, +1 closed).
//   obs (65): joint positions (18: left arm 7, fingers 2, right arm 7, fingers 2), joint
//      velocities (18), per hand (left, right) TCP position (3), sin and cos of the turn
//      (2), finger opening (1), the block's position (3) and quaternion w x y z (4, w >=
//      0), last action (10).
//   stages (stage(); task_state keeps the highest per episode, max_stage()):
//      0 start, 1 the left hand holds the block, 2 held by the left at the meeting point
//      (center within 2 cm of M), 3 both hands hold it, 4 only the right hand holds it
//      (handed over), 5 placed, 6 done. Every stage is a state check, so any state can
//      be labeled (DAgger) and the expert decides from the state, not from time.
//   placed: the block resting on the table (center under 2.5 cm up) within 2.5 cm (x y)
//      of the place spot (0.40, -0.25), held by neither hand.
//   done: placed, both hands open and back within 2 cm of home.
//   episode: ends (terminated) when done; at most 800 steps (16 s at 50 Hz).
//   reward (sparse, as openarm_stack): once, at the done step: success +20, aligned up
//      to +10 (10 (1 - the block's x y offset from the spot / 1.2 cm)); penalties per
//      second, both arms: joint_limits, arms, table, push (the block sliding on the
//      table, not held), approach (a hand fast near the block it does not hold), jerk,
//      action_rate (openarm_stack's weights).
//   progress (dense, for RL; potential-based, w_progress 1): phi(state) by stage:
//        nobody holds        up to 1: the left hand at the nearer block end
//        left holds          1 + 2 (grasp) + up to 2 + 2 (the block's center near M: 20
//                            cm and 2 cm scales) + up to 1 (the right hand at its end)
//        both hold           10 + 2
//        right holds alone   20 + 2 + up to 2 + 2 (the block near the place spot)
//        placed              30 + up to 2 per hand (TCP home, 5 cm scale)
//      At most 34; the start is about 0.3. Dropping the block takes back what holding
//      it paid.
//   reset: both arms at the start posture (joints +-0.03 rad), fingers open; the block
//      flat on the left half (y >= 0.17, anywhere in the left arm's work area, the
//      left hand's grip point too), its long side turned within +-30 degrees of the
//      left arm's preferred grip there. Randomization: table friction 0.6-1.2, block
//      mass x0.5-2.
//   expert (scripted, true state; each arm in its arm frame, as openarm_stack's).
//     left:  the right hand holds it too   -> open (stay)
//            holding                       -> lift to M's height, carry it to M, turn
//                                             the closing axis to x, wait there closed
//            the block on its half, free   -> pick it at its near end (the block reaches
//                                             along the hand's pointing; hover, turn, down,
//                                             close: as openarm_stack)
//            else                          -> home (straight up first after letting go)
//     right: holding, the left still holds  -> wait closed
//            holding alone                 -> carry over the place spot, land slowly,
//                                             open (as openarm_stack's place)
//            the block held at M by the left -> pick its other end from above (starts
//                                             when the block is within 2 cm of M, turned
//                                             within 0.2 rad, slower than 0.25 m/s; keeps
//                                             going while the hand is away from home and
//                                             the block within 4 cm of M)
//            else                          -> home
//     Every decision is from the state. Measured on the CPU backend: 128 / 128 done
//     (tests_cpu, about 5.7 s each); collect_app's default noise (0.2): 116 / 128
//     (openarm_stack under the same settings: 62 / 74); --push_prob 0.3 --slip_prob 0.3:
//     109 / 128.
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

struct OpenArmHandoverTask {
    static constexpr const char* kName = "openarm_handover";
    static constexpr const char* kUrdf = "robots/openarm/openarm_cell_bimanual.urdf";
    static constexpr bool kFloating = false;
    static constexpr int kArm = 7, kJoints = 9, kSides = 2, kPads = 6, kMaxQ = 32, kMaxCaps = 24;
    static constexpr int kObsDim = 65, kActDim = 10, kMaxEpisodeSteps = 800;
    static constexpr int kControlHz = 50;
    // task_state: the arm joints' velocities before this step's physics (14, jerk), the
    // progress potential after the last step (1), the highest stage this episode (1)
    static constexpr int kTaskState = 16, kPhiSlot = 14, kStageSlot = 15;
    static constexpr float kHalfX = 0.05f, kHalfYZ = 0.02f;  // the block's half sizes (long side: its x)
    enum { kStart = 0, kLeftHolds, kAtMeet, kBothHold, kHanded, kPlaced, kDone, kStages };

    struct Params {
        int q[kSides][kJoints] = {}, d[kSides][kJoints] = {};
        float sgn[kJoints] = {-1, -1, -1, 1, -1, -1, -1, -1, -1};  // right joint = sgn * the mirrored left joint
        IkJoints ik;
        float start[kJoints] = {};
        int hand = 0, hand_r = 0;
        int block_body = 0, block_q = 0, block_d = 0;
        float tcp[3] = {-0.00143f, 0.0f, -0.1635f};
        float home_tcp[3] = {};
        float turn0 = 0.0f;
        // the left arm's work area (arm frame; openarm_stack's)
        float area_x[2] = {0.30f, 0.48f}, area_y[2] = {-0.07f, 0.40f};
        float hole_x = 0.36f, hole_y[2] = {0.03f, 0.24f};
        float corner_x = 0.45f, corner_y = -0.02f;
        float box_lo[3] = {0.22f, -0.12f, 0.012f}, box_hi[3] = {0.58f, 0.45f, 0.19f};
        float turn_c[3] = {-1.52f, 0.689f, 3.142f};
        float max_turn = 0.85f;
        float pick_down[2] = {0.8f, 0.8f};
        float approach_z_lo = 0.13f, approach_z_hi = 0.16f, home_lift = 0.056f;
        float move = 0.02f, turn = 0.1f, finger_open = 0.5f, max_speed = 0.35f;
        int ik_steps = 3;
        float posture_gain = 0.2f, upright_gain = 0.5f;
        // the task
        float meet[3] = {0.40f, 0.0f, 0.10f};  // M: the block's center at the handover
        float meet_turn = -1.5707963f;         // both hands' turn there (arm frame): closing axis along x
        float meet_xy = 0.02f;                 // stage 2: the center within this of M
        float spot[2] = {0.40f, -0.25f};       // the place spot (right half)
        float spot_xy = 0.025f, align_xy = 0.012f;
        float grip_off = 0.025f;   // each hand holds the block this far from its center (along the long side)
        float spawn_y = 0.17f;     // the block starts at y >= this (the right arm does not reach it)
        float spawn_yaw = 0.52f;
        float home_joint_noise = 0.03f, done_dist = 0.02f;
        // reward (openarm_stack's)
        float w_success = 20.0f, w_aligned = 10.0f;
        float w_limits = 10.0f, limit_soft = 0.95f, w_arms = 2.0f, w_table = 10.0f, table_touch = 0.002f;
        float w_push = 80.0f, w_approach = 4.0f, approach_speed = 0.15f, approach_dist = 0.06f;
        float w_jerk = 1e-5f, w_rate = 1.0f;
        float w_progress = 1.0f, phi_stage = 10.0f, phi_reach = 1.0f, phi_grasp = 2.0f, phi_carry = 2.0f,
              phi_home = 2.0f;
        float phi_reach_dist = 0.10f, phi_carry_far = 0.20f, phi_carry_near = 0.02f, phi_home_dist = 0.05f;
        int ncap = 0;
        int cap_body[kMaxCaps] = {}, cap_side[kMaxCaps] = {};
        float cap_pos[kMaxCaps][3] = {}, cap_axis[kMaxCaps][3] = {}, cap_r[kMaxCaps] = {}, cap_hl[kMaxCaps] = {};
        float lim_lo[kSides][kArm] = {}, lim_hi[kSides][kArm] = {};
        float touch = 0.002f;
        int pad_body[kSides][kPads] = {}, pad_side[kSides][kPads] = {};
        float pad_pos[kSides][kPads][3] = {}, pad_r[kSides][kPads] = {};
        // expert (openarm_stack's, plus the handover)
        float exp_hold_lo = 0.14f, exp_hold_hi = 0.30f, exp_empty = 0.12f;
        float exp_xy = 0.008f, exp_turn = 0.08f, exp_z = 0.006f, carry_z = 0.09f, carry_clear = 0.025f;
        float hover_dz = 0.06f, return_z = 0.17f, place_xy = 0.004f, release_dz = 0.009f, support_dz = 0.004f;
        float expert_speed = 0.4f, expert_return_speed = 0.25f, expert_land_speed = 0.1f;
        float slow_dist = 0.04f, slow_min = 0.1f;
        float turn_near_lo = 0.04f, turn_near_hi = 0.10f;
        float meet_ready_xy = 0.02f, meet_ready_turn = 0.2f, meet_ready_speed = 0.25f;  // the right hand starts
        float meet_going_home = 0.03f, meet_going_xy = 0.04f;  // and keeps going (away from home, the block near M)
        float give_open = 0.25f;  // the left lets go once the right's fingers are this closed on the block
        bool randomize = true;
        float friction_lo = 0.6f, friction_hi = 1.2f, mass_lo = 0.5f, mass_hi = 2.0f;
    };

    // ---- host ----
    static void urdf_options(UrdfOptions&) {}
    static std::vector<int> joint_bodies(const RobotModel& m, int side) {
        const std::string s = side == 0 ? "openarm_left_" : "openarm_right_";
        std::vector<int> j;
        for (int k = 1; k <= kArm; ++k) j.push_back(m.joint_index(s + "joint" + std::to_string(k)));
        j.push_back(m.joint_index(s + "finger_joint1"));
        j.push_back(m.joint_index(s + "finger_joint2"));
        for (int b : j)
            if (b < 0) throw std::runtime_error("openarm_handover: needs " + s + "joint1-7 and finger_joint1-2");
        return j;
    }
    static void edit_model(RobotModel& m) {
        // collision filter (as openarm_stack): the arms touch each other, not themselves
        auto side_mask = [&](int body, int& ct, int& ca) {
            const std::string& n = m.body_name[size_t(body)];
            if (n.find("openarm_left") != std::string::npos) ct = 3, ca = 4;
            if (n.find("openarm_right") != std::string::npos) ct = 5, ca = 2;
        };
        for (auto& c : m.capsules) side_mask(c.body, c.contype, c.conaffinity);
        for (auto& sp : m.spheres) side_mask(sp.body, sp.contype, sp.conaffinity);
        add_free_box(m, "block", {kHalfX, kHalfYZ, kHalfYZ}, 0.1f);
        const float rgb[3] = {0.95f, 0.55f, 0.1f};
        for (int i = 0; i < 3; ++i) m.boxes.back().rgb[i] = rgb[i];
        m.validate();
        m.dof_armature.assign(size_t(m.nv), 0.0f);
        m.dof_damping.assign(size_t(m.nv), 0.0f);
        const float arm[kJoints] = {0.016f, 0.016f, 0.030f, 0.030f, 0.002f, 0.002f, 0.002f, 0.002f, 0.002f};
        const float damp[kJoints] = {1.0f, 1.0f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.6f, 0.6f};
        for (int s = 0; s < kSides; ++s) {
            const std::vector<int> j = joint_bodies(m, s);
            for (int k = 0; k < kJoints; ++k) {
                m.dof_armature[size_t(m.dadr[j[k]])] = arm[k];
                m.dof_damping[size_t(m.dadr[j[k]])] = damp[k];
            }
        }
        m.visuals.push_back({{0.55f, 0.0f, -1.015f}, {0.75f, 0.6f, 0.01f}, {0.42f, 0.43f, 0.45f}, -1});  // room floor
        const Params p0{};
        m.visuals.push_back({{p0.spot[0], p0.spot[1], 0.0005f}, {0.06f, 0.04f, 0.0005f}, {0.55f, 0.55f, 0.58f}, -1});  // the spot
        m.name = "openarm_cell";  // viewers draw it with openarm_cell.cmesh
        m.view_target = {0.40f, 0.0f, 0.10f};
        m.view_distance = 1.15f, m.view_yaw = 0.0f, m.view_pitch = 0.55f;
    }
    static Params make_params(const RobotModel& m) {
        Params p;
        if (m.nq > kMaxQ) throw std::runtime_error("openarm_handover: nq too large");
        for (int s = 0; s < kSides; ++s) {
            const std::vector<int> j = joint_bodies(m, s);
            for (int k = 0; k < kJoints; ++k) p.q[s][k] = m.qadr[j[k]], p.d[s][k] = m.dadr[j[k]];
            for (int k = 0; k < kArm; ++k) {
                const int b = j[size_t(k)];
                const float lo = m.joint_lower[size_t(b)], hi = m.joint_upper[size_t(b)];
                const float mid = 0.5f * (lo + hi), half = 0.5f * (hi - lo) * p.limit_soft;
                p.lim_lo[s][k] = m.joint_limited[size_t(b)] ? mid - half : -1e9f;
                p.lim_hi[s][k] = m.joint_limited[size_t(b)] ? mid + half : 1e9f;
            }
        }
        p.hand = m.body_index("openarm_left_ee_base_link");
        p.hand_r = m.body_index("openarm_right_ee_base_link");
        if (p.hand < 0 || p.hand_r < 0) throw std::runtime_error("openarm_handover: needs both ee_base_links");
        p.block_body = m.body_index("block");
        if (p.block_body < 0) throw std::runtime_error("openarm_handover: needs the block (edit_model)");
        p.block_q = m.qadr[p.block_body], p.block_d = m.dadr[p.block_body];
        // start posture: openarm_stack's (the cell's home TCP raised by home_lift)
        const std::vector<int> armj = joint_bodies(m, 0);
        const float home[kArm] = {0.0f, 0.0f, 0.0f, 1.5707963f, 0.0f, 0.0f, 0.0f};
        std::vector<float> q(size_t(m.nq), 0.0f);
        for (int b = 0; b < m.nbody(); ++b)
            if (m.jtype[b] == kJointFree) q[size_t(m.qadr[b] + 3)] = 1.0f;
        for (int k = 0; k < kArm; ++k) q[size_t(p.q[0][k])] = home[k];
        Fk f;
        forward_kinematics(m.view(), q.data(), f.w);
        const Vec3 h0 = tcp_pos(p, f) + Vec3{0.0f, 0.0f, p.home_lift};
        p.ik = make_ik_joints(m, std::vector<int>(armj.begin(), armj.begin() + kArm),
                              std::vector<float>(home, home + kArm));
        for (int it = 0; it < 200; ++it) ik_to(p, m.view(), h0, p.turn0, 1, q.data(), f, true, p.pick_down[0]);
        for (int it = 0; it < 50; ++it) ik_to(p, m.view(), h0, p.turn0, 1, q.data(), f, false, p.pick_down[0]);
        for (int k = 0; k < kArm; ++k) p.start[k] = q[size_t(p.q[0][k])];
        forward_kinematics(m.view(), q.data(), f.w);
        const Vec3 hs = tcp_pos(p, f);
        p.home_tcp[0] = hs.x, p.home_tcp[1] = hs.y, p.home_tcp[2] = hs.z;
        p.start[7] = p.start[8] = p.finger_open;
        p.ik = make_ik_joints(m, std::vector<int>(armj.begin(), armj.begin() + kArm),
                              std::vector<float>(p.start, p.start + kArm));
        int np[kSides] = {0, 0};
        for (const auto& sp : m.spheres) {
            const std::string& n = m.body_name[size_t(sp.body)];
            const int s = n.find("openarm_left") != std::string::npos ? 0 : 1;
            if (np[s] == kPads) throw std::runtime_error("openarm_handover: expected 6 pads per hand");
            const int k = np[s]++;
            p.pad_body[s][k] = sp.body;
            p.pad_side[s][k] = n.find("inner") != std::string::npos ? 0 : 1;
            p.pad_pos[s][k][0] = sp.pos.x, p.pad_pos[s][k][1] = sp.pos.y, p.pad_pos[s][k][2] = sp.pos.z;
            p.pad_r[s][k] = sp.radius;
        }
        if (np[0] != kPads || np[1] != kPads) throw std::runtime_error("openarm_handover: expected 6 pads per hand");
        for (const auto& c : m.capsules) {
            const std::string& n = m.body_name[size_t(c.body)];
            const int s = n.find("openarm_left") != std::string::npos ? 0 : (n.find("openarm_right") != std::string::npos ? 1 : -1);
            bool moves = false;
            for (int b = c.body; b >= 0; b = m.parent[size_t(b)]) moves = moves || m.jtype[size_t(b)] != kJointFixed;
            if (s < 0 || !moves) continue;
            if (p.ncap == kMaxCaps) throw std::runtime_error("openarm_handover: too many capsules");
            const int k = p.ncap++;
            const Vec3 ax = rotate(c.rot, Vec3{0, 0, 1});
            p.cap_body[k] = c.body, p.cap_side[k] = s;
            p.cap_pos[k][0] = c.pos.x, p.cap_pos[k][1] = c.pos.y, p.cap_pos[k][2] = c.pos.z;
            p.cap_axis[k][0] = ax.x, p.cap_axis[k][1] = ax.y, p.cap_axis[k][2] = ax.z;
            p.cap_r[k] = c.radius, p.cap_hl[k] = c.half_length;
        }
        return p;
    }
    static Actuators make_actuators(const RobotModel& m) {
        Actuators a;
        const float kp[kJoints] = {230, 230, 190, 190, 30, 30, 30, 10, 10};
        const float kv[kJoints] = {2.7f, 2.7f, 2.2f, 2.2f, 1.5f, 1.5f, 1.5f, 0.2f, 0.2f};
        for (int s = 0; s < kSides; ++s) {
            const std::vector<int> j = joint_bodies(m, s);
            for (int k = 0; k < kJoints; ++k) a.add_position(m, m.joint_name[size_t(j[k])], kp[k], kv[k], 0.0f, 0.0f);
        }
        return a;
    }
    static void setup_constraints(ConstraintModel& c) {
        c.opt.impratio = 10.0f;
        c.add_ground();
    }

    // ---- cameras (host): openarm_stack's four: 0 head, 1 left wrist, 2 ceiling, 3 right wrist
    struct Camera {
        Vec3 eye, target, up;
        float fovy;
    };
    static constexpr int kCameras = 4;
    static void cameras(const Params& p, const ModelView& m, const float* qpos, DynamicsWork& w, Camera* out) {
        const float deg = 0.01745329f;
        out[0] = Camera{{0.223f, 0.0315f, 0.445f}, Vec3{0.223f, 0.0315f, 0.445f} + Vec3{0.46947f, 0.0f, -0.88295f},
                        {0.88295f, 0.0f, 0.46947f}, 82.0f * deg};
        out[2] = Camera{{0.245f, 0.0f, 0.795f}, {0.245f, 0.0f, -0.205f}, {1.0f, 0.0f, 0.0f}, 92.0f * deg};
        forward_kinematics(m, qpos, w);
        auto wrist = [&](int body, float side) {
            const Vec3 h = load3(w.xpos + 3 * body);
            const Mat3 R = quat_to_mat3(load_quat(w.xquat + 4 * body));
            const Vec3 eye = h + R * Vec3{0.023f, side * 0.0182f, -0.10236f};
            return Camera{eye, eye + R * Vec3{-0.22495f, -side * 0.34918f, -0.90965f},
                          R * Vec3{0.97437f, -side * 0.08062f, -0.21001f}, 92.0f * deg};
        };
        out[1] = wrist(p.hand, 1.0f);
        out[3] = wrist(p.hand_r, -1.0f);
    }

    // ---- per env: openarm_stack's arm-frame helpers ----
    struct Fk {
        float xpos[3 * kMaxBodies], xquat[4 * kMaxBodies];
        SpatialVec S[kMaxDofs];
        DynamicsWork w{xpos, xquat, S, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    };
    FORGE_HD static float ms(int side) { return side == 0 ? 1.0f : -1.0f; }
    FORGE_HD static Vec3 mirror(int side, Vec3 v) { return Vec3{v.x, ms(side) * v.y, v.z}; }
    FORGE_HD static Vec3 tcp_vec(const Params& p) { return Vec3{p.tcp[0], p.tcp[1], p.tcp[2]}; }
    FORGE_HD static Vec3 axis_for_turn(float t) { return Vec3{-sinf(t), cosf(t), 0.0f}; }
    FORGE_HD static Vec3 dir_for_turn(float t) { return Vec3{cosf(t), sinf(t), 0.0f}; }  // pointing, level
    FORGE_HD static Vec3 approach_for_turn(const Params& p, float t, float z, float down0) {
        float s = fminf(fmaxf((z - p.approach_z_lo) / (p.approach_z_hi - p.approach_z_lo), 0.0f), 1.0f);
        s = s * s * (3.0f - 2.0f * s);
        const float down = (1.0f - s) * down0;
        const float c = cosf(down);
        return Vec3{c * cosf(t), c * sinf(t), -sinf(down)};
    }
    FORGE_HD static float wrap_pi(float a) {
        while (a > 3.14159265f) a -= 6.2831853f;
        while (a < -3.14159265f) a += 6.2831853f;
        return a;
    }
    FORGE_HD static Vec3 tcp_pos(const Params& p, const Fk& f) {
        return load3(f.xpos + 3 * p.hand) + rotate(load_quat(f.xquat + 4 * p.hand), tcp_vec(p));
    }
    FORGE_HD static float hand_turn(const Params& p, const Fk& f) {
        const Vec3 a = rotate(load_quat(f.xquat + 4 * p.hand), Vec3{0, 1, 0});
        const float t = atan2f(-a.x, a.y);
        return p.turn0 + wrap_pi(t - p.turn0);
    }
    FORGE_HD static void arm_qpos(const Params& p, const EnvView& v, int e, int s, bool plan, float* qa) {
        const float* q = v.qpos(e);
        for (int i = 0; i < v.model.nq; ++i) qa[i] = q[i];
        const float* ctrl = v.ctrl + size_t(e) * v.nu;
        for (int k = 0; k < kJoints; ++k) {
            const float x = plan && k < kArm ? ctrl[s * kJoints + k] : q[p.q[s][k]];
            qa[p.q[0][k]] = s == 0 ? x : p.sgn[k] * x;
        }
    }
    FORGE_HD static Vec3 block_pos(const Params& p, const EnvView& v, int e) { return load3(v.qpos(e) + p.block_q); }
    FORGE_HD static Quat block_quat(const Params& p, const EnvView& v, int e) {
        return normalize(load_quat_wxyz(v.qpos(e) + p.block_q + 3));
    }
    // world yaw of the block's long side (its x axis)
    FORGE_HD static float block_yaw(const Params& p, const EnvView& v, int e) {
        const Vec3 a = rotate(block_quat(p, v, e), Vec3{1, 0, 0});
        return atan2f(a.y, a.x);
    }
    FORGE_HD static float opening(const Params& p, const EnvView& v, int e, int s) {
        return ms(s) * 0.5f * (v.qpos(e)[p.q[s][7]] + v.qpos(e)[p.q[s][8]]);
    }
    FORGE_HD static Vec3 clamp_ws(const Params& p, Vec3 t) {
        t.x = fminf(fmaxf(t.x, p.box_lo[0]), p.box_hi[0]);
        t.y = fminf(fmaxf(t.y, p.box_lo[1]), p.box_hi[1]);
        t.z = fminf(fmaxf(t.z, p.box_lo[2]), p.box_hi[2]);
        return t;
    }
    FORGE_HD static float turn_pref(const Params& p, float x, float y) {
        return p.turn_c[0] + p.turn_c[1] * x + p.turn_c[2] * y;
    }
    FORGE_HD static float clamp_turn(const Params& p, float x, float y, float t) {
        const float c = turn_pref(p, x, y);
        return fminf(fmaxf(t, c - p.max_turn), c + p.max_turn);
    }
    FORGE_HD static bool in_area(const Params& p, float x, float y) {  // arm frame
        if (x < p.area_x[0] || x > p.area_x[1] || y < p.area_y[0] || y > p.area_y[1]) return false;
        if (x < p.hole_x && y > p.hole_y[0] && y < p.hole_y[1]) return false;
        return !(x > p.corner_x && y < p.corner_y);
    }
    FORGE_HD static void ik_to(const Params& p, const ModelView& m, Vec3 t, float turn, int steps, float* qa, Fk& f,
                               bool pull_all = false, float down0 = 1.5707963f, const Vec3* level = nullptr) {
        IkOptions o;
        IkResult r;
        for (int k = 0; k < steps; ++k) {
            const bool pull = pull_all || k == 0 || k < steps - 2;
            o.posture_gain = pull ? p.posture_gain : 0.0f;
            forward_kinematics(m, qa, f.w);
            ik_step_axis(m, f.w, p.hand, tcp_vec(p), t, Vec3{0, 1, 0}, axis_for_turn(turn), Vec3{0, 0, -1},
                         level ? *level : approach_for_turn(p, turn, t.z, down0),
                         pull || level ? p.upright_gain : 0.0f, p.ik, qa, o, r);
        }
    }

    FORGE_HD static void apply_action(const Params& p, const EnvView& v, int e, const float* a) {
        float* ctrl = v.ctrl + size_t(e) * v.nu;
        for (int s = 0; s < kSides; ++s) {
            const float* as = a + 5 * s;
            float qa[kMaxQ];
            arm_qpos(p, v, e, s, true, qa);
            Fk f;
            forward_kinematics(v.model, qa, f.w);
            const Vec3 h = tcp_pos(p, f);
            float m[4];
            for (int k = 0; k < 4; ++k) m[k] = fminf(fmaxf(as[k], -p.max_speed), p.max_speed);
            const Vec3 t = clamp_ws(p, Vec3{h.x + p.move * m[0], h.y + p.move * ms(s) * m[1], h.z + p.move * m[2]});
            const float turn = clamp_turn(p, t.x, t.y, hand_turn(p, f) + p.turn * ms(s) * m[3]);
            Vec3 level;
            const bool hold = level_pointing(p, v, e, s, f, level);
            ik_to(p, v.model, t, turn, p.ik_steps, qa, f, false, p.pick_down[s], hold ? &level : nullptr);
            for (int k = 0; k < kArm; ++k) ctrl[s * kJoints + k] = s == 0 ? qa[p.q[0][k]] : p.sgn[k] * qa[p.q[0][k]];
            const float g = fminf(fmaxf(as[4], -1.0f), 1.0f);
            ctrl[s * kJoints + 7] = ctrl[s * kJoints + 8] = ms(s) * p.finger_open * 0.5f * (1.0f - g);
        }
        Fk f;
        forward_kinematics(v.model, v.qpos(e), f.w);
        gravity_torques(v.model, f.w, 0, v.tau(e));
        if (v.task_state)
            for (int s = 0; s < kSides; ++s)
                for (int k = 0; k < kArm; ++k) v.task_state[size_t(e) * kTaskState + s * kArm + k] = v.qvel(e)[p.d[s][k]];
    }

    // The block's grip point for a hand at turn t (arm frame: c the center, ax the long
    // side): grip_off from the center along the long side, toward the hand (the block
    // reaches from the grip point along the hand's pointing). The long side in 3D, so a
    // block tilted in the grip still gives the true point.
    FORGE_HD static Vec3 grip_point(const Params& p, Vec3 c, Vec3 ax, float t) {
        if (dot(ax, dir_for_turn(t)) < 0.0f) ax = ax * -1.0f;
        return c - ax * p.grip_off;
    }
    FORGE_HD static Vec3 block_axis(const Params& p, const EnvView& v, int e, int s) {  // arm frame
        return mirror(s, rotate(block_quat(p, v, e), Vec3{1, 0, 0}));
    }
    // The block between hand s's pads (arm frame: hand TCP hm, turn tm): its grip point
    // near the TCP. Geometry only (the expert and the IK; the reward uses the pads' contacts).
    FORGE_HD static bool between(const Params& p, Vec3 c, Vec3 ax, Vec3 hm, float tm) {
        const Vec3 g = grip_point(p, c, ax, tm);
        const Vec3 d = g - hm;
        return sqrtf(d.x * d.x + d.y * d.y) < 0.02f && d.z > -0.035f && d.z < 0.02f;
    }
    // Holding the block (fingers in the holding range, the block between the pads), the
    // pointing that keeps it level (openarm_stack's rule: the planned pointing turned by
    // the block's tilt); just after letting go above the table, low by it: the hand's
    // pitch. Arm frame. False otherwise.
    FORGE_HD static bool level_pointing(const Params& p, const EnvView& v, int e, int s, const Fk& fp, Vec3& out) {
        const float op = opening(p, v, e, s);
        float qa[kMaxQ];
        arm_qpos(p, v, e, s, false, qa);
        Fk fm;
        forward_kinematics(v.model, qa, fm.w);
        const Vec3 hm = tcp_pos(p, fm);
        const Vec3 pt = rotate(load_quat(fp.xquat + 4 * p.hand), Vec3{0, 0, -1});
        const Vec3 x = mirror(s, block_pos(p, v, e));
        if (op < p.exp_hold_lo || op > p.exp_hold_hi) {
            const Vec3 d = x - hm;
            if (x.z > kHalfYZ + 0.01f && hm.z < x.z + kHalfYZ + 0.015f && sqrtf(d.x * d.x + d.y * d.y) < 0.045f) {
                out = pt;
                return true;
            }
            return false;
        }
        if (!between(p, x, block_axis(p, v, e, s), hm, hand_turn(p, fm))) return false;
        const Mat3 R = quat_to_mat3(block_quat(p, v, e));
        Vec3 u = mirror(s, col(R, 0));
        for (int k = 1; k < 3; ++k) {
            const Vec3 c = mirror(s, col(R, k));
            if (fabsf(c.z) > fabsf(u.z)) u = c;
        }
        if (u.z < 0.0f) u = u * -1.0f;
        const Vec3 ax = cross(u, Vec3{0, 0, 1});
        const float sn = length(ax), cs = u.z;
        if (sn < 1e-6f) {
            out = pt;
            return true;
        }
        const Vec3 k = ax / sn;
        out = pt * cs + cross(k, pt) * sn + k * (dot(k, pt) * (1.0f - cs));
        return true;
    }
    FORGE_HD static void hand_state(const Params& p, const EnvView& v, int e, int s, bool plan, Vec3& h, float& turn) {
        float qa[kMaxQ];
        arm_qpos(p, v, e, s, plan, qa);
        Fk f;
        forward_kinematics(v.model, qa, f.w);
        h = mirror(s, tcp_pos(p, f));
        turn = ms(s) * hand_turn(p, f);
    }
    FORGE_HD static Vec3 home_of(const Params& p, int s) {
        return mirror(s, Vec3{p.home_tcp[0], p.home_tcp[1], p.home_tcp[2]});
    }
    FORGE_HD static Vec3 meet_of(const Params& p) { return Vec3{p.meet[0], p.meet[1], p.meet[2]}; }

    FORGE_HD static void observe(const Params& p, const EnvView& v, int e, float* o) {
        const float* q = v.qpos(e);
        const float* qd = v.qvel(e);
        for (int s = 0; s < kSides; ++s)
            for (int k = 0; k < kJoints; ++k) {
                o[s * kJoints + k] = q[p.q[s][k]];
                o[18 + s * kJoints + k] = qd[p.d[s][k]];
            }
        for (int s = 0; s < kSides; ++s) {
            Vec3 h;
            float turn;
            hand_state(p, v, e, s, false, h, turn);
            float* os = o + 36 + 6 * s;
            os[0] = h.x, os[1] = h.y, os[2] = h.z;
            os[3] = sinf(turn), os[4] = cosf(turn);
            os[5] = opening(p, v, e, s);
        }
        const Vec3 x = block_pos(p, v, e);
        Quat bq = block_quat(p, v, e);
        if (bq.w < 0.0f) bq = Quat{-bq.x, -bq.y, -bq.z, -bq.w};
        o[48] = x.x, o[49] = x.y, o[50] = x.z;
        o[51] = bq.w, o[52] = bq.x, o[53] = bq.y, o[54] = bq.z;
        for (int k = 0; k < kActDim; ++k) o[55 + k] = v.last_action[size_t(e) * kActDim + k];
    }

    // hand s (world FK of the real qpos in f) holds the block: pads of both fingers touch it
    FORGE_HD static bool grasped(const Params& p, const EnvView& v, int e, const Fk& f, int s) {
        const Vec3 x = block_pos(p, v, e);
        const Mat3 R = quat_to_mat3(block_quat(p, v, e));
        bool side[2] = {false, false};
        for (int k = 0; k < kPads; ++k) {
            const int b = p.pad_body[s][k];
            const Vec3 sc = load3(f.xpos + 3 * b) + rotate(load_quat(f.xquat + 4 * b),
                                                            Vec3{p.pad_pos[s][k][0], p.pad_pos[s][k][1], p.pad_pos[s][k][2]});
            float dist;
            Vec3 n;
            if (sphere_box(sc, p.pad_r[s][k] + p.touch, x, R, Vec3{kHalfX, kHalfYZ, kHalfYZ}, dist, n))
                side[p.pad_side[s][k]] = true;
        }
        return side[0] && side[1];
    }
    FORGE_HD static bool hand_home(const Params& p, const EnvView& v, int e, int s) {
        Vec3 h;
        float turn;
        hand_state(p, v, e, s, false, h, turn);
        return opening(p, v, e, s) > p.finger_open - 0.1f && length(h - home_of(p, s)) < p.done_dist;
    }
    FORGE_HD static bool placed(const Params& p, const EnvView& v, int e, const Fk& f) {
        const Vec3 x = block_pos(p, v, e);
        const float d = sqrtf((x.x - p.spot[0]) * (x.x - p.spot[0]) + (x.y - p.spot[1]) * (x.y - p.spot[1]));
        return d < p.spot_xy && x.z < kHalfYZ + 0.005f && !grasped(p, v, e, f, 0) && !grasped(p, v, e, f, 1);
    }
    FORGE_HD static bool done(const Params& p, const EnvView& v, int e, const Fk& f) {
        return placed(p, v, e, f) && hand_home(p, v, e, 0) && hand_home(p, v, e, 1);
    }
    // The stage of a state (header). f: forward kinematics of the env's qpos.
    FORGE_HD static int stage(const Params& p, const EnvView& v, int e, const Fk& f) {
        if (placed(p, v, e, f)) return done(p, v, e, f) ? kDone : kPlaced;
        const bool gl = grasped(p, v, e, f, 0), gr = grasped(p, v, e, f, 1);
        if (gr) return gl ? kBothHold : kHanded;
        if (!gl) return kStart;
        const Vec3 x = block_pos(p, v, e), m = meet_of(p);
        return length(x - m) < p.meet_xy ? kAtMeet : kLeftHolds;
    }
    // the highest stage of the env's episode so far (task_state; -1 without it)
    FORGE_HD static int max_stage(const EnvView& v, int e) {
        return v.task_state ? int(v.task_state[size_t(e) * kTaskState + kStageSlot]) : -1;
    }
    FORGE_HD static bool failed(const Params& p, const EnvView& v, int e) {
        Fk f;
        forward_kinematics(v.model, v.qpos(e), f.w);
        return done(p, v, e, f);
    }
    FORGE_HD static bool success(const Params& p, const EnvView& v, int e) { return failed(p, v, e); }

    // ---- reward ----
    static constexpr int kRewardTerms = 10;
    static const char* reward_term_name(int k) {
        static const char* const n[kRewardTerms] = {"success", "aligned", "joint_limits", "arms",        "table",
                                                    "push",    "approach", "jerk",       "action_rate", "progress"};
        return n[k];
    }
    // the nearer of the block's two grip points (grip_off along its long side) to h
    FORGE_HD static float end_dist(const Params& p, const EnvView& v, int e, Vec3 h) {
        const Vec3 x = block_pos(p, v, e);
        const Vec3 a = rotate(block_quat(p, v, e), Vec3{p.grip_off, 0, 0});
        return fminf(length(x + a - h), length(x - a - h));
    }
    FORGE_HD static float potential(const Params& p, const EnvView& v, int e, const Fk& f) {
        Vec3 h[kSides];
        for (int s = 0; s < kSides; ++s) {
            float turn;
            hand_state(p, v, e, s, false, h[s], turn);
        }
        const Vec3 x = block_pos(p, v, e);
        if (placed(p, v, e, f)) {
            float phi = 3.0f * p.phi_stage;
            for (int s = 0; s < kSides; ++s) phi += p.phi_home * (1.0f - tanhf(length(h[s] - home_of(p, s)) / p.phi_home_dist));
            return phi;
        }
        const bool gl = grasped(p, v, e, f, 0), gr = grasped(p, v, e, f, 1);
        if (gr && gl) return p.phi_stage + p.phi_grasp;
        if (gr) {
            const float d = length(x - Vec3{p.spot[0], p.spot[1], kHalfYZ});
            return 2.0f * p.phi_stage + p.phi_grasp + p.phi_carry * (1.0f - tanhf(d / p.phi_carry_far)) +
                   p.phi_carry * (1.0f - tanhf(d / p.phi_carry_near));
        }
        if (gl) {
            const float d = length(x - meet_of(p));
            return p.phi_reach + p.phi_grasp + p.phi_carry * (1.0f - tanhf(d / p.phi_carry_far)) +
                   p.phi_carry * (1.0f - tanhf(d / p.phi_carry_near)) +
                   p.phi_reach * (1.0f - tanhf(end_dist(p, v, e, h[1]) / p.phi_reach_dist));
        }
        return p.phi_reach * (1.0f - tanhf(end_dist(p, v, e, h[0]) / p.phi_reach_dist));
    }
    FORGE_HD static float combine_terms(const Params&, const float* t) {
        float r = 0.0f;
        for (int k = 0; k < kRewardTerms; ++k) r += t[k];
        return r;
    }
    FORGE_HD static float segment_dist2(Vec3 p1, Vec3 q1, Vec3 p2, Vec3 q2) {
        float w[4];
        seg_weights(p1, q1, p2, q2, w);
        const Vec3 d = (p1 + (q1 - p1) * w[0]) - (p2 + (q2 - p2) * w[1]);
        return dot(d, d);
    }
    FORGE_HD static void capsule_ends(const Params& p, const Fk& f, int k, Vec3& a, Vec3& b) {
        const int bd = p.cap_body[k];
        const Quat q = load_quat(f.xquat + 4 * bd);
        const Vec3 c = load3(f.xpos + 3 * bd) + rotate(q, Vec3{p.cap_pos[k][0], p.cap_pos[k][1], p.cap_pos[k][2]});
        const Vec3 ax = rotate(q, Vec3{p.cap_axis[k][0], p.cap_axis[k][1], p.cap_axis[k][2]}) * p.cap_hl[k];
        a = c - ax, b = c + ax;
    }
    FORGE_HD static void reward_terms(const Params& p, const EnvView& v, int e, const float* act, float* t) {
        const float dt = 1.0f / float(kControlHz);
        const float* q = v.qpos(e);
        const float* qd = v.qvel(e);
        Fk f;
        forward_kinematics(v.model, q, f.w);
        for (int k = 0; k < kRewardTerms; ++k) t[k] = 0.0f;
        const Vec3 x = block_pos(p, v, e);
        if (done(p, v, e, f)) {
            const float o = sqrtf((x.x - p.spot[0]) * (x.x - p.spot[0]) + (x.y - p.spot[1]) * (x.y - p.spot[1]));
            t[0] = p.w_success;
            t[1] = p.w_aligned * fmaxf(0.0f, 1.0f - o / p.align_xy);
        }
        float over = 0.0f;
        for (int s = 0; s < kSides; ++s)
            for (int k = 0; k < kArm; ++k) {
                const float j = q[p.q[s][k]];
                over += fmaxf(p.lim_lo[s][k] - j, 0.0f) + fmaxf(j - p.lim_hi[s][k], 0.0f);
            }
        t[2] = -p.w_limits * over * dt;
        bool arms = false, table = false;
        for (int i = 0; i < p.ncap; ++i) {
            Vec3 a1, b1;
            capsule_ends(p, f, i, a1, b1);
            table = table || fminf(a1.z, b1.z) - p.cap_r[i] <= p.table_touch;
            if (p.cap_side[i] != 0) continue;
            for (int j = 0; j < p.ncap && !arms; ++j) {
                if (p.cap_side[j] != 1) continue;
                Vec3 a2, b2;
                capsule_ends(p, f, j, a2, b2);
                const float rr = p.cap_r[i] + p.cap_r[j];
                arms = segment_dist2(a1, b1, a2, b2) < rr * rr;
            }
        }
        for (int s = 0; s < kSides && !table; ++s)
            for (int k = 0; k < kPads && !table; ++k) {
                const int b = p.pad_body[s][k];
                const Vec3 sc = load3(f.xpos + 3 * b) + rotate(load_quat(f.xquat + 4 * b),
                                                                Vec3{p.pad_pos[s][k][0], p.pad_pos[s][k][1], p.pad_pos[s][k][2]});
                table = sc.z - p.pad_r[s][k] <= p.table_touch;
            }
        t[3] = arms ? -p.w_arms * dt : 0.0f;
        t[4] = table ? -p.w_table * dt : 0.0f;
        const bool grip[kSides] = {grasped(p, v, e, f, 0), grasped(p, v, e, f, 1)};
        {
            const float* bv = qd + p.block_d;
            const float vxy = sqrtf(bv[0] * bv[0] + bv[1] * bv[1]);
            if (!grip[0] && !grip[1] && x.z < kHalfYZ + 0.01f) t[5] -= p.w_push * vxy * dt;
        }
        for (int s = 0; s < kSides; ++s) {
            if (grip[s]) continue;
            Vec3 h;
            float turn;
            hand_state(p, v, e, s, false, h, turn);
            if (end_dist(p, v, e, h) >= p.approach_dist) continue;
            const float lead = 0.01f;
            float qa[kMaxQ];
            arm_qpos(p, v, e, s, false, qa);
            for (int k = 0; k < kArm; ++k) qa[p.q[0][k]] += lead * (s == 0 ? 1.0f : p.sgn[k]) * qd[p.d[s][k]];
            Fk fa;
            forward_kinematics(v.model, qa, fa.w);
            const float speed = length(mirror(s, tcp_pos(p, fa)) - h) / lead;
            t[6] -= p.w_approach * fmaxf(0.0f, speed - p.approach_speed) * dt;
        }
        if (v.task_state) {
            const float* q0 = v.task_state + size_t(e) * kTaskState;
            float s2 = 0.0f;
            for (int s = 0; s < kSides; ++s)
                for (int k = 0; k < kArm; ++k) {
                    const float acc = (qd[p.d[s][k]] - q0[s * kArm + k]) / dt;
                    s2 += acc * acc;
                }
            t[7] = -p.w_jerk * s2 * dt;
        }
        const float* prev = v.prev_action + size_t(e) * kActDim;
        float r2 = 0.0f;
        for (int s = 0; s < kSides; ++s)
            for (int k = 0; k < 4; ++k) r2 += (act[5 * s + k] - prev[5 * s + k]) * (act[5 * s + k] - prev[5 * s + k]);
        t[8] = -p.w_rate * r2 * dt;
        if (v.task_state) {
            float* ts = v.task_state + size_t(e) * kTaskState;
            if (p.w_progress != 0.0f) {
                const float phi = potential(p, v, e, f);
                t[9] = p.w_progress * (phi - ts[kPhiSlot]);
                ts[kPhiSlot] = phi;
            }
            ts[kStageSlot] = fmaxf(ts[kStageSlot], float(stage(p, v, e, f)));
        }
    }
    FORGE_HD static float reward(const Params& p, const EnvView& v, int e, const float* a) {
        float t[kRewardTerms];
        reward_terms(p, v, e, a, t);
        return combine_terms(p, t);
    }

    // ---- reset ----
    FORGE_HD static void reset(const Params& p, const EnvView& v, int e, Rng& rng) {
        float* q = v.qpos(e);
        float* ctrl = v.ctrl + size_t(e) * v.nu;
        // the block: center and grip point in the left arm's area (y >= spawn_y), its long
        // side along the left hand's pointing at the preferred turn there (+- spawn_yaw)
        float bx = 0.40f, by = 0.30f, yaw = turn_pref(p, 0.40f, 0.30f);
        for (int k = 0; k < 128; ++k) {
            const float cx = rng.uniform(p.area_x[0], p.area_x[1]);
            const float cy = rng.uniform(fmaxf(p.area_y[0], p.spawn_y), p.area_y[1]);
            const float t = turn_pref(p, cx, cy) + rng.uniform(-p.spawn_yaw, p.spawn_yaw);
            const Vec3 g = Vec3{cx, cy, 0.0f} - dir_for_turn(t) * p.grip_off;
            if (!in_area(p, cx, cy) || !in_area(p, g.x, g.y)) continue;
            bx = cx, by = cy, yaw = t;
            break;
        }
        float* bq = q + p.block_q;
        bq[0] = bx, bq[1] = by, bq[2] = kHalfYZ;
        bq[3] = cosf(0.5f * yaw), bq[4] = 0.0f, bq[5] = 0.0f, bq[6] = sinf(0.5f * yaw);
        for (int s = 0; s < kSides; ++s) {
            for (int k = 0; k < kArm; ++k) {
                const float j = p.start[k] + rng.uniform(-p.home_joint_noise, p.home_joint_noise);
                q[p.q[s][k]] = s == 0 ? j : p.sgn[k] * j;
            }
            q[p.q[s][7]] = q[p.q[s][8]] = ms(s) * p.finger_open;
            for (int k = 0; k < kJoints; ++k) ctrl[s * kJoints + k] = q[p.q[s][k]];
        }
        if (v.task_state) {
            Fk f;
            forward_kinematics(v.model, q, f.w);
            v.task_state[size_t(e) * kTaskState + kPhiSlot] = potential(p, v, e, f);
            v.task_state[size_t(e) * kTaskState + kStageSlot] = 0.0f;
        }
        if (!p.randomize) return;
        if (v.rand_friction) v.rand_friction[e] = rng.uniform(p.friction_lo, p.friction_hi);
        if (v.rand_body_scale) {
            float* sc = v.rand_body_scale + size_t(v.model.nbody) * e;
            for (int b = 0; b < v.model.nbody; ++b) sc[b] = 1.0f;
            sc[p.block_body] = rng.uniform(p.mass_lo, p.mass_hi);
        }
    }

    // ---- expert ----
    // one axis of the expert's move (openarm_stack's approach())
    FORGE_HD static float approach(const Params& p, float d, float cap) {
        if (p.slow_dist <= 0.0f) return fminf(fmaxf(d / p.move, -cap), cap);
        const float ad = fabsf(d);
        const float sv = fminf(fmaxf(cap * ad / p.slow_dist, fminf(cap, p.slow_min)), cap);
        return copysignf(fminf(ad / p.move, sv), d);
    }
    // The turn (arm frame) for picking the block (center c, arm-frame long-side yaw y):
    // closing across the long side, the one of the two nearest 'pref'
    FORGE_HD static float grip_turn(const Params& p, float yaw, float pref) {
        float d = wrap_pi(yaw - pref);
        d -= 3.14159265f * rintf(d / 3.14159265f);
        float want = pref + d;
        while (want > pref + p.max_turn) want -= 3.14159265f;
        while (want < pref - p.max_turn) want += 3.14159265f;
        return want;
    }
    // What one arm does (header). job: 0 pick (the block free on the table: left; held at
    // M: right), 1 hold still closed, 2 open (let go, stay), 3 carry to 'target' (world:
    // the block's center) and wait there closed, 4 carry to 'target' and put it down, -1
    // home. Arm frame inside; writes the arm's 5 world-frame actions.
    FORGE_HD static void expert_arm(const Params& p, const EnvView& v, int e, int s, int job, Vec3 target,
                                    float target_turn, float* a) {
        float qa[kMaxQ];
        arm_qpos(p, v, e, s, true, qa);
        Fk f;
        forward_kinematics(v.model, qa, f.w);
        const Vec3 h = tcp_pos(p, f);
        const float plan_turn = hand_turn(p, f);
        arm_qpos(p, v, e, s, false, qa);
        Fk fm;
        forward_kinematics(v.model, qa, fm.w);
        const Vec3 hm = tcp_pos(p, fm);
        const float turn_m = hand_turn(p, fm);
        const float open = opening(p, v, e, s);
        const Vec3 home{p.home_tcp[0], p.home_tcp[1], p.home_tcp[2]};
        const Vec3 c = mirror(s, block_pos(p, v, e));
        Vec3 goal = h;
        float grip = -1.0f, turn_goal = plan_turn, speed = p.expert_speed;
        bool landing = false;
        // just let go of the block above the table, the hand low by it: straight up first
        const bool near = c.z > kHalfYZ + 0.01f && fabsf(c.x - hm.x) < 0.045f && fabsf(c.y - hm.y) < 0.045f &&
                          hm.z < c.z + 0.05f;
        if (job == 1) {
            grip = 1.0f;
        } else if (job == 2) {
            grip = -1.0f;
        } else if (job == -1) {
            speed = p.expert_return_speed;
            const float dhome = sqrtf((h.x - home.x) * (h.x - home.x) + (h.y - home.y) * (h.y - home.y));
            if (open < p.exp_hold_hi)
                goal = h;
            else if (near)
                goal = Vec3{h.x, h.y, c.z + 0.065f};
            else if (dhome > 0.05f && h.z < p.return_z - 0.005f)
                goal = Vec3{h.x, h.y, p.return_z};
            else if (dhome > 0.05f)
                goal = Vec3{home.x, home.y, p.return_z}, turn_goal = p.turn0;
            else
                goal = home, turn_goal = p.turn0;
        } else if (job == 3 || job == 4) {  // holding: carry
            grip = 1.0f;
            const Vec3 tc = mirror(s, target);
            const Vec3 off = hm - c;
            const float ex = tc.x - c.x, ey = tc.y - c.y, dt = sqrtf(ex * ex + ey * ey);
            const float carry = job == 3 ? tc.z : fmaxf(p.carry_z, tc.z + p.carry_clear);
            if (job == 3) {
                turn_goal = target_turn;
                if (hm.z < carry - 0.02f && dt > 0.02f)
                    goal = Vec3{h.x, h.y, carry};  // lift first
                else
                    goal = Vec3{tc.x + off.x, tc.y + off.y, tc.z + off.z};
            } else {
                const float tp = turn_pref(p, h.x, h.y);
                turn_goal = fminf(fmaxf(plan_turn, tp - 0.4f), tp + 0.4f);
                landing = sqrtf(ex * ex + ey * ey) < 0.02f && c.z < tc.z + 0.025f;
                if (dt > (c.z > tc.z + 0.015f ? p.place_xy : 2.0f * p.place_xy)) {
                    if (hm.z < carry - 0.02f && dt > 0.02f)
                        goal = Vec3{h.x, h.y, carry};
                    else
                        goal = Vec3{tc.x + off.x, tc.y + off.y, carry};
                    if (dt < 0.02f && c.z < carry - 0.005f) goal.z = h.z;
                } else if (c.z > tc.z + p.release_dz && hm.z > c.z - p.support_dz) {
                    goal = Vec3{tc.x + off.x, tc.y + off.y, tc.z + p.release_dz * 0.5f + off.z};
                } else {
                    goal = h;
                    grip = -1.0f;
                }
            }
        } else {  // job 0: pick at the grip point (openarm_stack's pick)
            const float pref = turn_pref(p, c.x, c.y);
            const float want = grip_turn(p, ms(s) * block_yaw(p, v, e), pref);
            const Vec3 g = grip_point(p, c, block_axis(p, v, e, s), want);
            const float grasp_z = g.z;
            const float dxy = sqrtf((g.x - hm.x) * (g.x - hm.x) + (g.y - hm.y) * (g.y - hm.y));
            const float dturn = fabsf(wrap_pi(want - turn_m));
            const float dz = hm.z - grasp_z;
            goal = Vec3{g.x, g.y, grasp_z + p.hover_dz};
            float w = fminf(fmaxf((p.turn_near_hi - dxy) / (p.turn_near_hi - p.turn_near_lo), 0.0f), 1.0f);
            w = w * w * (3.0f - 2.0f * w);
            turn_goal = pref + w * (want - pref);
            if (open <= p.exp_empty) {
                goal = Vec3{h.x, h.y, fmaxf(h.z, grasp_z + p.hover_dz)}, turn_goal = plan_turn;
            } else if ((dxy < p.exp_xy && dturn < p.exp_turn && fabsf(dz) < p.exp_z) ||
                       (open < p.finger_open - 0.05f && fabsf(dz) < 2.0f * p.exp_z && dxy < 2.0f * p.exp_xy)) {
                goal = h, grip = 1.0f, turn_goal = plan_turn;
            } else if ((dxy < p.exp_xy && dturn < p.exp_turn) ||
                       (dz < p.hover_dz - 0.01f && dxy < 2.0f * p.exp_xy && dturn < 2.0f * p.exp_turn)) {
                goal = Vec3{g.x, g.y, grasp_z};
            }
        }
        const float sz = landing ? p.expert_land_speed : speed;
        a[0] = approach(p, goal.x - h.x, speed);
        a[1] = ms(s) * approach(p, goal.y - h.y, speed);
        a[2] = approach(p, goal.z - h.z, sz);
        a[3] = ms(s) * fminf(fmaxf(wrap_pi(turn_goal - plan_turn) / p.turn, -speed), speed);
        a[4] = grip;
    }
    // hand s holds the block by geometry (fingers in the holding range, the block between
    // the pads): the expert's test (the pads' contacts lag the closing fingers)
    FORGE_HD static bool holding(const Params& p, const EnvView& v, int e, int s) {
        const float op = opening(p, v, e, s);
        if (op < p.exp_hold_lo || op > p.exp_hold_hi) return false;
        float qa[kMaxQ];
        arm_qpos(p, v, e, s, false, qa);
        Fk fm;
        forward_kinematics(v.model, qa, fm.w);
        return between(p, mirror(s, block_pos(p, v, e)), block_axis(p, v, e, s), tcp_pos(p, fm), hand_turn(p, fm));
    }
    FORGE_HD static void expert(const Params& p, const EnvView& v, int e, float* a) {
        Fk f;
        forward_kinematics(v.model, v.qpos(e), f.w);
        const Vec3 x = block_pos(p, v, e), m = meet_of(p);
        const bool hl = holding(p, v, e, 0), hr = holding(p, v, e, 1);
        const bool gr = grasped(p, v, e, f, 1);
        const bool on_left = x.z < kHalfYZ + 0.01f && x.y > 0.05f;  // free on the left half
        // left
        int jl = -1;
        if (hl && gr && opening(p, v, e, 1) < p.give_open)
            jl = 2;  // the right hand has it: let go
        else if (hl)
            jl = 3;
        else if (on_left && !hr)
            jl = 0;
        // right
        int jr = -1;
        if (hr && hl)
            jr = 1;  // wait for the left to let go
        else if (hr)
            jr = 4;
        else if (hl) {  // the block at M, still, turned: take it
            const float* bv = v.qvel(e) + p.block_d;
            // the long side along the left hand's pointing there (meet_turn; either way round)
            const float dturn = fabsf(wrap_pi(2.0f * (block_yaw(p, v, e) - p.meet_turn))) * 0.5f;
            const bool ready = sqrtf((x.x - m.x) * (x.x - m.x) + (x.y - m.y) * (x.y - m.y)) < p.meet_ready_xy &&
                               fabsf(x.z - m.z) < p.meet_ready_xy && dturn < p.meet_ready_turn &&
                               sqrtf(bv[0] * bv[0] + bv[1] * bv[1] + bv[2] * bv[2]) < p.meet_ready_speed;
            // once on its way (the hand away from home), keep going while the block stays
            // near M: the left's small moves must not send it home (a state rule, no memory)
            Vec3 hrp;
            float tr;
            hand_state(p, v, e, 1, false, hrp, tr);
            const bool going = length(hrp - home_of(p, 1)) > p.meet_going_home && length(x - m) < p.meet_going_xy;
            if (ready || going) jr = 0;
        }
        expert_arm(p, v, e, 0, jl, m, p.meet_turn, a);
        expert_arm(p, v, e, 1, jr, Vec3{p.spot[0], p.spot[1], kHalfYZ}, 0.0f, a + 5);
    }
};

}  // namespace forge
