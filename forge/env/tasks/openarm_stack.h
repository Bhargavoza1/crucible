#pragma once
// openarm_stack (Phase 11): OpenArm v2 cell, both arms build a 3-cube tower on a spot
// in front of the robot: red on the spot, green on red, blue on green. The real cell
// (openarm_cell_bimanual.urdf: both arms hang from the lifter at the back of the
// table, z = 0 is the table top).
//
// The cubes start anywhere the arms reach, each on one arm's half of the table (that
// arm moves it; chosen at reset, kept in task_state). The cubes go on in order: the
// arm of the next cube places it once the cube below is in place and the other hand
// is away from the spot; meanwhile an arm may pick its next cube and wait holding it.
// Then both hands go home.
//
// The right arm is the left arm's mirror image across y = 0 (checked: with the right
// joints at (-q1, -q2, -q3, q4, -q5, -q6, -q7) and fingers -f of the left's, the right
// hand's pose is the left's mirrored, M R M with M = diag(1, -1, 1)). So the right
// arm's IK, reach and expert are the left arm's, run on a mirrored copy of the state
// ("arm frame": y and turns flipped, the right joints mapped onto the left joints in a
// scratch qpos). Both arms work as openarm_pick_place's left arm, except the hand
// points straight down everywhere (the tower is up to 12 cm high: the 'turn to level'
// band is moved above the work heights), and the start posture is the cell's home
// TCP with the hand pointing down (IK from the cell's home keyframe at setup).
//
//   action (10): left arm 5, right arm 5, each as openarm_pick_place's (world frame:
//      TCP move dx dy dz = 0.02 m * a[0..2] per step, turn of the closing axis about z
//      = 0.1 rad * a[3], all cut to +-0.35; gripper a[4]: -1 open, +1 closed).
//   control: the cell's position servos and armature / damping, both arms (as
//      openarm_pick_place's left arm); gravity compensation.
//   obs (79): joint positions (18: left arm 7, fingers 2, right arm 7, fingers 2),
//      joint velocities (18), per hand (left, right) TCP position (3), sin and cos of
//      the turn (2, world yaw of the closing axis), finger opening (1: 0 closed .. 0.79
//      open), per cube (red, green, blue) position (3) and quaternion w x y z (4, w >=
//      0), last action (10).
//   placed: red resting on the table within 2.5 cm of the spot (x 0.41, y 0, a gray
//      square on the table); green on red, blue on green: center within 1.8 cm of the
//      one below in x y and 4 cm (+-5 mm) above it; not grasped.
//   done: all three placed, both hands open and back within 2 cm of home.
//   episode: ends (terminated) when done; at most 800 steps (16 s at 50 Hz: one arm
//      may move all three cubes).
//   reward (sparse; as openarm_pick_place): once, at the done step:
//        success  +20
//        aligned  up to +10: 10 (1 - the largest x y offset (red from the spot,
//                 green from red, blue from green) / 1.2 cm)
//      penalties per second (x 0.02 s each step), both arms:
//        joint_limits  10 per rad beyond 95% of the arm joints' ranges
//        arms          2 while a left-arm capsule overlaps a right-arm capsule (they
//                      also collide in the physics)
//        table         10 while an arm capsule or finger pad touches the table (2 mm)
//        push          80 per m/s of a cube's x y speed while it is on the table (not
//                      lifted 1 cm) and not grasped
//        approach      4 per m/s of a hand's speed above 0.15 m/s within 6 cm of a cube
//                      it does not grasp
//        slam          5 per m/s of a cube's speed while it sits on another cube (not
//                      on the table) and is not grasped
//        jerk          1e-5 sum of the arm joints' squared accelerations (mean over
//                      the step)
//        action_rate   1 sum of the squared changes of the 8 move and turn actions
//   reset: both arms at the start posture (joints +-0.03 rad), fingers open; each cube
//      on a random arm's half of the table, anywhere in that arm's work area, at least
//      12 cm (in x or y) from the spot and 10 cm from the other cubes, turned within
//      +-30 degrees of its arm's preferred grip turn there (spawn_yaw). Randomization:
//      table friction 0.6-1.2, cube masses x0.5-2.
//   expert (scripted, true state; openarm_pick_place's for each arm, in its arm frame).
//     An arm's job: its first cube (in the order red, green, blue) not placed yet (a
//     cube with the next one on it counts as placed: no digging under the tower).
//     just let go of a cube above the table, hand low by it -> straight up
//     no job                           -> up to return_z, over to home there, down
//     holding: not allowed to place yet -> lift to carry height and wait there
//              allowed                  -> carry over the spot (red) or the cube below
//                                          at carry height, land slowly, open within
//                                          9 mm above it
//     else: pick as openarm_pick_place (hover, turn to a face, down, close)
//     Moves slow down over the last slow_dist (4 cm) to each goal (approach()).
//     Allowed: the cube below is placed (red: always) and the other hand is 8 cm or
//     more from the spot in x y. A hand that waits near the spot waits over home.
//     While a hand holds a cube the IK keeps the cube level; just after letting go it
//     keeps the hand's pitch until the TCP is 1.5 cm above the cube's top. 118 / 128
//     towers (the misses: a cube that slid off the one below, out of reach).
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

struct OpenArmStackTask {
    static constexpr const char* kName = "openarm_stack";
    static constexpr const char* kUrdf = "robots/openarm/openarm_cell_bimanual.urdf";
    static constexpr bool kFloating = false;
    static constexpr int kArm = 7, kJoints = 9, kSides = 2, kCubes = 3, kPads = 6, kMaxQ = 48, kMaxCaps = 24;
    static constexpr int kObsDim = 79, kActDim = 10, kMaxEpisodeSteps = 800;
    static constexpr int kControlHz = 50;
    // task_state: the arm joints' velocities before this step's physics (14, jerk),
    // then each cube's arm (3: 0 left, 1 right; set at reset)
    static constexpr int kTaskState = 17;
    static constexpr float kHalf = 0.02f;  // cube half size
    enum { kRed = 0, kGreen = 1, kBlue = 2 };

    struct Params {
        int q[kSides][kJoints] = {}, d[kSides][kJoints] = {};  // per side: arm joints 1-7, finger 1, finger 2
        float sgn[kJoints] = {-1, -1, -1, 1, -1, -1, -1, -1, -1};  // right joint = sgn * the mirrored left joint
        IkJoints ik;  // the left arm's 7 joints (the right arm runs on a mirrored copy), posture = start
        float start[kJoints] = {};  // the left arm's start posture (the right's: sgn * it)
        int hand = 0;               // the left hand (the right's pose is computed on the mirrored copy)
        int hand_r = 0;             // the right hand (cameras)
        int cube_body[kCubes] = {}, cube_q[kCubes] = {}, cube_d[kCubes] = {};
        float tcp[3] = {-0.00143f, 0.0f, -0.1635f};  // finger-pad center (closed), hand frame
        float home_tcp[3] = {};                      // the left hand's TCP at the start posture (arm frame)
        float turn0 = 0.0f;
        // the left arm's work area (arm frame; openarm_pick_place's)
        float area_x[2] = {0.30f, 0.48f}, area_y[2] = {-0.07f, 0.40f};
        float hole_x = 0.36f, hole_y[2] = {0.03f, 0.24f};
        float corner_x = 0.45f, corner_y = -0.02f;
        float box_lo[3] = {0.22f, -0.12f, 0.012f}, box_hi[3] = {0.58f, 0.45f, 0.19f};
        float turn_c[3] = {-1.52f, 0.689f, 3.142f};
        float max_turn = 0.85f;
        // pitch pulled toward low over the table, per side: the left hand straight down;
        // the right hand less (it carries the top cube, at 10 cm, where the arm cannot
        // point down: picked at a pitch the tower reaches too, the cube stays level)
        float pick_down[2] = {0.8f, 0.8f};
        // hand pointing down below approach_z_lo, level above approach_z_hi: above the
        // work heights (the tower's top cube goes on at 10 cm), below home (17 cm)
        float approach_z_lo = 0.13f, approach_z_hi = 0.16f, home_lift = 0.056f;
        float move = 0.02f, turn = 0.1f, finger_open = 0.5f, max_speed = 0.35f;
        int ik_steps = 3;
        float posture_gain = 0.2f, upright_gain = 0.5f;
        // spawn
        float spot[2] = {0.41f, 0.0f};  // the tower's spot (both arms reach it)
        float spot_xy = 0.025f;  // red placed: within this of the spot
        float spot_gap = 0.12f, cube_gap = 0.10f;
        // cube turns at reset: within +-spawn_yaw of the arm's preferred grip turn there
        // (turn_pref, in that arm's frame). Then the grip turn is unique (the cube's other
        // faces are 60-120 degrees away, outside the +-max_turn window) and follows the
        // cube's visible turn smoothly. Any turn (>= pi, the old rule): two cube turns a
        // few degrees apart could need opposite turns; a VLA's turn was a coin flip.
        float spawn_yaw = 0.52f;
        float home_joint_noise = 0.03f, done_dist = 0.02f;
        float stack_xy = 0.018f, stack_dz = 0.005f, align_xy = 0.012f;
        // reward
        float w_success = 20.0f, w_aligned = 10.0f;
        float w_limits = 10.0f, limit_soft = 0.95f, w_arms = 2.0f, w_table = 10.0f, table_touch = 0.002f;
        float w_push = 80.0f, w_approach = 4.0f, approach_speed = 0.15f, approach_dist = 0.06f, w_slam = 5.0f;
        float w_jerk = 1e-5f, w_rate = 1.0f;
        int ncap = 0;
        int cap_body[kMaxCaps] = {}, cap_side[kMaxCaps] = {};
        float cap_pos[kMaxCaps][3] = {}, cap_axis[kMaxCaps][3] = {}, cap_r[kMaxCaps] = {}, cap_hl[kMaxCaps] = {};
        float lim_lo[kSides][kArm] = {}, lim_hi[kSides][kArm] = {};
        float touch = 0.002f;
        int pad_body[kSides][kPads] = {}, pad_side[kSides][kPads] = {};
        float pad_pos[kSides][kPads][3] = {}, pad_r[kSides][kPads] = {};
        // expert
        float exp_hold_lo = 0.14f, exp_hold_hi = 0.30f, exp_empty = 0.12f;
        float exp_xy = 0.008f, exp_turn = 0.08f, exp_z = 0.006f, carry_z = 0.09f, carry_clear = 0.025f;
        float hover_dz = 0.06f, return_z = 0.17f, place_xy = 0.004f, release_dz = 0.009f, support_dz = 0.004f,
              clear_dist = 0.08f;
        float expert_speed = 0.4f, expert_return_speed = 0.25f, expert_land_speed = 0.1f;
        // slowing down: per axis, the speed falls with the distance to the goal over the
        // last slow_dist (full speed beyond it, at least slow_min, never past the goal in
        // one step). 0: the old rule (full speed until 8 mm, a hard stop: hard to imitate,
        // a VLA's timing error became an overshoot)
        float slow_dist = 0.04f, slow_min = 0.1f;
        bool randomize = true;
        float friction_lo = 0.6f, friction_hi = 1.2f, cube_mass_lo = 0.5f, cube_mass_hi = 2.0f;
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
            if (b < 0) throw std::runtime_error("openarm_stack: needs " + s + "joint1-7 and finger_joint1-2");
        return j;
    }
    static void edit_model(RobotModel& m) {
        // collision filter (as openarm_pick_place): the arms touch each other, not themselves
        auto side_mask = [&](int body, int& ct, int& ca) {
            const std::string& n = m.body_name[size_t(body)];
            if (n.find("openarm_left") != std::string::npos) ct = 3, ca = 4;
            if (n.find("openarm_right") != std::string::npos) ct = 5, ca = 2;
        };
        for (auto& c : m.capsules) side_mask(c.body, c.contype, c.conaffinity);
        for (auto& sp : m.spheres) side_mask(sp.body, sp.contype, sp.conaffinity);
        const char* names[kCubes] = {"cube_red", "cube_green", "cube_blue"};
        const float rgb[kCubes][3] = {{0.85f, 0.12f, 0.1f}, {0.15f, 0.7f, 0.2f}, {0.15f, 0.3f, 0.9f}};
        for (int c = 0; c < kCubes; ++c) {
            add_free_box(m, names[c], {kHalf, kHalf, kHalf}, 0.1f);
            for (int i = 0; i < 3; ++i) m.boxes.back().rgb[i] = rgb[c][i];
        }
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
        m.visuals.push_back({{p0.spot[0], p0.spot[1], 0.0005f}, {0.03f, 0.03f, 0.0005f}, {0.55f, 0.55f, 0.58f}, -1});  // the spot
        m.name = "openarm_cell";  // viewers draw it with openarm_cell.cmesh (the same links)
        m.view_target = {0.40f, 0.0f, 0.10f};
        m.view_distance = 1.15f, m.view_yaw = 0.0f, m.view_pitch = 0.55f;
    }
    static Params make_params(const RobotModel& m) {
        Params p;
        if (m.nq > kMaxQ) throw std::runtime_error("openarm_stack: nq too large");
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
        if (p.hand < 0 || p.hand_r < 0) throw std::runtime_error("openarm_stack: needs both ee_base_links");
        const char* names[kCubes] = {"cube_red", "cube_green", "cube_blue"};
        for (int c = 0; c < kCubes; ++c) {
            p.cube_body[c] = m.body_index(names[c]);
            if (p.cube_body[c] < 0) throw std::runtime_error("openarm_stack: needs the cubes (edit_model)");
            p.cube_q[c] = m.qadr[p.cube_body[c]], p.cube_d[c] = m.dadr[p.cube_body[c]];
        }
        // start posture: the cell's home TCP raised by home_lift (above the band where the
        // hand turns to level), IK from the cell's home keyframe (elbow at 90 degrees)
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
        const Vec3 hs = tcp_pos(p, f);  // home = where the start posture puts the TCP
        p.home_tcp[0] = hs.x, p.home_tcp[1] = hs.y, p.home_tcp[2] = hs.z;
        p.start[7] = p.start[8] = p.finger_open;
        p.ik = make_ik_joints(m, std::vector<int>(armj.begin(), armj.begin() + kArm),
                              std::vector<float>(p.start, p.start + kArm));
        // pads: 6 per hand, inner (0) or outer (1) finger
        int np[kSides] = {0, 0};
        for (const auto& sp : m.spheres) {
            const std::string& n = m.body_name[size_t(sp.body)];
            const int s = n.find("openarm_left") != std::string::npos ? 0 : 1;
            if (np[s] == kPads) throw std::runtime_error("openarm_stack: expected 6 pads per hand");
            const int k = np[s]++;
            p.pad_body[s][k] = sp.body;
            p.pad_side[s][k] = n.find("inner") != std::string::npos ? 0 : 1;
            p.pad_pos[s][k][0] = sp.pos.x, p.pad_pos[s][k][1] = sp.pos.y, p.pad_pos[s][k][2] = sp.pos.z;
            p.pad_r[s][k] = sp.radius;
        }
        if (np[0] != kPads || np[1] != kPads) throw std::runtime_error("openarm_stack: expected 6 pads per hand");
        for (const auto& c : m.capsules) {  // the arms' moving capsules (the bases are welded)
            const std::string& n = m.body_name[size_t(c.body)];
            const int s = n.find("openarm_left") != std::string::npos ? 0 : (n.find("openarm_right") != std::string::npos ? 1 : -1);
            bool moves = false;
            for (int b = c.body; b >= 0; b = m.parent[size_t(b)]) moves = moves || m.jtype[size_t(b)] != kJointFixed;
            if (s < 0 || !moves) continue;
            if (p.ncap == kMaxCaps) throw std::runtime_error("openarm_stack: too many capsules");
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

    // ---- cameras (host): openarm_pick_place's four (the cell's): 0 head, 1 left wrist,
    // 2 ceiling, 3 right wrist (both hands work here)
    struct Camera {
        Vec3 eye, target, up;
        float fovy;  // rad
    };
    static constexpr int kCameras = 4;
    static void cameras(const Params& p, const ModelView& m, const float* qpos, DynamicsWork& w, Camera* out) {
        const float deg = 0.01745329f;
        out[0] = Camera{{0.223f, 0.0315f, 0.445f}, Vec3{0.223f, 0.0315f, 0.445f} + Vec3{0.46947f, 0.0f, -0.88295f},
                        {0.88295f, 0.0f, 0.46947f}, 82.0f * deg};
        out[2] = Camera{{0.245f, 0.0f, 0.795f}, {0.245f, 0.0f, -0.205f}, {1.0f, 0.0f, 0.0f}, 92.0f * deg};
        forward_kinematics(m, qpos, w);
        auto wrist = [&](int body, float side) {  // side +1 left, -1 right (mirrored)
            const Vec3 h = load3(w.xpos + 3 * body);
            const Mat3 R = quat_to_mat3(load_quat(w.xquat + 4 * body));
            const Vec3 eye = h + R * Vec3{0.023f, side * 0.0182f, -0.10236f};
            return Camera{eye, eye + R * Vec3{-0.22495f, -side * 0.34918f, -0.90965f},
                          R * Vec3{0.97437f, -side * 0.08062f, -0.21001f}, 92.0f * deg};
        };
        out[1] = wrist(p.hand, 1.0f);
        out[3] = wrist(p.hand_r, -1.0f);
    }

    // ---- per env ----
    struct Fk {
        float xpos[3 * kMaxBodies], xquat[4 * kMaxBodies];
        SpatialVec S[kMaxDofs];
        DynamicsWork w{xpos, xquat, S, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    };
    // arm frame <-> world: side 0 as is, side 1 mirrored across y = 0
    FORGE_HD static float ms(int side) { return side == 0 ? 1.0f : -1.0f; }
    FORGE_HD static Vec3 mirror(int side, Vec3 v) { return Vec3{v.x, ms(side) * v.y, v.z}; }
    FORGE_HD static Vec3 tcp_vec(const Params& p) { return Vec3{p.tcp[0], p.tcp[1], p.tcp[2]}; }
    FORGE_HD static Vec3 axis_for_turn(float t) { return Vec3{-sinf(t), cosf(t), 0.0f}; }
    // pointing direction pulled toward: 'down0' below level low over the table (side's
    // pick_down), turning to level between approach_z_lo and approach_z_hi
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
    // the left hand's TCP and turn in a qpos (for side 1: the mirrored copy = the right
    // hand in the arm frame)
    FORGE_HD static Vec3 tcp_pos(const Params& p, const Fk& f) {
        return load3(f.xpos + 3 * p.hand) + rotate(load_quat(f.xquat + 4 * p.hand), tcp_vec(p));
    }
    FORGE_HD static float hand_turn(const Params& p, const Fk& f) {
        const Vec3 a = rotate(load_quat(f.xquat + 4 * p.hand), Vec3{0, 1, 0});
        const float t = atan2f(-a.x, a.y);
        return p.turn0 + wrap_pi(t - p.turn0);
    }
    // the arm-frame copy of side s: the qpos with the left arm joints = side s's
    // joints (mirrored for the right: sgn); 'plan' takes the arm joints from ctrl
    FORGE_HD static void arm_qpos(const Params& p, const EnvView& v, int e, int s, bool plan, float* qa) {
        const float* q = v.qpos(e);
        for (int i = 0; i < v.model.nq; ++i) qa[i] = q[i];
        const float* ctrl = v.ctrl + size_t(e) * v.nu;
        for (int k = 0; k < kJoints; ++k) {
            const float x = plan && k < kArm ? ctrl[s * kJoints + k] : q[p.q[s][k]];
            qa[p.q[0][k]] = s == 0 ? x : p.sgn[k] * x;
        }
    }
    FORGE_HD static Vec3 cube_pos(const Params& p, const EnvView& v, int e, int c) {
        return load3(v.qpos(e) + p.cube_q[c]);
    }
    FORGE_HD static float cube_yaw(const Params& p, const EnvView& v, int e, int c) {
        const Quat q = normalize(load_quat_wxyz(v.qpos(e) + p.cube_q[c] + 3));
        return atan2f(2.0f * (q.w * q.z + q.x * q.y), 1.0f - 2.0f * (q.y * q.y + q.z * q.z));
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
    // IK on the left arm joints of qa (an arm-frame copy) toward TCP t and turn; as
    // openarm_pick_place's ik_to ('pull_all': the posture and upright pulls in every step)
    // 'level': pull the pointing direction there instead (holding a cube: the one that
    // keeps the cube level), else approach_for_turn with 'down0'
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
                         pull || level ? p.upright_gain : 0.0f, p.ik, qa, o, r);  // holding: level it every step
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
            // world action -> arm frame: y and the turn flip for the right arm
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

    // Holding a cube (fingers in the holding range, the cube between the pads: the real
    // state), the hand's pointing direction that would make the cube level: the
    // planned hand's pointing (fp: the arm-frame plan) turned by the rotation that takes
    // the cube's most upright axis to z. Arm frame. False when not holding.
    FORGE_HD static bool level_pointing(const Params& p, const EnvView& v, int e, int s, const Fk& fp, Vec3& out) {
        const float op = opening(p, v, e, s);
        float qa[kMaxQ];
        arm_qpos(p, v, e, s, false, qa);
        Fk fm;
        forward_kinematics(v.model, qa, fm.w);
        const Vec3 hm = tcp_pos(p, fm);
        const Vec3 pt = rotate(load_quat(fp.xquat + 4 * p.hand), Vec3{0, 0, -1});
        if (op < p.exp_hold_lo || op > p.exp_hold_hi) {
            // just let go of a cube above the table, the TCP still below 1.5 cm over its
            // top: keep the hand's pitch (turning back to the pick pitch there swung the
            // fingers into the tower); above that the hand may turn
            for (int c = 0; c < kCubes; ++c) {
                const Vec3 x = mirror(s, cube_pos(p, v, e, c)), d = x - hm;
                if (x.z > kHalf + 0.01f && hm.z < x.z + kHalf + 0.015f && sqrtf(d.x * d.x + d.y * d.y) < 0.045f) {
                    out = pt;
                    return true;
                }
            }
            return false;
        }
        int best = -1;
        for (int c = 0; c < kCubes && best < 0; ++c) {
            const Vec3 d = mirror(s, cube_pos(p, v, e, c)) - hm;
            if (sqrtf(d.x * d.x + d.y * d.y) < 0.02f && d.z > -0.035f && d.z < 0.02f) best = c;
        }
        if (best < 0) return false;
        const Mat3 R = quat_to_mat3(normalize(load_quat_wxyz(v.qpos(e) + p.cube_q[best] + 3)));
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
        const Vec3 k = ax / sn;  // Rodrigues: pt turned by the tilt angle about k
        out = pt * cs + cross(k, pt) * sn + k * (dot(k, pt) * (1.0f - cs));
        return true;
    }
    // a hand's TCP (world) and turn (world yaw of the closing axis), measured or planned
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
        for (int c = 0; c < kCubes; ++c) {
            const Vec3 x = cube_pos(p, v, e, c);
            Quat cq = normalize(load_quat_wxyz(q + p.cube_q[c] + 3));
            if (cq.w < 0.0f) cq = Quat{-cq.x, -cq.y, -cq.z, -cq.w};
            float* oc = o + 48 + 7 * c;
            oc[0] = x.x, oc[1] = x.y, oc[2] = x.z;
            oc[3] = cq.w, oc[4] = cq.x, oc[5] = cq.y, oc[6] = cq.z;
        }
        for (int k = 0; k < kActDim; ++k) o[69 + k] = v.last_action[size_t(e) * kActDim + k];
    }

    // hand s (world FK of the real qpos in f) holds cube c: pads of both fingers touch it
    FORGE_HD static bool grasped(const Params& p, const EnvView& v, int e, const Fk& f, int s, int c) {
        const Vec3 x = cube_pos(p, v, e, c);
        const Mat3 R = quat_to_mat3(normalize(load_quat_wxyz(v.qpos(e) + p.cube_q[c] + 3)));
        bool side[2] = {false, false};
        for (int k = 0; k < kPads; ++k) {
            const int b = p.pad_body[s][k];
            const Vec3 sc = load3(f.xpos + 3 * b) + rotate(load_quat(f.xquat + 4 * b),
                                                            Vec3{p.pad_pos[s][k][0], p.pad_pos[s][k][1], p.pad_pos[s][k][2]});
            float dist;
            Vec3 n;
            if (sphere_box(sc, p.pad_r[s][k] + p.touch, x, R, Vec3{kHalf, kHalf, kHalf}, dist, n)) side[p.pad_side[s][k]] = true;
        }
        return side[0] && side[1];
    }
    FORGE_HD static bool held(const Params& p, const EnvView& v, int e, const Fk& f, int c) {
        return grasped(p, v, e, f, 0, c) || grasped(p, v, e, f, 1, c);
    }
    // cube a rests on cube b (not grasped)
    FORGE_HD static bool on_top(const Params& p, const EnvView& v, int e, const Fk& f, int a, int b) {
        const Vec3 xa = cube_pos(p, v, e, a), xb = cube_pos(p, v, e, b);
        const float dxy = sqrtf((xa.x - xb.x) * (xa.x - xb.x) + (xa.y - xb.y) * (xa.y - xb.y));
        return dxy < p.stack_xy && fabsf(xa.z - xb.z - 2.0f * kHalf) < p.stack_dz && !held(p, v, e, f, a);
    }
    FORGE_HD static bool hand_home(const Params& p, const EnvView& v, int e, int s) {
        Vec3 h;
        float turn;
        hand_state(p, v, e, s, false, h, turn);
        return opening(p, v, e, s) > p.finger_open - 0.1f && length(h - home_of(p, s)) < p.done_dist;
    }
    // cube c in its place: red on the spot (on the table), green on red, blue on green
    FORGE_HD static bool placed(const Params& p, const EnvView& v, int e, const Fk& f, int c) {
        if (c != kRed) return on_top(p, v, e, f, c, c - 1);
        const Vec3 r = cube_pos(p, v, e, kRed);
        const float d = sqrtf((r.x - p.spot[0]) * (r.x - p.spot[0]) + (r.y - p.spot[1]) * (r.y - p.spot[1]));
        return d < p.spot_xy && r.z < kHalf + 0.005f && !held(p, v, e, f, kRed);
    }
    // where cube c's center goes: the spot (red), on top of the cube below
    FORGE_HD static Vec3 place_target(const Params& p, const EnvView& v, int e, int c) {
        if (c == kRed) return Vec3{p.spot[0], p.spot[1], kHalf};
        return cube_pos(p, v, e, c - 1) + Vec3{0.0f, 0.0f, 2.0f * kHalf};
    }
    FORGE_HD static bool done(const Params& p, const EnvView& v, int e, const Fk& f) {
        return placed(p, v, e, f, kRed) && placed(p, v, e, f, kGreen) && placed(p, v, e, f, kBlue) &&
               hand_home(p, v, e, 0) && hand_home(p, v, e, 1);
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
        static const char* const n[kRewardTerms] = {"success",  "aligned",  "joint_limits", "arms", "table",
                                                    "push",     "approach", "slam",         "jerk", "action_rate"};
        return n[k];
    }
    FORGE_HD static float combine_terms(const Params&, const float* t) {
        float r = 0.0f;
        for (int k = 0; k < kRewardTerms; ++k) r += t[k];
        return r;
    }
    // squared distance between segments p1-q1 and p2-q2 (Ericson 5.1.9)
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
        if (done(p, v, e, f)) {
            const Vec3 r = cube_pos(p, v, e, kRed), g = cube_pos(p, v, e, kGreen), b = cube_pos(p, v, e, kBlue);
            const float o0 = sqrtf((r.x - p.spot[0]) * (r.x - p.spot[0]) + (r.y - p.spot[1]) * (r.y - p.spot[1]));
            const float o1 = sqrtf((g.x - r.x) * (g.x - r.x) + (g.y - r.y) * (g.y - r.y));
            const float o2 = sqrtf((b.x - g.x) * (b.x - g.x) + (b.y - g.y) * (b.y - g.y));
            t[0] = p.w_success;
            t[1] = p.w_aligned * fmaxf(0.0f, 1.0f - fmaxf(o0, fmaxf(o1, o2)) / p.align_xy);
        }
        float over = 0.0f;
        for (int s = 0; s < kSides; ++s)
            for (int k = 0; k < kArm; ++k) {
                const float x = q[p.q[s][k]];
                over += fmaxf(p.lim_lo[s][k] - x, 0.0f) + fmaxf(x - p.lim_hi[s][k], 0.0f);
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
        bool grip[kSides][kCubes];
        for (int s = 0; s < kSides; ++s)
            for (int c = 0; c < kCubes; ++c) grip[s][c] = grasped(p, v, e, f, s, c);
        for (int c = 0; c < kCubes; ++c) {
            const float* cv = qd + p.cube_d[c];
            const float vxy = sqrtf(cv[0] * cv[0] + cv[1] * cv[1]), v3 = sqrtf(vxy * vxy + cv[2] * cv[2]);
            const float z = cube_pos(p, v, e, c).z;
            const bool g = grip[0][c] || grip[1][c];
            if (g) continue;
            if (z < kHalf + 0.01f)
                t[5] -= p.w_push * vxy * dt;
            else if (z > 3.0f * kHalf - p.stack_dz)  // on another cube
                t[7] -= p.w_slam * v3 * dt;
        }
        for (int s = 0; s < kSides; ++s) {
            Vec3 h;
            float turn;
            hand_state(p, v, e, s, false, h, turn);
            bool near = false;
            for (int c = 0; c < kCubes; ++c) near = near || (!grip[s][c] && length(h - cube_pos(p, v, e, c)) < p.approach_dist);
            if (!near) continue;
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
            t[8] = -p.w_jerk * s2 * dt;
        }
        const float* prev = v.prev_action + size_t(e) * kActDim;
        float r2 = 0.0f;
        for (int s = 0; s < kSides; ++s)
            for (int k = 0; k < 4; ++k) r2 += (act[5 * s + k] - prev[5 * s + k]) * (act[5 * s + k] - prev[5 * s + k]);
        t[9] = -p.w_rate * r2 * dt;
    }
    FORGE_HD static float reward(const Params& p, const EnvView& v, int e, const float* a) {
        float t[kRewardTerms];
        reward_terms(p, v, e, a, t);
        return combine_terms(p, t);
    }

    // ---- reset ----
    FORGE_HD static void area_point(const Params& p, Rng& r, float y_min, float& x, float& y) {  // arm frame
        for (int k = 0; k < 64; ++k) {
            x = r.uniform(p.area_x[0], p.area_x[1]);
            y = r.uniform(fmaxf(p.area_y[0], y_min), p.area_y[1]);
            if (in_area(p, x, y)) return;
        }
        x = 0.40f, y = 0.30f;
    }
    FORGE_HD static void reset(const Params& p, const EnvView& v, int e, Rng& rng) {
        float* q = v.qpos(e);
        float* ctrl = v.ctrl + size_t(e) * v.nu;
        float x[kCubes], y[kCubes];
        int sides[kCubes];
        for (int c = 0; c < kCubes; ++c) {  // a random arm's half, anywhere in its work area
            float best = -1e9f;
            int side = 0;
            for (int tries = 0; tries < 128; ++tries) {
                const int sd = rng.uniform(0.0f, 1.0f) < 0.5f ? 0 : 1;
                float cx, cy;
                area_point(p, rng, 0.0f, cx, cy);
                cy *= ms(sd);
                // margin to the rule (>= 0 when it holds): spot_gap from the spot, cube_gap from the others
                float m = fmaxf(fabsf(cx - p.spot[0]), fabsf(cy - p.spot[1])) - p.spot_gap;
                for (int o = 0; o < c; ++o) m = fminf(m, fmaxf(fabsf(cx - x[o]), fabsf(cy - y[o])) - p.cube_gap);
                if (m > best) best = m, x[c] = cx, y[c] = cy, side = sd;
                if (m >= 0.0f) break;
            }
            sides[c] = side;
            if (v.task_state) v.task_state[size_t(e) * kTaskState + 14 + c] = float(side);
        }
        for (int c = 0; c < kCubes; ++c) {
            // world yaw: any, or near its arm's preferred grip turn (arm frame: y and turn mirrored)
            const float m = ms(sides[c]);
            const float yaw = p.spawn_yaw >= 3.14159265f
                                  ? rng.uniform(-3.14159265f, 3.14159265f)
                                  : m * (turn_pref(p, x[c], m * y[c]) + rng.uniform(-p.spawn_yaw, p.spawn_yaw));
            float* cq = q + p.cube_q[c];
            cq[0] = x[c], cq[1] = y[c], cq[2] = kHalf;
            cq[3] = cosf(0.5f * yaw), cq[4] = 0.0f, cq[5] = 0.0f, cq[6] = sinf(0.5f * yaw);
        }
        for (int s = 0; s < kSides; ++s) {
            for (int k = 0; k < kArm; ++k) {
                const float j = p.start[k] + rng.uniform(-p.home_joint_noise, p.home_joint_noise);
                q[p.q[s][k]] = s == 0 ? j : p.sgn[k] * j;
            }
            q[p.q[s][7]] = q[p.q[s][8]] = ms(s) * p.finger_open;
            for (int k = 0; k < kJoints; ++k) ctrl[s * kJoints + k] = q[p.q[s][k]];
        }
        if (!p.randomize) return;
        if (v.rand_friction) v.rand_friction[e] = rng.uniform(p.friction_lo, p.friction_hi);
        if (v.rand_body_scale) {
            float* sc = v.rand_body_scale + size_t(v.model.nbody) * e;
            for (int b = 0; b < v.model.nbody; ++b) sc[b] = 1.0f;
            for (int c = 0; c < kCubes; ++c) sc[p.cube_body[c]] = rng.uniform(p.cube_mass_lo, p.cube_mass_hi);
        }
    }

    // ---- expert ----
    // The arm that moves cube c (set at reset; without task_state: the half it is on)
    FORGE_HD static int owner(const Params& p, const EnvView& v, int e, int c) {
        if (v.task_state) return v.task_state[size_t(e) * kTaskState + 14 + c] > 0.5f ? 1 : 0;
        return cube_pos(p, v, e, c).y >= 0.0f ? 0 : 1;
    }
    // One arm (side s) moves cube 'mine' (-1: no job, go home) to 'target' (world: where
    // its center goes); may_place: allowed to go over the target. In the arm frame
    // (mirrored for the right arm), as openarm_pick_place's expert; writes the arm's 5
    // world-frame actions.
    FORGE_HD static void expert_arm(const Params& p, const EnvView& v, int e, int s, int mine, Vec3 target,
                                    bool may_place, float* a) {
        float qa[kMaxQ];
        arm_qpos(p, v, e, s, true, qa);
        Fk f;
        forward_kinematics(v.model, qa, f.w);
        const Vec3 h = tcp_pos(p, f);  // planned, arm frame
        const float plan_turn = hand_turn(p, f);
        arm_qpos(p, v, e, s, false, qa);
        Fk fm;
        forward_kinematics(v.model, qa, fm.w);
        const Vec3 hm = tcp_pos(p, fm);
        const float turn_m = hand_turn(p, fm);
        const float open = opening(p, v, e, s);
        const Vec3 home{p.home_tcp[0], p.home_tcp[1], p.home_tcp[2]};
        Vec3 goal = h;
        float grip = -1.0f, turn_goal = plan_turn, speed = p.expert_speed;
        // a cube just let go above the table, the hand low by it: straight up first,
        // never sideways through it
        // (the measured hand: the plan leads it, and deciding on the plan let the open
        // fingers sweep sideways through the tower)
        int near = -1;
        for (int k = 0; k < kCubes && near < 0; ++k) {
            const Vec3 ck = mirror(s, cube_pos(p, v, e, k));
            if (ck.z > kHalf + 0.01f && fabsf(ck.x - hm.x) < 0.045f && fabsf(ck.y - hm.y) < 0.045f &&
                hm.z < ck.z + 0.05f)
                near = k;
        }
        const Vec3 c = mine >= 0 ? mirror(s, cube_pos(p, v, e, mine)) : h;
        const Vec3 dpad = c - hm;
        const bool between = mine >= 0 && sqrtf(dpad.x * dpad.x + dpad.y * dpad.y) < 0.02f && dpad.z > -0.035f &&
                             dpad.z < 0.02f;
        const bool holding = open > p.exp_hold_lo && open < p.exp_hold_hi && between;
        const Vec3 tc = mirror(s, target);  // where mine's center goes, arm frame
        const bool landing = holding && sqrtf((tc.x - c.x) * (tc.x - c.x) + (tc.y - c.y) * (tc.y - c.y)) < 0.02f &&
                             c.z < tc.z + 0.025f;
        if (!holding && near >= 0) {
            const Vec3 ck = mirror(s, cube_pos(p, v, e, near));
            goal = Vec3{h.x, h.y, ck.z + 0.065f};
            speed = p.expert_return_speed;
        } else if (mine < 0) {  // no job: home (up to return_z, over, down)
            const float dhome = sqrtf((h.x - home.x) * (h.x - home.x) + (h.y - home.y) * (h.y - home.y));
            speed = p.expert_return_speed;
            if (open < p.exp_hold_hi)
                goal = h;
            else if (dhome > 0.05f && h.z < p.return_z - 0.005f)
                goal = Vec3{h.x, h.y, p.return_z};
            else if (dhome > 0.05f)
                goal = Vec3{home.x, home.y, p.return_z}, turn_goal = p.turn0;
            else
                goal = home, turn_goal = p.turn0;
        } else if (holding) {
            grip = 1.0f;
            const float tp = turn_pref(p, h.x, h.y);
            turn_goal = fminf(fmaxf(plan_turn, tp - 0.4f), tp + 0.4f);
            const Vec3 off = hm - c;
            const float ex = tc.x - c.x, ey = tc.y - c.y, dt = sqrtf(ex * ex + ey * ey);
            const float carry = fmaxf(p.carry_z, tc.z + p.carry_clear);  // under approach_z_lo
            if (!may_place) {  // lift and wait; near the spot, wait over home (out of the other arm's way)
                const float ds = sqrtf((h.x - p.spot[0]) * (h.x - p.spot[0]) + (h.y - ms(s) * p.spot[1]) * (h.y - ms(s) * p.spot[1]));
                goal = ds < p.clear_dist + 0.05f ? Vec3{home.x, home.y, carry} : Vec3{h.x, h.y, carry};
                if (hm.z < carry - 0.02f) goal.x = h.x, goal.y = h.y;  // lift first
            } else if (dt > (c.z > tc.z + 0.015f ? p.place_xy : 2.0f * p.place_xy)) {
                if (hm.z < carry - 0.02f && dt > 0.02f)
                    goal = Vec3{h.x, h.y, carry};  // lift first
                else
                    goal = Vec3{tc.x + off.x, tc.y + off.y, carry};  // carry over the target
                if (dt < 0.02f && c.z < carry - 0.005f) goal.z = h.z;  // on the way down: center at this height
            } else if (c.z > tc.z + p.release_dz && hm.z > c.z - p.support_dz) {
                goal = Vec3{tc.x + off.x, tc.y + off.y, tc.z + p.release_dz * 0.5f + off.z};  // down
            } else {
                goal = h;  // just above the target, or resting on it (the TCP went on down past
                grip = -1.0f;  // the cube's center: the cube tilts in the grip and sits on a corner)
            }
        } else {  // pick, as openarm_pick_place
            float d = wrap_pi(ms(s) * cube_yaw(p, v, e, mine) - turn_m);
            d -= 1.5707963f * rintf(d / 1.5707963f);
            float want = turn_m + d;
            const float pref = turn_pref(p, c.x, c.y);
            while (want > pref + p.max_turn) want -= 1.5707963f;
            while (want < pref - p.max_turn) want += 1.5707963f;
            const float grasp_z = c.z;
            const float dxy = sqrtf((c.x - hm.x) * (c.x - hm.x) + (c.y - hm.y) * (c.y - hm.y));
            const float dturn = fabsf(wrap_pi(want - turn_m));
            const float dz = hm.z - grasp_z;
            goal = Vec3{c.x, c.y, grasp_z + p.hover_dz};
            turn_goal = want;
            if (open <= p.exp_empty) {
                goal = Vec3{h.x, h.y, fmaxf(h.z, grasp_z + p.hover_dz)}, turn_goal = plan_turn;
            } else if ((dxy < p.exp_xy && dturn < p.exp_turn && fabsf(dz) < p.exp_z) ||
                       (open < p.finger_open - 0.05f && fabsf(dz) < 2.0f * p.exp_z && dxy < 2.0f * p.exp_xy)) {
                goal = h, grip = 1.0f, turn_goal = plan_turn;
            } else if ((dxy < p.exp_xy && dturn < p.exp_turn) ||
                       (dz < p.hover_dz - 0.01f && dxy < 2.0f * p.exp_xy && dturn < 2.0f * p.exp_turn)) {
                goal = Vec3{c.x, c.y, grasp_z};
            }
        }
        const float sp = speed;
        const float sz = landing ? p.expert_land_speed : sp;  // slowly down the last 2.5 cm (a soft landing)
        // arm frame -> world: y and the turn flip for the right arm
        a[0] = approach(p, goal.x - h.x, sp);
        a[1] = ms(s) * approach(p, goal.y - h.y, sp);
        a[2] = approach(p, goal.z - h.z, sz);
        a[3] = ms(s) * fminf(fmaxf(wrap_pi(turn_goal - plan_turn) / p.turn, -sp), sp);
        a[4] = grip;
    }
    // one axis of the expert's move (action units, at most 'cap'): to the goal 'd' metres away
    FORGE_HD static float approach(const Params& p, float d, float cap) {
        if (p.slow_dist <= 0.0f) return fminf(fmaxf(d / p.move, -cap), cap);
        const float ad = fabsf(d);
        const float v = fminf(fmaxf(cap * ad / p.slow_dist, fminf(cap, p.slow_min)), cap);
        return copysignf(fminf(ad / p.move, v), d);
    }
    FORGE_HD static void expert(const Params& p, const EnvView& v, int e, float* a) {
        Fk f;
        forward_kinematics(v.model, v.qpos(e), f.w);
        bool done_c[kCubes];
        for (int c = 0; c < kCubes; ++c) done_c[c] = placed(p, v, e, f, c);
        for (int c = 0; c + 1 < kCubes; ++c)  // a cube with the next one sitting on it stays (no digging)
            done_c[c] = done_c[c] || on_top(p, v, e, f, c + 1, c);
        float spot_dist[kSides];  // each hand's x y distance from the spot
        for (int s = 0; s < kSides; ++s) {
            Vec3 h;
            float t;
            hand_state(p, v, e, s, false, h, t);
            spot_dist[s] = sqrtf((h.x - p.spot[0]) * (h.x - p.spot[0]) + (h.y - p.spot[1]) * (h.y - p.spot[1]));
        }
        for (int s = 0; s < kSides; ++s) {
            int job = -1;
            for (int c = 0; c < kCubes && job < 0; ++c)
                if (owner(p, v, e, c) == s && !done_c[c]) job = c;
            const bool may = job >= 0 && (job == kRed || done_c[job - 1]) && spot_dist[1 - s] > p.clear_dist;
            const Vec3 target = job >= 0 ? place_target(p, v, e, job) : Vec3{0, 0, 0};
            expert_arm(p, v, e, s, job, target, may, a + 5 * s);
        }
    }
};

}  // namespace forge
