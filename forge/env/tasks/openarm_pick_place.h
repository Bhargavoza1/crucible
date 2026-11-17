#pragma once
// OpenArm pick and place (Phase 10): the OpenArm v2 Cell (enactic/openarm_mujoco
// cell.xml, as enactic/openarm_mjlab's pick_place task): both arms hang from the
// lifter at the back of the table; the LEFT arm picks a 4 cm cube from the table and
// puts it into the tray, the right arm and the lifter stay at the cell's home pose
// (frozen in the URDF: assets/robots/openarm/openarm_cell.urdf, tools/openarm_cell_to_urdf.py).
// The floor plane is the table top (z = 0; the world is moved down 1.005 m). The cell
// is drawn from its meshes; the room floor is drawn only (RobotModel::visuals).
// The tray (Phase 10b) is solid: a base and 4 walls (collision boxes: the cube
// touches them box-box, the fingers and the arm sphere- and capsule-box) on a heavy,
// damped body that slides in x and y (two slide joints, qpos after the cube's), 1 mm
// above the table (no table contacts); reset puts it at the episode's spot. It is
// drawn from visuals that follow obs 34-35 (the tray's center, from its qpos).
//
// Reach (scratch sweep of the axis IK walked from home, then tests/cpu/forge/env
// OpenArmPickPlace.WorkspaceReachable): low over the table the gripper points
// straight down; the left arm then reaches most of its half of the table and the
// middle up to 7 cm onto the right arm's side (y >= -0.07), except right under its
// shoulder. The closing-axis turns it reaches are a window about 100 degrees wide
// whose center moves with the spot (turn_pref: a plane in x, y); a window of 90
// degrees is enough for any cube turn (a cube looks the same every quarter turn).
// Work area (in_area): x 0.30-0.48, y -0.07-0.40, without x < 0.36 at y 0.03-0.24
// (under the shoulder) and x > 0.45 at y < -0.02.
//
//   action (5), a in [-1, 1]: TCP move dx dy dz = 0.02 m * a[0..2] per step, turn of
//      the closing axis about z = 0.1 rad * a[3] (a[0..3] cut to +-0.35: at most
//      0.35 m/s, 1.75 rad/s), gripper a[4]: -1 open (fingers at
//      0.5 rad: a 9.5 cm gap), +1 closed (target 0). The TCP is the finger-pad center
//      at closed fingers (hand frame (-0.00143, 0, -0.1635)). The target is FK(the
//      joint targets) + the move, kept in a box around the work area and home, the
//      turn within turn_pref +- max_turn there; 3 axis-IK steps. The IK keeps the
//      closing axis level and pulls the pointing direction to straight down below
//      approach_z_lo, to level (home) above approach_z_hi.
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
//   reward (sparse, for RL fine-tuning; Phase 10c): paid once, at the step the episode
//      is done (cube settled in the tray, the arm home):
//        success  +20
//        centre   up to +10: 10 (1 - |cube - tray center| in x y / 5 cm), 0 beyond
//      penalties, per second (x 0.02 s each step):
//        joint_limits  10 per rad of the arm joints beyond 95% of their range (the
//                      outer 2.5% at each end; joint 2's home, 0, is 0.17 from its limit)
//        right_arm     2 while a left-arm capsule overlaps a right-arm capsule (the
//                      arms do not collide in the physics: capsule-capsule rows do
//                      not exist)
//        table         10 while a left-arm capsule or finger pad touches the table
//                      (2 mm or closer)
//        push          80 per m/s of the cube's x y speed while it is on the table
//                      (not lifted 1 cm) and not grasped
//        approach      4 per m/s of TCP speed above 0.15 m/s within 6 cm of the cube,
//                      while not grasping it
//        slam          5 per m/s of cube speed while it is in the tray and not grasped
//        jerk          1e-5 sum of the arm joints' squared accelerations (rad/s^2,
//                      mean over the step: velocity change / 0.02 s)
//        action_rate   1 sum of the squared changes of the move and turn actions
//                      (a[0..3]) from the last step (smooth commands)
//   placed: cube in the tray (center within the inner square minus the cube's half
//      size plus 5 mm), resting (center below 5 mm above resting on the tray's
//      floor), not grasped.
//   episode: ends (terminated: done) when the cube is placed, the fingers open and
//      the TCP back within 2 cm of home; at most 400 steps (8 s at 50 Hz).
//   reset: the arm at the cell's home (joints +-0.03 rad), fingers open; the tray
//      anywhere in the work area, the cube anywhere in it at least 15 cm (in x or y)
//      from the tray's center, any turn. The tray's center is a function of (seed,
//      env, episode) (no stored state), drawn by viewers from obs 34-35.
//   randomization: table friction 0.6-1.2, cube mass x0.5-2 (0.05-0.2 kg).
//   expert (scripted, true state, no clock; Phase 9d lessons), in order:
//     placed, not held: open (fully, or until the tray's walls stop the fingers);
//                       straight up to return_z, over to home there,
//                       down to home (decided from the plan, so no flip-flop)
//     holding (opening 0.14-0.30 rad, cube between the pads):
//         not above the tray: lift to carry_z first, then carry over the tray
//         above the tray: down until the cube is 1 cm or less above the tray, then open
//     closed on nothing (opening < 0.12)          -> open, back up
//     at the grasp pose (or closing there)        -> close, hold still
//     above the cube and turned to a face         -> down, open
//     else -> hover_dz above the cube, turned to the face nearest the current turn
//             (inside the window there)
//   At most 0.4 of full speed (going home 0.25: the wrist turns the most there); up and down at 0.1 where the hand turns between
//   pointing down and level (approach_z_lo-hi), so the wrist follows smoothly.
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
    static constexpr int kArm = 7, kJoints = 9, kPads = 6, kMaxQ = 32, kMaxCaps = 32;
    static constexpr int kObsDim = 43, kActDim = 5, kMaxEpisodeSteps = 400;
    static constexpr int kControlHz = 50;
    static constexpr int kTaskState = 7;  // the arm joints' velocities before this step's physics (jerk)
    static constexpr float kHalf = 0.02f;  // cube half size

    struct Params {
        int q[kJoints] = {}, d[kJoints] = {};  // arm joints 1-7, finger 1, finger 2
        IkJoints ik;                           // the 7 arm joints, posture = start posture
        float start[kJoints] = {};
        int hand = 0, right_hand = 0, cube = 0, cube_q = 0, cube_d = 0, tray_q[2] = {};
        float tcp[3] = {-0.00143f, 0.0f, -0.1635f};  // finger-pad center (closed), hand frame
        float home_tcp[3] = {};  // TCP at the home posture
        float turn0 = 0.0f;      // closing-axis yaw at home (along y); turns are kept within turn0 +- pi
        // work area (see the top): box x0-x1, y0-y1 without the two corners
        float area_x[2] = {0.30f, 0.48f}, area_y[2] = {-0.07f, 0.40f};
        float hole_x = 0.36f, hole_y[2] = {0.03f, 0.24f};  // under the shoulder: x < hole_x, y in hole_y
        float corner_x = 0.45f, corner_y = -0.02f;          // front right: x > corner_x, y < corner_y
        // the TCP target's box (actions): the work area with a margin, and home
        float box_lo[3] = {0.22f, -0.12f, 0.012f}, box_hi[3] = {0.58f, 0.45f, 0.16f};
        // the reachable turns: turn_pref(x, y) +- max_turn
        float turn_c[3] = {-1.52f, 0.689f, 3.142f};  // turn_pref = c0 + c1 x + c2 y
        float max_turn = 0.85f;
        float approach_down = 1.5707963f;  // preferred pointing direction low over the table: straight down
        float approach_z_lo = 0.08f, approach_z_hi = 0.11f;  // ... blended to level (home) between these TCP heights
        float move = 0.02f, turn = 0.1f, finger_open = 0.5f;
        float max_speed = 0.35f;  // moves and turns are cut to this part of a full step (the expert asks for 0.4)
        int ik_steps = 3;
        float posture_gain = 0.2f, upright_gain = 0.5f;
        float tray_half = 0.075f;      // inner square half size (15 cm inside: the open hand fits; walls 5 mm thick, 4 cm high)
        float tray_lift = 0.001f;      // tray bottom above the table (no table contacts)
        float tray_base = 0.005f;      // base thickness: the floor is at tray_lift + tray_base
        float tray_mass = 200.0f;      // per slide body (kg): the cube and the arm barely move it
        float tray_damping = 2000.0f;  // per slide joint (N s / m): 20 N pushes it 1 cm/s
        float cube_tray_gap = 0.15f;   // cube center at least this far from the tray center, in x or y
                                       // (the open fingers around the cube clear the walls)
        float home_joint_noise = 0.03f;
        float done_dist = 0.02f;       // TCP to home at the end
        // sparse reward (see the top)
        float w_success = 20.0f, w_centre = 10.0f, centre_dist = 0.05f;
        float w_limits = 10.0f, limit_soft = 0.95f, w_right_arm = 2.0f, w_table = 10.0f, table_touch = 0.002f;
        float w_push = 80.0f, w_approach = 4.0f, approach_speed = 0.15f, approach_dist = 0.06f, w_slam = 5.0f;
        float w_jerk = 1e-5f, w_rate = 1.0f;
        // collision capsules of the arms (left: the working arm; right: the resting arm)
        int ncap = 0;
        int cap_body[kMaxCaps] = {}, cap_left[kMaxCaps] = {};
        float cap_pos[kMaxCaps][3] = {}, cap_axis[kMaxCaps][3] = {}, cap_r[kMaxCaps] = {}, cap_hl[kMaxCaps] = {};
        float lim_lo[kArm] = {}, lim_hi[kArm] = {};  // soft limits of the arm joints
        float touch = 0.002f;
        int pad_body[kPads] = {}, pad_side[kPads] = {};
        float pad_pos[kPads][3] = {}, pad_r[kPads] = {};
        // expert
        float exp_hold_lo = 0.14f, exp_hold_hi = 0.30f, exp_empty = 0.12f;
        float exp_xy = 0.008f, exp_turn = 0.08f, exp_z = 0.006f, carry_z = 0.09f, hover_dz = 0.06f, return_z = 0.14f;
        float expert_speed = 0.4f, expert_blend_speed = 0.1f, expert_return_speed = 0.25f;
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
    // the tray's 5 boxes in its frame (center at the table, z up): base, then the walls
    static void tray_boxes(const Params& p, Vec3* c, Vec3* h) {
        const float wall = 0.005f, wh = 0.02f, in = p.tray_half, z0 = p.tray_lift;
        c[0] = {0, 0, z0 + 0.5f * p.tray_base}, h[0] = {in + wall, in + wall, 0.5f * p.tray_base};
        c[1] = {in + 0.5f * wall, 0, z0 + wh}, h[1] = {0.5f * wall, in + wall, wh};
        c[2] = {-in - 0.5f * wall, 0, z0 + wh}, h[2] = {0.5f * wall, in + wall, wh};
        c[3] = {0, in + 0.5f * wall, z0 + wh}, h[3] = {in, 0.5f * wall, wh};
        c[4] = {0, -in - 0.5f * wall, z0 + wh}, h[4] = {in, 0.5f * wall, wh};
    }
    // a body on a slide joint along 'axis' (x or y), child of 'parent' (-1: the world)
    static int add_slide_body(RobotModel& m, const std::string& name, int parent, Vec3 axis, float mass) {
        const int b = m.nbody();
        m.body_name.push_back(name);
        m.joint_name.push_back(name);
        m.parent.push_back(parent);
        m.jtype.push_back(kJointPrismatic);
        m.qadr.push_back(m.nq);
        m.dadr.push_back(m.nv);
        m.dof_body.push_back(b);
        m.jaxis.insert(m.jaxis.end(), {axis.x, axis.y, axis.z});
        m.body_pos.insert(m.body_pos.end(), {0.0f, 0.0f, 0.0f});
        m.body_quat.insert(m.body_quat.end(), {0.0f, 0.0f, 0.0f, 1.0f});
        m.mass.push_back(mass);
        m.com.insert(m.com.end(), {0.0f, 0.0f, 0.0f});
        const float I = mass * 0.01f;  // any: the body only slides
        m.inertia.insert(m.inertia.end(), {I, 0.0f, 0.0f, 0.0f, I, 0.0f, 0.0f, 0.0f, I});
        m.joint_lower.push_back(0.0f);
        m.joint_upper.push_back(0.0f);
        m.joint_effort.push_back(0.0f);
        m.joint_damping.push_back(0.0f);
        m.joint_limited.push_back(false);
        if (!m.dof_armature.empty()) m.dof_armature.push_back(0.0f);
        if (!m.dof_damping.empty()) m.dof_damping.push_back(0.0f);
        m.nq += 1;
        m.nv += 1;
        return b;
    }
    static void edit_model(RobotModel& m) {
        const Params p{};
        add_free_box(m, "cube", {kHalf, kHalf, kHalf}, 0.1f);
        // the tray: slides in x (body tray_x), then in y (body tray), with the boxes
        const int tx = add_slide_body(m, "tray_x", -1, {1, 0, 0}, p.tray_mass);
        const int tb = add_slide_body(m, "tray", tx, {0, 1, 0}, p.tray_mass);
        Vec3 tc[5], th[5];
        tray_boxes(p, tc, th);
        for (int k = 0; k < 5; ++k) m.boxes.push_back({tb, tc[k], quat_identity(), th[k], false});
        m.validate();
        // armature / damping per joint (the cell's values)
        m.dof_armature.assign(size_t(m.nv), 0.0f);
        m.dof_damping.assign(size_t(m.nv), 0.0f);
        m.dof_damping[size_t(m.dadr[tx])] = m.dof_damping[size_t(m.dadr[tb])] = p.tray_damping;
        const float arm[kJoints] = {0.016f, 0.016f, 0.030f, 0.030f, 0.002f, 0.002f, 0.002f, 0.002f, 0.002f};
        const float damp[kJoints] = {1.0f, 1.0f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.6f, 0.6f};
        const std::vector<int> j = joint_bodies(m);
        for (int k = 0; k < kJoints; ++k) {
            m.dof_armature[size_t(m.dadr[j[k]])] = arm[k];
            m.dof_damping[size_t(m.dadr[j[k]])] = damp[k];
        }
        // drawn only: the room floor under the cell (its cabinet stands on it, 1.005 m
        // below the table top); the tray's boxes (black as the cell's), drawn centered on
        // obs 34-35 (its qpos: it moves every episode)
        auto vis = [&](Vec3 c, Vec3 h, float r, float g, float b, int obs_xy) {
            m.visuals.push_back({c, h, {r, g, b}, obs_xy});
        };
        vis({0.55f, 0.0f, -1.015f}, {0.75f, 0.6f, 0.01f}, 0.42f, 0.43f, 0.45f, -1);
        const float k = 0.1f;
        for (int i = 0; i < 5; ++i) vis(tc[i], th[i], k, k, k, 34);
        // viewer: from the front left of the table, looking at the work area
        m.view_target = {0.40f, 0.12f, 0.10f};
        m.view_distance = 1.05f, m.view_yaw = 0.45f, m.view_pitch = 0.50f;
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
        p.right_hand = m.body_index("openarm_right_ee_base_link");
        if (p.right_hand < 0) throw std::runtime_error("openarm_pick_place: needs openarm_right_ee_base_link");
        p.cube = m.body_index("cube");
        if (p.hand < 0 || p.cube < 0) throw std::runtime_error("openarm_pick_place: needs openarm_left_ee_base_link and cube");
        p.cube_q = m.qadr[p.cube];
        p.cube_d = m.dadr[p.cube];
        const int tx = m.body_index("tray_x"), tb = m.body_index("tray");
        if (tx < 0 || tb < 0) throw std::runtime_error("openarm_pick_place: needs the tray bodies (edit_model)");
        p.tray_q[0] = m.qadr[tx], p.tray_q[1] = m.qadr[tb];
        // Start posture: the cell's home (keyframe "home": elbow at 90 degrees, the
        // gripper pointing forward, level, closing axis along y).
        const float home[kArm] = {0.0f, 0.0f, 0.0f, 1.5707963f, 0.0f, 0.0f, 0.0f};
        for (int k = 0; k < kArm; ++k) p.start[k] = home[k];
        const std::vector<int> armj(j.begin(), j.begin() + kArm);
        {
            std::vector<float> q(m.nq, 0.0f);
            for (int k = 0; k < kArm; ++k) q[p.q[k]] = home[k];
            for (int b = 0; b < m.nbody(); ++b)
                if (m.jtype[b] == kJointFree) q[m.qadr[b] + 3] = 1.0f;
            RobotWork work(m);
            DynamicsWork w = work.view();
            forward_kinematics(m.view(), q.data(), w);
            const Vec3 h = load3(w.xpos + 3 * p.hand) + rotate(load_quat(w.xquat + 4 * p.hand), tcp_vec(p));
            p.home_tcp[0] = h.x, p.home_tcp[1] = h.y, p.home_tcp[2] = h.z;
        }
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
        for (const auto& c : m.capsules) {
            const std::string& n = m.body_name[size_t(c.body)];
            const bool left = n.find("openarm_left") != std::string::npos;
            if (!left && n.find("openarm_right") == std::string::npos) continue;
            bool moves = false;  // the left arm's base is welded to the lifter, next to the right arm's
            for (int b = c.body; b >= 0; b = m.parent[size_t(b)]) moves = moves || m.jtype[size_t(b)] != kJointFixed;
            if (left && !moves) continue;
            if (p.ncap == kMaxCaps) throw std::runtime_error("openarm_pick_place: too many capsules");
            const int k = p.ncap++;
            const Vec3 ax = rotate(c.rot, Vec3{0, 0, 1});
            p.cap_body[k] = c.body, p.cap_left[k] = left;
            p.cap_pos[k][0] = c.pos.x, p.cap_pos[k][1] = c.pos.y, p.cap_pos[k][2] = c.pos.z;
            p.cap_axis[k][0] = ax.x, p.cap_axis[k][1] = ax.y, p.cap_axis[k][2] = ax.z;
            p.cap_r[k] = c.radius, p.cap_hl[k] = c.half_length;
        }
        for (int k = 0; k < kArm; ++k) {
            const int b = j[size_t(k)];
            const float lo = m.joint_lower[size_t(b)], hi = m.joint_upper[size_t(b)];
            const float mid = 0.5f * (lo + hi), half = 0.5f * (hi - lo) * p.limit_soft;
            p.lim_lo[k] = m.joint_limited[size_t(b)] ? mid - half : -1e9f;
            p.lim_hi[k] = m.joint_limited[size_t(b)] ? mid + half : 1e9f;
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

    // ---- cameras (host): the cell's (openarm_mujoco cell.xml), z moved down by the
    // table top height. 0 head (camera_head_left: above the arms, looking forward and
    // down), 1 left wrist (on the working hand), 2 ceiling (straight down on the
    // table), 3 right wrist (on the resting hand). Square images that span the cell
    // cameras' horizontal field of view (head 1280 x 720 at 52 degrees vertical: 82
    // degrees across; wrists and ceiling 960 x 600 at 66: 92), so they see what the
    // real wide images see side to side (the cube area is near the head view's edge).
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
        out[3] = wrist(p.right_hand, -1.0f);
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
    FORGE_HD static Vec3 approach_for_turn(const Params& p, float t, float z) {
        float s = fminf(fmaxf((z - p.approach_z_lo) / (p.approach_z_hi - p.approach_z_lo), 0.0f), 1.0f);
        s = s * s * (3.0f - 2.0f * s);
        const float down = (1.0f - s) * p.approach_down;
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
        t.x = fminf(fmaxf(t.x, p.box_lo[0]), p.box_hi[0]);
        t.y = fminf(fmaxf(t.y, p.box_lo[1]), p.box_hi[1]);
        t.z = fminf(fmaxf(t.z, p.box_lo[2]), p.box_hi[2]);
        return t;
    }
    // center of the closing-axis turns the arm reaches at (x, y) (+- max_turn)
    FORGE_HD static float turn_pref(const Params& p, float x, float y) {
        return p.turn_c[0] + p.turn_c[1] * x + p.turn_c[2] * y;
    }
    FORGE_HD static float clamp_turn(const Params& p, float x, float y, float t) {
        const float c = turn_pref(p, x, y);
        return fminf(fmaxf(t, c - p.max_turn), c + p.max_turn);
    }
    FORGE_HD static bool in_area(const Params& p, float x, float y) {
        if (x < p.area_x[0] || x > p.area_x[1] || y < p.area_y[0] || y > p.area_y[1]) return false;
        if (x < p.hole_x && y > p.hole_y[0] && y < p.hole_y[1]) return false;
        return !(x > p.corner_x && y < p.corner_y);
    }
    // a uniform spot in the work area (rejection: about 80% of the box qualifies)
    FORGE_HD static void area_point(const Params& p, Rng& r, float& x, float& y) {
        for (int k = 0; k < 64; ++k) {
            x = r.uniform(p.area_x[0], p.area_x[1]);
            y = r.uniform(p.area_y[0], p.area_y[1]);
            if (in_area(p, x, y)) return;
        }
        x = 0.40f, y = 0.30f;
    }
    // The tray's spot this episode: from (seed, env, episode) (reset puts it there).
    FORGE_HD static void tray_spawn(const Params& p, const EnvView& v, int e, float& x, float& y) {
        Rng r(v.seed ^ 0x7A4A1Bull, uint64_t(e), uint64_t(v.episode_index[e]));
        area_point(p, r, x, y);
    }
    // The tray's center now (its slide joints: the cube or the arm may push it a little).
    FORGE_HD static void tray_center(const Params& p, const EnvView& v, int e, float& x, float& y) {
        x = v.qpos(e)[p.tray_q[0]];
        y = v.qpos(e)[p.tray_q[1]];
    }
    FORGE_HD static float tray_floor(const Params& p) { return p.tray_lift + p.tray_base; }
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
                         approach_for_turn(p, turn, t.z), pull ? p.upright_gain : 0.0f, p.ik, qp, o, r);
        }
    }

    FORGE_HD static void apply_action(const Params& p, const EnvView& v, int e, const float* a) {
        float qp[kMaxQ];
        plan_qpos(p, v, e, qp);
        Fk f;
        forward_kinematics(v.model, qp, f.w);
        const Vec3 h = tcp_pos(p, f);
        // at most max_speed of a full move or turn per step (a calm arm: 0.35 = 0.35 m/s)
        float m[4];
        for (int k = 0; k < 4; ++k) m[k] = fminf(fmaxf(a[k], -p.max_speed), p.max_speed);
        const Vec3 t = clamp_ws(p, Vec3{h.x + p.move * m[0], h.y + p.move * m[1], h.z + p.move * m[2]});
        const float turn = clamp_turn(p, t.x, t.y, hand_turn(p, f) + p.turn * m[3]);
        ik_to(p, v.model, t, turn, p.ik_steps, qp, f);
        float* ctrl = v.ctrl + size_t(e) * v.nu;
        for (int k = 0; k < kArm; ++k) ctrl[k] = qp[p.q[k]];
        const float g = fminf(fmaxf(a[4], -1.0f), 1.0f);
        ctrl[7] = ctrl[8] = p.finger_open * 0.5f * (1.0f - g);
        forward_kinematics(v.model, v.qpos(e), f.w);
        gravity_torques(v.model, f.w, 0, v.tau(e));
        if (v.task_state)
            for (int k = 0; k < kArm; ++k) v.task_state[size_t(e) * kTaskState + k] = v.qvel(e)[p.d[k]];
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
        float tx, ty;
        tray_center(p, v, e, tx, ty);
        o[34] = tx, o[35] = ty;
        o[36] = c.x - tx, o[37] = c.y - ty;
        for (int k = 0; k < kActDim; ++k) o[38 + k] = v.last_action[size_t(e) * kActDim + k];
    }
    // The episode's end when the work is done (terminated, the true end: no value after it).
    FORGE_HD static bool failed(const Params& p, const EnvView& v, int e) {
        Fk f;
        forward_kinematics(v.model, v.qpos(e), f.w);
        return done(p, v, e, f);
    }

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
    FORGE_HD static bool in_tray(const Params& p, Vec3 c, float tx, float ty) {
        const float lim = p.tray_half - kHalf + 0.005f;
        return fabsf(c.x - tx) < lim && fabsf(c.y - ty) < lim && c.z < tray_floor(p) + kHalf + 0.005f;
    }
    FORGE_HD static bool placed(const Params& p, const EnvView& v, int e, const Fk& f) {
        float tx, ty;
        tray_center(p, v, e, tx, ty);
        return in_tray(p, cube_pos(p, v, e), tx, ty) && !grasped(p, v, e, f);
    }
    FORGE_HD static Vec3 home_vec(const Params& p) { return Vec3{p.home_tcp[0], p.home_tcp[1], p.home_tcp[2]}; }
    // finished: placed, fingers open, the TCP back home
    FORGE_HD static bool done(const Params& p, const EnvView& v, int e, const Fk& f) {
        return placed(p, v, e, f) && opening(p, v, e) > p.finger_open - 0.1f &&
               length(tcp_pos(p, f) - home_vec(p)) < p.done_dist;
    }
    FORGE_HD static bool success(const Params& p, const EnvView& v, int e) {
        Fk f;
        forward_kinematics(v.model, v.qpos(e), f.w);
        return placed(p, v, e, f);
    }

    static constexpr int kRewardTerms = 10;
    static const char* reward_term_name(int k) {
        static const char* const n[kRewardTerms] = {"success",  "centre", "joint_limits", "right_arm", "table",
                                                    "push",     "approach", "slam",       "jerk",      "action_rate"};
        return n[k];
    }
    FORGE_HD static float combine_terms(const Params&, const float* t) {
        float r = 0.0f;
        for (int k = 0; k < kRewardTerms; ++k) r += t[k];
        return r;
    }
    // squared distance between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision
    // Detection 5.1.9)
    FORGE_HD static float segment_dist2(Vec3 p1, Vec3 q1, Vec3 p2, Vec3 q2) {
        const Vec3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
        const float a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
        float s = 0.0f, t = 0.0f;
        if (a <= 1e-12f && e <= 1e-12f) return dot(r, r);
        if (a <= 1e-12f) {
            t = fminf(fmaxf(f / e, 0.0f), 1.0f);
        } else {
            const float c = dot(d1, r);
            if (e <= 1e-12f) {
                s = fminf(fmaxf(-c / a, 0.0f), 1.0f);
            } else {
                const float b = dot(d1, d2), den = a * e - b * b;
                s = den > 1e-12f ? fminf(fmaxf((b * f - c * e) / den, 0.0f), 1.0f) : 0.0f;
                t = (b * s + f) / e;
                if (t < 0.0f) {
                    t = 0.0f;
                    s = fminf(fmaxf(-c / a, 0.0f), 1.0f);
                } else if (t > 1.0f) {
                    t = 1.0f;
                    s = fminf(fmaxf((b - c) / a, 0.0f), 1.0f);
                }
            }
        }
        const Vec3 w = (p1 + d1 * s) - (p2 + d2 * t);
        return dot(w, w);
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
        const Vec3 c = cube_pos(p, v, e), h = tcp_pos(p, f);
        float tx, ty;
        tray_center(p, v, e, tx, ty);
        const bool g = grasped(p, v, e, f);
        for (int k = 0; k < kRewardTerms; ++k) t[k] = 0.0f;
        // once, when done
        if (done(p, v, e, f)) {
            const float dc = sqrtf((c.x - tx) * (c.x - tx) + (c.y - ty) * (c.y - ty));
            t[0] = p.w_success;
            t[1] = p.w_centre * fmaxf(0.0f, 1.0f - dc / p.centre_dist);
        }
        // joint limits
        float over = 0.0f;
        for (int k = 0; k < kArm; ++k) {
            const float x = q[p.q[k]];
            over += fmaxf(p.lim_lo[k] - x, 0.0f) + fmaxf(x - p.lim_hi[k], 0.0f);
        }
        t[2] = -p.w_limits * over * dt;
        // the left arm into the right arm; the left arm on the table
        bool right = false, table = false;
        for (int i = 0; i < p.ncap; ++i) {
            if (!p.cap_left[i]) continue;
            Vec3 a1, b1;
            capsule_ends(p, f, i, a1, b1);
            table = table || fminf(a1.z, b1.z) - p.cap_r[i] <= p.table_touch;
            for (int j = 0; j < p.ncap && !right; ++j) {
                if (p.cap_left[j]) continue;
                Vec3 a2, b2;
                capsule_ends(p, f, j, a2, b2);
                const float r = p.cap_r[i] + p.cap_r[j];
                right = segment_dist2(a1, b1, a2, b2) < r * r;
            }
        }
        for (int s = 0; s < kPads && !table; ++s) {
            const int b = p.pad_body[s];
            const Vec3 sc = load3(f.xpos + 3 * b) +
                            rotate(load_quat(f.xquat + 4 * b), Vec3{p.pad_pos[s][0], p.pad_pos[s][1], p.pad_pos[s][2]});
            table = sc.z - p.pad_r[s] <= p.table_touch;
        }
        t[3] = right ? -p.w_right_arm * dt : 0.0f;
        t[4] = table ? -p.w_table * dt : 0.0f;
        // the cube: pushed on the table, slammed in the tray
        const float* cv = qd + p.cube_d;  // world-frame linear velocity of the cube's free joint
        const float vxy = sqrtf(cv[0] * cv[0] + cv[1] * cv[1]), v3 = sqrtf(vxy * vxy + cv[2] * cv[2]);
        const bool on_table = c.z < kHalf + 0.01f;
        const bool in_t = in_tray(p, c, tx, ty);
        if (on_table && !in_t && !g) t[5] = -p.w_push * vxy * dt;
        if (in_t && !g) t[7] = -p.w_slam * v3 * dt;
        // approach: the TCP's speed (FK a little ahead along the joint velocities)
        if (!g && length(h - c) < p.approach_dist) {
            const float lead = 0.01f;
            float qa[kMaxQ];
            for (int i = 0; i < v.model.nq; ++i) qa[i] = q[i];
            for (int k = 0; k < kArm; ++k) qa[p.q[k]] += lead * qd[p.d[k]];
            Fk fa;
            forward_kinematics(v.model, qa, fa.w);
            const float speed = length(tcp_pos(p, fa) - h) / lead;
            t[6] = -p.w_approach * fmaxf(0.0f, speed - p.approach_speed) * dt;
        }
        // jerk: the arm joints' mean accelerations over this step (velocity change / 0.02
        // s; the last physics step's qacc chatters with the grasp contacts)
        if (v.task_state) {
            const float* q0 = v.task_state + size_t(e) * kTaskState;
            float s2 = 0.0f;
            for (int k = 0; k < kArm; ++k) {
                const float acc = (qd[p.d[k]] - q0[k]) / dt;
                s2 += acc * acc;
            }
            t[8] = -p.w_jerk * s2 * dt;
        }
        // action rate: the change of the move and turn commands from the last step
        const float* prev = v.prev_action + size_t(e) * kActDim;
        float r2 = 0.0f;
        for (int k = 0; k < 4; ++k) r2 += (act[k] - prev[k]) * (act[k] - prev[k]);
        t[9] = -p.w_rate * r2 * dt;
    }
    FORGE_HD static float reward(const Params& p, const EnvView& v, int e, const float* a) {
        float t[kRewardTerms];
        reward_terms(p, v, e, a, t);
        return combine_terms(p, t);
    }

    FORGE_HD static void reset(const Params& p, const EnvView& v, int e, Rng& rng) {
        float* q = v.qpos(e);
        float* ctrl = v.ctrl + size_t(e) * v.nu;
        // cube: anywhere in the work area, at least cube_tray_gap from the tray's center
        // in x or y (else the try farthest from it)
        float tx, ty;
        tray_spawn(p, v, e, tx, ty);
        q[p.tray_q[0]] = tx, q[p.tray_q[1]] = ty;
        float x = 0.0f, y = 0.0f, best = -1.0f;
        for (int tries = 0; tries < 32; ++tries) {
            float cx, cy;
            area_point(p, rng, cx, cy);
            const float gap = fmaxf(fabsf(cx - tx), fabsf(cy - ty));
            if (gap > best) best = gap, x = cx, y = cy;
            if (gap >= p.cube_tray_gap) break;
        }
        const float yaw = rng.uniform(-3.14159265f, 3.14159265f);
        float* c = q + p.cube_q;
        c[0] = x, c[1] = y, c[2] = kHalf;
        c[3] = cosf(0.5f * yaw), c[4] = 0.0f, c[5] = 0.0f, c[6] = sinf(0.5f * yaw);
        // the arm at home, each joint a little off; fingers open
        for (int k = 0; k < kArm; ++k) q[p.q[k]] = p.start[k] + rng.uniform(-p.home_joint_noise, p.home_joint_noise);
        q[p.q[7]] = q[p.q[8]] = p.finger_open;
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
        float tx, ty;
        tray_center(p, v, e, tx, ty);
        const Vec3 tray{tx, ty, 0.0f};
        // closing axis across a pair of faces: the face nearest the current turn, inside
        // the turns the arm reaches above the cube
        float d = wrap_pi(cube_yaw(p, v, e) - turn_m);
        d -= 1.5707963f * rintf(d / 1.5707963f);
        float want = turn_m + d;
        const float pref = turn_pref(p, c.x, c.y);
        while (want > pref + p.max_turn) want -= 1.5707963f;
        while (want < pref - p.max_turn) want += 1.5707963f;
        const float grasp_z = c.z;
        const float dxy = sqrtf((c.x - hm.x) * (c.x - hm.x) + (c.y - hm.y) * (c.y - hm.y));
        const float dturn = fabsf(wrap_pi(want - turn_m));
        const float dz = hm.z - grasp_z;
        const Vec3 dpad = c - hm;
        const bool between = sqrtf(dpad.x * dpad.x + dpad.y * dpad.y) < 0.02f && dpad.z > -0.035f && dpad.z < 0.02f;
        const bool holding = open > p.exp_hold_lo && open < p.exp_hold_hi && between;
        Vec3 goal{c.x, c.y, grasp_z + p.hover_dz};
        float grip = -1.0f, turn_goal = want;
        float speed = p.expert_speed;
        if (in_tray(p, c, tx, ty) && !holding) {  // placed: open fully, up if low by the cube, then home
            turn_goal = plan_turn;
            // From the plan (not the lagging measured TCP: deciding on it flip-flopped
            // between going up and going home): straight up by the cube, over at return_z
            // (the hand turns level up there, clear of the cube), down to home at the end.
            const float dhome = sqrtf((h.x - p.home_tcp[0]) * (h.x - p.home_tcp[0]) +
                                      (h.y - p.home_tcp[1]) * (h.y - p.home_tcp[1]));
            const float dcube = sqrtf((h.x - c.x) * (h.x - c.x) + (h.y - c.y) * (h.y - c.y));
            speed = p.expert_return_speed;
            // once the fingers let go of the cube (past the holding range) go up: a wall
            // may stop a finger from opening fully when the cube sits near it; they open
            // fully above the walls
            if (open < p.exp_hold_hi)
                goal = h;
            else if (dhome > 0.05f && h.z < p.return_z - 0.005f && dcube < 0.08f)
                goal = Vec3{h.x, h.y, p.return_z};
            else if (dhome > 0.05f)
                goal = Vec3{p.home_tcp[0], p.home_tcp[1], p.return_z}, turn_goal = p.turn0;
            else
                goal = home_vec(p), turn_goal = p.turn0;
        } else if (holding) {
            grip = 1.0f;
            // keep the turn away from the window's edges (the arm tracks badly there)
            const float tp = turn_pref(p, h.x, h.y);
            turn_goal = fminf(fmaxf(plan_turn, tp - 0.4f), tp + 0.4f);
            // TCP goals from where the cube sits in the hand (measured): the goal does
            // not move with the arm's lag behind its plan (a goal of plan + (tray - cube)
            // circled around the tray).
            const Vec3 off = hm - c;
            const float ex = tray.x - c.x, ey = tray.y - c.y, dtray = sqrtf(ex * ex + ey * ey);
            // over the tray: within 1 cm when high, 2.5 cm when low (going down; not back
            // up for a small drift)
            const float high = fminf(fmaxf((c.z - kHalf) / 0.05f, 0.0f), 1.0f);
            if (dtray > 0.025f - 0.015f * high) {
                if (hm.z < p.carry_z - 0.02f && dtray > 0.03f)
                    goal = Vec3{h.x, h.y, p.carry_z};  // lift first
                else
                    goal = Vec3{tray.x + off.x, tray.y + off.y, p.carry_z};  // carry over the tray
            } else if (c.z > tray_floor(p) + kHalf + 0.010f) {
                goal = Vec3{tray.x + off.x, tray.y + off.y, tray_floor(p) + kHalf + 0.004f + off.z};  // down
            } else {
                goal = h;  // low over the tray: let go
                grip = -1.0f;
            }
        } else if (open <= p.exp_empty) {  // closed on nothing: open, back up
            goal = Vec3{h.x, h.y, fmaxf(h.z, grasp_z + p.hover_dz)};
        } else if ((dxy < p.exp_xy && dturn < p.exp_turn && fabsf(dz) < p.exp_z) ||
                   (open < p.finger_open - 0.05f && fabsf(dz) < 2.0f * p.exp_z && dxy < 2.0f * p.exp_xy)) {
            goal = h;  // at the grasp pose (or closing there): close, hold still
            grip = 1.0f;
            turn_goal = plan_turn;
        } else if ((dxy < p.exp_xy && dturn < p.exp_turn) ||
                   (dz < p.hover_dz - 0.01f && dxy < 2.0f * p.exp_xy && dturn < 2.0f * p.exp_turn)) {
            goal = Vec3{c.x, c.y, grasp_z};  // above and turned (or on the way down): go down
        }
        const float s = speed;
        // up or down slowly where the hand turns between pointing down and level (90
        // degrees over 3 cm: at full speed a jerk the wrist servos lag far behind)
        const bool turning = h.z > p.approach_z_lo - 0.005f && h.z < p.approach_z_hi + 0.005f;
        const float sz = turning ? p.expert_blend_speed : s;
        a[0] = fminf(fmaxf((goal.x - h.x) / p.move, -s), s);
        a[1] = fminf(fmaxf((goal.y - h.y) / p.move, -s), s);
        a[2] = fminf(fmaxf((goal.z - h.z) / p.move, -sz), sz);
        a[3] = fminf(fmaxf(wrap_pi(turn_goal - plan_turn) / p.turn, -s), s);
        a[4] = grip;
    }
};

}  // namespace forge
