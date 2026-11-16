#pragma once
// Panda pick task (Phase 9b): lift a cube from the floor with the Franka Panda.
// assets/robots/panda/panda.urdf (fixed base, 7 arm joints, 2 finger joints;
// Menagerie's armature 0.1 and damping 1) plus a free 5 cm cube (edit_model).
//
//   action (5), end-effector space, a in [-1, 1]:
//      hand move dx dy dz = 0.02 m * a[0..2] per step, turn about z = 0.1 rad * a[3],
//      gripper a[4]: -1 open (fingers at 0.04 m), +1 closed (target 0), in between linear.
//      The hand target is FK(the arm's current joint targets) + the move: the joint
//      targets (ctrl) are the plan, so no other state is kept. Three IK steps per
//      control step (forge/robots/ik.h, with the posture term toward the start
//      posture, so the elbow stays natural) give the new joint targets. The target
//      stays in a box over the table that the hand can reach pointing down (checked
//      on a 5 cm grid at turns 0 and +-1.2): x 0.30-0.65 m, y -0.30-0.30 m, z from
//      1 cm above the floor (lowest finger pad) to 0.40 m, turn within +-1.2 rad.
//      (A larger box let random actions push the arm against its joint limits,
//      stretched and tilted, from where IK could not come back.)
//   control: arm PD kp 1000 / kv 30, fingers kp 200 / kv 10 (about 5 N grip on the
//      cube), forces capped by the URDF effort; gravity compensation torques for
//      the arm and fingers (the real Panda's controller does the same).
//   obs (39): joint positions (9: arm, finger 1, finger 2), joint velocities (9),
//      hand position (3), sin and cos of the hand's turn (2), finger opening (1),
//      cube position (3), cube quaternion w x y z (4, w >= 0), cube minus the
//      finger-pad center (3), last action (5). No noise (state input; Phase 9c
//      adds camera images).
//   reward terms: reach 1 - tanh(10 |pad center - cube|), grasp 0.5 when both
//      fingers touch the cube, lift 2 clamp((cube height - rest height) / 0.1, 0, 1),
//      action_rate -0.01 sum (a - a_prev)^2. Success (for the data): cube lifted
//      more than 8 cm.
//   episode: 250 steps (5 s at 50 Hz), no early end.
//   reset: the hand at a random spot (x 0.35-0.60 m, y -0.25-0.25 m, z 0.20-0.35 m),
//      pointing down, turned -0.8..0.8 rad: IK (40 steps with the posture term)
//      from the start posture (hand 0.25 m above the table center); fingers open.
//      Cube on the floor anywhere the hand can reach it (x 0.32-0.63 m, y
//      -0.28-0.28 m: the reachable box minus 2 cm), turned anywhere (full circle).
//   randomization: floor friction 0.6-1.2, cube mass x0.5-2 (0.05-0.2 kg).
//   expert (scripted, uses the true cube pose): 0-1.8 s hand 10 cm above the cube,
//      turned to the cube's nearest face (the cube looks the same every 90
//      degrees); 1.8-2.8 s down to the cube; 2.8-3.3 s close; then lift 15 cm and
//      hold. Moves at most 0.4 of full speed (0.4 m/s, 2 rad/s).
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

struct PandaPickTask {
    static constexpr const char* kName = "panda_pick";
    static constexpr const char* kUrdf = "robots/panda/panda.urdf";
    static constexpr bool kFloating = false;
    static constexpr int kArm = 7, kJoints = 9, kPads = 6, kMaxQ = 32;
    static constexpr int kObsDim = 39, kActDim = 5, kMaxEpisodeSteps = 250;
    static constexpr int kControlHz = 50;
    static constexpr float kHalf = 0.025f;  // cube half size

    struct Params {
        int q[kJoints] = {}, d[kJoints] = {};  // arm joints 1-7, finger 1, finger 2
        float lo[kJoints] = {}, hi[kJoints] = {};
        IkJoints ik;                           // the 7 arm joints, posture = start posture
        float start[kJoints] = {};             // start posture
        int hand = 0, cube = 0, cube_q = 0, cube_d = 0;
        float down[4] = {};      // hand orientation at turn 0 (x y z w)
        float hand_yaw0 = 0.0f;  // world yaw of the hand's x axis at turn 0
        float pad_z = 0.0584f + 0.0445f;  // finger pad center below the hand origin
        int pad_body[kPads] = {}, pad_side[kPads] = {};
        float pad_pos[kPads][3] = {}, pad_r[kPads] = {};
        float move = 0.02f, turn = 0.1f, finger_open = 0.04f;
        float box_lo[3] = {0.30f, -0.30f, 0.0f}, box_hi[3] = {0.65f, 0.30f, 0.40f}, max_turn = 1.2f;
        int ik_steps = 3;
        float posture_gain = 0.2f;
        float w_reach = 1.0f, w_grasp = 0.5f, w_lift = 2.0f, w_action = 0.01f, touch = 0.002f;
        float success_height = 0.08f;
        float cube_x_lo = 0.32f, cube_x_hi = 0.63f, cube_y = 0.28f, cube_turn = 3.14159265f;
        float hand_lo[3] = {0.35f, -0.25f, 0.20f}, hand_hi[3] = {0.60f, 0.25f, 0.35f}, hand_turn0 = 0.8f;
        int reset_ik_steps = 40;
        int t_down = 90, t_close = 140, t_lift = 165;  // expert phases (control steps)
        bool randomize = true;
        float friction_lo = 0.6f, friction_hi = 1.2f, cube_mass_lo = 0.5f, cube_mass_hi = 2.0f;
        float expert_speed = 0.4f;
    };

    // ---- host ----
    static void urdf_options(UrdfOptions& u) {
        u.armature = 0.1f;  // Menagerie's panda.xml
        u.damping = 1.0f;
    }
    static void edit_model(RobotModel& m) { add_free_box(m, "cube", {kHalf, kHalf, kHalf}, 0.1f); }
    static std::vector<int> joint_bodies(const RobotModel& m) {
        std::vector<int> j;
        for (int k = 1; k <= kArm; ++k) j.push_back(m.joint_index("joint" + std::to_string(k)));
        j.push_back(m.joint_index("finger_joint1"));
        j.push_back(m.joint_index("finger_joint2"));
        for (int b : j)
            if (b < 0) throw std::runtime_error("panda_pick: the Panda URDF needs joint1-7 and finger_joint1-2");
        return j;
    }
    static Params make_params(const RobotModel& m) {
        Params p;
        if (m.nq > kMaxQ) throw std::runtime_error("panda_pick: nq too large");
        const std::vector<int> j = joint_bodies(m);
        for (int k = 0; k < kJoints; ++k) {
            p.q[k] = m.qadr[j[k]];
            p.d[k] = m.dadr[j[k]];
            p.lo[k] = m.joint_lower[j[k]];
            p.hi[k] = m.joint_upper[j[k]];
        }
        p.hand = m.body_index("hand");
        p.cube = m.body_index("cube");
        if (p.hand < 0 || p.cube < 0) throw std::runtime_error("panda_pick: needs bodies 'hand' and 'cube'");
        p.cube_q = m.qadr[p.cube];
        p.cube_d = m.dadr[p.cube];
        // Start posture: from Menagerie's home pose, the hand 0.25 m above the table center.
        std::vector<float> q(m.nq, 0.0f);
        const float home[kArm] = {0.0f, 0.0f, 0.0f, -1.57079f, 0.0f, 1.57079f, -0.7853f};
        for (int k = 0; k < kArm; ++k) q[p.q[k]] = home[k];
        q[p.cube_q + 3] = 1.0f;
        RobotWork work(m);
        DynamicsWork w = work.view();
        forward_kinematics(m.view(), q.data(), w);
        const Quat down = load_quat(w.xquat + 4 * p.hand);
        p.down[0] = down.x, p.down[1] = down.y, p.down[2] = down.z, p.down[3] = down.w;
        const Mat3 Rd = quat_to_mat3(down);
        p.hand_yaw0 = std::atan2(Rd(1, 0), Rd(0, 0));
        const std::vector<int> arm(j.begin(), j.begin() + kArm);
        const IkResult r = solve_ik(m, p.hand, {0.5f, 0.0f, 0.25f}, down, arm, q);
        if (r.pos_error > 1e-4f) throw std::runtime_error("panda_pick: start pose IK failed");
        for (int k = 0; k < kArm; ++k) p.start[k] = q[p.q[k]];
        p.start[7] = p.start[8] = p.finger_open;
        std::vector<float> posture(p.start, p.start + kArm);
        p.ik = make_ik_joints(m, arm, posture);
        p.box_lo[2] = p.pad_z + 0.01f;
        // finger pad spheres
        if (static_cast<int>(m.spheres.size()) != kPads) throw std::runtime_error("panda_pick: expected 6 finger pad spheres");
        for (int s = 0; s < kPads; ++s) {
            const auto& sp = m.spheres[s];
            p.pad_body[s] = sp.body;
            p.pad_side[s] = m.body_name[sp.body] == "left_finger" ? 0 : 1;
            p.pad_pos[s][0] = sp.pos.x, p.pad_pos[s][1] = sp.pos.y, p.pad_pos[s][2] = sp.pos.z;
            p.pad_r[s] = sp.radius;
        }
        return p;
    }
    static Actuators make_actuators(const RobotModel& m) {
        Actuators a;
        const std::vector<int> j = joint_bodies(m);
        for (int k = 0; k < kJoints; ++k) {
            const bool finger = k >= kArm;
            a.add_position(m, m.joint_name[j[k]], finger ? 200.0f : 1000.0f, finger ? 10.0f : 30.0f, 0.0f, 0.0f);
        }
        return a;
    }
    static void setup_constraints(ConstraintModel& c) {
        c.opt.impratio = 10.0f;  // MuJoCo's usual value for grasping: little friction creep
        c.add_ground();
    }

    // ---- cameras (Phase 9c, host) ----
    // Two cameras per env: 0 = front (fixed, across the table from the robot,
    // looking at the cube area: x 0.32-0.63, y +-0.28, floor to 20 cm up, all in
    // view: tests/flux/test_pick_cameras.cpp), 1 = wrist (on the hand, 7 cm beside
    // the fingers, aimed just past the finger pads; turns with the hand).
    struct Camera {
        Vec3 eye, target, up;
        float fovy;  // rad
    };
    static constexpr int kCameras = 2;
    static void cameras(const Params& p, const ModelView& m, const float* qpos, DynamicsWork& w, Camera* out) {
        out[0] = Camera{{0.92f, 0.0f, 0.80f}, {0.45f, 0.0f, 0.12f}, {0.0f, 0.0f, 1.0f}, 0.92f};
        forward_kinematics(m, qpos, w);
        const Vec3 h = load3(w.xpos + 3 * p.hand);
        const Mat3 R = quat_to_mat3(load_quat(w.xquat + 4 * p.hand));
        // hand frame: z along the fingers (down), y the closing direction, x across
        out[1] = Camera{h + R * Vec3{0.07f, 0.0f, 0.0f}, h + R * Vec3{0.0f, 0.0f, 0.12f}, R * Vec3{1.0f, 0.0f, 0.0f},
                        1.0f};  // aimed just past the finger pads: the cube is centered at the grasp
    }

    // ---- per env ----
    struct Fk {
        float xpos[3 * kMaxBodies], xquat[4 * kMaxBodies];
        SpatialVec S[kMaxDofs];
        DynamicsWork w{xpos, xquat, S, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    };
    FORGE_HD static Quat down_quat(const Params& p) { return Quat{p.down[0], p.down[1], p.down[2], p.down[3]}; }
    FORGE_HD static float wrap_pi(float a) {
        while (a > 3.14159265f) a -= 6.2831853f;
        while (a < -3.14159265f) a += 6.2831853f;
        return a;
    }
    // Turn of the hand about z relative to the start orientation.
    FORGE_HD static float hand_turn(const Params& p, const Fk& f) {
        const Mat3 R = quat_to_mat3(load_quat(f.xquat + 4 * p.hand)) * transpose(quat_to_mat3(down_quat(p)));
        return atan2f(R(1, 0), R(0, 0));
    }
    // qpos with the arm at its joint targets (the plan).
    FORGE_HD static void plan_qpos(const Params& p, const EnvView& v, int e, float* qp) {
        const float* q = v.qpos(e);
        for (int i = 0; i < v.model.nq; ++i) qp[i] = q[i];
        for (int k = 0; k < kArm; ++k) qp[p.q[k]] = v.ctrl[size_t(e) * v.nu + k];
    }
    FORGE_HD static Vec3 pad_center(const Params& p, const Fk& f) {
        return load3(f.xpos + 3 * p.hand) + rotate(load_quat(f.xquat + 4 * p.hand), Vec3{0.0f, 0.0f, p.pad_z});
    }
    FORGE_HD static Vec3 cube_pos(const Params& p, const EnvView& v, int e) { return load3(v.qpos(e) + p.cube_q); }
    FORGE_HD static float cube_yaw(const Params& p, const EnvView& v, int e) {
        const Quat c = normalize(load_quat_wxyz(v.qpos(e) + p.cube_q + 3));
        return atan2f(2.0f * (c.w * c.z + c.x * c.y), 1.0f - 2.0f * (c.y * c.y + c.z * c.z));
    }

    FORGE_HD static void apply_action(const Params& p, const EnvView& v, int e, const float* a) {
        float qp[kMaxQ];
        plan_qpos(p, v, e, qp);
        Fk f;
        forward_kinematics(v.model, qp, f.w);
        const Vec3 h = load3(f.xpos + 3 * p.hand);
        Vec3 t{h.x + p.move * a[0], h.y + p.move * a[1], h.z + p.move * a[2]};
        t.x = fminf(fmaxf(t.x, p.box_lo[0]), p.box_hi[0]);
        t.y = fminf(fmaxf(t.y, p.box_lo[1]), p.box_hi[1]);
        t.z = fminf(fmaxf(t.z, p.box_lo[2]), p.box_hi[2]);
        const float turn = fminf(fmaxf(hand_turn(p, f) + p.turn * a[3], -p.max_turn), p.max_turn);
        const Quat rot = quat_from_axis_angle({0, 0, 1}, turn) * down_quat(p);
        IkOptions o;
        o.posture_gain = p.posture_gain;
        IkResult r;
        for (int k = 0; k < p.ik_steps; ++k) {
            if (k > 0) forward_kinematics(v.model, qp, f.w);
            ik_step(v.model, f.w, p.hand, t, rot, p.ik, qp, o, r);
        }
        float* ctrl = v.ctrl + size_t(e) * v.nu;
        for (int k = 0; k < kArm; ++k) ctrl[k] = qp[p.q[k]];
        const float g = fminf(fmaxf(a[4], -1.0f), 1.0f);
        ctrl[7] = ctrl[8] = p.finger_open * 0.5f * (1.0f - g);
        // gravity compensation at the actual pose (arm tree = root body 0)
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
        const Vec3 h = load3(f.xpos + 3 * p.hand);
        const float turn = hand_turn(p, f);
        o[18] = h.x, o[19] = h.y, o[20] = h.z;
        o[21] = sinf(turn), o[22] = cosf(turn);
        o[23] = q[p.q[7]] + q[p.q[8]];
        const Vec3 c = cube_pos(p, v, e);
        Quat cq = normalize(load_quat_wxyz(q + p.cube_q + 3));
        if (cq.w < 0.0f) cq = Quat{-cq.x, -cq.y, -cq.z, -cq.w};
        o[24] = c.x, o[25] = c.y, o[26] = c.z;
        o[27] = cq.w, o[28] = cq.x, o[29] = cq.y, o[30] = cq.z;
        const Vec3 d = c - pad_center(p, f);
        o[31] = d.x, o[32] = d.y, o[33] = d.z;
        for (int k = 0; k < kActDim; ++k) o[34 + k] = v.last_action[size_t(e) * kActDim + k];
    }
    FORGE_HD static bool failed(const Params&, const EnvView&, int) { return false; }

    // Both fingers touch the cube: a pad sphere of each within 'touch' of the box.
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
    FORGE_HD static bool success(const Params& p, const EnvView& v, int e) {
        return cube_pos(p, v, e).z - kHalf > p.success_height;
    }

    static constexpr int kRewardTerms = 4;
    static const char* reward_term_name(int k) {
        static const char* const n[kRewardTerms] = {"reach", "grasp", "lift", "action_rate"};
        return n[k];
    }
    FORGE_HD static float combine_terms(const Params&, const float* t) { return t[0] + t[1] + t[2] + t[3]; }
    FORGE_HD static void reward_terms(const Params& p, const EnvView& v, int e, const float* a, float* t) {
        Fk f;
        forward_kinematics(v.model, v.qpos(e), f.w);
        const Vec3 c = cube_pos(p, v, e);
        t[0] = p.w_reach * (1.0f - tanhf(10.0f * length(c - pad_center(p, f))));
        t[1] = grasped(p, v, e, f) ? p.w_grasp : 0.0f;
        t[2] = p.w_lift * fminf(fmaxf((c.z - kHalf) / 0.1f, 0.0f), 1.0f);
        const float* prev = v.prev_action + size_t(e) * kActDim;
        float act2 = 0.0f;
        for (int k = 0; k < kActDim; ++k) act2 += (a[k] - prev[k]) * (a[k] - prev[k]);
        t[3] = -p.w_action * act2;
    }
    FORGE_HD static float reward(const Params& p, const EnvView& v, int e, const float* a) {
        float t[kRewardTerms];
        reward_terms(p, v, e, a, t);
        return combine_terms(p, t);
    }

    FORGE_HD static void reset(const Params& p, const EnvView& v, int e, Rng& rng) {
        float* q = v.qpos(e);
        float* ctrl = v.ctrl + size_t(e) * v.nu;
        const float x = rng.uniform(p.cube_x_lo, p.cube_x_hi), y = rng.uniform(-p.cube_y, p.cube_y);
        const float yaw = rng.uniform(-p.cube_turn, p.cube_turn);
        float* c = q + p.cube_q;
        c[0] = x, c[1] = y, c[2] = kHalf;
        c[3] = cosf(0.5f * yaw), c[4] = 0.0f, c[5] = 0.0f, c[6] = sinf(0.5f * yaw);
        // the hand at a random start spot: IK from the start posture
        for (int k = 0; k < kJoints; ++k) q[p.q[k]] = p.start[k];
        const Vec3 t{rng.uniform(p.hand_lo[0], p.hand_hi[0]), rng.uniform(p.hand_lo[1], p.hand_hi[1]),
                     rng.uniform(p.hand_lo[2], p.hand_hi[2])};
        const Quat rot = quat_from_axis_angle({0, 0, 1}, rng.uniform(-p.hand_turn0, p.hand_turn0)) * down_quat(p);
        IkOptions o;
        o.posture_gain = p.posture_gain;
        IkResult r;
        Fk f;
        for (int k = 0; k < p.reset_ik_steps; ++k) {
            forward_kinematics(v.model, q, f.w);
            ik_step(v.model, f.w, p.hand, t, rot, p.ik, q, o, r);
        }
        for (int k = 0; k < kJoints; ++k) ctrl[k] = q[p.q[k]];
        if (!p.randomize) return;
        if (v.rand_friction) v.rand_friction[e] = rng.uniform(p.friction_lo, p.friction_hi);
        if (v.rand_body_scale) {
            float* s = v.rand_body_scale + size_t(v.model.nbody) * e;
            for (int b = 0; b < v.model.nbody; ++b) s[b] = 1.0f;
            s[p.cube] = rng.uniform(p.cube_mass_lo, p.cube_mass_hi);
        }
    }

    // Scripted expert (see the top of this file). Proportional moves toward the
    // phase's hand target, from the planned hand pose (the joint targets).
    FORGE_HD static void expert(const Params& p, const EnvView& v, int e, float* a) {
        const int t = v.episode_step[e];
        float qp[kMaxQ];
        plan_qpos(p, v, e, qp);
        Fk f;
        forward_kinematics(v.model, qp, f.w);
        const Vec3 h = load3(f.xpos + 3 * p.hand);
        const Vec3 c = cube_pos(p, v, e);
        // cube face nearest the hand's start turn (the cube looks the same every 90 degrees)
        float want = wrap_pi(cube_yaw(p, v, e) - p.hand_yaw0);
        want -= 1.5707963f * rintf(want / 1.5707963f);
        Vec3 goal = h;
        float grip = -1.0f;
        float turn_goal = want;
        if (t < p.t_down) {
            goal = Vec3{c.x, c.y, c.z + p.pad_z + 0.10f};
        } else if (t < p.t_close) {
            goal = Vec3{c.x, c.y, c.z + p.pad_z};
        } else if (t < p.t_lift) {
            goal = Vec3{h.x, h.y, h.z};
            grip = 1.0f;
            turn_goal = hand_turn(p, f);
        } else {
            goal = Vec3{h.x, h.y, kHalf + p.pad_z + 0.15f};
            grip = 1.0f;
            turn_goal = hand_turn(p, f);
        }
        const float s = p.expert_speed;
        a[0] = fminf(fmaxf((goal.x - h.x) / p.move, -s), s);
        a[1] = fminf(fmaxf((goal.y - h.y) / p.move, -s), s);
        a[2] = fminf(fmaxf((goal.z - h.z) / p.move, -s), s);
        a[3] = fminf(fmaxf(wrap_pi(turn_goal - hand_turn(p, f)) / p.turn, -s), s);
        a[4] = grip;
    }
};

}  // namespace forge
