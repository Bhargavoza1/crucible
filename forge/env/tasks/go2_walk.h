#pragma once
// Go2 walk task (doc Phase 8, quadruped): follow a velocity command.
// assets/robots/go2/go2.urdf: free base + 12 leg joints (per leg: hip, thigh, calf;
// legs FL, FR, RL, RR), one foot sphere per leg (radius 22 mm). Joint armature 0.01
// and damping 0.1 as Unitree's MuJoCo Go2 (unitree_mujoco go2.xml).
// Settings follow unitree_rl_gym's Go2 (go2_config.py, legged_robot_config.py),
// with reward scales not multiplied by dt, as g1_walk.
//
//   command:   (vx, vy, wz) in the base frame, drawn per env every 10 s: vx, vy in
//              [-1, 1] m/s, wz in [-1, 1] rad/s (legged_gym's ranges); |(vx, vy)| < 0.2
//              gives vx = vy = 0. No stored state; a task input (3 numbers) replaces it.
//   gait clock: trot, phase = (t mod 0.5 s) / 0.5; FL and RR at phase, FR and RL at
//              phase + 0.5. (legged_gym's feet_air_time reward needs per-foot state
//              kept between steps; the clock, as in g1_walk, needs none.)
//   obs (50):  base angular velocity (3), gravity in the base frame (3), base linear
//              velocity (3, base frame), command (3), joint positions minus the
//              standing pose (12), joint velocities (12), last action (12),
//              sin and cos of 2 pi phase (2). Noise as g1_walk (legged_gym's levels).
//   action (12): PD targets = standing pose + a, a in [-1, 1] rad. PD kp 20, kv 0.5
//              (legged_gym Go2); force capped by each joint's URDF effort (23.7 / 23.7
//              / 45.43 Nm). Action delay 0 or 1 control step per episode.
//   reward:
//      + 1.0  exp(-|cmd_xy - v_xy|^2 / 0.25)       linear velocity tracking (base frame)
//      + 0.5  exp(-(cmd_wz - w_z)^2 / 0.25)        yaw rate tracking
//      - 2.0  v_z^2  - 0.05 (w_x^2 + w_y^2)        no bouncing, no rocking
//      - 2e-4 sum tau^2                            torques (legged_gym Go2)
//      - 2.5e-7 sum qacc^2                         joint acceleration
//      - 0.01 sum (a - a_prev)^2                   smooth actions
//      - 10   sum (distance outside 90% of each joint range)   (legged_gym Go2)
//      - 0.2  sum over feet touching the ground of |foot velocity|^2   no foot slip
//      - 20   sum over feet in the air of (foot height - 0.08)^2       lift the feet
//      + 0.1  per foot whose contact matches the trot clock (stance while phase < 0.5)
//      total clipped at 0 from below (legged_gym only_positive_rewards); a fall ends
//      the episode. Not here: legged_gym's thigh / calf collision penalty (the model
//      has collision spheres only on the feet).
//   randomization (per episode): ground friction 0.5-1.25 and base mass -1..+1 kg
//              (legged_gym), each link's mass x0.9-1.1, PD gains x0.9-1.1, motor
//              strength x0.9-1.1
//   pushes:    every 10 to 15 s (random per episode), a horizontal base velocity kick
//              of up to 1 m/s (legged_gym: every 15 s, up to 1 m/s)
//   terminated: base below 0.12 m above the ground under it, or tilted more than 60 degrees
//              (legged_gym ends the episode when the base touches the ground)
//   truncated: 1000 steps (20 s at 50 Hz)
//   reset:     standing pose, feet 2 cm above the ground, joints +-0.05 rad, small
//              velocities, facing +x; on terrain a random spot of the field
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "forge/dynamics/articulated.h"
#include "forge/dynamics/robot_work.h"
#include "forge/env/env_view.h"
#include "forge/env/terrain.h"
#include "forge/math/quat.h"
#include "forge/robots/actuators.h"
#include "forge/robots/urdf.h"
#include "forge/solver/constraint_model.h"
#include "forge/solver/heightfield.h"
#include "forge/util/rng.h"

namespace forge {

struct Go2WalkTask {
    static constexpr const char* kName = "go2_walk";
    static constexpr const char* kUrdf = "robots/go2/go2.urdf";
    static constexpr bool kFloating = true;
    static constexpr int kJoints = 12, kFeet = 4;
    static constexpr int kObsDim = 12 + 3 * kJoints + 2, kActDim = kJoints, kMaxEpisodeSteps = 1000;
    static constexpr int kControlHz = 50;
    static constexpr int kTaskInputDim = 3;  // the command: vx, vy, wz

    struct Params {
        int base_q = 0, base_d = 0, base = 0;
        float base_mass = 1.0f;
        int q[kJoints] = {}, d[kJoints] = {};
        float q0[kJoints] = {};  // standing pose
        float kp[kJoints] = {}, kv[kJoints] = {}, fmax[kJoints] = {};
        float lo[kJoints] = {}, hi[kJoints] = {};  // 90% of each joint's range
        float action_scale = 1.0f;
        float stand_height = 0.0f;  // base height when standing (set from the model)
        float min_height = 0.12f, min_up = 0.5f;  // 60 degrees
        // feet: the collision sphere of each *_foot link, in model order
        int foot_body[kFeet] = {};
        float foot_pos[kFeet][3] = {}, foot_r[kFeet] = {};
        float foot_offset[kFeet] = {};  // trot clock: 0 (FL, RR) or 0.5 (FR, RL)
        float contact_eps = 0.005f;     // touching: lowest point within 5 mm of the ground
        // command and gait
        float cmd_vx = 1.0f, cmd_vy = 1.0f, cmd_wz = 1.0f, cmd_small = 0.2f, cmd_every_s = 10.0f;
        float period_s = 0.5f, stance_phase = 0.5f;
        // reward weights
        float w_track = 1.0f, w_track_yaw = 0.5f, tracking_sigma = 0.25f;
        float w_vz = 2.0f, w_wxy = 0.05f, w_torque = 2e-4f, w_qacc = 2.5e-7f, w_action = 0.01f;
        float w_limits = 10.0f, w_slip = 0.2f, w_swing = 20.0f, swing_height = 0.08f, w_contact = 0.1f;
        // pushes
        float push_speed = 1.0f, push_min_s = 10.0f, push_max_s = 15.0f;
        // domain randomization
        bool randomize = true;
        float friction_lo = 0.5f, friction_hi = 1.25f, mass_lo = 0.9f, mass_hi = 1.1f;
        float load_lo = -1.0f, load_hi = 1.0f;
        float gain_lo = 0.9f, gain_hi = 1.1f, strength_lo = 0.9f, strength_hi = 1.1f;
        bool obs_noise = true, action_delay = true;
        float noise_ang_vel = 0.2f, noise_gravity = 0.05f, noise_lin_vel = 0.1f, noise_q = 0.01f, noise_qd = 1.5f;
        float reset_joint_noise = 0.05f, reset_vel_noise = 0.05f;
    };

    // ---- host ----
    static void urdf_options(UrdfOptions& u) {
        u.armature = 0.01f;  // Unitree's MuJoCo Go2 (go2.xml)
        u.damping = 0.1f;
    }
    // Standing pose by joint name (unitree_rl_gym go2_config.py).
    static float stand_angle(const std::string& name) {
        const bool left = name.rfind("FL", 0) == 0 || name.rfind("RL", 0) == 0;
        const bool rear = name.rfind("R", 0) == 0;
        if (name.find("hip") != std::string::npos) return left ? 0.1f : -0.1f;
        if (name.find("thigh") != std::string::npos) return rear ? 1.0f : 0.8f;
        if (name.find("calf") != std::string::npos) return -1.5f;
        throw std::runtime_error("go2_walk: unexpected joint '" + name + "'");
    }
    static Params make_params(const RobotModel& m) {
        Params p;
        int k = 0;
        for (int b = 0; b < m.nbody(); ++b) {
            if (m.jtype[b] == kJointFree) {
                p.base_q = m.qadr[b];
                p.base_d = m.dadr[b];
                p.base = b;
                p.base_mass = m.mass[b];
            } else if (m.jtype[b] == kJointRevolute) {
                if (k >= kJoints) throw std::runtime_error("go2_walk: more than 12 hinge joints");
                p.q[k] = m.qadr[b];
                p.d[k] = m.dadr[b];
                p.q0[k] = stand_angle(m.joint_name[b]);
                p.kp[k] = 20.0f;
                p.kv[k] = 0.5f;
                p.fmax[k] = m.joint_effort[b];
                const float mid = 0.5f * (m.joint_lower[b] + m.joint_upper[b]);
                const float half = 0.5f * (m.joint_upper[b] - m.joint_lower[b]);
                p.lo[k] = mid - 0.9f * half;
                p.hi[k] = mid + 0.9f * half;
                ++k;
            }
        }
        if (k != kJoints) throw std::runtime_error("go2_walk: needs 12 hinge joints, found " + std::to_string(k));
        if (static_cast<int>(m.spheres.size()) != kFeet)
            throw std::runtime_error("go2_walk: expected 4 foot spheres, found " + std::to_string(m.spheres.size()));
        for (int f = 0; f < kFeet; ++f) {
            const int b = m.spheres[f].body;
            const std::string& n = m.body_name[b];
            if (n.rfind("FL", 0) == 0 || n.rfind("RR", 0) == 0) p.foot_offset[f] = 0.0f;
            else if (n.rfind("FR", 0) == 0 || n.rfind("RL", 0) == 0) p.foot_offset[f] = 0.5f;
            else throw std::runtime_error("go2_walk: foot sphere on unexpected body '" + n + "'");
            p.foot_body[f] = b;
            p.foot_pos[f][0] = m.spheres[f].pos.x;
            p.foot_pos[f][1] = m.spheres[f].pos.y;
            p.foot_pos[f][2] = m.spheres[f].pos.z;
            p.foot_r[f] = m.spheres[f].radius;
        }
        // Base height so the lowest foot sphere just touches the ground in the standing pose.
        RobotWork work(m);
        DynamicsWork w = work.view();
        std::vector<float> q(m.nq, 0.0f);
        q[p.base_q + 3] = 1.0f;
        for (int j = 0; j < kJoints; ++j) q[p.q[j]] = p.q0[j];
        forward_kinematics(m.view(), q.data(), w);
        float lowest = 1e9f;
        for (const auto& s : m.spheres) {
            const Vec3 c = load3(w.xpos + 3 * s.body) + rotate(load_quat(w.xquat + 4 * s.body), s.pos);
            lowest = std::fmin(lowest, c.z - s.radius);
        }
        p.stand_height = -lowest;
        return p;
    }
    // One PD actuator per leg joint, in the same order as Params (body order).
    static Actuators make_actuators(const RobotModel& m) {
        Actuators a;
        for (int b = 0; b < m.nbody(); ++b)
            if (m.jtype[b] == kJointRevolute) a.add_position(m, m.joint_name[b], 20.0f, 0.5f, 0.0f, 0.0f);
        return a;
    }
    static void setup_constraints(ConstraintModel& c) { c.add_ground(); }

    // ---- per env ----
    FORGE_HD static Quat base_quat(const Params& p, const EnvView& v, int e) {
        return normalize(load_quat_wxyz(v.qpos(e) + p.base_q + 3));
    }
    FORGE_HD static float ground(const EnvView& v, float x, float y) {
        return v.terrain.h ? hfield_height(v.terrain, x, y) : 0.0f;
    }
    FORGE_HD static float base_height(const Params& p, const EnvView& v, int e) {
        const float* q = v.qpos(e) + p.base_q;
        return q[2] - ground(v, q[0], q[1]);
    }
    FORGE_HD static int steps_per_command(const Params& p) { return int(p.cmd_every_s * kControlHz + 0.5f); }
    FORGE_HD static Vec3 command(const Params& p, const EnvView& v, int e, int step) {
        if (v.task_input) return {v.task_input[0], v.task_input[1], v.task_input[2]};
        Rng r(v.seed ^ 0xC0AA4Dull, uint64_t(e),
              (uint64_t(v.episode_index[e]) << 20) + uint64_t(step / steps_per_command(p)));
        Vec3 c{r.uniform(-p.cmd_vx, p.cmd_vx), r.uniform(-p.cmd_vy, p.cmd_vy), r.uniform(-p.cmd_wz, p.cmd_wz)};
        if (c.x * c.x + c.y * c.y < p.cmd_small * p.cmd_small) c.x = c.y = 0.0f;
        return c;
    }
    FORGE_HD static float phase(const Params& p, int step) {
        const float t = float(step) / float(kControlHz);
        return fmodf(t, p.period_s) / p.period_s;
    }
    FORGE_HD static int delay_of(const EnvView& v, int e) {
        Rng r(v.seed ^ 0xDE1A7ull, uint64_t(e), uint64_t(v.episode_index[e]));
        return r.uniform() < 0.5f ? 0 : 1;
    }
    FORGE_HD static bool push_now(const Params& p, const EnvView& v, int e, int step) {
        Rng r(v.seed ^ 0x9054ull, uint64_t(e), uint64_t(v.episode_index[e]));
        const float lo = p.push_min_s * kControlHz, hi = p.push_max_s * kControlHz;
        int t = 0;
        for (int n = 0; n < 64; ++n) {
            t += int(r.uniform(lo, hi));
            if (t == step) return true;
            if (t > step) return false;
        }
        return false;
    }
    FORGE_HD static const float* applied(const Params& p, const EnvView& v, int e, const float* a) {
        return (p.action_delay && delay_of(v, e) == 1) ? v.prev_action + size_t(e) * kActDim : a;
    }
    FORGE_HD static void apply_action(const Params& p, const EnvView& v, int e, const float* a) {
        const float* used = applied(p, v, e, a);
        for (int k = 0; k < kJoints; ++k) v.ctrl[size_t(e) * v.nu + k] = p.q0[k] + p.action_scale * used[k];
        const int step = v.episode_step[e];
        if (p.push_speed > 0.0f && push_now(p, v, e, step)) {
            Rng r(v.seed ^ 0xB0057ull, uint64_t(e), (uint64_t(v.episode_index[e]) << 20) + uint64_t(step));
            const float ang = r.uniform(0.0f, 6.2831853f), s = p.push_speed * sqrtf(r.uniform());
            float* qd = v.qvel(e);
            qd[p.base_d + 0] += s * cosf(ang);  // qvel of a free joint: world linear velocity
            qd[p.base_d + 1] += s * sinf(ang);
        }
    }
    FORGE_HD static void observe(const Params& p, const EnvView& v, int e, float* o) {
        const float* q = v.qpos(e);
        const float* qd = v.qvel(e);
        const int step = v.episode_step[e];
        const Quat rq = base_quat(p, v, e);
        const Vec3 g = rotate_inv(rq, Vec3{0, 0, -1});
        const Vec3 vl = rotate_inv(rq, load3(qd + p.base_d));
        const Vec3 c = command(p, v, e, step);
        const float ph = phase(p, step);
        o[0] = qd[p.base_d + 3];  // body-frame angular velocity
        o[1] = qd[p.base_d + 4];
        o[2] = qd[p.base_d + 5];
        o[3] = g.x;
        o[4] = g.y;
        o[5] = g.z;
        o[6] = vl.x;
        o[7] = vl.y;
        o[8] = vl.z;
        o[9] = c.x;
        o[10] = c.y;
        o[11] = c.z;
        for (int k = 0; k < kJoints; ++k) {
            o[12 + k] = q[p.q[k]] - p.q0[k];
            o[12 + kJoints + k] = qd[p.d[k]];
            o[12 + 2 * kJoints + k] = v.last_action[size_t(e) * kActDim + k];
        }
        o[12 + 3 * kJoints] = sinf(6.2831853f * ph);
        o[13 + 3 * kJoints] = cosf(6.2831853f * ph);
        if (!p.obs_noise) return;
        Rng r(v.seed ^ 0x0B5ull, uint64_t(e), (uint64_t(v.episode_index[e]) << 20) + uint64_t(step));
        for (int i = 0; i < 3; ++i) o[i] += p.noise_ang_vel * r.uniform(-1.0f, 1.0f);
        for (int i = 3; i < 6; ++i) o[i] += p.noise_gravity * r.uniform(-1.0f, 1.0f);
        for (int i = 6; i < 9; ++i) o[i] += p.noise_lin_vel * r.uniform(-1.0f, 1.0f);
        for (int k = 0; k < kJoints; ++k) {
            o[12 + k] += p.noise_q * r.uniform(-1.0f, 1.0f);
            o[12 + kJoints + k] += p.noise_qd * r.uniform(-1.0f, 1.0f);
        }
    }
    FORGE_HD static bool failed(const Params& p, const EnvView& v, int e) {
        const Vec3 up = rotate(base_quat(p, v, e), Vec3{0, 0, 1});
        return base_height(p, v, e) < p.min_height || up.z < p.min_up;
    }
    // Foot sphere centers, velocities and contact, from forward kinematics.
    struct FootState {
        Vec3 c[kFeet], vel[kFeet];
        float h[kFeet];  // center height above the ground under it
        bool touch[kFeet];
    };
    FORGE_HD static void foot_state(const Params& p, const EnvView& v, int e, FootState& fs) {
        float xpos[3 * kMaxBodies], xquat[4 * kMaxBodies];
        SpatialVec S[kMaxDofs];
        DynamicsWork w{xpos, xquat, S, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        const ModelView& m = v.model;
        forward_kinematics(m, v.qpos(e), w);
        const float* qd = v.qvel(e);
        for (int f = 0; f < kFeet; ++f) {
            const int b = p.foot_body[f];
            const Vec3 c = load3(xpos + 3 * b) +
                           rotate(load_quat(xquat + 4 * b), Vec3{p.foot_pos[f][0], p.foot_pos[f][1], p.foot_pos[f][2]});
            Vec3 vel{0, 0, 0};
            for (int k = b; k >= 0; k = m.parent[k]) {
                if (m.jtype[k] == kJointFixed) continue;
                for (int i = m.dadr[k]; i < m.dadr[k] + joint_nv(m.jtype[k]); ++i)
                    vel = vel + (S[i].lin + cross(S[i].ang, c)) * qd[i];
            }
            fs.c[f] = c;
            fs.vel[f] = vel;
            fs.h[f] = c.z - ground(v, c.x, c.y);
            fs.touch[f] = fs.h[f] - p.foot_r[f] <= p.contact_eps;
        }
    }

    static constexpr int kRewardTerms = 11;
    static const char* reward_term_name(int k) {
        static const char* const n[kRewardTerms] = {"track_lin_vel", "track_yaw",   "lin_vel_z", "ang_vel_xy",
                                                    "torques",       "joint_acc",   "action_rate", "joint_limits",
                                                    "foot_slip",     "swing_height", "contact"};
        return n[k];
    }
    FORGE_HD static float combine_terms(const Params&, const float* t) {
        float r = 0.0f;
        for (int k = 0; k < kRewardTerms; ++k) r += t[k];
        return fmaxf(r, 0.0f);
    }
    FORGE_HD static void reward_terms(const Params& p, const EnvView& v, int e, const float* a, float* t) {
        const float* q = v.qpos(e);
        const float* qd = v.qvel(e);
        const float* qacc = v.state.qacc ? v.state.qacc + size_t(e) * v.model.nv : nullptr;
        const float* prev = v.prev_action + size_t(e) * kActDim;
        const float* used = applied(p, v, e, a);
        const int step = v.episode_step[e];
        const Quat rq = base_quat(p, v, e);
        const Vec3 vl = rotate_inv(rq, load3(qd + p.base_d));
        const Vec3 w{qd[p.base_d + 3], qd[p.base_d + 4], qd[p.base_d + 5]};  // base frame
        const Vec3 c = command(p, v, e, step);

        const float ex = c.x - vl.x, ey = c.y - vl.y, ew = c.z - w.z;
        t[0] = p.w_track * expf(-(ex * ex + ey * ey) / p.tracking_sigma);
        t[1] = p.w_track_yaw * expf(-ew * ew / p.tracking_sigma);
        t[2] = -p.w_vz * vl.z * vl.z;
        t[3] = -p.w_wxy * (w.x * w.x + w.y * w.y);
        const float ks = v.rand_kp ? v.rand_kp[e] : 1.0f, kvs = v.rand_kv ? v.rand_kv[e] : 1.0f;
        const float ms = v.rand_strength ? v.rand_strength[e] : 1.0f;
        float tau2 = 0.0f, acc2 = 0.0f, act2 = 0.0f, lim = 0.0f;
        for (int k = 0; k < kJoints; ++k) {
            const float target = p.q0[k] + p.action_scale * used[k];
            const float tau = ms * fminf(fmaxf(ks * p.kp[k] * (target - q[p.q[k]]) - kvs * p.kv[k] * qd[p.d[k]],
                                               -p.fmax[k]), p.fmax[k]);  // the same force as apply_actuators
            tau2 += tau * tau;
            if (qacc) acc2 += qacc[p.d[k]] * qacc[p.d[k]];
            act2 += (a[k] - prev[k]) * (a[k] - prev[k]);
            const float x = q[p.q[k]];
            lim += fmaxf(p.lo[k] - x, 0.0f) + fmaxf(x - p.hi[k], 0.0f);
        }
        t[4] = -p.w_torque * tau2;
        t[5] = -p.w_qacc * acc2;
        t[6] = -p.w_action * act2;
        t[7] = -p.w_limits * lim;

        FootState fs;
        foot_state(p, v, e, fs);
        const float ph = phase(p, step);
        t[8] = t[9] = t[10] = 0.0f;
        for (int f = 0; f < kFeet; ++f) {
            if (fs.touch[f]) {
                t[8] -= p.w_slip * (fs.vel[f].x * fs.vel[f].x + fs.vel[f].y * fs.vel[f].y + fs.vel[f].z * fs.vel[f].z);
            } else {
                const float hz = fs.h[f] - p.swing_height;
                t[9] -= p.w_swing * hz * hz;
            }
            const bool stance = fmodf(ph + p.foot_offset[f], 1.0f) < p.stance_phase;
            if (fs.touch[f] == stance) t[10] += p.w_contact;
        }
    }
    FORGE_HD static float reward(const Params& p, const EnvView& v, int e, const float* a) {
        float t[kRewardTerms];
        reward_terms(p, v, e, a, t);
        return combine_terms(p, t);
    }
    FORGE_HD static void reset(const Params& p, const EnvView& v, int e, Rng& rng) {
        float* q = v.qpos(e);
        float* qd = v.qvel(e);
        q[p.base_q + 0] = 0.0f;
        q[p.base_q + 1] = 0.0f;
        q[p.base_q + 2] = p.stand_height + 0.02f;
        if (v.terrain.h) {  // terrain: a random spot at least 2 m from the field's edge,
            const HeightfieldView& t = v.terrain;  // lifted by the highest ground under the feet + 5 cm
            const float m = 2.0f, wdt = float(t.nx - 1) * t.cell, hgt = float(t.ny - 1) * t.cell;
            const float x = t.x0 + rng.uniform(m, wdt - m), y = t.y0 + rng.uniform(m, hgt - m);
            float g = ground(v, x, y);
            for (int k = 0; k < 4; ++k)
                g = fmaxf(g, ground(v, x + (k & 1 ? 0.2f : -0.2f), y + (k & 2 ? 0.15f : -0.15f)));
            q[p.base_q + 0] = x;
            q[p.base_q + 1] = y;
            q[p.base_q + 2] = g + p.stand_height + 0.05f;
        }
        q[p.base_q + 3] = 1.0f;  // w x y z: upright, facing +x
        q[p.base_q + 4] = q[p.base_q + 5] = q[p.base_q + 6] = 0.0f;
        for (int k = 0; k < kJoints; ++k) {
            const float j = p.q0[k] + rng.uniform(-p.reset_joint_noise, p.reset_joint_noise);
            q[p.q[k]] = j;
            qd[p.d[k]] = rng.uniform(-p.reset_vel_noise, p.reset_vel_noise);
            v.ctrl[size_t(e) * v.nu + k] = j;  // PD holds the start pose until the first action
        }
        for (int i = 0; i < 6; ++i) qd[p.base_d + i] = rng.uniform(-p.reset_vel_noise, p.reset_vel_noise);
        if (!p.randomize) return;
        if (v.rand_friction) v.rand_friction[e] = rng.uniform(p.friction_lo, p.friction_hi);
        if (v.rand_body_scale) {
            float* s = v.rand_body_scale + size_t(v.model.nbody) * e;
            for (int b = 0; b < v.model.nbody; ++b) s[b] = rng.uniform(p.mass_lo, p.mass_hi);
            // added base mass: the base's mass factor changes (its rotational inertia with it)
            const float load = rng.uniform(p.load_lo, p.load_hi);
            s[p.base] *= (p.base_mass * s[p.base] + load) / (p.base_mass * s[p.base]);
        }
        if (v.rand_kp) v.rand_kp[e] = rng.uniform(p.gain_lo, p.gain_hi);
        if (v.rand_kv) v.rand_kv[e] = rng.uniform(p.gain_lo, p.gain_hi);
        if (v.rand_strength) v.rand_strength[e] = rng.uniform(p.strength_lo, p.strength_hi);
    }
};

// Go2 walk on rough terrain: the same task on the make_terrain() heightfield (as
// g1_walk_rough). Blind: no height scan in the obs.
struct Go2WalkRoughTask : Go2WalkTask {
    static constexpr const char* kName = "go2_walk_rough";
    static void setup_constraints(ConstraintModel& c) {
        const Terrain t = make_terrain();
        c.set_heightfield(t.nx, t.ny, t.cell, t.x0, t.y0, t.h);
    }
};

}  // namespace forge
