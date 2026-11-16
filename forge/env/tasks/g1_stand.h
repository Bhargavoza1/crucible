#pragma once
// G1 stand task (doc 6.1, Phase 8b-1). assets/robots/g1/g1_12dof.urdf: free pelvis +
// 12 leg joints (per leg: hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll).
// Joint armature 0.01 and damping 0.001 as Unitree's MuJoCo G1 model.
//
//   obs (45):  base angular velocity (3, base frame), gravity direction in the base
//              frame (3), base linear velocity (3, base frame), joint positions minus
//              the standing pose (12), joint velocities (12), last action (12). With
//              observation noise (uniform, unitree_rl_gym's levels) when obs_noise is on.
//              (doc 6.1 has no linear velocity: without it the policy cannot see that
//              it drifts, and a trained G1 crept backward at ~0.13 m/s. A real robot
//              needs a state estimator for it.)
//   action (12): PD targets = standing pose + action_scale * a, a in [-1, 1].
//              PD gains as unitree_rl_gym's G1: hip 100 / 2, knee 150 / 4, ankle 40 / 2;
//              force capped by each joint's URDF effort (88 / 139 / 50 Nm).
//              Action delay: each episode applies the action of this step or of the
//              step before (random, 0 or 1 control step), when action_delay is on.
//   reward:    upright_weight * up_z + height_weight * exp(-((z - h0) / 0.05)^2)
//              + pose_weight * exp(-sum (q - q0)^2 / 0.5)
//              - torque_cost * sum tau^2 - action_change_cost * sum (a - a_prev)^2
//              + still_weight * exp(-(vx^2 + vy^2) / 0.02) + yaw_weight * exp(-wz^2 / 0.05)
//                (stand still: drifting at 0.13 m/s loses about half of still_weight)
//              - foot_slip_cost * sum over foot spheres touching the floor of their
//                horizontal speed^2 (doc 6.1 foot slip; from forward kinematics)
//              - fall_penalty when it falls
//   randomization (doc 6.2, per episode, drawn at reset): floor friction 0.5-1.5,
//              each link's mass x0.9-1.2, extra base (pelvis) load 0-3 kg, PD gains
//              x0.9-1.1 (kp and kv drawn separately), motor strength x0.8-1.2
//   pushes:    every 5 to 10 s (random per episode), the pelvis gets a horizontal
//              velocity kick of up to push_speed (0.5 m/s, random direction)
//   terminated: pelvis below min_height (0.5 m) or torso tilted more than 45 degrees
//   truncated: 1000 steps (20 s at 50 Hz)
//   reset:     standing pose, feet 1 cm above the floor, joints +-0.05 rad, small
//              velocities, facing +x
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "forge/dynamics/articulated.h"
#include "forge/dynamics/robot_work.h"
#include "forge/env/env_view.h"
#include "forge/math/quat.h"
#include "forge/robots/actuators.h"
#include "forge/robots/urdf.h"
#include "forge/solver/constraint_model.h"
#include "forge/util/rng.h"

namespace forge {

struct G1StandTask {
    static constexpr const char* kName = "g1_stand";
    static constexpr const char* kUrdf = "robots/g1/g1_12dof.urdf";
    static constexpr bool kFloating = true;
    static constexpr int kJoints = 12;
    static constexpr int kObsDim = 9 + 3 * kJoints, kActDim = kJoints, kMaxEpisodeSteps = 1000;
    static constexpr int kControlHz = 50;  // pushes are scheduled in control steps

    struct Params {
        int base_q = 0, base_d = 0;
        int q[kJoints] = {}, d[kJoints] = {};
        float q0[kJoints] = {};  // standing pose
        float kp[kJoints] = {}, kv[kJoints] = {}, fmax[kJoints] = {};
        float action_scale = 0.5f;
        float stand_height = 0.0f;  // pelvis height when standing (set from the model)
        float min_height = 0.5f, min_up = 0.7071f;  // 45 degrees
        float upright_weight = 1.0f, height_weight = 1.0f, pose_weight = 0.5f;
        float still_weight = 1.0f, yaw_weight = 0.5f;
        float torque_cost = 1e-5f, action_change_cost = 0.01f, fall_penalty = 10.0f;
        float push_speed = 0.5f, push_min_s = 5.0f, push_max_s = 10.0f;
        // feet: every collision sphere of the model (4 per foot)
        static constexpr int kFeet = 8;
        int foot_body[kFeet] = {};
        int foot_of[kFeet] = {};  // 0 = left foot, 1 = right foot
        int foot_link[2] = {};    // ankle_roll_link body of the left and right foot
        float foot_pos[kFeet][3] = {}, foot_r[kFeet] = {};
        float foot_slip_cost = 0.25f, contact_eps = 0.005f;  // touching: lowest point within 5 mm of the floor
        // domain randomization
        bool randomize = true;
        int pelvis = 0;
        float pelvis_mass = 1.0f;
        float friction_lo = 0.5f, friction_hi = 1.5f, mass_lo = 0.9f, mass_hi = 1.2f, load_max = 3.0f;
        float gain_lo = 0.9f, gain_hi = 1.1f, strength_lo = 0.8f, strength_hi = 1.2f;
        bool obs_noise = true, action_delay = true;
        float noise_ang_vel = 0.2f, noise_gravity = 0.05f, noise_q = 0.01f, noise_qd = 1.5f;  // unitree_rl_gym
        float noise_lin_vel = 0.1f;  // unitree_rl_gym's level for base linear velocity
        float reset_joint_noise = 0.05f, reset_vel_noise = 0.05f;
    };

    // ---- host ----
    static void urdf_options(UrdfOptions& u) {
        u.armature = 0.01f;  // Unitree's MuJoCo G1 (g1_12dof.xml)
        u.damping = 0.001f;
    }
    // Standing pose and gains by joint name (unitree_rl_gym g1_config.py).
    static void joint_setup(const std::string& name, float& q0, float& kp, float& kv) {
        q0 = 0.0f;
        if (name.find("hip_pitch") != std::string::npos) q0 = -0.1f;
        if (name.find("knee") != std::string::npos) q0 = 0.3f;
        if (name.find("ankle_pitch") != std::string::npos) q0 = -0.2f;
        if (name.find("knee") != std::string::npos) {
            kp = 150.0f;
            kv = 4.0f;
        } else if (name.find("ankle") != std::string::npos) {
            kp = 40.0f;
            kv = 2.0f;
        } else if (name.find("hip") != std::string::npos) {
            kp = 100.0f;
            kv = 2.0f;
        } else {
            throw std::runtime_error("g1_stand: unexpected joint '" + name + "'");
        }
    }
    static Params make_params(const RobotModel& m) {
        Params p;
        int k = 0;
        for (int b = 0; b < m.nbody(); ++b) {
            if (m.jtype[b] == kJointFree) {
                p.base_q = m.qadr[b];
                p.base_d = m.dadr[b];
                p.pelvis = b;
                p.pelvis_mass = m.mass[b];
            } else if (m.jtype[b] == kJointRevolute) {
                if (k >= kJoints) throw std::runtime_error("g1_stand: more than 12 hinge joints");
                p.q[k] = m.qadr[b];
                p.d[k] = m.dadr[b];
                joint_setup(m.joint_name[b], p.q0[k], p.kp[k], p.kv[k]);
                p.fmax[k] = m.joint_effort[b];
                ++k;
            }
        }
        if (k != kJoints) throw std::runtime_error("g1_stand: needs 12 hinge joints, found " + std::to_string(k));
        // Pelvis height so the lowest foot sphere just touches the floor in the standing pose.
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
        if (static_cast<int>(m.spheres.size()) != Params::kFeet)
            throw std::runtime_error("g1_stand: expected 8 foot spheres, found " + std::to_string(m.spheres.size()));
        p.foot_link[0] = p.foot_link[1] = -1;
        for (int f = 0; f < Params::kFeet; ++f) {
            const int b = m.spheres[f].body;
            const int side = m.body_name[b].find("left") != std::string::npos ? 0 : 1;
            if (p.foot_link[side] >= 0 && p.foot_link[side] != b)
                throw std::runtime_error("g1_stand: foot spheres on more than one body per side");
            p.foot_link[side] = b;
            p.foot_of[f] = side;
            p.foot_body[f] = b;
            p.foot_pos[f][0] = m.spheres[f].pos.x;
            p.foot_pos[f][1] = m.spheres[f].pos.y;
            p.foot_pos[f][2] = m.spheres[f].pos.z;
            p.foot_r[f] = m.spheres[f].radius;
        }
        if (p.foot_link[0] < 0 || p.foot_link[1] < 0) throw std::runtime_error("g1_stand: needs a left and a right foot");
        p.stand_height = -lowest;
        return p;
    }
    // One PD actuator per leg joint, in the same order as Params (body order).
    static Actuators make_actuators(const RobotModel& m) {
        Actuators a;
        for (int b = 0; b < m.nbody(); ++b) {
            if (m.jtype[b] != kJointRevolute) continue;
            float q0, kp, kv;
            joint_setup(m.joint_name[b], q0, kp, kv);
            a.add_position(m, m.joint_name[b], kp, kv, 0.0f, 0.0f);  // force capped by the URDF effort
        }
        return a;
    }
    static void setup_constraints(ConstraintModel& c) { c.add_ground(); }

    // ---- per env ----
    FORGE_HD static Quat base_quat(const Params& p, const EnvView& v, int e) {
        return normalize(load_quat_wxyz(v.qpos(e) + p.base_q + 3));
    }
    // Per-episode random choices, from their own streams (no stored state).
    FORGE_HD static int delay_of(const EnvView& v, int e) {
        Rng r(v.seed ^ 0xDE1A7ull, uint64_t(e), uint64_t(v.episode_index[e]));
        return r.uniform() < 0.5f ? 0 : 1;
    }
    // True when a push happens at control step 'step' of this episode.
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
    // The action the PD targets use this step: this step's, or the one before (delay).
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
        const Quat rq = base_quat(p, v, e);
        const Vec3 g = rotate_inv(rq, Vec3{0, 0, -1});
        const Vec3 vl = rotate_inv(rq, load3(qd + p.base_d));  // free joint qvel: world linear velocity
        o[0] = qd[p.base_d + 3];  // body-frame angular velocity (MuJoCo convention)
        o[1] = qd[p.base_d + 4];
        o[2] = qd[p.base_d + 5];
        o[3] = g.x;
        o[4] = g.y;
        o[5] = g.z;
        o[6] = vl.x;
        o[7] = vl.y;
        o[8] = vl.z;
        for (int k = 0; k < kJoints; ++k) {
            o[9 + k] = q[p.q[k]] - p.q0[k];
            o[9 + kJoints + k] = qd[p.d[k]];
            o[9 + 2 * kJoints + k] = v.last_action[size_t(e) * kActDim + k];
        }
        if (!p.obs_noise) return;
        Rng r(v.seed ^ 0x0B5ull, uint64_t(e), (uint64_t(v.episode_index[e]) << 20) + uint64_t(v.episode_step[e]));
        for (int i = 0; i < 3; ++i) o[i] += p.noise_ang_vel * r.uniform(-1.0f, 1.0f);
        for (int i = 3; i < 6; ++i) o[i] += p.noise_gravity * r.uniform(-1.0f, 1.0f);
        for (int i = 6; i < 9; ++i) o[i] += p.noise_lin_vel * r.uniform(-1.0f, 1.0f);
        for (int k = 0; k < kJoints; ++k) {
            o[9 + k] += p.noise_q * r.uniform(-1.0f, 1.0f);
            o[9 + kJoints + k] += p.noise_qd * r.uniform(-1.0f, 1.0f);
        }
    }
    FORGE_HD static bool failed(const Params& p, const EnvView& v, int e) {
        const float z = v.qpos(e)[p.base_q + 2];
        const Vec3 up = rotate(base_quat(p, v, e), Vec3{0, 0, 1});
        return z < p.min_height || up.z < p.min_up;
    }
    // Foot kinematics from forward kinematics: every foot sphere's center, velocity
    // and whether it touches the floor, and each foot's ankle_roll_link origin.
    // Point velocity on body b: sum over the joints above b of (S.lin + S.ang x p) qdot.
    struct FootState {
        Vec3 c[Params::kFeet], vel[Params::kFeet];
        bool touch[Params::kFeet];
        Vec3 origin[2];  // ankle_roll_link origin, left and right
    };
    FORGE_HD static void foot_state(const Params& p, const EnvView& v, int e, FootState& fs) {
        float xpos[3 * kMaxBodies], xquat[4 * kMaxBodies];
        SpatialVec S[kMaxDofs];
        DynamicsWork w{xpos, xquat, S, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        const ModelView& m = v.model;
        forward_kinematics(m, v.qpos(e), w);
        const float* qd = v.qvel(e);
        for (int f = 0; f < Params::kFeet; ++f) {
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
            fs.touch[f] = c.z - p.foot_r[f] <= p.contact_eps;
        }
        for (int s = 0; s < 2; ++s) fs.origin[s] = load3(xpos + 3 * p.foot_link[s]);
    }
    // Sum over foot spheres touching the floor of their horizontal speed^2.
    FORGE_HD static float foot_slip2(const Params& p, const EnvView& v, int e) {
        FootState fs;
        foot_state(p, v, e, fs);
        float sum = 0.0f;
        for (int f = 0; f < Params::kFeet; ++f)
            if (fs.touch[f]) sum += fs.vel[f].x * fs.vel[f].x + fs.vel[f].y * fs.vel[f].y;
        return sum;
    }
    FORGE_HD static float reward(const Params& p, const EnvView& v, int e, const float* a) {
        const float* q = v.qpos(e);
        const float* qd = v.qvel(e);
        const float* prev = v.prev_action + size_t(e) * kActDim;
        const float* used = applied(p, v, e, a);
        const float ks = v.rand_kp ? v.rand_kp[e] : 1.0f, kvs = v.rand_kv ? v.rand_kv[e] : 1.0f;
        const float ms = v.rand_strength ? v.rand_strength[e] : 1.0f;
        float torque2 = 0.0f, change2 = 0.0f, pose2 = 0.0f;
        for (int k = 0; k < kJoints; ++k) {
            const float target = p.q0[k] + p.action_scale * used[k];
            const float t = ms * fminf(fmaxf(ks * p.kp[k] * (target - q[p.q[k]]) - kvs * p.kv[k] * qd[p.d[k]],
                                             -p.fmax[k]), p.fmax[k]);  // the same force as apply_actuators
            torque2 += t * t;
            change2 += (a[k] - prev[k]) * (a[k] - prev[k]);
            const float dq = q[p.q[k]] - p.q0[k];
            pose2 += dq * dq;
        }
        const Quat rq = base_quat(p, v, e);
        const float up = rotate(rq, Vec3{0, 0, 1}).z;
        const float dz = (q[p.base_q + 2] - p.stand_height) / 0.05f;
        const Vec3 w_world = rotate(rq, load3(qd + p.base_d + 3));
        const float vxy2 = qd[p.base_d] * qd[p.base_d] + qd[p.base_d + 1] * qd[p.base_d + 1];
        float r = p.upright_weight * up + p.height_weight * expf(-dz * dz) + p.pose_weight * expf(-pose2 / 0.5f) +
                  p.still_weight * expf(-vxy2 / 0.02f) + p.yaw_weight * expf(-w_world.z * w_world.z / 0.05f) -
                  p.torque_cost * torque2 - p.action_change_cost * change2 - p.foot_slip_cost * foot_slip2(p, v, e);
        if (failed(p, v, e)) r -= p.fall_penalty;
        return r;
    }
    FORGE_HD static void reset(const Params& p, const EnvView& v, int e, Rng& rng) {
        float* q = v.qpos(e);
        float* qd = v.qvel(e);
        q[p.base_q + 0] = 0.0f;
        q[p.base_q + 1] = 0.0f;
        q[p.base_q + 2] = p.stand_height + 0.01f;
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
            // extra pelvis load: the pelvis's mass factor grows (its rotational inertia
            // grows with it, so this is a heavier pelvis, not an exact point mass)
            const float load = rng.uniform(0.0f, p.load_max);
            s[p.pelvis] *= (p.pelvis_mass * s[p.pelvis] + load) / (p.pelvis_mass * s[p.pelvis]);
        }
        if (v.rand_kp) v.rand_kp[e] = rng.uniform(p.gain_lo, p.gain_hi);
        if (v.rand_kv) v.rand_kv[e] = rng.uniform(p.gain_lo, p.gain_hi);
        if (v.rand_strength) v.rand_strength[e] = rng.uniform(p.strength_lo, p.strength_hi);
    }
};

}  // namespace forge
