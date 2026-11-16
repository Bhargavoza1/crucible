#pragma once
// G1 walk task (doc Phase 8c): follow a velocity command. Same robot, PD gains,
// pushes, domain randomization, observation noise, action delay, falls and reset
// as the stand task (g1_stand.h); new obs and reward. The reward terms and
// weights follow unitree_rl_gym's G1 walking setup (g1_config.py, g1_env.py,
// legged_robot.py), with their scales not multiplied by dt (our PPO trains on
// per-step rewards of order 1, as the other tasks).
//
//   command:   (vx, vy, wz) in the base frame, drawn per env every 10 s:
//              vx in [-0.5, 1.0] m/s, vy in [-0.5, 0.5] m/s, wz in [-1, 1] rad/s;
//              a command with |(vx, vy)| < 0.2 gets vx = vy = 0 (so some envs learn
//              to stand and turn in place). No stored state: the command is a
//              function of (seed, env, episode, step / 500). A task input
//              (Env::set_task_input, 3 numbers: vx, vy, wz) replaces it, for
//              driving the robot from the keyboard in play mode.
//   gait clock: phase = (t mod 0.8 s) / 0.8; left leg at phase, right at phase + 0.5
//   obs (50):  base angular velocity (3), gravity in the base frame (3), base linear
//              velocity (3, base frame), command (3), joint positions minus the
//              standing pose (12), joint velocities (12), last action (12),
//              sin and cos of 2 pi phase (2). Noise as the stand task (command and
//              clock: none).
//   action (12): PD targets = standing pose + a, a in [-1, 1] rad.
//   reward (unitree_rl_gym G1 weights):
//      + 1.0  exp(-|cmd_xy - v_xy|^2 / 0.25)       linear velocity tracking (base frame)
//      + 0.5  exp(-(cmd_wz - w_z)^2 / 0.25)        yaw rate tracking
//      - 2.0  v_z^2  - 0.05 (w_x^2 + w_y^2)        no bouncing, no rocking
//      - 1.0  (g_x^2 + g_y^2)                      upright (projected gravity)
//      - 10   (z - 0.78)^2                         pelvis height
//      - 2.5e-7 sum qacc^2 - 1e-3 sum qdot^2       smooth joints
//      - 0.01 sum (a - a_prev)^2                   smooth actions
//      - 5.0  sum (distance outside 90% of each joint range)
//      + 0.15                                      alive
//      - 1.0  (hip roll^2 + hip yaw^2, both legs)  legs straight in the hips
//      - 0.2  sum over feet touching the floor of |foot velocity|^2   no foot slip
//      - 20   sum over feet in the air of (ankle height - 0.08)^2      lift the feet
//      + 0.18 per foot whose contact matches the clock (stance while phase < 0.55)
//      total clipped at 0 from below (unitree_rl_gym only_positive_rewards); a fall
//      ends the episode (the lost future reward is the penalty)
#include "forge/env/tasks/g1_stand.h"

namespace forge {

struct G1WalkTask : G1StandTask {
    static constexpr const char* kName = "g1_walk";
    static constexpr int kObsDim = 12 + 3 * kJoints + 2, kActDim = kJoints, kMaxEpisodeSteps = 1000;
    static constexpr int kTaskInputDim = 3;  // the command: vx, vy, wz

    struct Params : G1StandTask::Params {
        float cmd_vx_lo = -0.5f, cmd_vx_hi = 1.0f, cmd_vy = 0.5f, cmd_wz = 1.0f, cmd_small = 0.2f;
        float cmd_every_s = 10.0f;
        float period_s = 0.8f, stance_phase = 0.55f;
        float lo[kJoints] = {}, hi[kJoints] = {};  // 90% of each joint's range
        int hip_roll[2] = {}, hip_yaw[2] = {};     // joint index (k) of left / right hip roll and yaw
        float w_track = 1.0f, w_track_yaw = 0.5f, tracking_sigma = 0.25f;
        float w_vz = 2.0f, w_wxy = 0.05f, w_orient = 1.0f, w_height = 10.0f, height_target = 0.78f;
        float w_qacc = 2.5e-7f, w_qd = 1e-3f, w_action = 0.01f, w_limits = 5.0f, w_alive = 0.15f;
        float w_hip = 1.0f, w_slip = 0.2f, w_swing = 20.0f, swing_height = 0.08f, w_contact = 0.18f;
    };

    static Params make_params(const RobotModel& m) {
        Params p;
        static_cast<G1StandTask::Params&>(p) = G1StandTask::make_params(m);
        p.action_scale = 1.0f;
        p.fall_penalty = 0.0f;
        int k = 0, roll = 0, yaw = 0;
        for (int b = 0; b < m.nbody(); ++b) {
            if (m.jtype[b] != kJointRevolute) continue;
            const float mid = 0.5f * (m.joint_lower[b] + m.joint_upper[b]);
            const float half = 0.5f * (m.joint_upper[b] - m.joint_lower[b]);
            p.lo[k] = mid - 0.9f * half;
            p.hi[k] = mid + 0.9f * half;
            const std::string& n = m.joint_name[b];
            if (n.find("hip_roll") != std::string::npos && roll < 2) p.hip_roll[roll++] = k;
            if (n.find("hip_yaw") != std::string::npos && yaw < 2) p.hip_yaw[yaw++] = k;
            ++k;
        }
        if (roll != 2 || yaw != 2) throw std::runtime_error("g1_walk: needs hip roll and hip yaw joints on both legs");
        return p;
    }

    // ---- per env ----
    FORGE_HD static int steps_per_command(const Params& p) { return int(p.cmd_every_s * kControlHz + 0.5f); }
    // Command at control step 'step' of the current episode.
    FORGE_HD static Vec3 command(const Params& p, const EnvView& v, int e, int step) {
        if (v.task_input) return {v.task_input[0], v.task_input[1], v.task_input[2]};
        Rng r(v.seed ^ 0xC0AA4Dull, uint64_t(e),
              (uint64_t(v.episode_index[e]) << 20) + uint64_t(step / steps_per_command(p)));
        Vec3 c{r.uniform(p.cmd_vx_lo, p.cmd_vx_hi), r.uniform(-p.cmd_vy, p.cmd_vy), r.uniform(-p.cmd_wz, p.cmd_wz)};
        if (c.x * c.x + c.y * c.y < p.cmd_small * p.cmd_small) c.x = c.y = 0.0f;
        return c;
    }
    FORGE_HD static float phase(const Params& p, int step) {
        const float t = float(step) / float(kControlHz);
        return fmodf(t, p.period_s) / p.period_s;
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
        o[0] = qd[p.base_d + 3];
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

    // The reward split into its terms (log.csv gets one column each), each already
    // weighted and signed; the reward is their sum, clipped at 0 from below.
    static constexpr int kRewardTerms = 15;
    static const char* reward_term_name(int k) {
        static const char* const n[kRewardTerms] = {"track_lin_vel", "track_yaw", "lin_vel_z",  "ang_vel_xy",
                                                    "orientation",   "height",    "joint_acc",  "joint_vel",
                                                    "action_rate",   "joint_limits", "alive",   "hip_pos",
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
        const int step = v.episode_step[e];
        const Quat rq = base_quat(p, v, e);
        const Vec3 vl = rotate_inv(rq, load3(qd + p.base_d));
        const Vec3 w{qd[p.base_d + 3], qd[p.base_d + 4], qd[p.base_d + 5]};  // base frame
        const Vec3 g = rotate_inv(rq, Vec3{0, 0, -1});
        const Vec3 c = command(p, v, e, step);

        const float ex = c.x - vl.x, ey = c.y - vl.y, ew = c.z - w.z;
        t[0] = p.w_track * expf(-(ex * ex + ey * ey) / p.tracking_sigma);
        t[1] = p.w_track_yaw * expf(-ew * ew / p.tracking_sigma);
        t[2] = -p.w_vz * vl.z * vl.z;
        t[3] = -p.w_wxy * (w.x * w.x + w.y * w.y);
        t[4] = -p.w_orient * (g.x * g.x + g.y * g.y);
        const float dz = q[p.base_q + 2] - p.height_target;
        t[5] = -p.w_height * dz * dz;
        float acc2 = 0.0f, vel2 = 0.0f, act2 = 0.0f, lim = 0.0f;
        for (int k = 0; k < kJoints; ++k) {
            if (qacc) acc2 += qacc[p.d[k]] * qacc[p.d[k]];
            vel2 += qd[p.d[k]] * qd[p.d[k]];
            act2 += (a[k] - prev[k]) * (a[k] - prev[k]);
            const float x = q[p.q[k]];
            lim += fmaxf(p.lo[k] - x, 0.0f) + fmaxf(x - p.hi[k], 0.0f);
        }
        t[6] = -p.w_qacc * acc2;
        t[7] = -p.w_qd * vel2;
        t[8] = -p.w_action * act2;
        t[9] = -p.w_limits * lim;
        t[10] = p.w_alive;
        float hip2 = 0.0f;
        for (int s = 0; s < 2; ++s) {
            const float hr = q[p.q[p.hip_roll[s]]], hy = q[p.q[p.hip_yaw[s]]];
            hip2 += hr * hr + hy * hy;
        }
        t[11] = -p.w_hip * hip2;

        // Feet: a foot touches when any of its 4 spheres does; its velocity is the
        // mean of its spheres' velocities.
        FootState fs;
        foot_state(p, v, e, fs);
        const float ph = phase(p, step);
        t[12] = t[13] = t[14] = 0.0f;
        for (int s = 0; s < 2; ++s) {
            bool touch = false;
            Vec3 vel{0, 0, 0};
            int n = 0;
            for (int f = 0; f < Params::kFeet; ++f) {
                if (p.foot_of[f] != s) continue;
                touch = touch || fs.touch[f];
                vel = vel + fs.vel[f];
                ++n;
            }
            vel = vel * (1.0f / float(n > 0 ? n : 1));
            if (touch) {
                t[12] -= p.w_slip * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
            } else {
                const float hz = fs.origin[s].z - p.swing_height;
                t[13] -= p.w_swing * hz * hz;
            }
            const float leg_phase = s == 0 ? ph : fmodf(ph + 0.5f, 1.0f);
            const bool stance = leg_phase < p.stance_phase;
            if (touch == stance) t[14] += p.w_contact;
        }
    }
    FORGE_HD static float reward(const Params& p, const EnvView& v, int e, const float* a) {
        float t[kRewardTerms];
        reward_terms(p, v, e, a, t);
        return combine_terms(p, t);
    }
};

}  // namespace forge
