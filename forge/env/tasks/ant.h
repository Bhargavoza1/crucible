#pragma once
// Ant task (doc 6.1). assets/robots/ant.urdf: free base + 8 hinges (4 x hip, ankle).
//   obs (33):  gravity direction in the base frame (3), base linear velocity in the
//              base frame (3), base angular velocity (3, base frame), joint
//              positions minus the middle of their range (8), joint velocities (8),
//              last action (8)
//   action (8): joint position targets for the PD actuators; [-1, 1] covers each
//              joint's whole range: target = mid + half_range * a
//   reward:    forward_weight * (world x velocity of the base) + alive_bonus
//              - torque_cost * sum(PD torque^2) - action_change_cost * sum((a - a_prev)^2)
//              - fall_penalty when the episode ends because it fell
//   terminated: torso center below min_height, or flipped (base up axis points down)
//   truncated: 1000 steps (20 s at 50 Hz)
//   reset:     standing pose 5 cm above the floor (lowest sphere), joints +-0.1 rad
//              around the middle of their range, small velocities, facing +x
#include <stdexcept>
#include <string>

#include "forge/dynamics/articulated.h"
#include "forge/env/env_view.h"
#include "forge/math/quat.h"
#include "forge/robots/actuators.h"
#include "forge/solver/constraint_model.h"
#include "forge/util/rng.h"

namespace forge {

struct AntTask {
    static constexpr const char* kName = "ant";
    static constexpr const char* kUrdf = "robots/ant.urdf";
    static constexpr bool kFloating = true;
    static constexpr int kObsDim = 33, kActDim = 8, kMaxEpisodeSteps = 1000;
    static constexpr int kJoints = 8;

    struct Params {
        int base_q = 0, base_d = 0;
        int q[kJoints] = {}, d[kJoints] = {};  // qpos / dof index of each actuated joint
        float mid[kJoints] = {}, half[kJoints] = {};
        float kp = 20.0f, kv = 0.5f, force_limit = 5.0f;  // tuned in MuJoCo: stands, survives random targets
        float stand_height = 0.50f;      // torso center when standing (m)
        float min_height = 0.28f;        // below this the torso is on the floor: fallen
        float forward_weight = 1.0f, alive_bonus = 0.5f;
        float torque_cost = 0.005f, action_change_cost = 0.01f, fall_penalty = 2.0f;
        float reset_joint_noise = 0.1f, reset_vel_noise = 0.1f;
    };

    // ---- host ----
    static Params make_params(const RobotModel& m) {
        Params p;
        int k = 0;
        for (int b = 0; b < m.nbody(); ++b) {
            if (m.jtype[b] == kJointFree) {
                p.base_q = m.qadr[b];
                p.base_d = m.dadr[b];
            } else if (m.jtype[b] == kJointRevolute) {
                if (k >= kJoints) throw std::runtime_error("ant: more than 8 hinge joints");
                if (!m.joint_limited[b]) throw std::runtime_error("ant: joint '" + m.joint_name[b] + "' needs limits");
                p.q[k] = m.qadr[b];
                p.d[k] = m.dadr[b];
                p.mid[k] = 0.5f * (m.joint_lower[b] + m.joint_upper[b]);
                p.half[k] = 0.5f * (m.joint_upper[b] - m.joint_lower[b]);
                ++k;
            }
        }
        if (k != kJoints) throw std::runtime_error("ant: needs 8 hinge joints, found " + std::to_string(k));
        return p;
    }
    static Actuators make_actuators(const RobotModel& m) {
        const Params p{};
        return Actuators::position_all(m, p.kp, p.kv, -p.force_limit, p.force_limit);
    }
    static void setup_constraints(ConstraintModel& c) { c.add_ground(); }

    // ---- per env ----
    FORGE_HD static Quat base_quat(const Params& p, const EnvView& v, int e) {
        return normalize(load_quat_wxyz(v.qpos(e) + p.base_q + 3));
    }
    // Actuator k drives joint k (Actuators::position_all uses the same body order).
    FORGE_HD static void apply_action(const Params& p, const EnvView& v, int e, const float* a) {
        for (int k = 0; k < kJoints; ++k) v.ctrl[size_t(e) * v.nu + k] = p.mid[k] + p.half[k] * a[k];
    }
    FORGE_HD static void observe(const Params& p, const EnvView& v, int e, float* o) {
        const float* q = v.qpos(e);
        const float* qd = v.qvel(e);
        const Quat r = base_quat(p, v, e);
        const Vec3 g = rotate_inv(r, Vec3{0, 0, -1});
        const Vec3 vlin = rotate_inv(r, load3(qd + p.base_d));  // qvel: world linear velocity
        o[0] = g.x;
        o[1] = g.y;
        o[2] = g.z;
        o[3] = vlin.x;
        o[4] = vlin.y;
        o[5] = vlin.z;
        o[6] = qd[p.base_d + 3];  // body-frame angular velocity (MuJoCo convention)
        o[7] = qd[p.base_d + 4];
        o[8] = qd[p.base_d + 5];
        for (int k = 0; k < kJoints; ++k) {
            o[9 + k] = q[p.q[k]] - p.mid[k];
            o[17 + k] = qd[p.d[k]];
            o[25 + k] = v.last_action[size_t(e) * kActDim + k];
        }
    }
    FORGE_HD static bool failed(const Params& p, const EnvView& v, int e) {
        const float z = v.qpos(e)[p.base_q + 2];
        const Vec3 up = rotate(base_quat(p, v, e), Vec3{0, 0, 1});
        return z < p.min_height || up.z < 0.0f;
    }
    FORGE_HD static float reward(const Params& p, const EnvView& v, int e, const float* a) {
        const float* q = v.qpos(e);
        const float* qd = v.qvel(e);
        const float* prev = v.prev_action + size_t(e) * kActDim;
        float torque2 = 0.0f, change2 = 0.0f;
        for (int k = 0; k < kJoints; ++k) {
            const float target = p.mid[k] + p.half[k] * a[k];
            const float t = fminf(fmaxf(p.kp * (target - q[p.q[k]]) - p.kv * qd[p.d[k]], -p.force_limit), p.force_limit);
            torque2 += t * t;
            change2 += (a[k] - prev[k]) * (a[k] - prev[k]);
        }
        float r = p.forward_weight * qd[p.base_d] + p.alive_bonus - p.torque_cost * torque2 -
                  p.action_change_cost * change2;
        if (failed(p, v, e)) r -= p.fall_penalty;
        return r;
    }
    FORGE_HD static void reset(const Params& p, const EnvView& v, int e, Rng& rng) {
        float* q = v.qpos(e);
        float* qd = v.qvel(e);
        q[p.base_q + 0] = 0.0f;
        q[p.base_q + 1] = 0.0f;
        q[p.base_q + 2] = p.stand_height + 0.05f;
        q[p.base_q + 3] = 1.0f;  // w x y z: upright, facing +x
        q[p.base_q + 4] = q[p.base_q + 5] = q[p.base_q + 6] = 0.0f;
        for (int k = 0; k < kJoints; ++k) {
            const float j = p.mid[k] + rng.uniform(-p.reset_joint_noise, p.reset_joint_noise);
            q[p.q[k]] = j;
            qd[p.d[k]] = rng.uniform(-p.reset_vel_noise, p.reset_vel_noise);
            v.ctrl[size_t(e) * v.nu + k] = j;  // PD holds the start pose until the first action
        }
        for (int i = 0; i < 6; ++i) qd[p.base_d + i] = rng.uniform(-p.reset_vel_noise, p.reset_vel_noise);
    }
};

}  // namespace forge
