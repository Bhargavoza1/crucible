#pragma once
// What a task's per-env code sees (FORGE_HD): the physics state of all envs and
// the Env API arrays (doc 3.6). Host pointers on the CPU, device pointers on the GPU.
#include <cstdint>

#include "forge/dynamics/robot_batch.h"
#include "forge/math/hd.h"
#include "forge/robots/model.h"
#include "forge/solver/constraint_batch.h"

namespace forge {

struct EnvView {
    int num_envs = 0, obs_dim = 0, act_dim = 0;
    int max_episode_steps = 0;
    uint64_t seed = 0;
    ModelView model;
    RobotBatchView state;        // qpos, qvel, tau, qacc, status
    float* ctrl = nullptr;       // [num_envs * nu] PD targets (null if the task has no actuators)
    int nu = 0;
    float* obs = nullptr;        // [num_envs * obs_dim]
    float* final_obs = nullptr;  // [num_envs * obs_dim], valid where terminated or truncated
    float* reward = nullptr;     // [num_envs]
    int* terminated = nullptr;   // [num_envs] 1 = real end (failed): do not bootstrap
    int* truncated = nullptr;    // [num_envs] 1 = time limit: bootstrap
    int* episode_step = nullptr;      // [num_envs] steps in the current episode
    uint32_t* episode_index = nullptr; // [num_envs] episodes finished so far (RNG stream for resets)
    float* episode_return = nullptr;  // [num_envs] sum of rewards in the current episode
    float* last_return = nullptr;     // [num_envs] return of the last finished episode (for logs)
    int* last_length = nullptr;       // [num_envs] length of the last finished episode
    float* last_action = nullptr;     // [num_envs * act_dim] clipped action of the last step
    float* prev_action = nullptr;     // [num_envs * act_dim] the action before it (for "action change" penalties)
    const float* task_input = nullptr;  // [task input dim], same for all envs; null = the task's own values
    // Per-env domain randomization (doc 6.2), read by the physics. A task writes them
    // in reset(); untouched they hold neutral values (model friction, factors 1).
    float* rand_friction = nullptr;    // [num_envs]
    float* rand_body_scale = nullptr;  // [num_envs * model.nbody] mass factor per body
    float* rand_kp = nullptr;          // [num_envs] PD gain factors
    float* rand_kv = nullptr;
    float* rand_strength = nullptr;    // [num_envs] motor strength

    FORGE_HD float* qpos(int e) const { return state.qpos + size_t(e) * model.nq; }
    FORGE_HD float* qvel(int e) const { return state.qvel + size_t(e) * model.nv; }
    FORGE_HD float* tau(int e) const { return state.tau + size_t(e) * model.nv; }
};

}  // namespace forge
