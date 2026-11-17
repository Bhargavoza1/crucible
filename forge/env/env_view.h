#pragma once
// What a task's per-env code sees (FORGE_HD): the physics state of all envs and
// the Env API arrays (doc 3.6). Host pointers on the CPU, device pointers on the GPU.
#include <cstdint>
#include <type_traits>

#include "forge/dynamics/robot_batch.h"
#include "forge/math/hd.h"
#include "forge/robots/model.h"
#include "forge/solver/constraint_batch.h"
#include "forge/solver/heightfield.h"

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
    float* term_sum = nullptr;          // [num_envs * reward terms] sum of each reward term since the last read
    float* task_state = nullptr;        // [num_envs * T::kTaskState] the task's own per-env numbers (null: none)
    HeightfieldView terrain;            // the task's terrain (h null = flat ground at z = 0)
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

// A task may split its reward into named terms (log.csv gets one column each,
// doc 7.4): static constexpr int kRewardTerms, static const char* reward_term_name(int),
// FORGE_HD static void reward_terms(p, v, e, a, float* terms) and
// FORGE_HD static float combine_terms(p, const float* terms). Otherwise 0.
template <class T, class = void>
struct RewardTermCount {
    static constexpr int value = 0;
};
template <class T>
struct RewardTermCount<T, std::void_t<decltype(T::kRewardTerms)>> {
    static constexpr int value = T::kRewardTerms;
};

// A task may have a scripted expert (Phase 9b): FORGE_HD static void expert(p, v, e, a)
// writes act_dim actions for env e from the full state (Env::expert_actions).
template <class T, class = void>
struct TaskHasExpert {
    static constexpr bool value = false;
};
template <class T>
struct TaskHasExpert<T, std::void_t<decltype(&T::expert)>> {
    static constexpr bool value = true;
};

}  // namespace forge
