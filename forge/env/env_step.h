#pragma once
// The task-independent part of Env::step and Env::reset, per env (FORGE_HD).
// T is a task (for example CartpoleTask) with static FORGE_HD functions:
//   apply_action(p, v, e, a)   a: act_dim numbers in [-1, 1] -> tau or ctrl
//   observe(p, v, e, obs)
//   reward(p, v, e, a)         after the physics steps
//   failed(p, v, e)            true = terminated
//   reset(p, v, e, rng)        new start state (also randomization)
#include "forge/env/env_view.h"
#include "forge/math/hd.h"
#include "forge/util/rng.h"

namespace forge {

// Clip the policy's actions to [-1, 1], keep them as last_action, apply them.
template <class T>
FORGE_HD void env_apply_actions(const typename T::Params& p, const EnvView& v, int e, const float* actions) {
    float* a = v.last_action + size_t(e) * v.act_dim;
    float* prev = v.prev_action + size_t(e) * v.act_dim;
    for (int k = 0; k < v.act_dim; ++k) {
        prev[k] = a[k];
        float x = actions[size_t(e) * v.act_dim + k];
        x = x != x ? 0.0f : fminf(fmaxf(x, -1.0f), 1.0f);  // NaN -> 0
        a[k] = x;
    }
    T::apply_action(p, v, e, a);
}

// New episode for env e: reset the state with the stream (seed, e, episode_index).
template <class T>
FORGE_HD void env_reset_one(const typename T::Params& p, const EnvView& v, int e) {
    for (int i = 0; i < v.model.nq; ++i) v.qpos(e)[i] = 0.0f;
    for (int i = 0; i < v.model.nv; ++i) {
        v.qvel(e)[i] = 0.0f;
        v.tau(e)[i] = 0.0f;
    }
    if (v.state.status) v.state.status[e] = 0;
    Rng rng(v.seed, uint64_t(e), uint64_t(v.episode_index[e]));
    T::reset(p, v, e, rng);
    v.episode_step[e] = 0;
    v.episode_return[e] = 0.0f;
    for (int k = 0; k < v.act_dim; ++k) {
        v.last_action[size_t(e) * v.act_dim + k] = 0.0f;
        v.prev_action[size_t(e) * v.act_dim + k] = 0.0f;
    }
    T::observe(p, v, e, v.obs + size_t(e) * v.obs_dim);
}

// After the physics steps: reward, done flags, final_obs, auto-reset (doc 3.6).
// A dynamics error (status != 0) or a non-finite observation ends the episode as
// terminated, so one broken env never stops training.
template <class T>
FORGE_HD void env_after_physics(const typename T::Params& p, const EnvView& v, int e) {
    float* o = v.obs + size_t(e) * v.obs_dim;
    const float* a = v.last_action + size_t(e) * v.act_dim;
    const int step = ++v.episode_step[e];  // before observe: per-step noise streams see the new step
    T::observe(p, v, e, o);
    bool finite = true;
    for (int k = 0; k < v.obs_dim; ++k) finite = finite && (o[k] - o[k] == 0.0f);  // false for NaN and inf
    const bool broken = !finite || (v.state.status && v.state.status[e] != 0);
    float r = 0.0f;
    if constexpr (RewardTermCount<T>::value > 0) {
        constexpr int K = RewardTermCount<T>::value;
        float terms[K];
        if (broken) {
            for (int k = 0; k < K; ++k) terms[k] = 0.0f;
        } else {
            T::reward_terms(p, v, e, a, terms);
            r = T::combine_terms(p, terms);
        }
        if (v.term_sum)
            for (int k = 0; k < K; ++k) v.term_sum[size_t(e) * K + k] += terms[k];
    } else {
        r = broken ? 0.0f : T::reward(p, v, e, a);
    }
    v.reward[e] = r;
    v.episode_return[e] += r;
    const int term = broken || T::failed(p, v, e);
    const int trunc = !term && step >= v.max_episode_steps;
    v.terminated[e] = term;
    v.truncated[e] = trunc;
    if (term || trunc) {
        float* fo = v.final_obs + size_t(e) * v.obs_dim;
        for (int k = 0; k < v.obs_dim; ++k) fo[k] = broken ? 0.0f : o[k];
        v.last_return[e] = v.episode_return[e];
        v.last_length[e] = step;
        ++v.episode_index[e];
        env_reset_one<T>(p, v, e);  // obs now belongs to the new episode
    }
}

}  // namespace forge
