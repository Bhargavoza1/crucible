#pragma once
// Algorithm interface (doc 5.1), written now that there are two algorithms (PPO,
// SAC). The training loop asks every algorithm the same questions; each call
// covers all envs at once, so the virtual call costs nothing.
//
//   per env step:   act(obs) -> actions;  env.step(actions);  observe(transition)
//                   if ready_to_update(): update(obs)
//   per iteration:  read_stats() (the one sync), log, checkpoint
//
// PPO: ready after 'horizon' steps (one update per iteration).
// SAC: ready once the replay buffer has learning_starts transitions, then every
//      step (updates_per_step gradient steps each).
// All pointers are device arrays [num_envs x ...] on the algorithm's stream.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "forge/util/config.h"
#include "temper/common/checkpoint.h"

namespace temper {

struct TransitionView {
    const float* reward = nullptr;
    const int* terminated = nullptr;
    const int* truncated = nullptr;
    const float* final_obs = nullptr;  // real last obs where an env just ended
    const float* next_obs = nullptr;   // obs after the step (a new episode's first obs where one ended)
};

class Algorithm {
public:
    virtual ~Algorithm() = default;
    virtual const char* name() const = 0;
    virtual int steps_per_iteration() const = 0;  // env steps between two log lines

    virtual void act(const float* obs, float* actions) = 0;                          // with exploration
    virtual void act_deterministic(const float* obs, float* actions, int n) = 0;     // play mode
    virtual void observe(const TransitionView& t) = 0;
    virtual bool ready_to_update() const = 0;
    virtual void update(const float* obs) = 0;  // obs = current obs (PPO bootstraps from it)

    // Names and values of the algorithm's log columns (means since the last call).
    // read_stats() syncs once and resets the sums.
    virtual std::vector<std::string> stat_names() const = 0;
    virtual std::vector<double> read_stats() = 0;
    virtual double learning_rate() const = 0;

    virtual Checkpoint checkpoint(const std::string& task) const = 0;
    virtual void load(const Checkpoint& c, bool with_optimizer) = 0;
    virtual int64_t iteration() const = 0;  // finished iterations (read_stats calls), kept in checkpoints
};

// algo = "ppo" or "sac". cfg may be null (defaults). Throws for an unknown name.
std::unique_ptr<Algorithm> make_algorithm(const std::string& algo, int obs_dim, int act_dim, int num_envs,
                                          const forge::Config* cfg, cudaStream_t stream);
std::vector<std::string> algorithm_names();

}  // namespace temper
