#pragma once
// The on-policy training loop (doc 5.2) for PPO, used by train_app.
// Writes the run folder (doc 7.3): config.txt, log.csv, ckpt_NNNNNN.bin every
// save_every iterations, best.bin (highest mean episode return), final.bin.
#include <string>

#include "forge/env/env.h"
#include "forge/util/config.h"
#include "temper/algos/ppo/ppo.h"

namespace temper {

struct TrainOptions {
    int max_iterations = 100;
    int save_every = 50;        // iterations; 0 = only best and final
    std::string run_dir;        // empty = no files (no log.csv, no checkpoints)
    std::string resume;         // checkpoint to continue from (weights, normalizer, Adam, iteration)
    bool quiet = false;         // no console line per iteration

    // Keys: max_iterations, save_every, resume.
    static TrainOptions from_config(const forge::Config& cfg);
};

struct TrainResult {
    int iterations = 0;
    double best_mean_return = -1e30;
    double last_mean_return = 0;
    double last_mean_length = 0;
    double env_steps_per_s = 0;
    bool stopped_on_nan = false;
};

// env must use the gpu backend. Returns after max_iterations, or early on NaN.
TrainResult train_ppo(forge::Env& env, const PpoConfig& cfg, const TrainOptions& opt);

}  // namespace temper
