#pragma once
// The training loop (doc 5.2) for any Algorithm (ppo, sac), used by train_app.
// Writes the run folder (doc 7.3): log.csv (with one r_<term> column per reward
// term when the task names them, doc 7.4), ckpt_NNNNNN.bin every save_every
// iterations, best.bin (highest mean episode return), final.bin.
// One iteration = algo.steps_per_iteration() env steps, then one log line.
#include <string>

#include "forge/env/env.h"
#include "forge/util/config.h"
#include "temper/algos/algorithm.h"

namespace temper {

struct TrainOptions {
    int max_iterations = 100;
    int save_every = 50;  // iterations; 0 = only best and final
    std::string run_dir;  // empty = no files
    std::string resume;   // checkpoint to continue from
    bool quiet = false;
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

// env must use the gpu backend and the same stream as the algorithm.
TrainResult train(forge::Env& env, Algorithm& algo, const TrainOptions& opt);

}  // namespace temper
