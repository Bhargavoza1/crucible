#include "temper/common/train_loop.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"
#include "forge/util/nvtx.h"
#include "temper/common/train_log.h"

namespace temper {

TrainOptions TrainOptions::from_config(const forge::Config& cfg) {
    TrainOptions o;
    o.max_iterations = cfg.get_int("max_iterations", o.max_iterations);
    o.save_every = cfg.get_int("save_every", o.save_every);
    o.resume = cfg.get_string("resume", "");
    return o;
}

TrainResult train(forge::Env& env, Algorithm& algo, const TrainOptions& opt) {
    if (env.backend() != forge::EnvBackend::kGpu) throw std::runtime_error("training needs the gpu env backend");
    cudaStream_t stream = static_cast<cudaStream_t>(env.stream());
    const int N = env.num_envs(), A = env.act_dim();
    if (!opt.resume.empty()) {
        algo.load(Checkpoint::load(opt.resume), true);
        std::printf("resumed from %s at iteration %lld (episodes restart; an off-policy replay buffer starts empty)\n",
                    opt.resume.c_str(), static_cast<long long>(algo.iteration()));
    }
    forge::DeviceBuffer<float> actions(size_t(N) * A);
    std::vector<float> last_return(N);
    std::vector<int> last_length(N);

    std::vector<std::string> cols = {"iteration", "total_steps", "time_s", "steps_per_s", "mean_reward",
                                     "mean_episode_len", "lr"};
    const std::vector<std::string> names = algo.stat_names();
    cols.insert(cols.end(), names.begin(), names.end());
    for (const auto& t : env.reward_term_names()) cols.push_back("r_" + t);  // doc 7.4: one column per reward term
    env.read_reward_terms();  // start the term sums at 0
    std::unique_ptr<TrainLog> log;
    if (!opt.run_dir.empty()) log = std::make_unique<TrainLog>(opt.run_dir + "/log.csv", cols);
    auto save = [&](const std::string& name) {
        if (!opt.run_dir.empty()) algo.checkpoint(env.options().task).save(opt.run_dir + "/" + name);
    };

    const TransitionView tv{env.reward(), env.terminated(), env.truncated(), env.final_obs(), env.obs()};
    TrainResult res;
    const auto t_start = std::chrono::steady_clock::now();
    long long total_steps = 0;
    const int steps = algo.steps_per_iteration();
    for (int it = 0; it < opt.max_iterations; ++it) {
        const auto t_it = std::chrono::steady_clock::now();
        for (int t = 0; t < steps; ++t) {
            algo.act(env.obs(), actions.data());
            env.step(actions.data());
            algo.observe(tv);
            if (algo.ready_to_update()) algo.update(env.obs());
        }
        const std::vector<double> st = algo.read_stats();  // syncs once
        const std::vector<double> terms = env.read_reward_terms();  // mean per env-step this iteration
        total_steps += static_cast<long long>(steps) * N;

        CUDA_CHECK(cudaMemcpyAsync(last_return.data(), env.last_return(), N * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(last_length.data(), env.last_length(), N * sizeof(int), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double sum_r = 0, sum_l = 0;
        int finished = 0;
        for (int e = 0; e < N; ++e)
            if (last_length[e] > 0) {
                sum_r += last_return[e];
                sum_l += last_length[e];
                ++finished;
            }
        const double mean_r = finished ? sum_r / finished : std::nan("");
        const double mean_l = finished ? sum_l / finished : std::nan("");
        const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        const double sps = double(steps) * N / std::chrono::duration<double>(std::chrono::steady_clock::now() - t_it).count();
        res.iterations = it + 1;
        res.last_mean_return = mean_r;
        res.last_mean_length = mean_l;
        res.env_steps_per_s = total_steps / now;

        bool bad = false;
        for (double v : st) bad = bad || !std::isfinite(v);
        if (log) {
            std::vector<double> row = {double(algo.iteration()), double(total_steps), now, sps, mean_r, mean_l,
                                       algo.learning_rate()};
            row.insert(row.end(), st.begin(), st.end());
            row.insert(row.end(), terms.begin(), terms.end());
            log->row(row);
        }
        if (!opt.quiet) {
            std::printf("it %5lld  steps %10lld  %8.3g steps/s  return %8.2f  ep_len %7.1f ",
                        static_cast<long long>(algo.iteration()), total_steps, sps, mean_r, mean_l);
            for (size_t i = 0; i < st.size() && i < 4; ++i) std::printf(" %s %.4g", names[i].c_str(), st[i]);
            std::printf("\n");
        }
        if (bad) {
            std::fprintf(stderr, "stop: NaN or inf in the training stats at iteration %lld\n",
                         static_cast<long long>(algo.iteration()));
            res.stopped_on_nan = true;
            break;
        }
        if (finished && mean_r > res.best_mean_return) {
            res.best_mean_return = mean_r;
            save("best.bin");
        }
        if (opt.save_every > 0 && algo.iteration() % opt.save_every == 0) {
            char name[32];
            std::snprintf(name, sizeof(name), "ckpt_%06lld.bin", static_cast<long long>(algo.iteration()));
            save(name);
        }
    }
    save("final.bin");
    return res;
}

}  // namespace temper
