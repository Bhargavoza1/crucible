#include "temper/algos/ppo/train_ppo.h"

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

TrainResult train_ppo(forge::Env& env, const PpoConfig& cfg, const TrainOptions& opt) {
    if (env.backend() != forge::EnvBackend::kGpu) throw std::runtime_error("train_ppo needs the gpu env backend");
    cudaStream_t stream = static_cast<cudaStream_t>(env.stream());
    const int N = env.num_envs(), A = env.act_dim();
    Ppo ppo(env.obs_dim(), A, N, cfg, stream);
    if (!opt.resume.empty()) {
        ppo.load(Checkpoint::load(opt.resume), true);
        std::printf("resumed from %s at iteration %lld (episodes restart: the env state is not in the checkpoint)\n",
                    opt.resume.c_str(), static_cast<long long>(ppo.iteration()));
    }
    forge::DeviceBuffer<float> actions(size_t(N) * A);
    std::vector<float> last_return(N);
    std::vector<int> last_length(N);

    std::unique_ptr<TrainLog> log;
    if (!opt.run_dir.empty())
        log = std::make_unique<TrainLog>(
            opt.run_dir + "/log.csv",
            std::vector<std::string>{"iteration", "total_steps", "time_s", "steps_per_s", "mean_reward",
                                     "mean_episode_len", "value_loss", "policy_loss", "entropy", "lr", "kl",
                                     "clip_fraction", "grad_norm", "action_std"});
    auto save = [&](const std::string& name) {
        if (!opt.run_dir.empty()) ppo.checkpoint(env.options().task).save(opt.run_dir + "/" + name);
    };

    TrainResult res;
    const auto t_start = std::chrono::steady_clock::now();
    long long total_steps = 0;
    for (int it = 0; it < opt.max_iterations; ++it) {
        const auto t_it = std::chrono::steady_clock::now();
        {
            FORGE_NVTX_RANGE("rollout");
            for (int t = 0; t < cfg.horizon; ++t) {
                ppo.act(env.obs(), actions.data());
                env.step(actions.data());
                ppo.observe(env.reward(), env.terminated(), env.truncated(), env.final_obs());
            }
        }
        const PpoStats st = ppo.update(env.obs());  // syncs once
        total_steps += static_cast<long long>(cfg.horizon) * N;

        // Episode stats: every env's last finished episode (an iteration where no
        // episode ends still has a number).
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
        const double it_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_it).count();
        const double sps = double(cfg.horizon) * N / it_s;
        res.iterations = it + 1;
        res.last_mean_return = mean_r;
        res.last_mean_length = mean_l;
        res.env_steps_per_s = total_steps / now;

        const bool bad = !std::isfinite(st.policy_loss) || !std::isfinite(st.value_loss) || !std::isfinite(st.entropy);
        if (log)
            log->row({double(ppo.iteration()), double(total_steps), now, sps, mean_r, mean_l, st.value_loss,
                      st.policy_loss, st.entropy, cfg.lr, st.approx_kl, st.clip_fraction, st.grad_norm, st.std});
        if (!opt.quiet)
            std::printf("it %5lld  steps %10lld  %8.3g steps/s  return %8.2f  ep_len %7.1f  vloss %.4f  kl %.4f  std %.3f\n",
                        static_cast<long long>(ppo.iteration()), total_steps, sps, mean_r, mean_l, st.value_loss,
                        st.approx_kl, st.std);
        if (bad) {
            std::fprintf(stderr, "stop: NaN or inf in the losses at iteration %lld\n",
                         static_cast<long long>(ppo.iteration()));
            res.stopped_on_nan = true;
            break;
        }
        if (finished && mean_r > res.best_mean_return) {
            res.best_mean_return = mean_r;
            save("best.bin");
        }
        if (opt.save_every > 0 && ppo.iteration() % opt.save_every == 0) {
            char name[32];
            std::snprintf(name, sizeof(name), "ckpt_%06lld.bin", static_cast<long long>(ppo.iteration()));
            save(name);
        }
    }
    save("final.bin");
    return res;
}

}  // namespace temper
