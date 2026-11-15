// train_app: headless training. Phase 7a: makes the Env from the config and can
// benchmark it with random actions. PPO comes in Phase 7c.
//
//   ./train_app --task cartpole --num_envs 4096 --bench_env 1000     env steps/s, random actions
//   ./train_app --task cartpole --backend cpu --num_envs 256 --bench_env 200
//   ./train_app --config configs/cartpole.txt --save runs/
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "forge/env/env.h"
#include "forge/util/config.h"
#include "forge/util/device_info.h"
#ifdef CRUCIBLE_HAS_CUDA
#include <cuda_runtime.h>

#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"
#endif

namespace {

// Random actions, all envs, 'steps' env steps (each = decimation physics steps).
void bench_env(forge::Env& env, int steps) {
    const forge::EnvOptions& o = env.options();
    const size_t na = size_t(o.num_envs) * env.act_dim();
    std::vector<float> host_act(na);
    float* act = host_act.data();
#ifdef CRUCIBLE_HAS_CUDA
    forge::DeviceBuffer<float> dev_act;
    if (o.backend == forge::EnvBackend::kGpu) {
        dev_act.resize(na);
        act = dev_act.data();
    }
#endif
    auto sync = [&env] {
#ifdef CRUCIBLE_HAS_CUDA
        if (env.stream()) CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(env.stream())));
#endif
    };
    for (int t = 0; t < 10; ++t) {  // warm up
        env.random_actions(act, uint64_t(t));
        env.step(act);
    }
    sync();
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < steps; ++t) {
        env.random_actions(act, uint64_t(t + 10));
        env.step(act);
    }
    sync();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double sps = double(o.num_envs) * steps / s;
    std::printf("bench_env: %d steps x %d envs in %.3f s: %.3g env-steps/s (%.3g physics steps/s)\n", steps,
                o.num_envs, s, sps, sps * o.decimation);
}

}  // namespace

int main(int argc, char** argv) {
    forge::Config cfg;
    try {
        cfg = forge::Config::from_cli(argc, argv);
    } catch (const forge::ConfigError& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }

    try {
        const forge::EnvOptions eopt = forge::EnvOptions::from_config(cfg, CRUCIBLE_ASSET_DIR);
        const bool viz = cfg.get_bool("viz", false);
        const bool headless = cfg.get_bool("headless", !viz);
        const std::string save = cfg.get_string("save", "");
        const int bench_steps = cfg.get_int("bench_env", 0);

        if (viz && headless) {
            std::fprintf(stderr, "error: --viz and --headless cannot be used together\n");
            return 2;
        }
#ifndef CRUCIBLE_HAS_FLUX
        if (viz) {
            std::fprintf(stderr, "error: --viz needs a build with -DBUILD_FLUX=ON\n");
            return 2;
        }
#endif

        // PPO keys (doc 7.2, prefix ppo.): read now so they count as used; used in Phase 7c.
        cfg.get_string("algo", "ppo");
        cfg.get_int("ppo.horizon", 24);
        cfg.get_int("ppo.epochs", 5);
        cfg.get_int("ppo.minibatches", 4);
        cfg.get_float("ppo.lr", 3e-4f);
        cfg.get_float("ppo.clip", 0.2f);
        cfg.get_float("ppo.gamma", 0.99f);
        cfg.get_float("ppo.lambda", 0.95f);
        cfg.get_int("save_every", 50);

        std::printf("crucible train_app\n");
        std::printf("  task      %s\n", eopt.task.c_str());
        std::printf("  num_envs  %d\n", eopt.num_envs);
        std::printf("  seed      %llu\n", static_cast<unsigned long long>(eopt.seed));
        std::printf("  mode      %s\n", viz ? "viz" : "headless");
        if (!forge::built_with_cuda()) {
            std::printf("  cuda      not built (CPU only build)\n");
        } else if (forge::cuda_device_count() == 0) {
            std::printf("  cuda      built, but no CUDA device found\n");
        } else {
            std::printf("  cuda      %s\n", forge::cuda_device_description(0).c_str());
        }

        for (const auto& k : cfg.unused_keys())
            std::fprintf(stderr, "warning: unknown config key '%s' (typo?)\n", k.c_str());

        if (!save.empty()) {
            const std::string dir = forge::make_run_dir(save, eopt.task, static_cast<int>(eopt.seed));
            cfg.save(dir + "/config.txt");
            std::printf("  run dir   %s\n", dir.c_str());
        }

        auto env = forge::make_env(eopt);
        std::printf("  env       obs %d, act %d, %d steps per episode, physics dt %.4f x %d, solver %s, %s\n",
                    env->obs_dim(), env->act_dim(), env->max_episode_steps(), eopt.physics_dt, eopt.decimation,
                    env->solver_name(), eopt.backend == forge::EnvBackend::kGpu ? "gpu" : "cpu");

        if (bench_steps > 0)
            bench_env(*env, bench_steps);
        else
            std::printf("Phase 7a: Env ready. Training (PPO) comes in Phase 7c.\n");
    } catch (const forge::ConfigError& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
