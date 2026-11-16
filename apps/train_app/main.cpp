// train_app: headless training (doc 7.1).
//
//   ./train_app --config ../configs/cartpole.txt --save runs/       train PPO, write the run folder
//   ./train_app --config ../configs/sac_cartpole.txt --save runs/   train SAC
//   ./train_app --task cartpole --max_iterations 100                train, no files
//   ./train_app --config ../configs/cartpole.txt --resume runs/cartpole_.../ckpt_000050.bin --save runs/
//   ./train_app --task cartpole --num_envs 4096 --bench_env 1000     env steps/s with random actions
//   ./train_app --task cartpole --backend cpu --num_envs 256 --bench_env 200
//   ./train_app --config ../configs/g1_walk.txt --save runs/ --viz true   watch while training
//                (like Isaac Gym's viewer: full speed, V = drawing on / off; train_viewer.h)
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "forge/env/env.h"
#include "forge/util/config.h"
#include "forge/util/device_info.h"
#ifdef CRUCIBLE_HAS_FLUX
#include "train_viewer.h"
#endif
#ifdef CRUCIBLE_HAS_TEMPER
#include "temper/algos/algorithm.h"
#include "temper/algos/ppo/ppo.h"
#include "temper/algos/sac/sac.h"
#include "temper/common/train_loop.h"
#include "temper/nn/gemm.h"
#endif
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
        const int viz_envs = viz ? cfg.get_int("viz_envs", 16) : 0;  // envs drawn by --viz
        (void)viz_envs;  // used only in builds with flux
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

        const std::string algo = cfg.get_string("algo", "ppo");
        if (algo != "ppo" && algo != "sac" && algo != "tqc")
            throw forge::ConfigError("algo must be ppo, sac or tqc, not '" + algo + "'");
#ifdef CRUCIBLE_HAS_TEMPER
        temper::TrainOptions topt = temper::TrainOptions::from_config(cfg);
        const std::string gemm = cfg.get_string("gemm", "cublas");  // cublas | cutlass (bench_gemm compares them)
        if (gemm != "cublas" && gemm != "cutlass") throw forge::ConfigError("gemm must be cublas or cutlass, not '" + gemm + "'");
        temper::set_gemm_backend(gemm == "cutlass" ? temper::GemmBackend::kCutlass : temper::GemmBackend::kCublas);
        // Read the algorithm's keys now (so typos are reported), build it after the env.
        if (algo == "ppo") temper::PpoConfig::from_config(cfg);
        else temper::SacConfig::from_config(&cfg, algo == "tqc");
#endif

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

        std::string run_dir;
        if (!save.empty() && bench_steps == 0) {
            run_dir = forge::make_run_dir(save, eopt.task, static_cast<int>(eopt.seed));
            cfg.save(run_dir + "/config.txt");
            std::printf("  run dir   %s\n", run_dir.c_str());
        }

        auto env = forge::make_env(eopt);
        std::printf("  env       obs %d, act %d, %d steps per episode, physics dt %.4f x %d, solver %s, %s\n",
                    env->obs_dim(), env->act_dim(), env->max_episode_steps(), eopt.physics_dt, eopt.decimation,
                    env->solver_name(), eopt.backend == forge::EnvBackend::kGpu ? "gpu" : "cpu");

        if (bench_steps > 0) {
            bench_env(*env, bench_steps);
            return 0;
        }
#ifdef CRUCIBLE_HAS_TEMPER
        topt.run_dir = run_dir;
        auto learner = temper::make_algorithm(algo, env->obs_dim(), env->act_dim(), env->num_envs(), &cfg,
                                              static_cast<cudaStream_t>(env->stream()));
        std::printf("  algo      %s, %d env steps per iteration, %d iterations, gemm %s\n", learner->name(),
                    learner->steps_per_iteration(), topt.max_iterations,
                    temper::gemm_backend_name(temper::gemm_backend()));
#ifdef CRUCIBLE_HAS_FLUX
        std::unique_ptr<TrainViewer> viewer;
        if (viz) {
            flux::RendererOptions ro;
            ro.vsync = false;       // never wait for the screen: training runs at full speed
            ro.validation = false;
            ro.shader_dir = FLUX_SHADER_DIR;
            viewer = std::make_unique<TrainViewer>(*env, viz_envs, ro, 1280, 720);
            topt.after_step = [&viewer]() { return viewer->after_step(); };
        }
#endif
        const temper::TrainResult r = temper::train(*env, *learner, topt);
        std::printf("done: %d iterations, last mean return %.2f (episode length %.1f), best %.2f, %.3g env-steps/s "
                    "overall%s\n", r.iterations, r.last_mean_return, r.last_mean_length, r.best_mean_return,
                    r.env_steps_per_s, r.stopped_on_nan ? ", STOPPED ON NaN" : (r.stopped_by_user ? ", stopped (window closed)" : ""));
        if (!run_dir.empty()) std::printf("run folder: %s\n", run_dir.c_str());
        return r.stopped_on_nan ? 3 : 0;
#else
        std::fprintf(stderr, "error: training needs a build with CUDA (temper)\n");
        return 1;
#endif
    } catch (const forge::ConfigError& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
