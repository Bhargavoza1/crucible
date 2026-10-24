// train_app: headless training. Phase 1 only reads config and flags, checks the
// GPU, and makes the run folder. Physics and PPO come in later phases.
//
//   ./train_app --task ant --num_envs 4096 --headless --save runs/
//   ./train_app --config configs/ant.txt --lr 1e-4
#include <cstdio>
#include <string>

#include "forge/util/config.h"
#include "forge/util/device_info.h"

int main(int argc, char** argv) {
    forge::Config cfg;
    try {
        cfg = forge::Config::from_cli(argc, argv);
    } catch (const forge::ConfigError& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }

    try {
        const std::string task = cfg.get_string("task", "cartpole");
        const int num_envs = cfg.get_int("num_envs", 1024);
        const int seed = cfg.get_int("seed", 1);
        const bool viz = cfg.get_bool("viz", false);
        const bool headless = cfg.get_bool("headless", !viz);
        const std::string save = cfg.get_string("save", "");

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
        if (num_envs <= 0) {
            std::fprintf(stderr, "error: num_envs must be > 0\n");
            return 2;
        }

        // Read the remaining doc 7.2 keys now, so they count as used and typos show up.
        cfg.get_float("physics_dt", 0.005f);
        cfg.get_int("decimation", 4);
        cfg.get_int("horizon", 24);
        cfg.get_int("epochs", 5);
        cfg.get_int("minibatches", 4);
        cfg.get_float("lr", 3e-4f);
        cfg.get_float("clip", 0.2f);
        cfg.get_float("gamma", 0.99f);
        cfg.get_float("lambda", 0.95f);
        cfg.get_int("save_every", 50);

        std::printf("crucible train_app\n");
        std::printf("  task      %s\n", task.c_str());
        std::printf("  num_envs  %d\n", num_envs);
        std::printf("  seed      %d\n", seed);
        std::printf("  mode      %s\n", viz ? "viz" : "headless");

        if (!forge::built_with_cuda()) {
            std::printf("  cuda      not built (CPU only build)\n");
        } else {
            const int n = forge::cuda_device_count();
            if (n == 0) {
                std::printf("  cuda      built, but no CUDA device found\n");
            } else {
                std::printf("  cuda      %s\n", forge::cuda_device_description(0).c_str());
            }
        }

        for (const auto& k : cfg.unused_keys())
            std::fprintf(stderr, "warning: unknown config key '%s' (typo?)\n", k.c_str());

        if (!save.empty()) {
            const std::string dir = forge::make_run_dir(save, task, seed);
            cfg.save(dir + "/config.txt");
            std::printf("  run dir   %s\n", dir.c_str());
        }

        std::printf("Phase 1: no physics yet. Done.\n");
    } catch (const forge::ConfigError& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
