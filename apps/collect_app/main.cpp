// collect_app (Phase 9b): records a task's scripted expert into a demonstration
// file (forge/env/demo_file.h) for imitation learning.
//
//   ./collect_app --config ../configs/panda_pick.txt --episodes 10000 --out demos/pick.bin
//   ./collect_app --task panda_pick --num_envs 256 --backend cpu --episodes 512 --out demos/small.bin
//
// DART noise (--noise, default 0.2): each episode draws a noise level s uniform in
// [0, noise]; the env runs expert + Gaussian(0, s) (clipped to [-1, 1]), and the
// file keeps both the expert's action (the label) and the executed one. The noise
// shows the learner states a little off the expert's path, with the expert's fix
// as the label (Laskey et al. 2017, DART). --noise 0 records clean runs.
// Success per episode (panda_pick): the cube is more than 8 cm above its rest
// height at the end. --only_success true drops the failed episodes.
// Camera images (Phase 9c, builds with flux): --images true records the task's
// cameras (panda_pick: front and wrist) every --image_every steps (2) at
// --image_size x --image_size pixels (64) into the file (version 2, zlib).
// Options: the env keys (task, num_envs, seed, physics_dt, decimation, solver,
// backend), episodes, out, noise, only_success, images, image_every, image_size.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "forge/env/demo_file.h"
#include "forge/env/env.h"
#include "forge/util/config.h"
#ifdef CRUCIBLE_HAS_CUDA
#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"
#endif
#ifdef CRUCIBLE_HAS_FLUX
#include "camera_rig.h"
#endif

namespace {

// Did the episode succeed, from its final obs? Only tasks with a known rule.
bool episode_success(const std::string& task, const float* final_obs) {
    if (task == "panda_pick") return final_obs[26] - 0.025f > 0.08f;  // cube z (obs 26) - rest height > 8 cm
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        forge::Config cfg = forge::Config::from_cli(argc, argv);
        const forge::EnvOptions eopt = forge::EnvOptions::from_config(cfg, CRUCIBLE_ASSET_DIR);
        const int want = cfg.get_int("episodes", 1000);
        const std::string out = cfg.get_string("out", "demos.bin");
        const float noise = cfg.get_float("noise", 0.2f);
        const bool only_success = cfg.get_bool("only_success", false);
        const bool images = cfg.get_bool("images", false);
        const int image_every = cfg.get_int("image_every", 2), image_size = cfg.get_int("image_size", 64);
        // keys of other apps in a shared config file (algo, ppo.*) are fine here
        for (const auto& k : cfg.unused_keys())
            if (k.rfind("ppo.", 0) != 0 && k.rfind("sac.", 0) != 0 && k != "algo" && k != "max_iterations" &&
                k != "save_every")
                std::fprintf(stderr, "warning: unknown config key '%s' (typo?)\n", k.c_str());
        if (want <= 0 || noise < 0.0f) throw std::runtime_error("episodes must be > 0 and noise >= 0");
        if (images && (image_every <= 0 || image_size <= 0)) throw std::runtime_error("image_every and image_size must be > 0");
#ifndef CRUCIBLE_HAS_FLUX
        if (images) throw std::runtime_error("--images needs a build with -DBUILD_FLUX=ON");
#endif
        if (images && eopt.task != "panda_pick") throw std::runtime_error("--images: only panda_pick has cameras");

        auto env = forge::make_env(eopt);
        if (!env->has_expert()) throw std::runtime_error("task '" + eopt.task + "' has no scripted expert");
        const int N = env->num_envs(), D = env->obs_dim(), A = env->act_dim();
        std::printf("collect_app: task %s, %d envs (%s), obs %d, act %d, DART noise up to %.2f -> %s\n",
                    eopt.task.c_str(), N, eopt.backend == forge::EnvBackend::kGpu ? "gpu" : "cpu", D, A, noise, out.c_str());

        const bool gpu = eopt.backend == forge::EnvBackend::kGpu;
        std::vector<float> expert(size_t(N) * A), run(size_t(N) * A);
#ifdef CRUCIBLE_HAS_CUDA
        forge::DeviceBuffer<float> dact(gpu ? size_t(N) * A : 0);
#endif
        auto get_expert = [&]() {
            if (!gpu) return env->expert_actions(expert.data());
#ifdef CRUCIBLE_HAS_CUDA
            env->expert_actions(dact.data());
            CUDA_CHECK(cudaMemcpyAsync(expert.data(), dact.data(), expert.size() * 4, cudaMemcpyDeviceToHost,
                                       static_cast<cudaStream_t>(env->stream())));
            CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(env->stream())));
#endif
        };
        auto step = [&]() {
            if (!gpu) return env->step(run.data());
#ifdef CRUCIBLE_HAS_CUDA
            CUDA_CHECK(cudaMemcpyAsync(dact.data(), run.data(), run.size() * 4, cudaMemcpyHostToDevice,
                                       static_cast<cudaStream_t>(env->stream())));
            env->step(dact.data());
#endif
        };

        forge::DemoImages im;
#ifdef CRUCIBLE_HAS_FLUX
        std::unique_ptr<PickCameraRig> rig;
        if (images) {
            rig = std::make_unique<PickCameraRig>(env->model(), N, image_size, FLUX_SHADER_DIR);
            im = forge::DemoImages{PickCameraRig::kCameras, image_size, image_size, image_every};
            std::printf("collect_app: images: %d cameras x %d x %d every %d steps, rendered on %s\n", im.cameras,
                        image_size, image_size, image_every, rig->camera().device_name());
        }
        double render_s = 0.0;
        long long frames_rendered = 0;
#endif
        forge::DemoWriter writer(out, eopt.task, D, A, im);
        std::vector<forge::DemoEpisode> cur(N);
        std::vector<float> level(N);
        std::mt19937 rng(static_cast<unsigned>(eopt.seed) * 7919u + 1u);
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (int e = 0; e < N; ++e) level[e] = noise * uni(rng);
        forge::Env::HostCopy h;
        env->download(h);
        int written = 0, finished = 0, succeeded = 0;
        const auto t0 = std::chrono::steady_clock::now();
        auto t_print = t0;
        while (written < want) {
            get_expert();
            for (int e = 0; e < N; ++e)
                for (int k = 0; k < A; ++k) {
                    const size_t i = size_t(e) * A + k;
                    run[i] = std::clamp(expert[i] + level[e] * gauss(rng), -1.0f, 1.0f);
                }
#ifdef CRUCIBLE_HAS_FLUX
            if (rig) {  // camera frames of the envs at a recorded step (before the step, with its obs)
                bool any = false;
                for (int e = 0; e < N && !any; ++e) any = cur[e].reward.size() % size_t(image_every) == 0;
                if (any) {
                    const auto r0 = std::chrono::steady_clock::now();
                    rig->render(h.qpos.data());
                    render_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count();
                    ++frames_rendered;
                    for (int e = 0; e < N; ++e) {
                        if (cur[e].reward.size() % size_t(image_every) != 0) continue;
                        auto& img = cur[e].images;
                        const size_t at = img.size(), one = size_t(image_size) * image_size * 3;
                        img.resize(at + im.frame_bytes());
                        for (int c = 0; c < im.cameras; ++c) rig->camera().copy_rgb(e, c, img.data() + at + c * one);
                    }
                }
            }
#endif
            for (int e = 0; e < N; ++e) {
                auto& ep = cur[e];
                ep.obs.insert(ep.obs.end(), h.obs.begin() + size_t(e) * D, h.obs.begin() + size_t(e + 1) * D);
                ep.expert.insert(ep.expert.end(), expert.begin() + size_t(e) * A, expert.begin() + size_t(e + 1) * A);
                ep.executed.insert(ep.executed.end(), run.begin() + size_t(e) * A, run.begin() + size_t(e + 1) * A);
            }
            step();
            env->download(h);
            for (int e = 0; e < N && written < want; ++e) {
                auto& ep = cur[e];
                ep.reward.push_back(h.reward[e]);
                if (!h.terminated[e] && !h.truncated[e]) continue;
                ep.final_obs.assign(h.final_obs.begin() + size_t(e) * D, h.final_obs.begin() + size_t(e + 1) * D);
                ep.success = episode_success(eopt.task, ep.final_obs.data());
                ep.noise = level[e];
                ++finished;
                succeeded += ep.success;
                if (ep.success || !only_success) {
                    writer.write(ep);
                    ++written;
                }
                ep = forge::DemoEpisode{};
                level[e] = noise * uni(rng);
            }
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - t_print).count() >= 5.0) {
                t_print = now;
                std::printf("collect_app: %d / %d episodes written, success %.1f%% of %d finished\n", written, want,
                            finished ? 100.0 * succeeded / finished : 0.0, finished);
            }
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("collect_app: done: %d episodes in %s (%.1f s); success %d of %d finished (%.1f%%)\n", written,
                    out.c_str(), s, succeeded, finished, finished ? 100.0 * succeeded / finished : 0.0);
#ifdef CRUCIBLE_HAS_FLUX
        if (rig && frames_rendered > 0)
            std::printf("collect_app: images: %lld renders of %d envs x %d cameras, %.1f ms each (%.0f images/s); "
                        "stored %.1f MB of %.1f MB raw (zlib %.1fx)\n",
                        frames_rendered, N, im.cameras, 1000.0 * render_s / frames_rendered,
                        double(frames_rendered) * N * im.cameras / render_s, writer.image_bytes_stored() / 1e6,
                        writer.image_bytes_raw() / 1e6,
                        double(writer.image_bytes_raw()) / std::max<size_t>(1, writer.image_bytes_stored()));
#endif
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
