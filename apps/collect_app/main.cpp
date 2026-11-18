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
// height at the end; (openarm_pick_place): the episode finished (cube in the tray,
// the arm back home); (openarm_stack): the episode finished (the tower red, green,
// blue on the spot, both hands home). --only_success true drops the failed episodes
// and runs on until --episodes successful ones are written.
// Camera images (Phase 9c, builds with flux): --images true records the task's
// cameras (panda_pick: front and wrist) every --image_every steps (2) at
// --image_size x --image_size pixels (64) into the file (version 2, zlib).
// openarm_stack (Phase 11): its 4 cameras (head, left wrist, ceiling, right wrist);
// each env's frames go into its own zlib stream as they are rendered (several cores),
// not kept raw: its episodes are long (up to 800 steps, 1024 envs x 400 frames x 4
// images of 128 x 128 would be about 80 GB of memory).
// Mistakes with recovery (openarm_pick_place, openarm_stack; off by default): DART
// noise keeps the arm near the expert's path, so the demos show little recovery from
// bigger errors. --push_prob P: an episode is pushed with probability P: once, at a
// random step in 10-299, for push_steps/2 .. --push_steps (20) steps, every move and
// turn action is a random constant in +-0.35 (both arms; the grippers stay the
// expert's), then the expert takes over and recovers. --slip_prob P: with probability
// P a gripper opens for --slip_steps (6) steps after it has been closed (the expert's
// grip) for 10-59 steps: the held cube slips or drops, the expert picks it up again.
// The labels (expert) are always the expert's action, also during a mistake; the
// executed actions are what ran. With --only_success only the recovered episodes are
// kept. Prints how many episodes were pushed or slipped and how many of them succeeded.
// Options: the env keys (task, num_envs, seed, physics_dt, decimation, solver,
// backend), episodes, out, noise, only_success, images, image_every, image_size,
// push_prob, push_steps, slip_prob, slip_steps.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <zlib.h>

#include "forge/env/demo_file.h"
#include "forge/env/env.h"
#include "forge/util/config.h"
#ifdef CRUCIBLE_HAS_CUDA
#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"
#endif
#ifdef CRUCIBLE_HAS_FLUX
#include "camera_rig.h"
#include "openarm_camera_rig.h"
#include "stack_camera_rig.h"
#endif

namespace {

// Did the episode succeed, from its final obs (and how it ended)? Only tasks with a known rule.
bool episode_success(const std::string& task, const float* final_obs, bool terminated) {
    // panda_pick(_color): the (target) cube's z (obs 26) - rest height > 8 cm
    if (task == "panda_pick" || task == "panda_pick_color") return final_obs[26] - 0.025f > 0.08f;
    // openarm_pick_place: the episode finished (terminated: placed and the arm home), cube
    // minus tray center (obs 36, 37) inside the tray, resting (z, obs 26), TCP up (obs 20;
    // a broken env's final obs is all zeros)
    if (task == "openarm_pick_place")
        return terminated && std::fabs(final_obs[36]) < 0.06f && std::fabs(final_obs[37]) < 0.06f &&
               final_obs[26] < 0.031f && final_obs[20] > 0.08f;
    // openarm_stack: the episode finished (terminated: the tower built, both hands home),
    // and the final obs agree: red (obs 48-50) on the table within 2.5 cm of the spot
    // (0.41, 0), green (55-57) on red and blue (62-64) on green (1.8 cm in x y, 4 cm +- 5 mm
    // up), both TCPs up (obs 38, 44; a broken env's final obs is all zeros)
    if (task == "openarm_stack") {
        const float* r = final_obs + 48;
        const float* g = final_obs + 55;
        const float* b = final_obs + 62;
        auto on = [](const float* a, const float* u) {
            return std::hypot(a[0] - u[0], a[1] - u[1]) < 0.018f && std::fabs(a[2] - u[2] - 0.04f) < 0.005f;
        };
        return terminated && std::hypot(r[0] - 0.41f, r[1]) < 0.025f && r[2] < 0.025f && on(g, r) && on(b, g) &&
               final_obs[38] > 0.12f && final_obs[44] > 0.12f;
    }
    return false;
}

// One zlib stream: an episode's frames compressed as they come (openarm_stack).
class FrameStream {
public:
    FrameStream() {
        z_ = z_stream{};
        if (deflateInit(&z_, 1) != Z_OK) throw std::runtime_error("zlib deflateInit failed");
    }
    ~FrameStream() { deflateEnd(&z_); }
    FrameStream(const FrameStream&) = delete;
    FrameStream& operator=(const FrameStream&) = delete;
    void add(const uint8_t* data, size_t n) { pump(data, n, Z_NO_FLUSH); }
    std::vector<uint8_t> finish() {  // the stream; then ready for the next episode
        pump(nullptr, 0, Z_FINISH);
        std::vector<uint8_t> done(out_.begin(), out_.end());
        out_.clear();
        out_.shrink_to_fit();
        deflateReset(&z_);
        return done;
    }

private:
    void pump(const uint8_t* data, size_t n, int flush) {
        z_.next_in = const_cast<Bytef*>(data);
        z_.avail_in = uInt(n);
        uint8_t buf[1 << 16];
        int ret;
        do {
            z_.next_out = buf;
            z_.avail_out = sizeof(buf);
            ret = deflate(&z_, flush);
            if (ret == Z_STREAM_ERROR) throw std::runtime_error("zlib deflate failed");
            out_.insert(out_.end(), buf, buf + (sizeof(buf) - z_.avail_out));
        } while (z_.avail_out == 0 || (flush == Z_FINISH && ret != Z_STREAM_END));
    }
    z_stream z_;
    std::vector<uint8_t> out_;
};

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
        const float push_prob = cfg.get_float("push_prob", 0.0f), slip_prob = cfg.get_float("slip_prob", 0.0f);
        const int push_steps = cfg.get_int("push_steps", 20), slip_steps = cfg.get_int("slip_steps", 6);
        // keys of other apps in a shared config file (algo, ppo.*) are fine here
        for (const auto& k : cfg.unused_keys())
            if (k.rfind("ppo.", 0) != 0 && k.rfind("sac.", 0) != 0 && k != "algo" && k != "max_iterations" &&
                k != "save_every")
                std::fprintf(stderr, "warning: unknown config key '%s' (typo?)\n", k.c_str());
        if (want <= 0 || noise < 0.0f) throw std::runtime_error("episodes must be > 0 and noise >= 0");
        if (images && (image_every <= 0 || image_size <= 0)) throw std::runtime_error("image_every and image_size must be > 0");
        if (push_prob < 0.0f || push_prob > 1.0f || slip_prob < 0.0f || slip_prob > 1.0f || push_steps < 2 || slip_steps < 1)
            throw std::runtime_error("push_prob and slip_prob in 0..1, push_steps >= 2, slip_steps >= 1");
        // the gripper actions (the rest are moves and turns): OpenArm tasks only
        const std::vector<int> grips = eopt.task == "openarm_stack"        ? std::vector<int>{4, 9}
                                       : eopt.task == "openarm_pick_place" ? std::vector<int>{4}
                                                                            : std::vector<int>{};
        if ((push_prob > 0.0f || slip_prob > 0.0f) && grips.empty())
            throw std::runtime_error("--push_prob / --slip_prob: only openarm_pick_place and openarm_stack");
#ifndef CRUCIBLE_HAS_FLUX
        if (images) throw std::runtime_error("--images needs a build with -DBUILD_FLUX=ON");
#endif
        if (images && eopt.task != "panda_pick" && eopt.task != "panda_pick_color" && eopt.task != "openarm_pick_place" &&
            eopt.task != "openarm_stack")
            throw std::runtime_error(
                "--images: only panda_pick, panda_pick_color, openarm_pick_place and openarm_stack have cameras");

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
        std::unique_ptr<OpenArmCameraRig> oa_rig;  // openarm_pick_place: the cell's 4 cameras, meshes
        std::unique_ptr<StackCameraRig> st_rig;    // openarm_stack: the same cameras, both arms, 3 cubes
        const flux::BatchCamera* cams = nullptr;
        if (images && eopt.task == "openarm_stack") {
            st_rig = std::make_unique<StackCameraRig>(env->model(), N, image_size, FLUX_SHADER_DIR,
                                                      std::string(CRUCIBLE_ASSET_DIR) + "/robots/openarm/openarm_cell.cmesh");
            im = forge::DemoImages{StackCameraRig::kCameras, image_size, image_size, image_every};
            cams = &st_rig->camera();
        } else if (images && eopt.task == "openarm_pick_place") {
            oa_rig = std::make_unique<OpenArmCameraRig>(env->model(), N, image_size, FLUX_SHADER_DIR,
                                                        std::string(CRUCIBLE_ASSET_DIR) + "/robots/openarm/openarm_cell.cmesh");
            im = forge::DemoImages{OpenArmCameraRig::kCameras, image_size, image_size, image_every};
            cams = &oa_rig->camera();
        } else if (images) {
            rig = std::make_unique<PickCameraRig>(env->model(), N, image_size, FLUX_SHADER_DIR);
            im = forge::DemoImages{PickCameraRig::kCameras, image_size, image_size, image_every};
            cams = &rig->camera();
        }
        if (images)
            std::printf("collect_app: images: %d cameras x %d x %d every %d steps, rendered on %s\n", im.cameras,
                        image_size, image_size, image_every, cams->device_name());
        double render_s = 0.0, copy_s = 0.0;
        long long frames_rendered = 0;
#endif
        // openarm_stack: each env's frames into its own zlib stream as they are rendered
        const bool stream = images && eopt.task == "openarm_stack";
        std::vector<std::unique_ptr<FrameStream>> zs(stream ? N : 0);
        for (auto& z : zs) z = std::make_unique<FrameStream>();
        std::vector<int> frames(N, 0);
        [[maybe_unused]] const int nthreads = std::max(1, int(std::thread::hardware_concurrency()));
        forge::DemoWriter writer(out, eopt.task, D, A, im);
        std::vector<forge::DemoEpisode> cur(N);
        std::vector<float> level(N);
        std::mt19937 rng(static_cast<unsigned>(eopt.seed) * 7919u + 1u);
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (int e = 0; e < N; ++e) level[e] = noise * uni(rng);
        // per env: this episode's planned mistakes (--push_prob, --slip_prob)
        struct Mishap {
            int push_at = -1, push_len = 0;  // push: first step, steps
            std::vector<float> push;         // the pushed actions [A] (grippers unused)
            int slip_after = -1, slip_left = 0, slip_dim = -1;  // slip: after this many closed steps
            std::vector<int> closed;                            // per gripper: steps closed so far
            bool pushed = false, slipped = false;
        };
        std::vector<Mishap> mis(N);
        auto plan_mishap = [&](int e) {  // draws nothing when both are off (old runs stay the same)
            Mishap m;
            m.closed.assign(grips.size(), 0);
            if (push_prob > 0.0f && uni(rng) < push_prob) {
                m.push_at = 10 + std::min(289, int(uni(rng) * 290.0f));
                m.push_len = push_steps / 2 + std::min(push_steps - push_steps / 2, int(uni(rng) * float(push_steps - push_steps / 2 + 1)));
                m.push.resize(size_t(A));
                for (auto& x : m.push) x = 0.35f * (2.0f * uni(rng) - 1.0f);
            }
            if (slip_prob > 0.0f && uni(rng) < slip_prob) m.slip_after = 10 + std::min(49, int(uni(rng) * 50.0f));
            mis[size_t(e)] = std::move(m);
        };
        for (int e = 0; e < N; ++e) plan_mishap(e);
        int pushed_n = 0, pushed_ok = 0, slipped_n = 0, slipped_ok = 0;
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
            for (int e = 0; e < N; ++e) {  // the planned mistakes (labels stay the expert's)
                Mishap& m = mis[size_t(e)];
                const int t = int(cur[e].reward.size());
                float* r = run.data() + size_t(e) * A;
                const float* x = expert.data() + size_t(e) * A;
                if (m.push_at >= 0 && t >= m.push_at && t < m.push_at + m.push_len) {
                    for (int k = 0; k < A; ++k)
                        if (std::find(grips.begin(), grips.end(), k) == grips.end()) r[k] = m.push[size_t(k)];
                    m.pushed = true;
                }
                if (m.slip_after >= 0 && !m.slipped)
                    for (size_t g = 0; g < grips.size(); ++g) {
                        m.closed[g] = x[grips[g]] > 0.0f ? m.closed[g] + 1 : 0;
                        if (m.closed[g] >= m.slip_after) {
                            m.slipped = true, m.slip_dim = grips[g], m.slip_left = slip_steps;
                            break;
                        }
                    }
                if (m.slip_left > 0) r[m.slip_dim] = -1.0f, --m.slip_left;  // open: the cube slips
            }
#ifdef CRUCIBLE_HAS_FLUX
            if (cams) {  // camera frames of the envs at a recorded step (before the step, with its obs)
                bool any = false;
                for (int e = 0; e < N && !any; ++e) any = cur[e].reward.size() % size_t(image_every) == 0;
                if (any) {
                    const auto r0 = std::chrono::steady_clock::now();
                    if (st_rig)
                        st_rig->render(h.qpos.data());
                    else if (oa_rig)
                        oa_rig->render(h.qpos.data(), h.obs.data(), D);
                    else
                        rig->render(h.qpos.data(), h.obs.data(), D);
                    const auto r1 = std::chrono::steady_clock::now();
                    render_s += std::chrono::duration<double>(r1 - r0).count();
                    ++frames_rendered;
                    if (stream) {  // copy and compress on all cores, env by env
                        std::vector<std::thread> pool;
                        for (int t = 0; t < nthreads; ++t)
                            pool.emplace_back([&, t]() {
                                std::vector<uint8_t> frame(im.frame_bytes());
                                const size_t one = size_t(image_size) * image_size * 3;
                                for (int e = t; e < N; e += nthreads) {
                                    if (cur[e].reward.size() % size_t(image_every) != 0) continue;
                                    for (int c = 0; c < im.cameras; ++c) cams->copy_rgb(e, c, frame.data() + c * one);
                                    zs[size_t(e)]->add(frame.data(), frame.size());
                                    ++frames[size_t(e)];
                                }
                            });
                        for (auto& th : pool) th.join();
                    } else {
                        for (int e = 0; e < N; ++e) {
                            if (cur[e].reward.size() % size_t(image_every) != 0) continue;
                            auto& img = cur[e].images;
                            const size_t at = img.size(), one = size_t(image_size) * image_size * 3;
                            img.resize(at + im.frame_bytes());
                            for (int c = 0; c < im.cameras; ++c) cams->copy_rgb(e, c, img.data() + at + c * one);
                        }
                    }
                    copy_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - r1).count();
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
            // Episodes that ended this step: compress their images on all cores at
            // once (all envs often end together), then write them in env order.
            std::vector<int> done;
            for (int e = 0; e < N; ++e) {
                cur[e].reward.push_back(h.reward[e]);
                if (h.terminated[e] || h.truncated[e]) done.push_back(e);
            }
            std::vector<int> keep;
            for (int e : done) {
                auto& ep = cur[e];
                ep.final_obs.assign(h.final_obs.begin() + size_t(e) * D, h.final_obs.begin() + size_t(e + 1) * D);
                ep.success = episode_success(eopt.task, ep.final_obs.data(), h.terminated[e] != 0);
                ep.noise = level[e];
                ++finished;
                succeeded += ep.success;
                if (mis[size_t(e)].pushed) ++pushed_n, pushed_ok += ep.success;
                if (mis[size_t(e)].slipped) ++slipped_n, slipped_ok += ep.success;
                if ((ep.success || !only_success) && written + int(keep.size()) < want) keep.push_back(e);
            }
            std::vector<std::vector<uint8_t>> packed(keep.size());
            if (stream) {  // finish every ended env's stream (kept or not: the next episode starts empty)
                for (int e : done) {
                    std::vector<uint8_t> z = zs[size_t(e)]->finish();
                    const auto it = std::find(keep.begin(), keep.end(), e);
                    if (it != keep.end()) {
                        const int want_frames = (cur[size_t(e)].steps() + image_every - 1) / image_every;
                        if (frames[size_t(e)] != want_frames)
                            throw std::runtime_error("collect_app: env " + std::to_string(e) + " recorded " +
                                                     std::to_string(frames[size_t(e)]) + " frames, its episode needs " +
                                                     std::to_string(want_frames));
                        packed[size_t(it - keep.begin())] = std::move(z);
                    }
                    frames[size_t(e)] = 0;
                }
            } else if (im.cameras > 0 && !keep.empty()) {
                const int nt = std::max(1, std::min(int(keep.size()), int(std::thread::hardware_concurrency())));
                std::vector<std::thread> pool;
                for (int t = 0; t < nt; ++t)
                    pool.emplace_back([&, t]() {
                        for (size_t k = size_t(t); k < keep.size(); k += size_t(nt))
                            packed[k] = forge::DemoWriter::compress_images(cur[keep[k]].images);
                    });
                for (auto& th : pool) th.join();
            }
            for (size_t k = 0; k < keep.size(); ++k) {
                writer.write(cur[keep[k]], im.cameras > 0 ? &packed[k] : nullptr);
                ++written;
            }
            for (int e : done) {
                cur[e] = forge::DemoEpisode{};
                level[e] = noise * uni(rng);
                plan_mishap(e);
            }
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - t_print).count() >= 5.0) {
                t_print = now;
                std::printf("collect_app: %d / %d episodes written, success %.1f%% of %d finished", written, want,
                            finished ? 100.0 * succeeded / finished : 0.0, finished);
#ifdef CRUCIBLE_HAS_FLUX
                if (frames_rendered > 0)
                    std::printf("; images: %lld renders, %.1f ms render + %.1f ms copy each", frames_rendered,
                                1000.0 * render_s / frames_rendered, 1000.0 * copy_s / frames_rendered);
#endif
                std::printf("\n");
            }
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("collect_app: done: %d episodes in %s (%.1f s); success %d of %d finished (%.1f%%)\n", written,
                    out.c_str(), s, succeeded, finished, finished ? 100.0 * succeeded / finished : 0.0);
        if (push_prob > 0.0f || slip_prob > 0.0f)
            std::printf("collect_app: mistakes: pushed %d finished episodes (%d succeeded, %.1f%%), slipped %d (%d "
                        "succeeded, %.1f%%)\n",
                        pushed_n, pushed_ok, pushed_n ? 100.0 * pushed_ok / pushed_n : 0.0, slipped_n, slipped_ok,
                        slipped_n ? 100.0 * slipped_ok / slipped_n : 0.0);
#ifdef CRUCIBLE_HAS_FLUX
        if (cams && frames_rendered > 0)
            std::printf("collect_app: images: %lld renders of %d envs x %d cameras, %.1f ms render + %.1f ms copy each "
                        "(%.0f images/s); stored %.1f MB of %.1f MB raw (zlib %.1fx)\n",
                        frames_rendered, N, im.cameras, 1000.0 * render_s / frames_rendered,
                        1000.0 * copy_s / frames_rendered,
                        double(frames_rendered) * N * im.cameras / (render_s + copy_s), writer.image_bytes_stored() / 1e6,
                        writer.image_bytes_raw() / 1e6,
                        double(writer.image_bytes_raw()) / std::max<size_t>(1, writer.image_bytes_stored()));
#endif
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
