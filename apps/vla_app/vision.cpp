// vla_app --task <vision task> (Phases 10, 11; one runner for every task of
// apps/common/vision_task): the VLA from tools/vla_openarm.py, closed loop. Each env runs
// one episode from its start: the task's cameras render, the model turns the images it
// was trained on (<onnx without .onnx>_openarm.txt: "cameras 0 1 3") + the robot's state
// (the first VisionTask::state numbers of obs) + the instruction into an action chunk,
// the env runs the first --exec actions of it, then the policy plans again.
//
//   ./vla_app --task openarm_pick_place --onnx runs/oa_vla.onnx --eval_envs 500
//   ./vla_app --task openarm_stack --onnx runs/st_vla.onnx --eval_envs 500
//
// Success: the task's done rule (VisionTask::done_ok, as collect_app, distill_app and
// rl_app). --eval_expert true (default): the scripted expert on the same starts, for
// comparison. Failures (--save_failures K, default 20; 0 = off) are counted by where
// they stopped (VisionTask::fail_kind; openarm_pick_place: never reached the cube ...
// placed, did not go home; openarm_stack: never lifted a cube ... tower built, hands not
// home), each printed with the task's numbers (fail_detail); the first K are saved as
// camera strips <onnx without .onnx>_fail_<env>.png (rows: the model's cameras; one
// column per VisionTask::strip_every steps). Physics as the task's config file.
// Flow noise (--noise): fresh (default): new random noise at every plan | episode: one
// noise per env for the whole episode | zero: noise 0 (the flow's middle path).
// DAgger (--record FILE): every env's episode as a demo file (collect_app's format, all
// the task's cameras every 2 steps): the states the POLICY visited, labeled with the
// EXPERT's action in each (executed = the policy's). Use another --eval_seed than the one
// you test with. The images are compressed frame by frame as they come (one zlib stream
// per env, all cores), so thousands of envs fit in memory. Then train on demos + it:
// vla_openarm.py demos.bin dagger1.bin --init vla.pt.
// Options: task, onnx, info, eval_envs, eval_seed, eval_backend, exec, device,
// eval_expert, seed, save_failures, noise, record.
#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "openarm_common.h"
#include "vla_model.h"
#include "common/vision_task.h"
#include "forge/env/demo_file.h"
#include "forge/env/env.h"
#include "forge/util/config.h"
#include "forge/util/device_buffer.h"

namespace {

using pipeline::VisionTask;
using vla::Clock;
using vla::FrameStream;
using vla::Info;
using vla::Model;
using vla::since;

struct Trace {
    std::vector<float> progress;  // VisionTask::progress_dim numbers
    std::vector<uint8_t> frames;  // [frame][cam][size][size][3]
};

struct Result {
    int recorded = 0, episodes = 0, done = 0;
    double seconds = 0, steps_done = 0;
};

struct RunOptions {
    int envs = 500, exec = 4, save_failures = 0;
    uint64_t seed = 1000, rng_seed = 1;
    forge::EnvBackend backend = forge::EnvBackend::kGpu;
    std::string noise = "fresh", prefix;
    forge::DemoWriter* record = nullptr;  // DAgger: the policy's states with the expert's actions
};

// Every env runs one episode. model null = the scripted expert.
Result run(const VisionTask& task, Model* model, const Info& info, const RunOptions& o) {
    auto env = forge::make_env(task.env_options(o.envs, o.seed, o.backend));
    const int N = env->num_envs(), D = env->obs_dim(), A = env->act_dim();
    if (A != task.act || D != task.obs)
        throw std::runtime_error("expected " + task.name + " (obs " + std::to_string(task.obs) + ", act " +
                                 std::to_string(task.act) + ")");
    const int kState = task.state, kAct = task.act;
    const bool gpu = o.backend == forge::EnvBackend::kGpu;
    const bool trace = model && o.save_failures > 0, record = model && o.record;
    constexpr int kImageEvery = 2;  // recorded frames: every 2 steps (as collect_app)
    const int strip_envs = task.strip_envs > 0 ? std::min(N, task.strip_envs) : N;
    const int cams = int(info.cameras.size()), size = model ? model->size : 0, C = model ? model->chunk : 1;
    std::unique_ptr<pipeline::CameraRig> rig;
    if (model) {
        if (model->cams != cams) throw std::runtime_error("the model's camera count differs from the info file's");
        rig = task.make_rig(env->model(), N, size);
    }
    const int E = int(info.text.size());
    const size_t one = size_t(size) * size * 3, fb = one * cams;
    std::vector<uint8_t> img(size_t(N) * fb);
    std::vector<float> st(size_t(N) * kState), tx(size_t(N) * E), noise(size_t(N) * C * kAct),
        chunk(size_t(N) * C * kAct), act(size_t(N) * A), ex(record ? size_t(N) * A : 0);
    const size_t fb_all = one * task.cameras;  // recorded: all the task's cameras
    std::vector<uint8_t> all(record ? size_t(N) * fb_all : 0);
    std::vector<forge::DemoEpisode> rec(record ? N : 0);
    std::vector<std::unique_ptr<FrameStream>> zs(record ? N : 0);
    for (auto& z : zs) z = std::make_unique<FrameStream>();
    const int nthreads = std::max(1, int(std::thread::hardware_concurrency()));
    for (int e = 0; e < N; ++e) std::copy(info.text.begin(), info.text.end(), tx.begin() + size_t(e) * E);
    forge::DeviceBuffer<float> dact;
    if (gpu) dact.resize(size_t(N) * A);
    std::mt19937_64 rng(o.rng_seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<char> ended(N, 0);
    std::vector<Trace> tr(N);
    for (auto& q : tr) {
        q.progress.resize(size_t(task.progress_dim));
        if (task.progress_reset) task.progress_reset(q.progress.data());
    }
    std::map<std::string, int> kinds;
    int saved = 0;
    Result r;
    const auto t0 = Clock::now();
    forge::Env::HostCopy h;
    env->download(h);
    for (int t = 0; r.episodes < N && t < 2 * task.max_steps; ++t) {
        if (task.progress_step)
            for (int e = 0; e < N; ++e)  // trace from the obs before the step
                if (!ended[e]) task.progress_step(h.obs.data() + size_t(e) * D, tr[e].progress.data());
        if (model) {
            const int k = t % o.exec;
            const bool plan = k == 0, strip = trace && t % task.strip_every == 0, keep = record && t % kImageEvery == 0;
            if (plan || strip || keep) {
                rig->render(h, D);
                for (int e = 0; e < N; ++e)
                    for (int c = 0; c < cams; ++c) rig->camera().copy_rgb(e, info.cameras[c], img.data() + e * fb + c * one);
                if (keep) {  // all cameras of the envs still in their episode, compressed on all cores
                    for (int e = 0; e < N; ++e)
                        if (!ended[e])
                            for (int c = 0; c < task.cameras; ++c)
                                rig->camera().copy_rgb(e, c, all.data() + e * fb_all + c * one);
                    std::vector<std::thread> pool;
                    for (int w = 0; w < nthreads; ++w)
                        pool.emplace_back([&, w]() {
                            for (int e = w; e < N; e += nthreads)
                                if (!ended[e]) zs[e]->add(all.data() + e * fb_all, fb_all);
                        });
                    for (auto& th : pool) th.join();
                }
                if (strip)
                    for (int e = 0; e < strip_envs; ++e)
                        if (!ended[e]) tr[e].frames.insert(tr[e].frames.end(), img.data() + e * fb, img.data() + (e + 1) * fb);
            }
            if (plan) {
                for (int e = 0; e < N; ++e)
                    std::copy(h.obs.begin() + size_t(e) * D, h.obs.begin() + size_t(e) * D + kState,
                              st.begin() + size_t(e) * kState);
                if (o.noise == "fresh" || (o.noise == "episode" && t == 0))
                    for (auto& z : noise) z = gauss(rng);  // "zero": stays 0
                model->run(N, img.data(), st.data(), tx.data(), noise.data(), chunk.data());
            }
            for (int e = 0; e < N; ++e)
                std::copy(chunk.begin() + (size_t(e) * C + k) * kAct, chunk.begin() + (size_t(e) * C + k + 1) * kAct,
                          act.begin() + size_t(e) * A);
            if (record) {  // the expert's action in the policy's state: the label
                if (gpu) {
                    env->expert_actions(dact.data());
                    ex = dact.download();
                } else {
                    env->expert_actions(ex.data());
                }
                for (int e = 0; e < N; ++e) {
                    if (ended[e]) continue;
                    const float* ob = h.obs.data() + size_t(e) * D;
                    rec[e].obs.insert(rec[e].obs.end(), ob, ob + D);
                    rec[e].expert.insert(rec[e].expert.end(), ex.begin() + size_t(e) * A, ex.begin() + size_t(e + 1) * A);
                    rec[e].executed.insert(rec[e].executed.end(), act.begin() + size_t(e) * A,
                                           act.begin() + size_t(e + 1) * A);
                }
            }
            if (gpu) {
                dact.upload(act);
                env->step(dact.data());
            } else {
                env->step(act.data());
            }
        } else if (gpu) {
            env->expert_actions(dact.data());
            env->step(dact.data());
        } else {
            env->expert_actions(act.data());
            env->step(act.data());
        }
        env->download(h);
        for (int e = 0; e < N; ++e) {
            if (ended[e]) continue;
            if (record) rec[e].reward.push_back(h.reward[e]);
            if (!(h.terminated[e] || h.truncated[e])) continue;
            ended[e] = 1;
            ++r.episodes;
            const float* fo = h.final_obs.data() + size_t(e) * D;
            const bool ok = task.done_ok(fo, h.terminated[e]);
            if (record) {
                auto& ep = rec[e];
                ep.final_obs.assign(fo, fo + D);
                ep.success = ok;
                const std::vector<uint8_t> packed = zs[e]->finish();
                o.record->write(ep, &packed);
                ep = forge::DemoEpisode{};
                ++r.recorded;
            }
            if (ok) {
                ++r.done;
                r.steps_done += t + 1;
            }
            Trace& q = tr[e];
            if (!ok && model && task.fail_kind) {
                const char* kind = task.fail_kind(q.progress.data(), fo);
                ++kinds[kind];
                if (trace) {
                    const std::string detail =
                        task.fail_detail ? task.fail_detail(env->model(), q.progress.data(), fo) : std::string();
                    std::printf("  fail env %4d: %-38s %s\n", e, kind, detail.c_str());
                    if (saved < o.save_failures && e < strip_envs) {
                        vla::save_strip(o.prefix + "_fail_" + std::to_string(e) + ".png", q.frames, cams, size);
                        ++saved;
                    }
                }
            }
            q.frames.clear();
            q.frames.shrink_to_fit();
            if (task.progress_reset) task.progress_reset(q.progress.data());
        }
    }
    r.seconds = since(t0);
    for (const auto& [k, n] : kinds) std::printf("  %3d x %s\n", n, k.c_str());
    if (saved > 0) {
        std::printf("  saved %d camera strips %s_fail_<env>.png (rows: cameras", saved, o.prefix.c_str());
        for (int c : info.cameras) std::printf(" %d", c);
        std::printf("; one column per %d steps)\n", task.strip_every);
    }
    return r;
}

void print(const VisionTask& task, const char* who, const Result& r) {
    std::printf("vla_app: %s: done (%s) %d / %d (%.1f%%), mean %.0f steps  (%.1f s)\n", who, task.done_text.c_str(),
                r.done, r.episodes, 100.0 * r.done / std::max(1, r.episodes), r.steps_done / std::max(1, r.done),
                r.seconds);
}

}  // namespace

int run_vision(forge::Config& cfg, const VisionTask& task) {
    const std::string onnx = cfg.get_string("onnx", "vla.onnx");
    const size_t dot = onnx.rfind(".onnx");
    const std::string base = dot == std::string::npos ? onnx : onnx.substr(0, dot);
    const std::string info_path = cfg.get_string("info", base + "_openarm.txt");
    const int envs = cfg.get_int("eval_envs", 500), eval_seed = cfg.get_int("eval_seed", 1000);
    const std::string backend = cfg.get_string("eval_backend", "gpu"), device = cfg.get_string("device", "cpu");
    const int exec = cfg.get_int("exec", 4), seed = cfg.get_int("seed", 1);
    const bool eval_expert = cfg.get_bool("eval_expert", true);
    const int save_failures = cfg.get_int("save_failures", 20);
    const std::string noise_mode = cfg.get_string("noise", "fresh");
    const std::string record_path = cfg.get_string("record", "");
    for (const auto& k : cfg.unused_keys()) std::fprintf(stderr, "warning: unknown option '%s' (typo?)\n", k.c_str());
    if (backend != "gpu" && backend != "cpu") throw std::runtime_error("eval_backend must be gpu or cpu");
    if (device != "cpu" && device != "cuda") throw std::runtime_error("device must be cpu or cuda");
    if (noise_mode != "fresh" && noise_mode != "episode" && noise_mode != "zero")
        throw std::runtime_error("noise must be fresh, episode or zero");
    if (!task.make_rig) throw std::runtime_error("vla_app: task " + task.name + " has no cameras in this build");
    const Info info = vla::read_info(info_path, task.cameras, task.name.c_str());
    if (task.info_needs_task && info.task != task.name)
        throw std::runtime_error(info_path + ": not an " + task.name + " model (no 'task' line)");
    Model model(onnx, device == "cuda");
    if (model.state_dim != task.state || model.act != task.act)
        throw std::runtime_error("the model is not an " + task.name + " model (state " + std::to_string(task.state) +
                                 ", act " + std::to_string(task.act) + ")");
    if (exec < 1 || exec > model.chunk) throw std::runtime_error("exec must be 1 .. chunk");
    if (int(info.text.size()) != model.text_dim) throw std::runtime_error("text vector size does not match the model");
    std::printf("vla_app: %s (%d cameras %dx%d, text %d, chunk %d; runs %d of each chunk), ONNX Runtime on %s; %d "
                "episodes of %s (seed %d, %s env)\n",
                onnx.c_str(), model.cams, model.size, model.size, model.text_dim, model.chunk, exec, device.c_str(),
                envs, task.name.c_str(), eval_seed, backend.c_str());
    std::fflush(stdout);
    RunOptions ro;
    ro.envs = envs;
    ro.exec = exec;
    ro.seed = uint64_t(eval_seed);
    ro.rng_seed = uint64_t(seed);
    ro.backend = backend == "gpu" ? forge::EnvBackend::kGpu : forge::EnvBackend::kCpu;
    ro.noise = noise_mode;
    ro.save_failures = save_failures;
    ro.prefix = base;
    std::unique_ptr<forge::DemoWriter> writer;
    if (!record_path.empty()) {
        const forge::DemoImages im{task.cameras, model.size, model.size, 2};
        writer = std::make_unique<forge::DemoWriter>(record_path, task.name, task.obs, task.act, im);
        ro.record = writer.get();
    }
    const Result r = run(task, &model, info, ro);
    print(task, "policy", r);
    std::printf("vla_app: model %.1f ms per call (%d envs at once, %d calls)\n",
                1e3 * model.seconds / std::max(1, model.calls), envs, model.calls);
    if (writer)
        std::printf("vla_app: recorded %d episodes (the policy's states, the expert's actions, %d cameras) to %s\n",
                    r.recorded, task.cameras, record_path.c_str());
    if (eval_expert) {
        RunOptions xo = ro;
        xo.save_failures = 0;
        xo.record = nullptr;
        print(task, "expert", run(task, nullptr, info, xo));
    }
    return 0;
}
