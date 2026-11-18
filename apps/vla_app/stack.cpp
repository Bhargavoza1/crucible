// vla_app --task openarm_stack (Phase 11): the VLA from tools/vla_openarm.py in the
// two-arm OpenArm cell, closed loop. Each env runs one episode from its start (the red,
// green and blue cubes anywhere in reach, both arms at home): the cell's cameras render,
// the model turns the images it was trained on (head, left wrist, right wrist by
// default; <onnx without .onnx>_openarm.txt) + both arms' state (obs 0-47) + the
// instruction into an action chunk (10 actions per step: left arm 5, right arm 5), the
// env runs the first --exec actions of it, then the policy plans again.
//
//   ./vla_app --task openarm_stack --onnx runs/st_vla.onnx --eval_envs 500
//
// Success: collect_app's rule (the episode ended done; in the final obs red on the
// spot, green on red, blue on green, both TCPs up). --eval_expert true (default): the
// scripted expert on the same starts, for comparison. Failures (--save_failures K,
// default 20; 0 = off) are sorted by how far the tower got: never lifted a cube,
// lifted but red never on the spot, red placed but green never on it, red and green
// but blue never on top, the tower built then knocked down, the tower standing but
// the hands not home. Each is printed with the tower height at the end, red's offset
// from the spot and both TCPs' heights; the first K (of the first 256 envs) are saved
// as camera strips <onnx without .onnx>_fail_<env>.png (rows: the model's cameras; one
// column per 20 steps). Physics as configs/openarm_stack.txt (0.002 s x 10).
// Flow noise (--noise): fresh (default) | episode | zero.
// DAgger (--record FILE): every env's episode as a demo file (collect_app's format, all
// 4 cameras every 2 steps): the states the POLICY visited, labeled with the EXPERT's
// action in each (executed = the policy's). Use another --eval_seed than the one you
// test with. Then: vla_openarm.py demos.bin dagger1.bin --init st_vla.pt.
// Options: task, onnx, info, eval_envs, eval_seed, eval_backend, exec, device,
// eval_expert, seed, save_failures, noise, record.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "openarm_common.h"
#include "vla_model.h"
#include "../collect_app/stack_camera_rig.h"
#include "forge/env/demo_file.h"
#include "forge/env/env.h"
#include "forge/env/tasks/openarm_stack.h"
#include "forge/util/config.h"
#include "forge/util/device_buffer.h"

namespace {

using vla::Clock;
using vla::FrameStream;
using vla::Info;
using vla::Model;
using vla::since;
using Task = forge::OpenArmStackTask;

constexpr int kState = 48, kAct = Task::kActDim;  // obs 0-47: both arms' joints, velocities, TCPs, turns, openings
constexpr float kSpotX = 0.41f, kSpotY = 0.0f;
constexpr int kTcpL = 36, kTcpR = 42, kRed = 48, kGreen = 55, kBlue = 62;  // obs columns

bool on(const float* a, const float* u) {  // cube a on cube u (collect_app's rule)
    return std::hypot(a[0] - u[0], a[1] - u[1]) < 0.018f && std::fabs(a[2] - u[2] - 0.04f) < 0.005f;
}
// tower height from an obs: 0 none, 1 red on the spot, 2 + green on red, 3 + blue on green
int tower(const float* o) {
    const float *r = o + kRed, *g = o + kGreen, *b = o + kBlue;
    if (!(std::hypot(r[0] - kSpotX, r[1] - kSpotY) < 0.025f && r[2] < 0.025f)) return 0;
    if (!on(g, r)) return 1;
    return on(b, g) ? 3 : 2;
}
bool success(const float* fo, bool terminated) {
    return terminated && tower(fo) == 3 && fo[kTcpL + 2] > 0.12f && fo[kTcpR + 2] > 0.12f;
}

struct Trace {
    int max_tower = 0;
    float max_cube_z = 0.0f;
    std::vector<uint8_t> frames;  // [frame][cam][size][size][3]
};
constexpr int kFrameEvery = 20, kStripEnvs = 256;  // strips: 800-step episodes, so fewer frames and envs

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
Result run(Model* model, const Info& info, const RunOptions& o) {
    forge::EnvOptions eo;
    eo.task = Task::kName;
    eo.num_envs = o.envs;
    eo.seed = o.seed;
    eo.backend = o.backend;
    eo.physics_dt = 0.002f;  // configs/openarm_stack.txt
    eo.decimation = 10;
    eo.asset_dir = CRUCIBLE_ASSET_DIR;
    auto env = forge::make_env(eo);
    const int N = env->num_envs(), D = env->obs_dim(), A = env->act_dim();
    if (A != kAct || D != Task::kObsDim) throw std::runtime_error("expected openarm_stack (obs 79, act 10)");
    const bool gpu = o.backend == forge::EnvBackend::kGpu;
    const bool trace = model && o.save_failures > 0, record = model && o.record;
    constexpr int kImageEvery = 2;  // recorded frames: every 2 steps (as collect_app)
    const int cams = int(info.cameras.size()), size = model ? model->size : 0, C = model ? model->chunk : 1;
    std::unique_ptr<StackCameraRig> rig;
    if (model) {
        if (model->cams != cams) throw std::runtime_error("the model's camera count differs from the info file's");
        rig = std::make_unique<StackCameraRig>(env->model(), N, size, FLUX_SHADER_DIR,
                                               std::string(CRUCIBLE_ASSET_DIR) + "/robots/openarm/openarm_cell.cmesh");
    }
    const int E = int(info.text.size());
    const size_t one = size_t(size) * size * 3, fb = one * cams;
    std::vector<uint8_t> img(size_t(N) * fb);
    std::vector<float> st(size_t(N) * kState), tx(size_t(N) * E), noise(size_t(N) * C * kAct),
        chunk(size_t(N) * C * kAct), act(size_t(N) * A), ex(record ? size_t(N) * A : 0);
    const size_t fb_all = one * Task::kCameras;  // recorded: all the cell's cameras
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
    std::map<std::string, int> kinds;
    int saved = 0;
    Result r;
    const auto t0 = Clock::now();
    forge::Env::HostCopy h;
    env->download(h);
    for (int t = 0; r.episodes < N && t < 2 * Task::kMaxEpisodeSteps; ++t) {
        for (int e = 0; e < N; ++e) {  // trace from the obs before the step
            if (ended[e]) continue;
            const float* ob = h.obs.data() + size_t(e) * D;
            Trace& q = tr[e];
            q.max_tower = std::max(q.max_tower, tower(ob));
            for (int c : {kRed, kGreen, kBlue}) q.max_cube_z = std::max(q.max_cube_z, ob[c + 2]);
        }
        if (model) {
            const int k = t % o.exec;
            const bool plan = k == 0, strip = trace && t % kFrameEvery == 0, keep = record && t % kImageEvery == 0;
            if (plan || strip || keep) {
                rig->render(h.qpos.data());
                for (int e = 0; e < N; ++e)
                    for (int c = 0; c < cams; ++c) rig->camera().copy_rgb(e, info.cameras[c], img.data() + e * fb + c * one);
                if (keep) {  // all cameras of the envs still in their episode, compressed on all cores
                    for (int e = 0; e < N; ++e)
                        if (!ended[e])
                            for (int c = 0; c < Task::kCameras; ++c)
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
                    for (int e = 0; e < std::min(N, kStripEnvs); ++e)
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
            const bool ok = success(fo, h.terminated[e]);
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
            const Trace& q = tr[e];
            if (!ok && model) {
                const int end_tower = tower(fo);
                const bool lifted = q.max_cube_z >= Task::kHalf + 0.02f;
                const char* kind = end_tower == 3       ? "tower built, hands not home"
                                   : q.max_tower == 3   ? "tower built, then knocked down"
                                   : q.max_tower == 2   ? "red + green placed, blue never on top"
                                   : q.max_tower == 1   ? "red placed, green never on it"
                                   : lifted             ? "lifted a cube, red never on the spot"
                                                        : "never lifted a cube";
                ++kinds[kind];
                if (trace) {
                    std::printf("  fail env %4d: %-38s tower %d at the end (max %d), red - spot (%.1f, %.1f) cm, "
                                "TCPs %.1f / %.1f cm up\n",
                                e, kind, end_tower, q.max_tower, 100.0f * (fo[kRed] - kSpotX),
                                100.0f * (fo[kRed + 1] - kSpotY), 100.0f * fo[kTcpL + 2], 100.0f * fo[kTcpR + 2]);
                    if (saved < o.save_failures && e < kStripEnvs) {
                        vla::save_strip(o.prefix + "_fail_" + std::to_string(e) + ".png", q.frames, cams, size);
                        ++saved;
                    }
                }
            }
            tr[e] = Trace{};
        }
    }
    r.seconds = since(t0);
    for (const auto& [k, n] : kinds) std::printf("  %3d x %s\n", n, k.c_str());
    if (saved > 0) {
        std::printf("  saved %d camera strips %s_fail_<env>.png (rows: cameras", saved, o.prefix.c_str());
        for (int c : info.cameras) std::printf(" %d", c);
        std::printf("; one column per %d steps)\n", kFrameEvery);
    }
    return r;
}

void print(const char* who, const Result& r) {
    std::printf("vla_app: %s: done (tower built, hands home) %d / %d (%.1f%%), mean %.0f steps  (%.1f s)\n", who,
                r.done, r.episodes, 100.0 * r.done / std::max(1, r.episodes), r.steps_done / std::max(1, r.done),
                r.seconds);
}

}  // namespace

int run_stack(forge::Config& cfg) {
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
    const Info info = vla::read_info(info_path, Task::kCameras, Task::kName);
    if (info.task != Task::kName) throw std::runtime_error(info_path + ": not an openarm_stack model (no 'task' line)");
    Model model(onnx, device == "cuda");
    if (model.state_dim != kState || model.act != kAct)
        throw std::runtime_error("the model is not an openarm_stack model (state 48, act 10)");
    if (exec < 1 || exec > model.chunk) throw std::runtime_error("exec must be 1 .. chunk");
    if (int(info.text.size()) != model.text_dim) throw std::runtime_error("text vector size does not match the model");
    std::printf("vla_app: %s (%d cameras %dx%d, text %d, chunk %d; runs %d of each chunk), ONNX Runtime on %s; %d "
                "episodes of openarm_stack (seed %d, %s env)\n",
                onnx.c_str(), model.cams, model.size, model.size, model.text_dim, model.chunk, exec, device.c_str(),
                envs, eval_seed, backend.c_str());
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
        const forge::DemoImages im{Task::kCameras, model.size, model.size, 2};
        writer = std::make_unique<forge::DemoWriter>(record_path, Task::kName, Task::kObsDim, kAct, im);
        ro.record = writer.get();
    }
    const Result r = run(&model, info, ro);
    print("policy", r);
    std::printf("vla_app: model %.1f ms per call (%d envs at once, %d calls)\n",
                1e3 * model.seconds / std::max(1, model.calls), envs, model.calls);
    if (writer)
        std::printf("vla_app: recorded %d episodes (the policy's states, the expert's actions, %d cameras) to %s\n",
                    r.recorded, Task::kCameras, record_path.c_str());
    if (eval_expert) {
        RunOptions xo = ro;
        xo.save_failures = 0;
        xo.record = nullptr;
        print("expert", run(nullptr, info, xo));
    }
    return 0;
}
