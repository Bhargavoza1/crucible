// vla_app --task openarm_pick_place (Phase 10): the VLA from tools/vla_openarm.py in the
// OpenArm Cell, closed loop. Each env runs one episode from its start (cube and tray
// somewhere in reach, the arm at home): the cell's cameras render, the model turns the
// images it was trained on (head and left wrist; <onnx without .onnx>_openarm.txt) +
// the arm's state + the instruction into an action chunk, the env runs the first
// --exec actions of it, then the policy plans again.
//
//   ./vla_app --task openarm_pick_place --onnx runs/oa_vla.onnx --eval_envs 500
//
// Success: the episode ended done (the cube in the tray, the fingers open, the arm
// back home; collect_app's rule). --eval_expert true (default): the scripted expert on
// the same starts, for comparison. Failures (--save_failures K, default 20; 0 = off):
// printed with where they stopped (never reached the cube, reached but never lifted
// it, lifted but never put it in the tray, put it in then moved it out, placed but not
// home), the closest the finger pads got to the cube, the highest the cube got and,
// at the end, the cube's offset from the tray and the TCP's from home; the first K
// are saved as camera strips <onnx without .onnx>_fail_<env>.png (rows: the model's
// cameras; one column per 10 steps). Physics as configs/openarm_pick_place.txt (0.002
// s x 10). Flow noise (--noise): fresh (default) | episode | zero, as for panda.
// Options: task, onnx, info, eval_envs, eval_seed, eval_backend, exec, device, eval_expert,
// seed, save_failures, noise.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "vla_model.h"
#include "../collect_app/openarm_camera_rig.h"
#include "forge/env/env.h"
#include "forge/env/tasks/openarm_pick_place.h"
#include "forge/util/config.h"
#include "forge/util/device_buffer.h"
#include "forge/util/png.h"

namespace {

using vla::Clock;
using vla::kAct;
using vla::kState;
using vla::Model;
using vla::since;
using Task = forge::OpenArmPickPlaceTask;

// <onnx>_openarm.txt (vla_openarm.py --onnx): "cameras 0 1" and "text v1 v2 ..."
struct Info {
    std::vector<int> cameras;
    std::vector<float> text;
};
Info read_info(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path + " (written by vla_openarm.py --onnx)");
    Info info;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "cameras") {
            int c;
            while (ss >> c) info.cameras.push_back(c);
        } else if (key == "text") {
            float v;
            while (ss >> v) info.text.push_back(v);
        }
    }
    if (info.cameras.empty() || info.text.empty()) throw std::runtime_error(path + ": needs 'cameras' and 'text' lines");
    for (int c : info.cameras)
        if (c < 0 || c >= Task::kCameras) throw std::runtime_error(path + ": camera ids are 0-3");
    return info;
}

// obs: TCP 18-20, cube 24-26, cube - TCP 31-33, cube - tray center 36-37
bool placed(const float* o) { return std::fabs(o[36]) < 0.03f && std::fabs(o[37]) < 0.03f && o[26] < 0.025f; }

struct Trace {
    float min_dist = 1e9f, max_cube_z = 0.0f;
    bool ever_placed = false;
    std::vector<uint8_t> frames;  // [frame][cam][size][size][3]
};
constexpr int kFrameEvery = 10;

struct Result {
    int episodes = 0, done = 0;
    double seconds = 0, steps_done = 0;
};

void save_strip(const std::string& path, const std::vector<uint8_t>& frames, int cams, int size) {
    const size_t one = size_t(size) * size * 3, frame = one * cams;
    const int nf = int(frames.size() / frame);
    if (nf == 0) return;
    const int W = nf * size, H = cams * size;
    std::vector<uint8_t> img(size_t(W) * H * 3);
    for (int f = 0; f < nf; ++f)
        for (int c = 0; c < cams; ++c)
            for (int y = 0; y < size; ++y)
                std::copy_n(frames.data() + f * frame + c * one + size_t(y) * size * 3, size_t(size) * 3,
                            img.data() + ((size_t(c) * size + y) * W + size_t(f) * size) * 3);
    forge::write_png(path, img.data(), W, H);
}

struct RunOptions {
    int envs = 500, exec = 4, save_failures = 0;
    uint64_t seed = 1000, rng_seed = 1;
    forge::EnvBackend backend = forge::EnvBackend::kGpu;
    std::string noise = "fresh", prefix;
};

// Every env runs one episode. model null = the scripted expert.
Result run(Model* model, const Info& info, const RunOptions& o) {
    forge::EnvOptions eo;
    eo.task = Task::kName;
    eo.num_envs = o.envs;
    eo.seed = o.seed;
    eo.backend = o.backend;
    eo.physics_dt = 0.002f;  // configs/openarm_pick_place.txt
    eo.decimation = 10;
    eo.asset_dir = CRUCIBLE_ASSET_DIR;
    auto env = forge::make_env(eo);
    const int N = env->num_envs(), D = env->obs_dim(), A = env->act_dim();
    if (A != kAct || D != Task::kObsDim) throw std::runtime_error("expected openarm_pick_place (obs 43, act 5)");
    const auto p = Task::make_params(env->model());
    const forge::Vec3 home{p.home_tcp[0], p.home_tcp[1], p.home_tcp[2]};
    const bool gpu = o.backend == forge::EnvBackend::kGpu;
    const bool trace = model && o.save_failures > 0;
    const int cams = int(info.cameras.size()), size = model ? model->size : 0, C = model ? model->chunk : 1;
    std::unique_ptr<OpenArmCameraRig> rig;
    if (model) {
        if (model->cams != cams) throw std::runtime_error("the model's camera count differs from the info file's");
        rig = std::make_unique<OpenArmCameraRig>(env->model(), N, size, FLUX_SHADER_DIR,
                                                 std::string(CRUCIBLE_ASSET_DIR) + "/robots/openarm/openarm_cell.cmesh");
    }
    const int E = int(info.text.size());
    const size_t one = size_t(size) * size * 3, fb = one * cams;
    std::vector<uint8_t> img(size_t(N) * fb);
    std::vector<float> st(size_t(N) * kState), tx(size_t(N) * E), noise(size_t(N) * C * kAct),
        chunk(size_t(N) * C * kAct), act(size_t(N) * A);
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
            q.min_dist = std::min(q.min_dist, std::sqrt(ob[31] * ob[31] + ob[32] * ob[32] + ob[33] * ob[33]));
            q.max_cube_z = std::max(q.max_cube_z, ob[26]);
            q.ever_placed = q.ever_placed || placed(ob);
        }
        if (model) {
            const int k = t % o.exec;
            const bool plan = k == 0, strip = trace && t % kFrameEvery == 0;
            if (plan || strip) {
                rig->render(h.qpos.data(), h.obs.data(), D);
                for (int e = 0; e < N; ++e)
                    for (int c = 0; c < cams; ++c) rig->camera().copy_rgb(e, info.cameras[c], img.data() + e * fb + c * one);
                if (strip)
                    for (int e = 0; e < N; ++e)
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
            if (ended[e] || !(h.terminated[e] || h.truncated[e])) continue;
            ended[e] = 1;
            ++r.episodes;
            const float* fo = h.final_obs.data() + size_t(e) * D;
            const bool ok = h.terminated[e] && placed(fo) && fo[20] > 0.08f;
            if (ok) {
                ++r.done;
                r.steps_done += t + 1;
            }
            const Trace& q = tr[e];
            if (!ok && model) {
                const char* kind = q.min_dist > 0.02f            ? "never reached the cube"
                                   : q.max_cube_z < Task::kHalf + 0.02f ? "reached, never lifted it"
                                   : !q.ever_placed              ? "lifted, never put it in the tray"
                                   : !placed(fo)                 ? "put it in the tray, then moved it out"
                                                                 : "placed, did not go home";
                ++kinds[kind];
                if (trace) {
                    const forge::Vec3 tcp{fo[18], fo[19], fo[20]};
                    std::printf("  fail env %4d: %-38s closest pads-cube %.1f cm, cube max %.1f cm up; at the end "
                                "cube - tray (%.1f, %.1f) cm, cube %.1f cm up, TCP %.1f cm from home\n",
                                e, kind, 100.0f * q.min_dist, 100.0f * (q.max_cube_z - Task::kHalf), 100.0f * fo[36],
                                100.0f * fo[37], 100.0f * (fo[26] - Task::kHalf), 100.0f * forge::length(tcp - home));
                    if (saved < o.save_failures) {
                        save_strip(o.prefix + "_fail_" + std::to_string(e) + ".png", q.frames, cams, size);
                        ++saved;
                    }
                }
            }
            tr[e] = Trace{};
        }
    }
    r.seconds = since(t0);
    for (const auto& [k, n] : kinds) std::printf("  %3d x %s\n", n, k.c_str());
    if (saved > 0)
        std::printf("  saved %d camera strips %s_fail_<env>.png (rows: cameras", saved, o.prefix.c_str());
    if (saved > 0) {
        for (int c : info.cameras) std::printf(" %d", c);
        std::printf("; one column per %d steps)\n", kFrameEvery);
    }
    return r;
}

void print(const char* who, const Result& r) {
    std::printf("vla_app: %s: done (cube in the tray, arm home) %d / %d (%.1f%%), mean %.0f steps  (%.1f s)\n", who,
                r.done, r.episodes, 100.0 * r.done / std::max(1, r.episodes), r.steps_done / std::max(1, r.done),
                r.seconds);
}

}  // namespace

int run_openarm(forge::Config& cfg) {
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
    for (const auto& k : cfg.unused_keys()) std::fprintf(stderr, "warning: unknown option '%s' (typo?)\n", k.c_str());
    if (backend != "gpu" && backend != "cpu") throw std::runtime_error("eval_backend must be gpu or cpu");
    if (device != "cpu" && device != "cuda") throw std::runtime_error("device must be cpu or cuda");
    if (noise_mode != "fresh" && noise_mode != "episode" && noise_mode != "zero")
        throw std::runtime_error("noise must be fresh, episode or zero");
    const Info info = read_info(info_path);
    Model model(onnx, device == "cuda");
    if (exec < 1 || exec > model.chunk) throw std::runtime_error("exec must be 1 .. chunk");
    if (int(info.text.size()) != model.text_dim) throw std::runtime_error("text vector size does not match the model");
    std::printf("vla_app: %s (%d cameras %dx%d, text %d, chunk %d; runs %d of each chunk), ONNX Runtime on %s; %d "
                "episodes of openarm_pick_place (seed %d, %s env)\n",
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
    print("policy", run(&model, info, ro));
    std::printf("vla_app: model %.1f ms per call (%d envs at once, %d calls)\n",
                1e3 * model.seconds / std::max(1, model.calls), envs, model.calls);
    if (eval_expert) {
        RunOptions xo = ro;
        xo.save_failures = 0;
        print("expert", run(nullptr, info, xo));
    }
    return 0;
}
