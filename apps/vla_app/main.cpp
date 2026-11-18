// vla_app (Phase 9e): the small VLA from tools/vla_train.py drives panda_pick_color in
// our engine (--task openarm_pick_place: tools/vla_openarm.py's VLA in the OpenArm
// Cell, Phase 10: see openarm.cpp; --task openarm_stack: both arms build a 3-cube
// tower, Phase 11: see stack.cpp). Each env gets an instruction ("get the red cube"), the cameras render,
// the ONNX model (ONNX Runtime) turns images + robot state + instruction into an
// action chunk, the env runs the first --exec actions of it, then the policy plans
// again. We count which cube ends up lifted: the named one, another one, or none.
//
//   ./vla_app --onnx vla.onnx --eval_envs 500
//   ./vla_app --onnx vla.onnx --eval_envs 500 --instruct other    # language test
//
// --instruct target (default): the instruction names the task's target color (what
// the expert picks). --instruct other: it names one of the OTHER two cubes, picked
// at random: a policy that reads the instruction must now lift that cube instead; a
// policy that ignores it keeps lifting the task's target ("picked the task target").
// The text vectors come from <onnx without .onnx>_text.txt (vla_train.py --onnx).
// Lifted: the cube's center more than 8 cm up at the end (collect_app's rule).
// --eval_expert true (default, with --instruct target): the scripted expert on the
// same starts, for comparison.
// Failures (--save_failures K, default 20; 0 = off): each failed episode is printed
// with where it stopped (never reached the named cube, reached but never closed,
// closed on nothing, holding but not lifted, lifted then dropped, wrong cube), the
// closest the pads got, the first close step, and at its last step the cube offset and
// the policy's action next to the expert's (when the named cube is the expert's
// target); the first K are saved as camera strips <onnx without .onnx>_fail_<env>.png.
// DAgger (--record FILE): writes every episode as a demo file (collect_app's format):
// the states the POLICY visited, with camera images every 2 steps, and as labels the
// EXPERT's action for each of those states (executed = the policy's). Train on it
// with vla_train.py together with the demos (--init from the current model). Needs
// --instruct target.
// Flow noise (--noise): fresh (default): new random noise at every plan, so two
// plans may pick different answers where two are good (turn left / right);
// episode: one noise per env for the whole episode, so similar states get the same
// answer and the robot keeps one choice; zero: noise 0 (the flow's middle path).
// Options: noise, onnx, text, eval_envs, eval_seed, eval_backend, exec, device (cpu |
// cuda: ONNX Runtime's CUDA provider, needs the GPU build of ONNX Runtime), instruct,
// eval_expert, seed (noise and instruction choice), save_failures, record.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "vla_model.h"
#include "../collect_app/camera_rig.h"
#include "forge/env/demo_file.h"
#include "forge/env/env.h"
#include "forge/util/config.h"
#include "forge/util/device_buffer.h"
#include "forge/util/png.h"

// openarm.cpp: vla_app --task openarm_pick_place; stack.cpp: --task openarm_stack
int run_openarm(forge::Config& cfg);
int run_stack(forge::Config& cfg);

namespace {

using vla::Clock;
using vla::kAct;
using vla::kState;
using vla::Model;
using vla::since;

constexpr int kColors = 3;
const char* kColorNames[kColors] = {"red", "green", "yellow"};
constexpr int kEpisodeSteps = 250;

// panda_pick_color obs: target color one-hot 39-41, other cubes' z 44 / 47, their color ids 48 / 49
int target_color(const float* o) {
    int t = 0;
    for (int k = 1; k < kColors; ++k)
        if (o[39 + k] > o[39 + t]) t = k;
    return t;
}

// Which color ended up lifted (-1: none). Several lifted: the highest.
int lifted_color(const float* fo) {
    const float z[3] = {fo[26], fo[44], fo[47]};
    const int col[3] = {target_color(fo), int(fo[48] + 0.5f), int(fo[49] + 0.5f)};
    int best = -1;
    float bz = 0.025f + 0.08f;
    for (int k = 0; k < 3; ++k)
        if (z[k] > bz) bz = z[k], best = col[k];
    return best;
}

std::map<std::string, std::vector<float>> read_text_vectors(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path + " (written by vla_train.py --onnx)");
    std::map<std::string, std::vector<float>> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string name;
        ss >> name;
        float v;
        while (ss >> v) out[name].push_back(v);
    }
    for (const char* c : kColorNames)
        if (!out.count(c)) throw std::runtime_error(path + ": no vector for " + std::string(c));
    return out;
}

struct Result {
    int episodes = 0, named = 0, task_target = 0, other = 0, none = 0;
    int recorded = 0;
    double seconds = 0;
};

// Positions of the three cubes by color, and the finger-pad center, from the obs.
struct Scene {
    forge::Vec3 cube[kColors];
    forge::Vec3 pad;
    float z_of(int color) const { return cube[color].z; }
};
Scene scene(const float* o) {
    Scene s;
    s.cube[target_color(o)] = {o[24], o[25], o[26]};
    s.cube[int(o[48] + 0.5f)] = {o[42], o[43], o[44]};
    s.cube[int(o[49] + 0.5f)] = {o[45], o[46], o[47]};
    s.pad = {o[24] - o[31], o[25] - o[32], o[26] - o[33]};
    return s;
}

// What a failed episode did (first episode of each env).
struct Trace {
    float min_pad_dist = 1e9f;  // closest the pad center came to the named cube (m)
    int first_close = -1, grip_switches = 0;
    float max_named_z = 0.0f, prev_grip = -1.0f;
    float policy_a[kAct] = {}, expert_a[kAct] = {};
    forge::Vec3 offset{};  // named cube - pad center at the last step
    std::vector<uint8_t> frames;  // camera strip: [frame][cam][size][size][3]
};
constexpr int kFrameEvery = 10;

const char* failure_kind(const Trace& t, float final_open, float final_named_z) {
    if (t.min_pad_dist > 0.02f) return "never reached the cube";
    if (t.first_close < 0) return "reached, never closed";
    if (t.max_named_z - 0.025f > 0.08f) return "lifted, then dropped";
    if (final_open <= 0.03f) return "closed on nothing";
    if (final_open < 0.065f && final_named_z - 0.025f < 0.02f) return "holding, not lifted";
    return "closed, cube not held";
}

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
    bool instruct_other = false;
    std::string noise = "fresh";  // fresh | episode | zero (see the top of this file)
    std::string prefix;  // failure strips: <prefix>_fail_<env>.png
    forge::DemoWriter* record = nullptr;  // DAgger: the policy's states with the expert's actions
};

// Every env runs one episode. model null = the scripted expert (instruct ignored).
Result run(Model* model, const std::map<std::string, std::vector<float>>& text, const RunOptions& o) {
    forge::EnvOptions eo;
    eo.task = "panda_pick_color";
    eo.num_envs = o.envs;
    eo.seed = o.seed;
    eo.backend = o.backend;
    eo.asset_dir = CRUCIBLE_ASSET_DIR;
    auto env = forge::make_env(eo);
    const int N = env->num_envs(), D = env->obs_dim(), A = env->act_dim();
    if (A != kAct || D < 50) throw std::runtime_error("expected panda_pick_color (obs 50, act 5)");
    const bool gpu = o.backend == forge::EnvBackend::kGpu;
    const bool trace = model && o.save_failures > 0, record = model && o.record;
    const bool label = trace || record;  // the expert's action in the policy's state
    std::unique_ptr<PickCameraRig> rig;
    if (model) {
        rig = std::make_unique<PickCameraRig>(env->model(), N, model->size, FLUX_SHADER_DIR);
        if (PickCameraRig::kCameras != model->cams) throw std::runtime_error("model and task cameras differ");
    }
    const int C = model ? model->chunk : 1, E = model ? model->text_dim : 0, cams = PickCameraRig::kCameras;
    const int size = model ? model->size : 0, image_every = 2;  // recorded images: every 2 steps (as collect_app)
    const size_t one = size_t(size) * size * 3, fb = one * cams;
    std::vector<uint8_t> img(size_t(N) * fb);
    std::vector<float> st(size_t(N) * kState), tx(size_t(N) * E), noise(size_t(N) * C * kAct),
        chunk(size_t(N) * C * kAct), act(size_t(N) * A), ex(size_t(N) * A);
    forge::DeviceBuffer<float> dact;
    if (gpu) dact.resize(size_t(N) * A);
    std::mt19937_64 rng(o.rng_seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<int> instructed(N, -1), task_col(N, -1);
    std::vector<char> done(N, 0);
    std::vector<Trace> tr(trace ? N : 0);
    std::vector<forge::DemoEpisode> rec(record ? N : 0);
    std::map<std::string, int> kinds;
    int saved = 0;
    Result r;
    const auto t0 = Clock::now();
    forge::Env::HostCopy h;
    env->download(h);
    for (int e = 0; e < N; ++e) {  // the instruction of each env's first episode
        task_col[e] = target_color(h.obs.data() + size_t(e) * D);
        instructed[e] = task_col[e];
        if (o.instruct_other) instructed[e] = (task_col[e] + 1 + int(rng() % 2)) % kColors;
        if (model) {
            const auto& v = text.at(kColorNames[instructed[e]]);
            std::copy(v.begin(), v.end(), tx.begin() + size_t(e) * E);
        }
    }
    for (int t = 0; r.episodes < N && t < 4 * kEpisodeSteps; ++t) {
        if (model) {
            const int k = t % o.exec;
            const bool plan = k == 0, keep_frame = record && t % image_every == 0;
            const bool strip_frame = trace && t % kFrameEvery == 0;
            if (plan || keep_frame || strip_frame) {
                rig->render(h.qpos.data(), h.obs.data(), D);
                for (int e = 0; e < N; ++e)
                    for (int c = 0; c < cams; ++c) rig->camera().copy_rgb(e, c, img.data() + e * fb + c * one);
                for (int e = 0; e < N; ++e) {
                    if (done[e]) continue;
                    if (keep_frame) rec[e].images.insert(rec[e].images.end(), img.data() + e * fb, img.data() + (e + 1) * fb);
                    if (strip_frame) tr[e].frames.insert(tr[e].frames.end(), img.data() + e * fb, img.data() + (e + 1) * fb);
                }
            }
            if (plan) {  // run the model for every env
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
            if (label) {
                if (gpu) {
                    env->expert_actions(dact.data());
                    ex = dact.download();
                } else {
                    env->expert_actions(ex.data());
                }
            }
            for (int e = 0; e < N; ++e) {
                if (done[e]) continue;
                const float* ob = h.obs.data() + size_t(e) * D;
                if (record) {
                    auto& ep = rec[e];
                    ep.obs.insert(ep.obs.end(), ob, ob + D);
                    ep.expert.insert(ep.expert.end(), ex.begin() + size_t(e) * A, ex.begin() + size_t(e + 1) * A);
                    ep.executed.insert(ep.executed.end(), act.begin() + size_t(e) * A, act.begin() + size_t(e + 1) * A);
                }
                if (trace) {
                    Trace& q = tr[e];
                    const Scene sc = scene(ob);
                    const forge::Vec3 d = sc.cube[instructed[e]] - sc.pad;
                    q.min_pad_dist = std::min(q.min_pad_dist, forge::length(d));
                    const float g = act[size_t(e) * A + 4];
                    if (g > 0.0f && q.first_close < 0) q.first_close = t;
                    q.grip_switches += (g > 0.0f) != (q.prev_grip > 0.0f);
                    q.prev_grip = g;
                    q.max_named_z = std::max(q.max_named_z, sc.z_of(instructed[e]));
                    for (int j = 0; j < A; ++j) q.policy_a[j] = act[size_t(e) * A + j], q.expert_a[j] = ex[size_t(e) * A + j];
                    q.offset = d;
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
        std::vector<int> ended;
        for (int e = 0; e < N; ++e) {
            if (done[e]) continue;
            if (record) rec[e].reward.push_back(h.reward[e]);
            if (!(h.terminated[e] || h.truncated[e])) continue;
            done[e] = 1;
            ended.push_back(e);
            ++r.episodes;
            const float* fo = h.final_obs.data() + size_t(e) * D;
            const int got = lifted_color(fo);
            const int want = model ? instructed[e] : task_col[e];
            if (got < 0)
                ++r.none;
            else if (got == want)
                ++r.named;
            else if (got == task_col[e])
                ++r.task_target;
            else
                ++r.other;
            if (trace && got != want) {
                const Trace& q = tr[e];
                const Scene sc = scene(fo);
                const char* kind = got >= 0 ? "lifted the wrong cube" : failure_kind(q, fo[23], sc.z_of(instructed[e]));
                ++kinds[kind];
                std::printf("  fail env %4d (%s): %-24s closest pads-cube %.1f cm, first close step %d, gripper "
                            "switches %d, cube max %.1f cm, fingers %.1f cm apart at the end\n",
                            e, kColorNames[instructed[e]], kind, 100.0f * q.min_pad_dist, q.first_close,
                            q.grip_switches, 100.0f * (q.max_named_z - 0.025f), 100.0f * fo[23]);
                std::printf("      last step: cube - pads (%.1f, %.1f, %.1f) cm; action x y z turn grip: policy "
                            "(%.2f %.2f %.2f %.2f %.2f)",
                            100.0f * q.offset.x, 100.0f * q.offset.y, 100.0f * q.offset.z, q.policy_a[0],
                            q.policy_a[1], q.policy_a[2], q.policy_a[3], q.policy_a[4]);
                if (instructed[e] == task_col[e])  // the expert goes for the task's target only
                    std::printf(", expert (%.2f %.2f %.2f %.2f %.2f)", q.expert_a[0], q.expert_a[1], q.expert_a[2],
                                q.expert_a[3], q.expert_a[4]);
                std::printf("\n");
                if (saved < o.save_failures) {
                    save_strip(o.prefix + "_fail_" + std::to_string(e) + ".png", q.frames, cams, size);
                    ++saved;
                }
            }
            if (trace) tr[e] = Trace{};
        }
        if (record && !ended.empty()) {  // compress on all cores (envs end together), write in env order
            std::vector<std::vector<uint8_t>> packed(ended.size());
            const int nt = std::max(1, std::min(int(ended.size()), int(std::thread::hardware_concurrency())));
            std::vector<std::thread> pool;
            for (int w = 0; w < nt; ++w)
                pool.emplace_back([&, w]() {
                    for (size_t k = size_t(w); k < ended.size(); k += size_t(nt))
                        packed[k] = forge::DemoWriter::compress_images(rec[ended[k]].images);
                });
            for (auto& th : pool) th.join();
            for (size_t k = 0; k < ended.size(); ++k) {
                auto& ep = rec[ended[k]];
                ep.final_obs.assign(h.final_obs.begin() + size_t(ended[k]) * D,
                                    h.final_obs.begin() + size_t(ended[k] + 1) * D);
                ep.success = lifted_color(ep.final_obs.data()) == task_col[ended[k]];
                o.record->write(ep, &packed[k]);
                ep = forge::DemoEpisode{};
                ++r.recorded;
            }
        }
    }
    r.seconds = since(t0);
    for (const auto& [k, n] : kinds) std::printf("  %3d x %s\n", n, k.c_str());
    if (saved > 0)
        std::printf("  saved %d camera strips %s_fail_<env>.png (rows: front, wrist camera; one column per %d steps)\n",
                    saved, o.prefix.c_str(), kFrameEvery);
    return r;
}

void print(const char* who, const Result& r, bool instruct_other) {
    auto pct = [&](int n) { return 100.0 * n / std::max(1, r.episodes); };
    std::printf("vla_app: %s: lifted the named cube %d / %d (%.1f%%)", who, r.named, r.episodes, pct(r.named));
    if (instruct_other) std::printf(", the task's target instead %.1f%%", pct(r.task_target));
    std::printf(", another cube %.1f%%, nothing %.1f%%  (%.1f s)\n", pct(r.other), pct(r.none), r.seconds);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        forge::Config cfg = forge::Config::from_cli(argc, argv);
        const std::string task = cfg.get_string("task", "panda_pick_color");
        if (task == "openarm_pick_place") return run_openarm(cfg);
        if (task == "openarm_stack") return run_stack(cfg);
        if (task != "panda_pick_color")
            throw std::runtime_error("task must be panda_pick_color, openarm_pick_place or openarm_stack");
        const std::string onnx = cfg.get_string("onnx", "vla.onnx");
        std::string text_path = cfg.get_string("text", "");
        const int envs = cfg.get_int("eval_envs", 500), eval_seed = cfg.get_int("eval_seed", 1000);
        const std::string backend = cfg.get_string("eval_backend", "gpu"), device = cfg.get_string("device", "cpu");
        const int exec = cfg.get_int("exec", 4), seed = cfg.get_int("seed", 1);
        const std::string instruct = cfg.get_string("instruct", "target");
        const bool eval_expert = cfg.get_bool("eval_expert", true);
        const int save_failures = cfg.get_int("save_failures", 20);
        const std::string record_path = cfg.get_string("record", "");
        const std::string noise_mode = cfg.get_string("noise", "fresh");
        for (const auto& k : cfg.unused_keys()) std::fprintf(stderr, "warning: unknown option '%s' (typo?)\n", k.c_str());
        if (backend != "gpu" && backend != "cpu") throw std::runtime_error("eval_backend must be gpu or cpu");
        if (device != "cpu" && device != "cuda") throw std::runtime_error("device must be cpu or cuda");
        if (instruct != "target" && instruct != "other") throw std::runtime_error("instruct must be target or other");
        if (noise_mode != "fresh" && noise_mode != "episode" && noise_mode != "zero")
            throw std::runtime_error("noise must be fresh, episode or zero");
        if (!record_path.empty() && instruct != "target")
            throw std::runtime_error("--record needs --instruct target (the expert only goes for the task's target)");
        if (text_path.empty()) {
            const size_t dot = onnx.rfind(".onnx");
            text_path = (dot == std::string::npos ? onnx : onnx.substr(0, dot)) + "_text.txt";
        }
        const auto text = read_text_vectors(text_path);
        Model model(onnx, device == "cuda");
        if (model.state_dim != kState || model.act != kAct)
            throw std::runtime_error("the model is not a panda_pick_color model (state 24, act 5)");
        if (exec < 1 || exec > model.chunk) throw std::runtime_error("exec must be 1 .. chunk");
        if (int(text.at("red").size()) != model.text_dim) throw std::runtime_error("text vector size does not match");
        std::printf("vla_app: %s (%d cameras %dx%d, text %d, chunk %d; runs %d of each chunk), ONNX Runtime on %s; "
                    "%d episodes of panda_pick_color (seed %d, %s env), instruction names %s\n",
                    onnx.c_str(), model.cams, model.size, model.size, model.text_dim, model.chunk, exec, device.c_str(), envs,
                    eval_seed, backend.c_str(), instruct == "target" ? "the task's target" : "another cube");
        std::fflush(stdout);
        const size_t dot = onnx.rfind(".onnx");
        RunOptions ro;
        ro.envs = envs;
        ro.exec = exec;
        ro.seed = uint64_t(eval_seed);
        ro.rng_seed = uint64_t(seed);
        ro.backend = backend == "gpu" ? forge::EnvBackend::kGpu : forge::EnvBackend::kCpu;
        ro.instruct_other = instruct == "other";
        ro.noise = noise_mode;
        ro.save_failures = save_failures;
        ro.prefix = dot == std::string::npos ? onnx : onnx.substr(0, dot);
        std::unique_ptr<forge::DemoWriter> writer;
        if (!record_path.empty()) {
            const forge::DemoImages im{PickCameraRig::kCameras, model.size, model.size, 2};
            writer = std::make_unique<forge::DemoWriter>(record_path, "panda_pick_color", 50, kAct, im);
            ro.record = writer.get();
        }
        const Result r = run(&model, text, ro);
        print("policy", r, ro.instruct_other);
        std::printf("vla_app: model %.1f ms per call (%d envs at once, %d calls)\n",
                    1e3 * model.seconds / std::max(1, model.calls), envs, model.calls);
        if (writer)
            std::printf("vla_app: recorded %d episodes (the policy's states, the expert's actions) to %s\n", r.recorded,
                        record_path.c_str());
        if (eval_expert && !ro.instruct_other) {
            RunOptions xo = ro;
            xo.save_failures = 0;
            xo.record = nullptr;
            print("expert", run(nullptr, text, xo), false);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
