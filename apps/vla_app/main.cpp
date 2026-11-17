// vla_app (Phase 9e): the small VLA from tools/vla_train.py drives panda_pick_color in
// our engine. Each env gets an instruction ("get the red cube"), the cameras render,
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
// Options: onnx, text, eval_envs, eval_seed, eval_backend, exec, device (cpu |
// cuda: ONNX Runtime's CUDA provider, needs the GPU build of ONNX Runtime), instruct,
// eval_expert, seed (noise and instruction choice).
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
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "../collect_app/camera_rig.h"
#include "forge/env/env.h"
#include "forge/util/config.h"
#include "forge/util/device_buffer.h"

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

constexpr int kAct = 5, kState = 24, kColors = 3;
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

struct Model {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "vla_app"};
    std::unique_ptr<Ort::Session> session;
    int cams = 0, size = 0, text_dim = 0, chunk = 0;
    double seconds = 0;
    int calls = 0;

    Model(const std::string& path, bool cuda) {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(0);
        if (cuda) {
            OrtCUDAProviderOptions o{};
            so.AppendExecutionProvider_CUDA(o);
        }
        session = std::make_unique<Ort::Session>(env, path.c_str(), so);
        auto in_shape = [&](size_t i) { return session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape(); };
        const auto img = in_shape(0), txt = in_shape(2);
        const auto out = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (img.size() != 5 || txt.size() != 2 || out.size() != 3) throw std::runtime_error("unexpected ONNX inputs");
        cams = int(img[1]), size = int(img[2]), text_dim = int(txt[1]), chunk = int(out[1]);
    }

    // images uint8 [n][cams][size][size][3], state [n][24] (raw), text [n][E], noise [n][chunk*5]
    // -> chunk [n][chunk][5]
    void run(int n, uint8_t* images, float* state, float* text, float* noise, float* out) {
        const auto t0 = Clock::now();
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 5> si{n, cams, size, size, 3};
        const std::array<int64_t, 2> ss{n, kState}, st{n, text_dim}, sn{n, int64_t(chunk) * kAct};
        const std::array<int64_t, 3> so{n, chunk, kAct};
        std::array<Ort::Value, 4> in = {
            Ort::Value::CreateTensor<uint8_t>(mem, images, size_t(n) * cams * size * size * 3, si.data(), si.size()),
            Ort::Value::CreateTensor<float>(mem, state, size_t(n) * kState, ss.data(), ss.size()),
            Ort::Value::CreateTensor<float>(mem, text, size_t(n) * text_dim, st.data(), st.size()),
            Ort::Value::CreateTensor<float>(mem, noise, size_t(n) * chunk * kAct, sn.data(), sn.size())};
        Ort::Value res = Ort::Value::CreateTensor<float>(mem, out, size_t(n) * chunk * kAct, so.data(), so.size());
        const char* in_names[4] = {"images", "state", "text", "noise"};
        const char* out_names[1] = {"chunk"};
        Ort::RunOptions ro;
        session->Run(ro, in_names, in.data(), 4, out_names, &res, 1);
        seconds += since(t0);
        ++calls;
    }
};

struct Result {
    int episodes = 0, named = 0, task_target = 0, other = 0, none = 0;
    double seconds = 0;
};

// Every env runs one episode. model null = the scripted expert (instruct ignored).
Result run(Model* model, const std::map<std::string, std::vector<float>>& text, int envs, uint64_t seed,
           forge::EnvBackend backend, int exec, bool instruct_other, uint64_t rng_seed) {
    forge::EnvOptions eo;
    eo.task = "panda_pick_color";
    eo.num_envs = envs;
    eo.seed = seed;
    eo.backend = backend;
    eo.asset_dir = CRUCIBLE_ASSET_DIR;
    auto env = forge::make_env(eo);
    const int N = env->num_envs(), D = env->obs_dim(), A = env->act_dim();
    if (A != kAct || D < 50) throw std::runtime_error("expected panda_pick_color (obs 50, act 5)");
    const bool gpu = backend == forge::EnvBackend::kGpu;
    std::unique_ptr<PickCameraRig> rig;
    if (model) {
        rig = std::make_unique<PickCameraRig>(env->model(), N, model->size, FLUX_SHADER_DIR);
        if (PickCameraRig::kCameras != model->cams) throw std::runtime_error("model and task cameras differ");
    }
    const int C = model ? model->chunk : 1, E = model ? model->text_dim : 0;
    const size_t one = model ? size_t(model->size) * model->size * 3 : 0, fb = one * PickCameraRig::kCameras;
    std::vector<uint8_t> img(size_t(N) * fb);
    std::vector<float> st(size_t(N) * kState), tx(size_t(N) * E), noise(size_t(N) * C * kAct),
        chunk(size_t(N) * C * kAct), act(size_t(N) * A);
    forge::DeviceBuffer<float> dact;
    if (gpu) dact.resize(size_t(N) * A);
    std::mt19937_64 rng(rng_seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<int> instructed(N, -1), task_col(N, -1);
    std::vector<char> done(N, 0);
    Result r;
    const auto t0 = Clock::now();
    forge::Env::HostCopy h;
    env->download(h);
    for (int e = 0; e < N; ++e) {  // the instruction of each env's first episode
        task_col[e] = target_color(h.obs.data() + size_t(e) * D);
        instructed[e] = task_col[e];
        if (instruct_other) instructed[e] = (task_col[e] + 1 + int(rng() % 2)) % kColors;
        if (model) {
            const auto& v = text.at(kColorNames[instructed[e]]);
            std::copy(v.begin(), v.end(), tx.begin() + size_t(e) * E);
        }
    }
    for (int t = 0; r.episodes < N && t < 4 * kEpisodeSteps; ++t) {
        if (model) {
            const int k = t % exec;
            if (k == 0) {  // plan: render, run the model for every env
                rig->render(h.qpos.data(), h.obs.data(), D);
                for (int e = 0; e < N; ++e) {
                    for (int c = 0; c < PickCameraRig::kCameras; ++c)
                        rig->camera().copy_rgb(e, c, img.data() + e * fb + c * one);
                    std::copy(h.obs.begin() + size_t(e) * D, h.obs.begin() + size_t(e) * D + kState,
                              st.begin() + size_t(e) * kState);
                }
                for (auto& z : noise) z = gauss(rng);
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
            if (done[e] || !(h.terminated[e] || h.truncated[e])) continue;
            done[e] = 1;
            ++r.episodes;
            const int got = lifted_color(h.final_obs.data() + size_t(e) * D);
            const int want = model ? instructed[e] : task_col[e];
            if (got < 0)
                ++r.none;
            else if (got == want)
                ++r.named;
            else if (got == task_col[e])
                ++r.task_target;
            else
                ++r.other;
        }
    }
    r.seconds = since(t0);
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
        const std::string onnx = cfg.get_string("onnx", "vla.onnx");
        std::string text_path = cfg.get_string("text", "");
        const int envs = cfg.get_int("eval_envs", 500), eval_seed = cfg.get_int("eval_seed", 1000);
        const std::string backend = cfg.get_string("eval_backend", "gpu"), device = cfg.get_string("device", "cpu");
        const int exec = cfg.get_int("exec", 4), seed = cfg.get_int("seed", 1);
        const std::string instruct = cfg.get_string("instruct", "target");
        const bool eval_expert = cfg.get_bool("eval_expert", true);
        for (const auto& k : cfg.unused_keys()) std::fprintf(stderr, "warning: unknown option '%s' (typo?)\n", k.c_str());
        if (backend != "gpu" && backend != "cpu") throw std::runtime_error("eval_backend must be gpu or cpu");
        if (device != "cpu" && device != "cuda") throw std::runtime_error("device must be cpu or cuda");
        if (instruct != "target" && instruct != "other") throw std::runtime_error("instruct must be target or other");
        if (text_path.empty()) {
            const size_t dot = onnx.rfind(".onnx");
            text_path = (dot == std::string::npos ? onnx : onnx.substr(0, dot)) + "_text.txt";
        }
        const auto text = read_text_vectors(text_path);
        Model model(onnx, device == "cuda");
        if (exec < 1 || exec > model.chunk) throw std::runtime_error("exec must be 1 .. chunk");
        if (int(text.at("red").size()) != model.text_dim) throw std::runtime_error("text vector size does not match");
        std::printf("vla_app: %s (%d cameras %dx%d, text %d, chunk %d; runs %d of each chunk), ONNX Runtime on %s; "
                    "%d episodes of panda_pick_color (seed %d, %s env), instruction names the %s\n",
                    onnx.c_str(), model.cams, model.size, model.size, model.text_dim, model.chunk, exec, device.c_str(), envs,
                    eval_seed, backend.c_str(), instruct == "target" ? "task's target" : "another cube");
        std::fflush(stdout);
        const auto be = backend == "gpu" ? forge::EnvBackend::kGpu : forge::EnvBackend::kCpu;
        const bool other = instruct == "other";
        const Result r = run(&model, text, envs, uint64_t(eval_seed), be, exec, other, uint64_t(seed));
        print("policy", r, other);
        std::printf("vla_app: model %.1f ms per call (%d envs at once, %d calls)\n",
                    1e3 * model.seconds / std::max(1, model.calls), envs, model.calls);
        if (eval_expert && !other) print("expert", run(nullptr, text, envs, uint64_t(eval_seed), be, exec, false, 0), false);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
