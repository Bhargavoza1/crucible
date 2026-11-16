// bc_app (Phase 9d): behavior cloning for panda_pick in our own engine. The same
// network, data, split and metrics as tools/bc_train.py (PyTorch), so the two can
// be compared; then the closed-loop test: the policy drives the robot in the env,
// with live camera images, and we count how often it lifts the cube.
//
//   ./bc_app --demos demos/pick_img.bin --out runs/bc_pick.bin --eval_envs 500
//   ./bc_app --demos demos/pick_img.bin --input state          # full state, no images
//   ./bc_app --load runs/bc_pick.bin --eval_envs 500           # closed loop only
//
// Data: every image_every-th step of each episode (the steps with camera frames;
// every 2nd step with --input state). Inputs: --input image: front + wrist images
// and obs 0-23 (robot state); --input state: obs 0-33 (+ the true cube pose). Label:
// the expert's clean action. State is normalized by the training frames' mean / std.
// Split: episode i is held out when i % val_every == 0 (bc_train.py --val_every:
// the same episodes). Metrics on the held-out frames as bc_train.py: MSE per action,
// gripper right (sign of a[4]), move error in cm per step (a[0..2] x 2 cm).
// Training: Adam (lr 3e-4), MSE, batch 256, random shifts of up to 4 pixels (DrQ).
// Writes the policy (checkpoint) to --out and the metrics per epoch to
// <out without .bin>_metrics.json.
//
// Closed loop (--eval_envs N > 0): N envs (seed --eval_seed, not the demo seed),
// each runs one episode with the policy (images from the task's cameras each step,
// rendered like collect_app's). Success: the cube is more than 8 cm up at the end
// (collect_app's rule). With --eval_expert true (default) the scripted expert runs
// the same N starts too, for comparison.
// Options: demos, input, epochs, batch, lr, val_every, max_episodes, seed, out,
// eval_batch, load, eval_envs, eval_seed, eval_backend, eval_expert.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "forge/env/demo_file.h"
#include "forge/env/env.h"
#include "forge/util/config.h"
#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"
#include "temper/algos/bc/bc_ops.h"
#include "temper/algos/bc/bc_policy.h"
#include "temper/common/checkpoint.h"
#include "temper/optim/adam.h"
#ifdef CRUCIBLE_HAS_FLUX
#include "../collect_app/camera_rig.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

constexpr int kAct = 5;
constexpr float kCmPerUnit = 2.0f;  // panda_pick: a[0..2] = 1 moves the hand 2 cm per step

// Page-locked host memory (fast, asynchronous uploads).
template <class T>
struct Pinned {
    T* p = nullptr;
    size_t n = 0;
    explicit Pinned(size_t count) : n(count) {
        if (n) CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&p), n * sizeof(T)));
    }
    ~Pinned() {
        if (p) cudaFreeHost(p);
    }
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
};

struct Data {
    forge::DemoFile file;
    int S = 0, cams = 0, size = 0;
    std::vector<int> cols;               // obs columns used as state
    std::vector<float> state;            // [frames][S], normalized
    std::vector<float> act;              // [frames][kAct], expert labels
    std::vector<const uint8_t*> img;     // per frame: [cams][size][size][3] in the file's episode
    std::vector<int> ep;                 // per frame: episode index
    std::vector<int> train, val;         // frame indices
    std::vector<float> mean, stdev;      // per state column
    size_t frame_bytes() const { return size_t(cams) * size * size * 3; }
};

Data load(const std::string& path, bool images, int max_episodes, int val_every) {
    Data d;
    d.file = forge::read_demo_file(path, max_episodes);
    const auto& f = d.file;
    if (f.task != "panda_pick") throw std::runtime_error("expected panda_pick demos, got " + f.task);
    if (f.act_dim != kAct) throw std::runtime_error("expected 5 actions");
    if (f.episodes.empty()) throw std::runtime_error("no episodes in " + path);
    if (images && f.images.cameras == 0)
        throw std::runtime_error("no camera images in this file: record with collect_app --images true");
    if (images && f.images.width != f.images.height) throw std::runtime_error("images must be square");
    const int every = images ? f.images.every : 2;
    d.cols.resize(images ? 24 : 34);
    std::iota(d.cols.begin(), d.cols.end(), 0);
    d.S = int(d.cols.size());
    d.cams = images ? f.images.cameras : 0;
    d.size = images ? f.images.width : 0;
    std::vector<float> raw;  // un-normalized state
    for (int i = 0; i < int(f.episodes.size()); ++i) {
        const auto& e = f.episodes[i];
        const int T = e.steps();
        int n = (T + every - 1) / every;
        if (images) n = std::min<int>(n, int(e.images.size() / d.frame_bytes()));
        for (int k = 0; k < n; ++k) {
            const int t = k * every;
            for (int c : d.cols) raw.push_back(e.obs[size_t(t) * f.obs_dim + c]);
            for (int a = 0; a < kAct; ++a) d.act.push_back(e.expert[size_t(t) * kAct + a]);
            d.img.push_back(images ? e.images.data() + size_t(k) * d.frame_bytes() : nullptr);
            d.ep.push_back(i);
            (i % val_every == 0 ? d.val : d.train).push_back(int(d.ep.size()) - 1);
        }
    }
    if (d.train.empty() || d.val.empty()) throw std::runtime_error("need training and held-out episodes (val_every?)");
    // mean / std (population, + 1e-6) over the training frames, as bc_train.py
    std::vector<double> m(d.S, 0.0), v(d.S, 0.0);
    for (int r : d.train)
        for (int c = 0; c < d.S; ++c) m[c] += raw[size_t(r) * d.S + c];
    for (auto& x : m) x /= double(d.train.size());
    for (int r : d.train)
        for (int c = 0; c < d.S; ++c) {
            const double q = raw[size_t(r) * d.S + c] - m[c];
            v[c] += q * q;
        }
    d.mean.resize(d.S);
    d.stdev.resize(d.S);
    for (int c = 0; c < d.S; ++c) {
        d.mean[c] = float(m[c]);
        d.stdev[c] = float(std::sqrt(v[c] / double(d.train.size())) + 1e-6);
    }
    d.state.resize(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        const int c = int(i % d.S);
        d.state[i] = (raw[i] - d.mean[c]) / d.stdev[c];
    }
    return d;
}

// Host staging for one batch (double-buffered while training) and its device copy.
struct Batch {
    Pinned<float> state, act;
    Pinned<uint8_t> img;
    Pinned<int> shifts;
    cudaEvent_t uploaded = nullptr;
    Batch(int B, int S, size_t frame_bytes, int cams)
        : state(size_t(B) * S), act(size_t(B) * kAct), img(size_t(B) * frame_bytes), shifts(size_t(B) * cams * 2) {
        CUDA_CHECK(cudaEventCreateWithFlags(&uploaded, cudaEventDisableTiming));
    }
    ~Batch() { cudaEventDestroy(uploaded); }
};

struct DeviceBatch {
    forge::DeviceBuffer<float> state, act;
    forge::DeviceBuffer<uint8_t> img;
    forge::DeviceBuffer<int> shifts;
    DeviceBatch(int B, int S, size_t frame_bytes, int cams) {
        state.resize(size_t(B) * S);
        act.resize(size_t(B) * kAct);
        img.resize(std::max<size_t>(1, size_t(B) * frame_bytes));
        shifts.resize(std::max<size_t>(1, size_t(B) * cams * 2));
    }
};

// Fill 'b' with frames idx[0..n) (and random shifts if rng), then upload.
void stage(const Data& d, const int* idx, int n, Batch& b, DeviceBatch& g, std::mt19937_64* rng, cudaStream_t s) {
    CUDA_CHECK(cudaEventSynchronize(b.uploaded));  // its last upload is done: safe to overwrite
    const size_t fb = d.frame_bytes();
    for (int i = 0; i < n; ++i) {
        const int r = idx[i];
        std::memcpy(b.state.p + size_t(i) * d.S, d.state.data() + size_t(r) * d.S, d.S * sizeof(float));
        std::memcpy(b.act.p + size_t(i) * kAct, d.act.data() + size_t(r) * kAct, kAct * sizeof(float));
        if (fb) std::memcpy(b.img.p + i * fb, d.img[r], fb);
    }
    CUDA_CHECK(cudaMemcpyAsync(g.state.data(), b.state.p, size_t(n) * d.S * 4, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(g.act.data(), b.act.p, size_t(n) * kAct * 4, cudaMemcpyHostToDevice, s));
    if (fb) CUDA_CHECK(cudaMemcpyAsync(g.img.data(), b.img.p, size_t(n) * fb, cudaMemcpyHostToDevice, s));
    if (rng && d.cams) {
        std::uniform_int_distribution<int> u(0, 2 * temper::BcPolicy::kShiftPad);
        for (int i = 0; i < n * d.cams * 2; ++i) b.shifts.p[i] = u(*rng);
        CUDA_CHECK(cudaMemcpyAsync(g.shifts.data(), b.shifts.p, size_t(n) * d.cams * 2 * 4, cudaMemcpyHostToDevice, s));
    }
    CUDA_CHECK(cudaEventRecord(b.uploaded, s));
}

struct Metrics {
    double mse[kAct] = {}, mse_mean = 0, gripper_right = 0, move_error_cm = 0;
};

Metrics evaluate(temper::BcPolicy& pol, const Data& d, int eval_batch, Batch& b, DeviceBatch& g, cudaStream_t s) {
    Metrics m;
    std::vector<float> pred(size_t(eval_batch) * kAct);
    size_t n = 0, grip = 0;
    for (size_t k = 0; k < d.val.size(); k += size_t(eval_batch)) {
        const int nb = int(std::min<size_t>(eval_batch, d.val.size() - k));
        stage(d, d.val.data() + k, nb, b, g, nullptr, s);
        const float* y = pol.forward(g.state.data(), d.cams ? g.img.data() : nullptr, nullptr, nb, s);
        CUDA_CHECK(cudaMemcpyAsync(pred.data(), y, size_t(nb) * kAct * 4, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        for (int i = 0; i < nb; ++i) {
            const float* p = pred.data() + size_t(i) * kAct;
            const float* t = d.act.data() + size_t(d.val[k + i]) * kAct;
            double move = 0;
            for (int a = 0; a < kAct; ++a) {
                const double e = double(p[a]) - t[a];
                m.mse[a] += e * e;
                if (a < 3) move += e * e;
            }
            m.move_error_cm += std::sqrt(move) * kCmPerUnit;
            auto sign = [](float v) { return (v > 0.0f) - (v < 0.0f); };
            grip += sign(p[4]) == sign(t[4]);
        }
        n += size_t(nb);
    }
    for (double& v : m.mse) {
        v /= double(n);
        m.mse_mean += v / kAct;
    }
    m.gripper_right = double(grip) / double(n);
    m.move_error_cm /= double(n);
    return m;
}

std::string strip_bin(const std::string& p) {
    return p.size() > 4 && p.compare(p.size() - 4, 4, ".bin") == 0 ? p.substr(0, p.size() - 4) : p;
}

// ---- closed loop --------------------------------------------------------------------

struct LoopResult {
    int success = 0, episodes = 0;
    double seconds = 0;
};

// Every env runs one episode; 'policy' null = the scripted expert.
LoopResult closed_loop(temper::BcPolicy* policy, const std::vector<int>& cols, const std::vector<float>& mean,
                       const std::vector<float>& stdev, int envs, uint64_t seed, forge::EnvBackend backend,
                       cudaStream_t s) {
    forge::EnvOptions eo;
    eo.task = "panda_pick";
    eo.num_envs = envs;
    eo.seed = seed;
    eo.backend = backend;
    eo.asset_dir = CRUCIBLE_ASSET_DIR;
    auto env = forge::make_env(eo);
    const int N = env->num_envs(), D = env->obs_dim(), A = env->act_dim();
    if (A != kAct) throw std::runtime_error("panda_pick should have 5 actions");
    const bool gpu = backend == forge::EnvBackend::kGpu;
    const int cams = policy ? policy->config().cams : 0, S = int(cols.size());
    const int B = policy ? policy->max_batch() : 0;
#ifdef CRUCIBLE_HAS_FLUX
    std::unique_ptr<PickCameraRig> rig;
    if (cams > 0) {
        rig = std::make_unique<PickCameraRig>(env->model(), N, policy->config().size, FLUX_SHADER_DIR);
        if (PickCameraRig::kCameras != cams) throw std::runtime_error("policy and task cameras differ");
    }
#else
    if (cams > 0) throw std::runtime_error("closed loop with images needs a build with -DBUILD_FLUX=ON");
#endif
    const int size = policy ? policy->config().size : 0;
    const size_t one = size_t(size) * size * 3, fb = one * cams;
    Pinned<float> st(size_t(N) * S), act(size_t(N) * A);
    Pinned<uint8_t> img(size_t(N) * fb);
    forge::DeviceBuffer<float> dst(std::max<size_t>(1, size_t(B) * S)), dact(size_t(N) * A);
    forge::DeviceBuffer<uint8_t> dimg(std::max<size_t>(1, size_t(B) * fb));
    cudaStream_t es = static_cast<cudaStream_t>(env->stream());

    LoopResult r;
    const auto t0 = Clock::now();
    std::vector<char> done(N, 0);
    forge::Env::HostCopy h;
    env->download(h);
    for (int guard = 0; r.episodes < N && guard < 100000; ++guard) {
        if (policy) {
#ifdef CRUCIBLE_HAS_FLUX
            if (rig) {
                rig->render(h.qpos.data());
                for (int e = 0; e < N; ++e)
                    for (int c = 0; c < cams; ++c) rig->camera().copy_rgb(e, c, img.p + e * fb + c * one);
            }
#endif
            for (int e = 0; e < N; ++e)
                for (int k = 0; k < S; ++k)
                    st.p[size_t(e) * S + k] = (h.obs[size_t(e) * D + cols[k]] - mean[k]) / stdev[k];
            for (int e0 = 0; e0 < N; e0 += B) {
                const int nb = std::min(B, N - e0);
                CUDA_CHECK(cudaMemcpyAsync(dst.data(), st.p + size_t(e0) * S, size_t(nb) * S * 4, cudaMemcpyHostToDevice, s));
                if (fb) CUDA_CHECK(cudaMemcpyAsync(dimg.data(), img.p + e0 * fb, nb * fb, cudaMemcpyHostToDevice, s));
                const float* y = policy->forward(dst.data(), fb ? dimg.data() : nullptr, nullptr, nb, s);
                CUDA_CHECK(cudaMemcpyAsync(act.p + size_t(e0) * A, y, size_t(nb) * A * 4, cudaMemcpyDeviceToHost, s));
            }
            CUDA_CHECK(cudaStreamSynchronize(s));
            if (gpu) {
                CUDA_CHECK(cudaMemcpyAsync(dact.data(), act.p, size_t(N) * A * 4, cudaMemcpyHostToDevice, es));
                env->step(dact.data());
            } else {
                env->step(act.p);
            }
        } else if (gpu) {
            env->expert_actions(dact.data());
            env->step(dact.data());
        } else {
            env->expert_actions(act.p);
            env->step(act.p);
        }
        env->download(h);
        for (int e = 0; e < N; ++e) {
            if (done[e] || !(h.terminated[e] || h.truncated[e])) continue;
            done[e] = 1;
            ++r.episodes;
            r.success += h.final_obs[size_t(e) * D + 26] - 0.025f > 0.08f;  // cube z - rest height > 8 cm
        }
    }
    r.seconds = since(t0);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        forge::Config cfg = forge::Config::from_cli(argc, argv);
        const std::string demos = cfg.get_string("demos", "");
        const std::string input = cfg.get_string("input", "image");
        const int epochs = cfg.get_int("epochs", 20), B = cfg.get_int("batch", 256);
        const float lr = cfg.get_float("lr", 3e-4f);
        const int val_every = cfg.get_int("val_every", 10), max_episodes = cfg.get_int("max_episodes", 0);
        const int seed = cfg.get_int("seed", 1), eval_batch = cfg.get_int("eval_batch", 1024);
        const std::string out = cfg.get_string("out", "bc_pick.bin"), load_path = cfg.get_string("load", "");
        const int eval_envs = cfg.get_int("eval_envs", 0), eval_seed = cfg.get_int("eval_seed", 1000);
        const std::string eval_backend = cfg.get_string("eval_backend", "gpu");
        const bool eval_expert = cfg.get_bool("eval_expert", true);
        for (const auto& k : cfg.unused_keys()) std::fprintf(stderr, "warning: unknown option '%s' (typo?)\n", k.c_str());
        if (input != "image" && input != "state") throw std::runtime_error("input must be image or state");
        if (B <= 0 || eval_batch <= 0 || epochs < 0 || val_every < 2) throw std::runtime_error("bad batch / epochs / val_every");
        if (eval_backend != "gpu" && eval_backend != "cpu") throw std::runtime_error("eval_backend must be gpu or cpu");
        if (demos.empty() && load_path.empty()) throw std::runtime_error("give --demos FILE (train) or --load CHECKPOINT");

        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        std::unique_ptr<temper::BcPolicy> pol;
        std::vector<int> cols;
        std::vector<float> mean, stdev;
        const int max_batch = std::max({B, eval_batch, 1});

        if (!load_path.empty()) {
            const auto c = temper::Checkpoint::load(load_path);
            if (c.algo != "bc") throw std::runtime_error(load_path + " is not a bc_app checkpoint");
            pol = std::make_unique<temper::BcPolicy>(temper::BcPolicy::config_from(c), max_batch, uint64_t(seed));
            pol->load(c);
            mean = c.array("state_mean");
            stdev = c.array("state_std");
            for (float v : c.array("state_cols")) cols.push_back(int(v));
            std::printf("bc_app: loaded %s (%s, %d state numbers, %d cameras)\n", load_path.c_str(),
                        pol->config().cams ? "images" : "state", pol->config().state_dim, pol->config().cams);
        } else {
            const auto t0 = Clock::now();
            Data d = load(demos, input == "image", max_episodes, val_every);
            const int n_eps = int(d.file.episodes.size());
            const int n_val = (n_eps + val_every - 1) / val_every;
            std::printf("bc_app: %d episodes (%d train, %d held out), %zu / %zu frames, input %s (%d state numbers",
                        n_eps, n_eps - n_val, n_val, d.train.size(), d.val.size(), input.c_str(), d.S);
            if (d.cams) std::printf(", %d cameras %dx%d", d.cams, d.size, d.size);
            std::printf("), loaded in %.0f s\n", since(t0));

            pol = std::make_unique<temper::BcPolicy>(temper::BcPolicyConfig{d.S, d.cams, d.size, kAct}, max_batch,
                                                     uint64_t(seed));
            std::printf("bc_app: %.2f M parameters\n", double(pol->num_params()) / 1e6);
            std::vector<temper::Tensor*> P, G;
            pol->params(P, G);
            temper::Adam adam(P, G, temper::AdamOptions{lr});
            Batch b0(max_batch, d.S, d.frame_bytes(), d.cams), b1(max_batch, d.S, d.frame_bytes(), d.cams);
            Batch* bufs[2] = {&b0, &b1};
            DeviceBatch g(max_batch, d.S, d.frame_bytes(), d.cams);
            forge::DeviceBuffer<float> dy(size_t(max_batch) * kAct), loss(1);
            std::mt19937_64 rng{static_cast<uint64_t>(seed)};
            std::vector<int> order = d.train;
            std::string json = "[";
            for (int epoch = 1; epoch <= epochs; ++epoch) {
                const auto te = Clock::now();
                std::shuffle(order.begin(), order.end(), rng);
                CUDA_CHECK(cudaMemsetAsync(loss.data(), 0, 4, s));
                const int nb = int(order.size()) / B;
                for (int k = 0; k < nb; ++k) {
                    // fills one host buffer while the GPU still works on the other's batch
                    stage(d, order.data() + size_t(k) * B, B, *bufs[k % 2], g, &rng, s);
                    const float* y = pol->forward(g.state.data(), d.cams ? g.img.data() : nullptr,
                                                  d.cams ? g.shifts.data() : nullptr, B, s);
                    temper::bc::mse_loss(y, g.act.data(), B, kAct, dy.data(), loss.data(), s);
                    pol->backward(dy.data(), B, s);
                    adam.step(s);
                }
                float lsum = 0;
                CUDA_CHECK(cudaMemcpyAsync(&lsum, loss.data(), 4, cudaMemcpyDeviceToHost, s));
                CUDA_CHECK(cudaStreamSynchronize(s));
                const double train_s = since(te);
                const Metrics m = evaluate(*pol, d, eval_batch, b0, g, s);
                const double train_loss = lsum / std::max(1, nb);
                std::printf("epoch %3d  train loss %.4f  held-out: mse %.4f (x %.3f y %.3f z %.3f turn %.3f grip %.3f)  "
                            "gripper right %.1f%%  move error %.2f cm/step  (%.0f s, %.0f frames/s)\n",
                            epoch, train_loss, m.mse_mean, m.mse[0], m.mse[1], m.mse[2], m.mse[3], m.mse[4],
                            100.0 * m.gripper_right, m.move_error_cm, since(te), double(nb) * B / train_s);
                std::fflush(stdout);
                char buf[512];
                std::snprintf(buf, sizeof(buf),
                              "%s\n {\"epoch\": %d, \"train_loss\": %.6g, \"mse\": [%.6g, %.6g, %.6g, %.6g, %.6g], "
                              "\"mse_mean\": %.6g, \"gripper_right\": %.6g, \"move_error_cm\": %.6g, \"seconds\": %.2f}",
                              epoch > 1 ? "," : "", epoch, train_loss, m.mse[0], m.mse[1], m.mse[2], m.mse[3],
                              m.mse[4], m.mse_mean, m.gripper_right, m.move_error_cm, since(te));
                json += buf;
            }
            json += "\n]\n";
            cols = d.cols;
            mean = d.mean;
            stdev = d.stdev;
            temper::Checkpoint c;
            c.algo = "bc";
            c.task = "panda_pick";
            c.obs_dim = d.file.obs_dim;
            c.act_dim = kAct;
            c.iteration = epochs;
            c.seed = uint64_t(seed);
            c.optimizer_steps = adam.steps();
            pol->save(c);
            c.arrays["state_mean"] = mean;
            c.arrays["state_std"] = stdev;
            c.arrays["state_cols"] = std::vector<float>(cols.begin(), cols.end());
            c.save(out);
            const std::string jpath = strip_bin(out) + "_metrics.json";
            if (std::FILE* f = std::fopen(jpath.c_str(), "w")) {
                std::fputs(json.c_str(), f);
                std::fclose(f);
            }
            std::printf("bc_app: wrote %s and %s\n", out.c_str(), jpath.c_str());
        }

        if (eval_envs > 0) {
            const auto be = eval_backend == "gpu" ? forge::EnvBackend::kGpu : forge::EnvBackend::kCpu;
            std::printf("bc_app: closed loop: %d episodes of panda_pick (seed %d, %s env)...\n", eval_envs, eval_seed,
                        eval_backend.c_str());
            std::fflush(stdout);
            const LoopResult r = closed_loop(pol.get(), cols, mean, stdev, eval_envs, uint64_t(eval_seed), be, s);
            std::printf("bc_app: policy success %d / %d (%.1f%%)  (%.1f s)\n", r.success, r.episodes,
                        100.0 * r.success / std::max(1, r.episodes), r.seconds);
            if (eval_expert) {
                const LoopResult x = closed_loop(nullptr, cols, mean, stdev, eval_envs, uint64_t(eval_seed), be, s);
                std::printf("bc_app: expert success %d / %d (%.1f%%) on the same starts\n", x.success, x.episodes,
                            100.0 * x.success / std::max(1, x.episodes));
            }
        }
        cudaStreamDestroy(s);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
