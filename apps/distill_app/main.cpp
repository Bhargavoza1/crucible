// distill_app (Phase 10): DAgger distillation of the OpenArm VLA (the teacher: an
// ONNX graph from tools/vla_openarm.py, run by ONNX Runtime) into the small CNN + GRU
// student (temper/algos/student: 64 x 64 images, trained in our engine).
//
//   ./distill_app --teacher runs/oa_vla_d1c.onnx --out runs/student.bin
//   ./distill_app --load runs/student.bin --rounds 0 --eval_envs 500      # closed loop only
//
// Rounds (--rounds, default 6). Each round, --envs (512) envs each run one episode
// in openarm_pick_place (new starts every round: seed --data_seed + round):
//   round 0: the TEACHER drives;  later rounds: the STUDENT drives (DAgger)
// and at every step the teacher labels the state: the first action of its chunk
// (the mode the VLA works best in: replanning every step, fresh flow noise). Every
// step is kept (state obs 0-23, the teacher's action, the head and left wrist images
// at 64 x 64: the teacher's 128 x 128 halved) in a buffer of up to --buffer_frames
// steps (oldest episodes dropped; about 25 KB a step: 200 000 steps = 5 GB of RAM).
// Then --train_steps (1500) Adam steps (lr --lr 3e-4, gradient norm clipped at 1)
// on --batch (32) windows of --seq (48) steps of the buffer's episodes, through the
// GRU (BPTT): the loss (MSE of the action mean vs the teacher's action) counts every
// step of a window that starts at an episode's start, else only the steps after the
// first --burn_in (16), so the memory has warmed up. Random shifts of 4 pixels.
// Each round prints how often the episodes it ran ended done (cube in the tray, arm
// home): the teacher's in round 0, the student's after. The state is normalized by
// the mean / std of round 0's steps.
// At the end: the student in --eval_envs (500) envs of the test starts (--eval_seed
// 1000, as vla_app), the actions its mean, its memory from the episode's start; the
// result next to the teacher's on the same starts with --eval_teacher true (slow).
// Writes --out (checkpoint: the student, its state mean / std) after every round.
// Options: teacher, info, device (ONNX Runtime: cpu | cuda), envs, rounds, data_seed,
// buffer_frames, train_steps, batch, seq, burn_in, lr, seed, out, load, eval_envs,
// eval_seed, eval_teacher, backend (gpu | cpu env).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../collect_app/openarm_camera_rig.h"
#include "../vla_app/vla_model.h"
#include "forge/env/env.h"
#include "forge/env/tasks/openarm_pick_place.h"
#include "forge/util/config.h"
#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"
#include "temper/algos/student/student_ops.h"
#include "temper/algos/student/student_policy.h"
#include "temper/common/checkpoint.h"
#include "temper/optim/adam.h"

namespace {

using vla::Clock;
using vla::since;
using Task = forge::OpenArmPickPlaceTask;
constexpr int kAct = 5, kState = 24, kCams = 2;

// <teacher without .onnx>_openarm.txt (vla_openarm.py --onnx): "cameras 0 1", "text ..."
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
    if (int(info.cameras.size()) != kCams || info.text.empty())
        throw std::runtime_error(path + ": needs 'cameras' (two ids) and 'text'");
    return info;
}

// obs: TCP z 20, cube z 26, cube - tray center 36-37 (vla_app's rule)
bool done_ok(const float* fo, bool terminated) {
    return terminated && std::fabs(fo[36]) < 0.03f && std::fabs(fo[37]) < 0.03f && fo[26] < 0.025f && fo[20] > 0.08f;
}

// 2 x 2 box filter: [S][S][3] -> [S/2][S/2][3]
void halve(const uint8_t* in, int S, uint8_t* out) {
    const int h = S / 2;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < h; ++x)
            for (int c = 0; c < 3; ++c) {
                const int s = in[((2 * y) * S + 2 * x) * 3 + c] + in[((2 * y) * S + 2 * x + 1) * 3 + c] +
                              in[((2 * y + 1) * S + 2 * x) * 3 + c] + in[((2 * y + 1) * S + 2 * x + 1) * 3 + c];
                out[(y * h + x) * 3 + c] = uint8_t((s + 2) / 4);
            }
}

struct Episode {
    int T = 0;
    std::vector<float> state, label;  // [T][kState] raw, [T][kAct]
    std::vector<uint8_t> img;         // [T][kCams][s][s][3]
};

struct Buffer {
    std::deque<Episode> eps;
    size_t frames = 0, cap = 200000;
    void add(Episode&& e) {
        frames += size_t(e.T);
        eps.push_back(std::move(e));
        while (frames > cap && eps.size() > 1) {
            frames -= size_t(eps.front().T);
            eps.pop_front();
        }
    }
};

struct Ctx {
    vla::Model* teacher = nullptr;
    Info info;
    temper::StudentPolicy* student = nullptr;
    std::vector<float> mean, stdev;  // state normalization
    forge::EnvBackend backend = forge::EnvBackend::kGpu;
    cudaStream_t stream = nullptr;
    std::mt19937_64 rng{1};
};

struct RolloutResult {
    int episodes = 0, done = 0;
    double steps_done = 0, seconds = 0;
};

// Every env runs one episode from its start. Actor: the student (student_acts) or
// the teacher. The teacher labels every step when 'rec' is set (or when it acts).
RolloutResult rollout(Ctx& c, bool student_acts, uint64_t seed, int envs, Buffer* rec) {
    forge::EnvOptions eo;
    eo.task = Task::kName;
    eo.num_envs = envs;
    eo.seed = seed;
    eo.backend = c.backend;
    eo.physics_dt = 0.002f;  // configs/openarm_pick_place.txt
    eo.decimation = 10;
    eo.asset_dir = CRUCIBLE_ASSET_DIR;
    auto env = forge::make_env(eo);
    const int N = env->num_envs(), D = env->obs_dim();
    const bool gpu = c.backend == forge::EnvBackend::kGpu;
    const int TS = c.teacher->size, SS = TS / 2;  // teacher and student image sizes
    const bool need_teacher = rec || !student_acts;
    if (student_acts && c.student->config().size != SS)
        throw std::runtime_error("the student's image size is not half the teacher's");
    OpenArmCameraRig rig(env->model(), N, TS, FLUX_SHADER_DIR,
                         std::string(CRUCIBLE_ASSET_DIR) + "/robots/openarm/openarm_cell.cmesh");
    const size_t tone = size_t(TS) * TS * 3, sone = size_t(SS) * SS * 3;
    std::vector<uint8_t> timg(size_t(N) * kCams * tone), simg(size_t(N) * kCams * sone);
    const int C = c.teacher->chunk, E = c.teacher->text_dim;
    std::vector<float> st(size_t(N) * kState), tx(size_t(N) * E), noise(size_t(N) * C * kAct),
        chunk(size_t(N) * C * kAct), act(size_t(N) * kAct), nst(size_t(N) * kState);
    for (int e = 0; e < N; ++e) std::copy(c.info.text.begin(), c.info.text.end(), tx.begin() + size_t(e) * E);
    const int H = c.student->config().hidden;
    forge::DeviceBuffer<float> dact(gpu ? size_t(N) * kAct : 0), dstate(size_t(N) * kState), hid(size_t(N) * H);
    forge::DeviceBuffer<uint8_t> dimg(size_t(N) * kCams * sone);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<char> ended(N, 0);
    std::vector<Episode> cur(rec ? N : 0);
    RolloutResult r;
    const auto t0 = Clock::now();
    forge::Env::HostCopy h;
    env->download(h);
    for (int t = 0; r.episodes < N && t < 2 * Task::kMaxEpisodeSteps; ++t) {
        rig.render(h.qpos.data(), h.obs.data(), D);
        for (int e = 0; e < N; ++e)
            for (int k = 0; k < kCams; ++k) {
                uint8_t* ti = timg.data() + (size_t(e) * kCams + k) * tone;
                rig.camera().copy_rgb(e, c.info.cameras[k], ti);
                halve(ti, TS, simg.data() + (size_t(e) * kCams + k) * sone);
            }
        for (int e = 0; e < N; ++e)
            std::copy(h.obs.begin() + size_t(e) * D, h.obs.begin() + size_t(e) * D + kState,
                      st.begin() + size_t(e) * kState);
        if (need_teacher) {  // the first action of a fresh chunk
            for (auto& z : noise) z = gauss(c.rng);
            c.teacher->run(N, timg.data(), st.data(), tx.data(), noise.data(), chunk.data());
        }
        if (student_acts) {
            for (size_t i = 0; i < st.size(); ++i) nst[i] = (st[i] - c.mean[i % kState]) / c.stdev[i % kState];
            CUDA_CHECK(cudaMemcpyAsync(dstate.data(), nst.data(), nst.size() * 4, cudaMemcpyHostToDevice, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(dimg.data(), simg.data(), simg.size(), cudaMemcpyHostToDevice, c.stream));
            const float* h0 = t == 0 ? nullptr : hid.data();
            const float* a = c.student->forward(dstate.data(), dimg.data(), nullptr, nullptr, h0, 1, N, c.stream);
            CUDA_CHECK(cudaMemcpyAsync(act.data(), a, act.size() * 4, cudaMemcpyDeviceToHost, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(hid.data(), c.student->last_hidden(), hid.size() * 4, cudaMemcpyDeviceToDevice,
                                       c.stream));
            CUDA_CHECK(cudaStreamSynchronize(c.stream));
        } else {
            for (int e = 0; e < N; ++e)
                std::copy(chunk.begin() + size_t(e) * C * kAct, chunk.begin() + size_t(e) * C * kAct + kAct,
                          act.begin() + size_t(e) * kAct);
        }
        if (rec)
            for (int e = 0; e < N; ++e) {
                if (ended[e]) continue;
                Episode& ep = cur[e];
                ep.state.insert(ep.state.end(), st.begin() + size_t(e) * kState, st.begin() + size_t(e + 1) * kState);
                ep.label.insert(ep.label.end(), chunk.begin() + size_t(e) * C * kAct,
                                chunk.begin() + size_t(e) * C * kAct + kAct);
                ep.img.insert(ep.img.end(), simg.begin() + size_t(e) * kCams * sone,
                              simg.begin() + size_t(e + 1) * kCams * sone);
                ++ep.T;
            }
        if (gpu) {
            dact.upload(act);
            env->step(dact.data());
        } else {
            env->step(act.data());
        }
        env->download(h);
        for (int e = 0; e < N; ++e) {
            if (ended[e] || !(h.terminated[e] || h.truncated[e])) continue;
            ended[e] = 1;
            ++r.episodes;
            if (done_ok(h.final_obs.data() + size_t(e) * D, h.terminated[e] != 0)) {
                ++r.done;
                r.steps_done += t + 1;
            }
            if (rec) {
                rec->add(std::move(cur[e]));
                cur[e] = Episode{};
            }
        }
    }
    r.seconds = since(t0);
    return r;
}

// Host staging of one training batch, time major (row k * B + b), and its device copy.
struct TrainBatch {
    std::vector<float> state, label, mask;
    std::vector<uint8_t> img;
    std::vector<int> shifts;
    forge::DeviceBuffer<float> dstate, dlabel, dmask, dy, loss, count;
    forge::DeviceBuffer<uint8_t> dimg;
    forge::DeviceBuffer<int> dshifts;
    TrainBatch(int L, int B, size_t frame) {
        const size_t n = size_t(L) * B;
        state.resize(n * kState), label.resize(n * kAct), mask.resize(n), img.resize(n * frame);
        shifts.resize(n * kCams * 2);
        dstate.resize(state.size()), dlabel.resize(label.size()), dmask.resize(n), dy.resize(label.size());
        loss.resize(1), count.resize(1), dimg.resize(img.size()), dshifts.resize(shifts.size());
    }
};

// Train 'steps' Adam steps on windows of the buffer. Returns the mean loss.
double train(Ctx& c, Buffer& buf, temper::Adam& adam, temper::GradClip& clip, TrainBatch& tb, int steps, int L, int B,
             int burn_in) {
    std::vector<size_t> starts;  // cumulative steps of the episodes long enough for a window
    std::vector<int> idx;
    size_t total = 0;
    for (int i = 0; i < int(buf.eps.size()); ++i)
        if (buf.eps[i].T >= L) {
            idx.push_back(i);
            total += size_t(buf.eps[i].T);
            starts.push_back(total);
        }
    if (idx.empty()) throw std::runtime_error("no episode is as long as --seq");
    const int S = c.student->config().size;
    const size_t frame = size_t(kCams) * S * S * 3;
    std::uniform_int_distribution<int> shift(0, 2 * temper::StudentPolicy::kShiftPad);
    double sum = 0;
    for (int it = 0; it < steps; ++it) {
        float masked = 0;
        for (int b = 0; b < B; ++b) {
            // an episode with probability by its length, then a window in it
            const size_t pick = std::uniform_int_distribution<size_t>(0, total - 1)(c.rng);
            const int i = idx[size_t(std::upper_bound(starts.begin(), starts.end(), pick) - starts.begin())];
            const Episode& ep = buf.eps[i];
            const int s0 = std::uniform_int_distribution<int>(0, ep.T - L)(c.rng);
            for (int k = 0; k < L; ++k) {
                const size_t row = size_t(k) * B + b, t = size_t(s0 + k);
                for (int j = 0; j < kState; ++j)
                    tb.state[row * kState + j] = (ep.state[t * kState + j] - c.mean[j]) / c.stdev[j];
                std::memcpy(&tb.label[row * kAct], &ep.label[t * kAct], kAct * sizeof(float));
                std::memcpy(&tb.img[row * frame], &ep.img[t * frame], frame);
                tb.mask[row] = (s0 == 0 || k >= burn_in) ? 1.0f : 0.0f;
                masked += tb.mask[row];
            }
        }
        for (auto& v : tb.shifts) v = shift(c.rng);
        auto up = [&](void* dst, const void* src, size_t bytes) {
            CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, c.stream));
        };
        up(tb.dstate.data(), tb.state.data(), tb.state.size() * 4);
        up(tb.dlabel.data(), tb.label.data(), tb.label.size() * 4);
        up(tb.dmask.data(), tb.mask.data(), tb.mask.size() * 4);
        up(tb.dimg.data(), tb.img.data(), tb.img.size());
        up(tb.dshifts.data(), tb.shifts.data(), tb.shifts.size() * 4);
        up(tb.count.data(), &masked, 4);
        CUDA_CHECK(cudaMemsetAsync(tb.loss.data(), 0, 4, c.stream));
        const float* y = c.student->forward(tb.dstate.data(), tb.dimg.data(), tb.dshifts.data(), nullptr, nullptr, L, B,
                                            c.stream);
        temper::student::masked_mse(y, tb.dlabel.data(), tb.dmask.data(), tb.count.data(), L * B, kAct, tb.dy.data(),
                                    tb.loss.data(), c.stream);
        c.student->backward(tb.dy.data(), c.stream);
        clip.compute(1.0f, c.stream);
        adam.step(c.stream, clip.scale());
        float l = 0;
        CUDA_CHECK(cudaMemcpyAsync(&l, tb.loss.data(), 4, cudaMemcpyDeviceToHost, c.stream));
        CUDA_CHECK(cudaStreamSynchronize(c.stream));  // also: the host buffers are free again
        sum += l;
    }
    return sum / std::max(1, steps);
}

void save(Ctx& c, const std::string& path, int round) {
    temper::Checkpoint k;
    k.algo = "student";
    k.task = Task::kName;
    k.obs_dim = Task::kObsDim;
    k.act_dim = kAct;
    k.iteration = round;
    c.student->save(k);
    k.arrays["state_mean"] = c.mean;
    k.arrays["state_std"] = c.stdev;
    k.save(path);
}

void print(const char* who, const RolloutResult& r) {
    std::printf("%s: done (cube in the tray, arm home) %d / %d (%.1f%%), mean %.0f steps (%.0f s)\n", who, r.done,
                r.episodes, 100.0 * r.done / std::max(1, r.episodes), r.steps_done / std::max(1, r.done), r.seconds);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        forge::Config cfg = forge::Config::from_cli(argc, argv);
        const std::string teacher_path = cfg.get_string("teacher", "runs/oa_vla_d1c.onnx");
        const size_t dot = teacher_path.rfind(".onnx");
        const std::string base = dot == std::string::npos ? teacher_path : teacher_path.substr(0, dot);
        const std::string info_path = cfg.get_string("info", base + "_openarm.txt");
        const std::string device = cfg.get_string("device", "cpu"), backend = cfg.get_string("backend", "gpu");
        const int envs = cfg.get_int("envs", 512), rounds = cfg.get_int("rounds", 6);
        const int data_seed = cfg.get_int("data_seed", 5000), buffer_frames = cfg.get_int("buffer_frames", 200000);
        const int train_steps = cfg.get_int("train_steps", 1500), B = cfg.get_int("batch", 32);
        const int L = cfg.get_int("seq", 48), burn_in = cfg.get_int("burn_in", 16);
        const float lr = cfg.get_float("lr", 3e-4f);
        const int seed = cfg.get_int("seed", 1);
        const std::string out = cfg.get_string("out", "runs/student.bin"), load = cfg.get_string("load", "");
        const int eval_envs = cfg.get_int("eval_envs", 500), eval_seed = cfg.get_int("eval_seed", 1000);
        const bool eval_teacher = cfg.get_bool("eval_teacher", false);
        for (const auto& k : cfg.unused_keys())
            std::fprintf(stderr, "warning: unknown option '%s' (typo?)\n", k.c_str());
        if (backend != "gpu" && backend != "cpu") throw std::runtime_error("backend must be gpu or cpu");
        if (device != "cpu" && device != "cuda") throw std::runtime_error("device must be cpu or cuda");
        if (envs <= 0 || rounds < 0 || B <= 0 || L <= 0 || burn_in < 0 || burn_in >= L)
            throw std::runtime_error("bad sizes (envs, rounds, batch, seq, burn_in < seq)");

        Ctx c;
        c.info = read_info(info_path);
        vla::Model teacher(teacher_path, device == "cuda");
        if (teacher.cams != kCams || teacher.size % 2 != 0) throw std::runtime_error("expected a 2-camera teacher");
        if (int(c.info.text.size()) != teacher.text_dim) throw std::runtime_error("text vector size does not match");
        c.teacher = &teacher;
        c.backend = backend == "gpu" ? forge::EnvBackend::kGpu : forge::EnvBackend::kCpu;
        c.rng.seed(uint64_t(seed));
        CUDA_CHECK(cudaStreamCreate(&c.stream));
        const int max_tb = std::max({envs, eval_envs, L * B});
        temper::StudentConfig sc{kState, kCams, teacher.size / 2, kAct, 256};
        std::unique_ptr<temper::StudentPolicy> student;
        if (!load.empty()) {
            const temper::Checkpoint k = temper::Checkpoint::load(load);
            sc = temper::StudentPolicy::config_from(k);
            student = std::make_unique<temper::StudentPolicy>(sc, max_tb, uint64_t(seed));
            student->load(k);
            c.mean = k.array("state_mean");
            c.stdev = k.array("state_std");
        } else {
            student = std::make_unique<temper::StudentPolicy>(sc, max_tb, uint64_t(seed));
        }
        c.student = student.get();
        std::printf("distill_app: teacher %s (cameras %d %d, %dx%d; first action of a fresh chunk each step, ONNX "
                    "Runtime on %s); student CNN + GRU %.2f M parameters (%dx%d images, GRU %d)%s\n",
                    teacher_path.c_str(), c.info.cameras[0], c.info.cameras[1], teacher.size, teacher.size,
                    device.c_str(), double(student->num_params()) / 1e6, sc.size, sc.size, sc.hidden,
                    load.empty() ? "" : (", from " + load).c_str());
        std::printf("distill_app: %d rounds of %d envs (round 0: the teacher drives; then the student), %d Adam steps "
                    "of %d x %d-step windows (burn-in %d) each; buffer %d steps\n",
                    rounds, envs, train_steps, B, L, burn_in, buffer_frames);
        std::fflush(stdout);

        std::vector<temper::Tensor*> P, G;
        student->params(P, G);
        temper::Adam adam(P, G, temper::AdamOptions{lr});
        temper::GradClip clip(G);
        TrainBatch tb(L, B, size_t(kCams) * sc.size * sc.size * 3);
        Buffer buf;
        buf.cap = size_t(buffer_frames);
        for (int round = 0; round < rounds; ++round) {
            const bool student_acts = round > 0 || !load.empty();
            const RolloutResult r = rollout(c, student_acts, uint64_t(data_seed + round), envs, &buf);
            char who[96];
            std::snprintf(who, sizeof(who), "round %d, %s drives", round, student_acts ? "the student" : "the teacher");
            print(who, r);
            if (c.mean.empty()) {  // state normalization from the first data
                std::vector<double> m(kState, 0.0), v(kState, 0.0);
                size_t n = 0;
                for (const auto& ep : buf.eps)
                    for (int t = 0; t < ep.T; ++t, ++n)
                        for (int j = 0; j < kState; ++j) m[j] += ep.state[size_t(t) * kState + j];
                for (auto& x : m) x /= double(std::max<size_t>(1, n));
                for (const auto& ep : buf.eps)
                    for (int t = 0; t < ep.T; ++t)
                        for (int j = 0; j < kState; ++j) {
                            const double q = ep.state[size_t(t) * kState + j] - m[j];
                            v[j] += q * q;
                        }
                c.mean.resize(kState);
                c.stdev.resize(kState);
                for (int j = 0; j < kState; ++j) {
                    c.mean[j] = float(m[j]);
                    c.stdev[j] = float(std::sqrt(v[j] / double(std::max<size_t>(1, n))) + 1e-6);
                }
            }
            const auto t0 = Clock::now();
            const double loss = train(c, buf, adam, clip, tb, train_steps, L, B, burn_in);
            std::printf("round %d: buffer %zu episodes (%zu steps); %d Adam steps, loss %.4f (%.0f s)\n", round,
                        buf.eps.size(), buf.frames, train_steps, loss, since(t0));
            std::fflush(stdout);
            save(c, out, round);
        }
        if (rounds > 0) std::printf("wrote %s\n", out.c_str());
        if (eval_envs > 0) {
            if (c.mean.empty()) throw std::runtime_error("nothing trained or loaded to evaluate");
            print("eval: student", rollout(c, true, uint64_t(eval_seed), eval_envs, nullptr));
            if (eval_teacher) print("eval: teacher", rollout(c, false, uint64_t(eval_seed), eval_envs, nullptr));
        }
        cudaStreamDestroy(c.stream);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
