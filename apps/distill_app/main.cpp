// distill_app (Phase 10): DAgger distillation of the OpenArm VLA (the teacher: an
// ONNX graph from tools/vla_openarm.py, run by ONNX Runtime) into the small CNN + GRU
// student (temper/algos/student: 64 x 64 images, trained in our engine).
//
//   ./distill_app --teacher runs/oa_vla12_d1.onnx --out runs/student.bin
//   ./distill_app --load runs/student.bin --rounds 0 --eval_envs 500      # closed loop only
//
// Decisions: every --exec (3) env steps the driver takes fresh camera images and
// plans an action chunk (the teacher's length, 12 steps); the robot runs the first
// --exec actions of it, then decides again. The student's GRU steps once per
// decision: its memory carries from one decision to the next. The student sees the
// cell's cameras --student_cameras (0,1,2: head, left wrist, ceiling) at 64 x 64
// (the 128 x 128 renders halved) and the arm's state (obs 0-23); its head gives the
// whole chunk (Tanh, chunk x 5 numbers).
// Rounds (--rounds, default 6), each on new starts (seed --data_seed + round):
//   round 0: the TEACHER drives --teacher_envs (1024) envs, one episode each;
//   later rounds: the STUDENT drives --envs (512) envs (DAgger)
// and at every decision the teacher labels the state with the chunk it plans there
// (one call, fresh flow noise: one plan from one state). Each decision is kept (the
// state, the label chunk, the student's images; 37 KB): round 0's in the teacher
// buffer (up to --teacher_frames decisions, 100 000: 3.7 GB of RAM; kept for good),
// the student rounds' in the DAgger buffer (--dagger_frames, 100 000; oldest
// episodes dropped). A failed student episode is long (it runs to the step limit):
// so that the student's failures do not crowd out the teacher's work, every window
// comes from the teacher buffer with probability --teacher_frac (0.5; 1 while the
// DAgger buffer is empty), and every episode of a buffer is picked equally often,
// whatever its length.
// Training: after each round --train_steps (1500) Adam steps (lr --lr 3e-4,
// gradient norm clipped at 1) on --batch (32) windows of --seq (16) decisions,
// through the GRU (BPTT; the GRU only ever uses the current and earlier decisions):
// MSE of the chunk vs the teacher's, on every decision of a window that starts at an
// episode's start, else after the first --burn_in (5) decisions (the memory warmed
// up). Random shifts of 4 pixels. The state is normalized by the teacher buffer's
// mean / std. Each round prints how often its episodes ended done (cube in the tray,
// arm home). At the end: the student in --eval_envs (500) envs of the test starts
// (--eval_seed 1000, as vla_app); with --eval_teacher true also the teacher on the
// same starts and decisions every --exec steps.
// Writes --out (checkpoint: the student, its state mean / std, cameras, exec) after
// every round.
// --task openarm_stack (Phase 11): the two-arm tower task instead (state obs 0-47,
// 10 actions per step, default student cameras 0,1,3: head and both wrists; done =
// the tower built and both hands home, as vla_app). The info file's "task" line must
// match. Example:
//   ./distill_app --task openarm_stack --teacher runs/st5_d1_e1.onnx --exec 4 --out runs/st_student.bin
// --record FILE (with the eval): every eval episode of the student as a demo file
// (collect_app's format, no images): obs every step, "expert" = the teacher's action
// planned in that state (it labels every decision), "executed" = the student's action.
// Then: python tools/read_demos.py FILE --npz FILE.npz (joint positions: obs 0-17).
//   ./distill_app --task openarm_stack --teacher runs/st5_d3_e7.onnx --load runs/st_student.bin
//       --rounds 0 --eval_envs 4 --record runs/st_student_track.bin
// --aux_weight W (openarm_stack; default 1 there, 0 = off; pick-place: off): the
// student also guesses from its images where each cube is (x y z and the yaw as
// sin / cos of 4 x yaw: a cube looks the same every 90 degrees; 15 numbers, normalized
// by the teacher buffer's mean / std, from obs 48-68) at every decision, an MSE loss
// of weight W added to the chunk's. The exact cube positions make the CNN learn to
// find the cubes (without it the stack student copied the start, then froze: it never
// learned to see them). The guess is not used to act.
// Options: task (openarm_pick_place | openarm_stack), teacher, info, device (ONNX Runtime: cpu | cuda), student_cameras, exec,
// envs, teacher_envs, rounds, data_seed, teacher_frames, dagger_frames, teacher_frac,
// train_steps, batch, seq, burn_in, lr, seed, out, load, eval_envs, eval_seed,
// eval_teacher, record, aux_weight, backend (gpu | cpu env).
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
#include "../collect_app/stack_camera_rig.h"
#include "../vla_app/vla_model.h"
#include "forge/env/demo_file.h"
#include "forge/env/env.h"
#include "forge/env/tasks/openarm_pick_place.h"
#include "forge/env/tasks/openarm_stack.h"
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
using PickPlace = forge::OpenArmPickPlaceTask;
using Stack = forge::OpenArmStackTask;
static_assert(PickPlace::kCameras == Stack::kCameras, "both tasks: 4 cameras");

// The task's sizes (set once in main from --task).
struct Spec {
    bool stack = false;
    std::string name = PickPlace::kName;
    int act = 5, state = 24, obs = PickPlace::kObsDim, max_steps = PickPlace::kMaxEpisodeSteps;
};
Spec spec;
void set_task(const std::string& task) {
    if (task == PickPlace::kName) return;
    if (task != Stack::kName) throw std::runtime_error("task must be openarm_pick_place or openarm_stack");
    spec = Spec{true, Stack::kName, 10, 48, Stack::kObsDim, Stack::kMaxEpisodeSteps};
}

// <teacher without .onnx>_openarm.txt (vla_openarm.py --onnx): "cameras 0 1", "text ..."
struct Info {
    std::vector<int> cameras;
    std::vector<float> text;
    std::string task;  // empty in files from before openarm_stack (then openarm_pick_place)
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
        } else if (key == "task") {
            ss >> info.task;
        } else if (key == "text") {
            float v;
            while (ss >> v) info.text.push_back(v);
        }
    }
    if (info.cameras.empty() || info.text.empty()) throw std::runtime_error(path + ": needs 'cameras' and 'text'");
    const std::string task = info.task.empty() ? std::string(PickPlace::kName) : info.task;
    if (task != spec.name)
        throw std::runtime_error(path + ": a teacher for " + task + ", not " + spec.name + " (use --task " + task + ")");
    return info;
}

std::vector<int> parse_cameras(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const int c = std::stoi(tok);
        if (c < 0 || c >= PickPlace::kCameras) throw std::runtime_error("camera ids are 0-3");
        out.push_back(c);
    }
    if (out.empty()) throw std::runtime_error("student_cameras: give at least one camera id");
    return out;
}

// pick-place obs: TCP z 20, cube z 26, cube - tray center 36-37 (vla_app's rule)
// stack obs: red 48, green 55, blue 62 (xyz first); TCP z 38, 44 (vla_app's rule)
bool done_ok(const float* fo, bool terminated) {
    if (!terminated) return false;
    if (!spec.stack)
        return std::fabs(fo[36]) < 0.06f && std::fabs(fo[37]) < 0.06f && fo[26] < 0.031f && fo[20] > 0.08f;
    const float *r = fo + 48, *g = fo + 55, *b = fo + 62;
    auto on = [](const float* x, const float* u) {
        return std::hypot(x[0] - u[0], x[1] - u[1]) < 0.018f && std::fabs(x[2] - u[2] - 0.04f) < 0.005f;
    };
    return std::hypot(r[0] - 0.41f, r[1]) < 0.025f && r[2] < 0.025f && on(g, r) && on(b, g) && fo[38] > 0.12f &&
           fo[44] > 0.12f;
}

// The aux targets of one obs (openarm_stack): per cube x y z, sin and cos of 4 x yaw.
constexpr int kStackAux = 15;
void stack_aux(const float* o, float* out) {
    for (int c = 0; c < 3; ++c) {
        const float* q = o + 48 + 7 * c;  // x y z, quat w x y z
        const float yaw = std::atan2(2.0f * (q[3] * q[6] + q[4] * q[5]), 1.0f - 2.0f * (q[5] * q[5] + q[6] * q[6]));
        float* a = out + 5 * c;
        a[0] = q[0], a[1] = q[1], a[2] = q[2];
        a[3] = std::sin(4.0f * yaw), a[4] = std::cos(4.0f * yaw);
    }
}

// The camera rig of the task.
struct Rig {
    Rig(const forge::RobotModel& m, int n, int size) {
        const std::string cmesh = std::string(CRUCIBLE_ASSET_DIR) + "/robots/openarm/openarm_cell.cmesh";
        if (spec.stack) st = std::make_unique<StackCameraRig>(m, n, size, FLUX_SHADER_DIR, cmesh);
        else pp = std::make_unique<OpenArmCameraRig>(m, n, size, FLUX_SHADER_DIR, cmesh);
    }
    void render(const forge::Env::HostCopy& h, int D) {
        if (st) st->render(h.qpos.data());
        else pp->render(h.qpos.data(), h.obs.data(), D);
    }
    const flux::BatchCamera& camera() const { return st ? st->camera() : pp->camera(); }
    std::unique_ptr<OpenArmCameraRig> pp;
    std::unique_ptr<StackCameraRig> st;
};

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

// One episode's decisions.
struct Episode {
    int T = 0;                        // decisions
    std::vector<float> state, label;  // [T][spec.state] raw, [T][chunk * spec.act]
    std::vector<float> aux;           // [T][aux_dim] raw (aux_dim 0: empty)
    std::vector<uint8_t> img;         // [T][student cams][s][s][3]
};

struct Buffer {
    std::deque<Episode> eps;
    size_t frames = 0, cap = 100000;  // decisions
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
    std::vector<int> scams;  // the student's cameras
    int exec = 3;            // env steps per decision
    temper::StudentPolicy* student = nullptr;
    std::vector<float> mean, stdev;  // state normalization
    int aux_dim = 0;                 // the student's aux head (0: none)
    float aux_weight = 0.0f;         // its loss weight in training (0: not trained)
    std::vector<float> aux_mean, aux_std;
    forge::EnvBackend backend = forge::EnvBackend::kGpu;
    cudaStream_t stream = nullptr;
    std::mt19937_64 rng{1};
};

struct RolloutResult {
    int episodes = 0, done = 0;
    double steps_done = 0, seconds = 0;
};

// Every env runs one episode from its start. Driver: the student (student_acts) or
// the teacher; a decision every c.exec steps. The teacher labels every decision
// when 'rec' or 'track' is set (or plans, when it drives). track: every episode, every
// step to a demo file (obs, the teacher's action, the executed action).
RolloutResult rollout(Ctx& c, bool student_acts, uint64_t seed, int envs, Buffer* rec,
                      forge::DemoWriter* track = nullptr) {
    forge::EnvOptions eo;
    eo.task = spec.name;
    eo.num_envs = envs;
    eo.seed = seed;
    eo.backend = c.backend;
    eo.physics_dt = 0.002f;  // configs/openarm_pick_place.txt, openarm_stack.txt (the same)
    eo.decimation = 10;
    eo.asset_dir = CRUCIBLE_ASSET_DIR;
    auto env = forge::make_env(eo);
    const int N = env->num_envs(), D = env->obs_dim();
    const bool gpu = c.backend == forge::EnvBackend::kGpu;
    const int TS = c.teacher->size, SS = TS / 2, TC = int(c.info.cameras.size()), SC = int(c.scams.size());
    const int C = c.teacher->chunk, E = c.teacher->text_dim, CA = C * spec.act;
    const bool need_teacher = rec || track || !student_acts;
    Rig rig(env->model(), N, TS);
    const size_t tone = size_t(TS) * TS * 3, sone = size_t(SS) * SS * 3;
    std::vector<uint8_t> full(tone), timg(size_t(N) * TC * tone), simg(size_t(N) * SC * sone);
    std::vector<float> st(size_t(N) * spec.state), tx(size_t(N) * E), noise(size_t(N) * CA), tchunk(size_t(N) * CA),
        plan(size_t(N) * CA), act(size_t(N) * spec.act), nst(size_t(N) * spec.state);
    for (int e = 0; e < N; ++e) std::copy(c.info.text.begin(), c.info.text.end(), tx.begin() + size_t(e) * E);
    const int H = c.student->config().hidden;
    forge::DeviceBuffer<float> dact(gpu ? size_t(N) * spec.act : 0), dstate(size_t(N) * spec.state), hid(size_t(N) * H);
    forge::DeviceBuffer<uint8_t> dimg(size_t(N) * SC * sone);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<char> ended(N, 0);
    std::vector<Episode> cur(rec ? N : 0);
    std::vector<forge::DemoEpisode> tr(track ? N : 0);
    RolloutResult r;
    const auto t0 = Clock::now();
    forge::Env::HostCopy h;
    env->download(h);
    for (int t = 0, d = 0; r.episodes < N && t < 2 * spec.max_steps; ++t) {
        const int k = t % c.exec;
        if (k == 0) {  // a decision: fresh images, a new chunk
            rig.render(h, D);
            for (int e = 0; e < N; ++e) {
                for (int j = 0; j < TC; ++j)
                    rig.camera().copy_rgb(e, c.info.cameras[j], timg.data() + (size_t(e) * TC + j) * tone);
                for (int j = 0; j < SC; ++j) {
                    rig.camera().copy_rgb(e, c.scams[j], full.data());
                    halve(full.data(), TS, simg.data() + (size_t(e) * SC + j) * sone);
                }
            }
            for (int e = 0; e < N; ++e)
                std::copy(h.obs.begin() + size_t(e) * D, h.obs.begin() + size_t(e) * D + spec.state,
                          st.begin() + size_t(e) * spec.state);
            if (need_teacher) {
                for (auto& z : noise) z = gauss(c.rng);
                c.teacher->run(N, timg.data(), st.data(), tx.data(), noise.data(), tchunk.data());
            }
            if (student_acts) {
                for (size_t i = 0; i < st.size(); ++i) nst[i] = (st[i] - c.mean[i % spec.state]) / c.stdev[i % spec.state];
                CUDA_CHECK(
                    cudaMemcpyAsync(dstate.data(), nst.data(), nst.size() * 4, cudaMemcpyHostToDevice, c.stream));
                CUDA_CHECK(cudaMemcpyAsync(dimg.data(), simg.data(), simg.size(), cudaMemcpyHostToDevice, c.stream));
                const float* h0 = d == 0 ? nullptr : hid.data();
                const float* a = c.student->forward(dstate.data(), dimg.data(), nullptr, nullptr, h0, 1, N, c.stream);
                CUDA_CHECK(cudaMemcpyAsync(plan.data(), a, plan.size() * 4, cudaMemcpyDeviceToHost, c.stream));
                CUDA_CHECK(cudaMemcpyAsync(hid.data(), c.student->last_hidden(), hid.size() * 4,
                                           cudaMemcpyDeviceToDevice, c.stream));
                CUDA_CHECK(cudaStreamSynchronize(c.stream));
            } else {
                plan = tchunk;
            }
            if (rec)
                for (int e = 0; e < N; ++e) {
                    if (ended[e]) continue;
                    Episode& ep = cur[e];
                    ep.state.insert(ep.state.end(), st.begin() + size_t(e) * spec.state,
                                    st.begin() + size_t(e + 1) * spec.state);
                    ep.label.insert(ep.label.end(), tchunk.begin() + size_t(e) * CA,
                                    tchunk.begin() + size_t(e + 1) * CA);
                    ep.img.insert(ep.img.end(), simg.begin() + size_t(e) * SC * sone,
                                  simg.begin() + size_t(e + 1) * SC * sone);
                    if (c.aux_dim > 0) {
                        float a[kStackAux];
                        stack_aux(h.obs.data() + size_t(e) * D, a);
                        ep.aux.insert(ep.aux.end(), a, a + kStackAux);
                    }
                    ++ep.T;
                }
            ++d;
        }
        for (int e = 0; e < N; ++e)  // action k of the current chunk
            std::copy(plan.begin() + size_t(e) * CA + size_t(k) * spec.act,
                      plan.begin() + size_t(e) * CA + size_t(k + 1) * spec.act, act.begin() + size_t(e) * spec.act);
        if (track)
            for (int e = 0; e < N; ++e) {
                if (ended[e]) continue;
                auto& ep = tr[e];
                const auto o = h.obs.begin() + size_t(e) * D;
                ep.obs.insert(ep.obs.end(), o, o + D);
                const auto x = tchunk.begin() + size_t(e) * CA + size_t(k) * spec.act;
                ep.expert.insert(ep.expert.end(), x, x + spec.act);
                const auto a = act.begin() + size_t(e) * spec.act;
                ep.executed.insert(ep.executed.end(), a, a + spec.act);
            }
        if (gpu) {
            dact.upload(act);
            env->step(dact.data());
        } else {
            env->step(act.data());
        }
        env->download(h);
        for (int e = 0; e < N; ++e) {
            if (ended[e]) continue;
            if (track) tr[e].reward.push_back(h.reward[e]);
            if (!(h.terminated[e] || h.truncated[e])) continue;
            ended[e] = 1;
            ++r.episodes;
            const float* fo = h.final_obs.data() + size_t(e) * D;
            const bool ok = done_ok(fo, h.terminated[e] != 0);
            if (ok) {
                ++r.done;
                r.steps_done += t + 1;
            }
            if (track) {
                tr[e].final_obs.assign(fo, fo + D);
                tr[e].success = ok;
                track->write(tr[e]);
                tr[e] = forge::DemoEpisode{};
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
    std::vector<float> state, label, mask, aux;
    std::vector<uint8_t> img;
    std::vector<int> shifts;
    forge::DeviceBuffer<float> dstate, dlabel, dmask, dy, loss, count, daux, dauxy, aux_loss;
    forge::DeviceBuffer<uint8_t> dimg;
    forge::DeviceBuffer<int> dshifts;
    TrainBatch(int L, int B, int out, int cams, size_t frame, int aux_dim) {
        const size_t n = size_t(L) * B;
        state.resize(n * spec.state), label.resize(n * out), mask.resize(n), img.resize(n * frame);
        aux.resize(n * aux_dim);
        shifts.resize(n * cams * 2);
        dstate.resize(state.size()), dlabel.resize(label.size()), dmask.resize(n), dy.resize(label.size());
        loss.resize(1), count.resize(1), dimg.resize(img.size()), dshifts.resize(shifts.size());
        daux.resize(aux.size()), dauxy.resize(aux.size()), aux_loss.resize(1);
    }
};

// Episodes of a buffer with at least L decisions, each picked equally often.
struct Picker {
    std::vector<const Episode*> eps;
    Picker(const Buffer& b, int L) {
        for (const auto& e : b.eps)
            if (e.T >= L) eps.push_back(&e);
    }
    const Episode& pick(std::mt19937_64& rng) const {
        return *eps[std::uniform_int_distribution<size_t>(0, eps.size() - 1)(rng)];
    }
};

struct TrainLoss {
    double chunk = 0, aux = 0;  // means over the steps
};

// 'steps' Adam steps on windows of the two buffers. Returns the mean losses.
TrainLoss train(Ctx& c, const Buffer& teach, const Buffer& dag, float teacher_frac, temper::Adam& adam,
                temper::GradClip& clip, TrainBatch& tb, int steps, int L, int B, int burn_in) {
    const Picker pt(teach, L), pd(dag, L);
    if (pt.eps.empty() && pd.eps.empty()) throw std::runtime_error("no episode has --seq decisions");
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const int S = c.student->config().size, SC = int(c.scams.size()), out = c.student->config().act_dim;
    const size_t frame = size_t(SC) * S * S * 3;
    std::uniform_int_distribution<int> shift(0, 2 * temper::StudentPolicy::kShiftPad);
    const bool use_aux = c.aux_dim > 0 && c.aux_weight > 0.0f;
    const int X = c.aux_dim;
    double sum = 0, aux_sum = 0;
    for (int it = 0; it < steps; ++it) {
        float masked = 0;
        for (int b = 0; b < B; ++b) {
            const bool from_teacher = pd.eps.empty() || (!pt.eps.empty() && u01(c.rng) < teacher_frac);
            const Episode& ep = (from_teacher ? pt : pd).pick(c.rng);
            const int s0 = std::uniform_int_distribution<int>(0, ep.T - L)(c.rng);
            for (int k = 0; k < L; ++k) {
                const size_t row = size_t(k) * B + b, t = size_t(s0 + k);
                for (int j = 0; j < spec.state; ++j)
                    tb.state[row * spec.state + j] = (ep.state[t * spec.state + j] - c.mean[j]) / c.stdev[j];
                std::memcpy(&tb.label[row * out], &ep.label[t * out], size_t(out) * sizeof(float));
                std::memcpy(&tb.img[row * frame], &ep.img[t * frame], frame);
                if (use_aux)
                    for (int j = 0; j < X; ++j)
                        tb.aux[row * X + j] = (ep.aux[t * X + j] - c.aux_mean[j]) / c.aux_std[j];
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
        if (use_aux) {
            up(tb.daux.data(), tb.aux.data(), tb.aux.size() * 4);
            CUDA_CHECK(cudaMemsetAsync(tb.aux_loss.data(), 0, 4, c.stream));
        }
        const float* y = c.student->forward(tb.dstate.data(), tb.dimg.data(), tb.dshifts.data(), nullptr, nullptr, L, B,
                                            c.stream);
        temper::student::masked_mse(y, tb.dlabel.data(), tb.dmask.data(), tb.count.data(), L * B, out, tb.dy.data(),
                                    tb.loss.data(), c.stream);
        if (use_aux)
            temper::student::masked_mse(c.student->aux(), tb.daux.data(), tb.dmask.data(), tb.count.data(), L * B, X,
                                        tb.dauxy.data(), tb.aux_loss.data(), c.stream);
        c.student->backward(tb.dy.data(), c.stream, use_aux ? tb.dauxy.data() : nullptr, c.aux_weight);
        clip.compute(1.0f, c.stream);
        adam.step(c.stream, clip.scale());
        float l = 0, la = 0;
        CUDA_CHECK(cudaMemcpyAsync(&l, tb.loss.data(), 4, cudaMemcpyDeviceToHost, c.stream));
        if (use_aux) CUDA_CHECK(cudaMemcpyAsync(&la, tb.aux_loss.data(), 4, cudaMemcpyDeviceToHost, c.stream));
        CUDA_CHECK(cudaStreamSynchronize(c.stream));  // also: the host buffers are free again
        sum += l;
        aux_sum += la;
    }
    return TrainLoss{sum / std::max(1, steps), aux_sum / std::max(1, steps)};
}

void save(Ctx& c, const std::string& path, int round) {
    temper::Checkpoint k;
    k.algo = "student";
    k.task = spec.name;
    k.obs_dim = spec.obs;
    k.act_dim = spec.act;
    k.iteration = round;
    c.student->save(k);
    k.layer_sizes["student_run"] = c.scams;  // the cameras, then exec
    k.layer_sizes["student_run"].push_back(c.exec);
    k.arrays["state_mean"] = c.mean;
    k.arrays["state_std"] = c.stdev;
    if (c.aux_dim > 0) {
        k.arrays["aux_mean"] = c.aux_mean;
        k.arrays["aux_std"] = c.aux_std;
    }
    k.save(path);
}

void print(const char* who, const RolloutResult& r) {
    std::printf("%s: done (%s) %d / %d (%.1f%%), mean %.0f steps (%.0f s)\n", who,
                spec.stack ? "tower built, hands home" : "cube in the tray, arm home", r.done,
                r.episodes, 100.0 * r.done / std::max(1, r.episodes), r.steps_done / std::max(1, r.done), r.seconds);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        forge::Config cfg = forge::Config::from_cli(argc, argv);
        set_task(cfg.get_string("task", PickPlace::kName));
        const std::string teacher_path = cfg.get_string("teacher", "runs/oa_vla12_d1.onnx");
        const size_t dot = teacher_path.rfind(".onnx");
        const std::string base = dot == std::string::npos ? teacher_path : teacher_path.substr(0, dot);
        const std::string info_path = cfg.get_string("info", base + "_openarm.txt");
        const std::string device = cfg.get_string("device", "cpu"), backend = cfg.get_string("backend", "gpu");
        std::vector<int> scams = parse_cameras(cfg.get_string("student_cameras", spec.stack ? "0,1,3" : "0,1,2"));
        int exec = cfg.get_int("exec", 3);
        const int envs = cfg.get_int("envs", 512), teacher_envs = cfg.get_int("teacher_envs", 1024);
        const int rounds = cfg.get_int("rounds", 6), data_seed = cfg.get_int("data_seed", 5000);
        const int teacher_frames = cfg.get_int("teacher_frames", 100000);
        const int dagger_frames = cfg.get_int("dagger_frames", 100000);
        const float teacher_frac = cfg.get_float("teacher_frac", 0.5f);
        const int train_steps = cfg.get_int("train_steps", 1500), B = cfg.get_int("batch", 32);
        const int L = cfg.get_int("seq", 16), burn_in = cfg.get_int("burn_in", 5);
        const float lr = cfg.get_float("lr", 3e-4f);
        const int seed = cfg.get_int("seed", 1);
        const std::string out = cfg.get_string("out", "runs/student.bin"), load = cfg.get_string("load", "");
        const int eval_envs = cfg.get_int("eval_envs", 500), eval_seed = cfg.get_int("eval_seed", 1000);
        const bool eval_teacher = cfg.get_bool("eval_teacher", false);
        const std::string record = cfg.get_string("record", "");
        const bool aux_given = cfg.has("aux_weight");
        float aux_weight = cfg.get_float("aux_weight", spec.stack ? 1.0f : 0.0f);
        for (const auto& k : cfg.unused_keys())
            std::fprintf(stderr, "warning: unknown option '%s' (typo?)\n", k.c_str());
        if (backend != "gpu" && backend != "cpu") throw std::runtime_error("backend must be gpu or cpu");
        if (!record.empty() && eval_envs <= 0) throw std::runtime_error("--record needs --eval_envs > 0");
        if (aux_weight < 0.0f) throw std::runtime_error("aux_weight must be >= 0");
        if (aux_weight > 0.0f && !spec.stack) throw std::runtime_error("--aux_weight: only for openarm_stack");
        if (device != "cpu" && device != "cuda") throw std::runtime_error("device must be cpu or cuda");
        if (envs <= 0 || teacher_envs <= 0 || rounds < 0 || B <= 0 || L <= 0 || burn_in < 0 || burn_in >= L)
            throw std::runtime_error("bad sizes (envs, teacher_envs, rounds, batch, seq, burn_in < seq)");

        Ctx c;
        c.info = read_info(info_path);
        vla::Model teacher(teacher_path, device == "cuda");
        if (int(c.info.cameras.size()) != teacher.cams || teacher.size % 2 != 0)
            throw std::runtime_error("the teacher's cameras do not match its info file");
        if (int(c.info.text.size()) != teacher.text_dim) throw std::runtime_error("text vector size does not match");
        if (teacher.state_dim != spec.state || teacher.act != spec.act)
            throw std::runtime_error("the teacher takes state " + std::to_string(teacher.state_dim) + " and gives " +
                                     std::to_string(teacher.act) + " actions; " + spec.name + " needs " +
                                     std::to_string(spec.state) + " and " + std::to_string(spec.act));
        c.teacher = &teacher;
        c.backend = backend == "gpu" ? forge::EnvBackend::kGpu : forge::EnvBackend::kCpu;
        c.rng.seed(uint64_t(seed));
        CUDA_CHECK(cudaStreamCreate(&c.stream));
        const int max_tb = std::max({envs, teacher_envs, eval_envs, L * B});
        std::unique_ptr<temper::StudentPolicy> student;
        if (!load.empty() && !aux_given) {  // --load: the default follows the checkpoint
            const temper::StudentConfig lc = temper::StudentPolicy::config_from(temper::Checkpoint::load(load));
            if (lc.aux_dim == 0) aux_weight = 0.0f;
        }
        temper::StudentConfig sc{spec.state, int(scams.size()), teacher.size / 2, teacher.chunk * spec.act, 256,
                                 aux_weight > 0.0f ? kStackAux : 0};
        if (!load.empty()) {
            const temper::Checkpoint k = temper::Checkpoint::load(load);
            sc = temper::StudentPolicy::config_from(k);
            auto it = k.layer_sizes.find("student_run");
            if (it == k.layer_sizes.end() || it->second.size() < 2) throw std::runtime_error(load + ": no student_run");
            scams.assign(it->second.begin(), it->second.end() - 1);
            exec = it->second.back();
            student = std::make_unique<temper::StudentPolicy>(sc, max_tb, uint64_t(seed));
            student->load(k);
            c.mean = k.array("state_mean");
            c.stdev = k.array("state_std");
            if (sc.aux_dim > 0) {
                c.aux_mean = k.array("aux_mean");
                c.aux_std = k.array("aux_std");
            }
            if (aux_weight > 0.0f && sc.aux_dim != kStackAux)
                throw std::runtime_error(load + ": this student has no aux head (use --aux_weight 0)");
        } else {
            student = std::make_unique<temper::StudentPolicy>(sc, max_tb, uint64_t(seed));
        }
        if (sc.state_dim != spec.state) throw std::runtime_error("the student is not for " + spec.name);
        if (sc.act_dim != teacher.chunk * spec.act || sc.size != teacher.size / 2)
            throw std::runtime_error("the student's chunk or image size does not fit this teacher");
        if (exec < 1 || exec > teacher.chunk) throw std::runtime_error("exec must be 1 .. the chunk length");
        c.scams = scams;
        c.exec = exec;
        c.aux_dim = sc.aux_dim;
        c.aux_weight = sc.aux_dim > 0 ? aux_weight : 0.0f;
        c.student = student.get();
        std::string cams_txt;
        for (int k : scams) cams_txt += (cams_txt.empty() ? "" : ",") + std::to_string(k);
        std::printf("distill_app: teacher %s (cameras %d..., %dx%d, chunk %d, ONNX Runtime on %s); student CNN + GRU "
                    "%.2f M parameters (cameras %s at %dx%d, GRU %d, chunk %d); a decision every %d steps%s\n",
                    teacher_path.c_str(), c.info.cameras[0], teacher.size, teacher.size, teacher.chunk, device.c_str(),
                    double(student->num_params()) / 1e6, cams_txt.c_str(), sc.size, sc.size, sc.hidden,
                    sc.act_dim / spec.act, exec, load.empty() ? "" : (", from " + load).c_str());
        if (c.aux_weight > 0.0f)
            std::printf("distill_app: aux head: the cubes' x y z, sin / cos of 4 x yaw from the images (%d numbers), "
                        "loss weight %.2f\n",
                        c.aux_dim, double(c.aux_weight));
        std::printf("distill_app: %d rounds (round 0: the teacher drives %d envs; then the student %d), %d Adam steps "
                    "of %d x %d-decision windows (burn-in %d, %.0f%% from the teacher's episodes); buffers %d + %d "
                    "decisions\n",
                    rounds, teacher_envs, envs, train_steps, B, L, burn_in, 100.0 * teacher_frac, teacher_frames,
                    dagger_frames);
        std::fflush(stdout);

        std::vector<temper::Tensor*> P, G;
        student->params(P, G);
        temper::Adam adam(P, G, temper::AdamOptions{lr});
        temper::GradClip clip(G);
        TrainBatch tb(L, B, sc.act_dim, sc.cams, size_t(sc.cams) * sc.size * sc.size * 3, sc.aux_dim);
        Buffer teach, dag;
        teach.cap = size_t(teacher_frames);
        dag.cap = size_t(dagger_frames);
        for (int round = 0; round < rounds; ++round) {
            const bool student_acts = round > 0 || !load.empty();
            const RolloutResult r = rollout(c, student_acts, uint64_t(data_seed + round),
                                            student_acts ? envs : teacher_envs, student_acts ? &dag : &teach);
            char who[96];
            std::snprintf(who, sizeof(who), "round %d, %s drives", round, student_acts ? "the student" : "the teacher");
            print(who, r);
            if (c.mean.empty()) {  // state normalization from the teacher's decisions
                std::vector<double> m(spec.state, 0.0), v(spec.state, 0.0);
                size_t n = 0;
                for (const auto& ep : teach.eps)
                    for (int t = 0; t < ep.T; ++t, ++n)
                        for (int j = 0; j < spec.state; ++j) m[j] += ep.state[size_t(t) * spec.state + j];
                for (auto& x : m) x /= double(std::max<size_t>(1, n));
                for (const auto& ep : teach.eps)
                    for (int t = 0; t < ep.T; ++t)
                        for (int j = 0; j < spec.state; ++j) {
                            const double q = ep.state[size_t(t) * spec.state + j] - m[j];
                            v[j] += q * q;
                        }
                c.mean.resize(spec.state);
                c.stdev.resize(spec.state);
                for (int j = 0; j < spec.state; ++j) {
                    c.mean[j] = float(m[j]);
                    c.stdev[j] = float(std::sqrt(v[j] / double(std::max<size_t>(1, n))) + 1e-6);
                }
            }
            if (c.aux_dim > 0 && c.aux_mean.empty()) {  // aux normalization, the same way
                const int X = c.aux_dim;
                std::vector<double> m(X, 0.0), v(X, 0.0);
                size_t n = 0;
                for (const auto& ep : teach.eps)
                    for (int t = 0; t < ep.T; ++t, ++n)
                        for (int j = 0; j < X; ++j) m[j] += ep.aux[size_t(t) * X + j];
                for (auto& x : m) x /= double(std::max<size_t>(1, n));
                for (const auto& ep : teach.eps)
                    for (int t = 0; t < ep.T; ++t)
                        for (int j = 0; j < X; ++j) {
                            const double q = ep.aux[size_t(t) * X + j] - m[j];
                            v[j] += q * q;
                        }
                c.aux_mean.resize(X);
                c.aux_std.resize(X);
                for (int j = 0; j < X; ++j) {
                    c.aux_mean[j] = float(m[j]);
                    c.aux_std[j] = float(std::sqrt(v[j] / double(std::max<size_t>(1, n))) + 1e-3);
                }
            }
            const auto t0 = Clock::now();
            const TrainLoss loss = train(c, teach, dag, teacher_frac, adam, clip, tb, train_steps, L, B, burn_in);
            char aux_txt[48] = "";
            if (c.aux_weight > 0.0f) std::snprintf(aux_txt, sizeof(aux_txt), ", aux loss %.4f", loss.aux);
            std::printf("round %d: buffers %zu + %zu episodes (%zu + %zu decisions); %d Adam steps, loss %.4f%s "
                        "(%.0f s)\n",
                        round, teach.eps.size(), dag.eps.size(), teach.frames, dag.frames, train_steps, loss.chunk,
                        aux_txt, since(t0));
            std::fflush(stdout);
            save(c, out, round);
        }
        if (rounds > 0) std::printf("wrote %s\n", out.c_str());
        if (eval_envs > 0) {
            if (c.mean.empty()) throw std::runtime_error("nothing trained or loaded to evaluate");
            std::unique_ptr<forge::DemoWriter> writer;
            if (!record.empty()) writer = std::make_unique<forge::DemoWriter>(record, spec.name, spec.obs, spec.act);
            print("eval: student", rollout(c, true, uint64_t(eval_seed), eval_envs, nullptr, writer.get()));
            if (writer)
                std::printf("recorded %d episodes (obs, the teacher's action, the student's action) to %s\n",
                            writer->episodes(), record.c_str());
            if (eval_teacher) print("eval: teacher", rollout(c, false, uint64_t(eval_seed), eval_envs, nullptr));
        }
        cudaStreamDestroy(c.stream);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
