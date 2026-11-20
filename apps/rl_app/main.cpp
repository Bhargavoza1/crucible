// rl_app (Phase 10c, 11): PPO fine-tuning of the OpenArm CNN + GRU student (from
// distill_app) on the task's reward. The task is the checkpoint's:
//   openarm_pick_place (Phase 10): the sparse reward (openarm_pick_place.h: success
//     +20, centre up to +10, and the penalties: joint limits, right arm, table, push,
//     approach, slam, jerk); state obs 0-23, 5 actions per step.
//   openarm_stack (Phase 11): both arms (state obs 0-47, 10 actions per step, cameras
//     from the checkpoint: 0,1,3), its reward (openarm_stack.h: success, aligned, the
//     penalties and the dense progress potential); done = the tower built, both hands
//     home. Its episodes are long (about 440 steps when done): use --gamma 0.997.
//     A student with an aux head (distill_app) keeps it as it is (not trained here).
//
//   ./rl_app --load runs/student.bin --out runs/student_rl.bin
//   ./rl_app --load runs/student_rl.bin --iters 0 --eval_envs 500        # eval only
//   ./rl_app --load runs/st_student_z2_r29.bin --gamma 0.997 --out runs/st_rl.bin
//   --start_mix 0.15,0.2,0.15 (openarm_stack): the training envs' start curriculum
//     (openarm_stack.h, Env::set_task_input): the chances of red already placed, of red
//     and green placed, of a cube at the outer edge; the rest normal. The eval keeps
//     normal starts only (comparable with every eval before).
//
// The policy is the student as distill_app runs it: a decision every --exec steps
// (from the checkpoint: 3) with fresh camera images, the GRU steps once per decision,
// the head gives a chunk of actions and the robot runs the first exec of them. For
// RL the executed actions (exec x the task's action numbers) get Gaussian noise: action = the head's
// mean + std * noise, std a learned vector (start --init_std 0.1). The rest of the
// chunk is not used. Evaluation runs the means (no noise).
// The critic (asymmetric: it is only used in training) is an MLP 256-256 on the full
// state the student does not see (the whole obs: 43 numbers for pick-place, 79 for
// stack: the cubes, ...) and the episode's time (step / the task's max steps), so a
// time limit is the end of the episode for it too.
// Each iteration: a rollout of --horizon (32) decisions in each of --envs (256) envs
// (the envs keep running from one iteration to the next; one decision's reward is the
// sum of its steps' rewards; an episode that ends inside a decision's steps ends the
// decision there, the leftover steps of that decision hold the new episode's arm
// still with the grippers open), GAE (gamma --gamma 0.99 per step, so gamma^exec per
// decision; lambda --lambda 0.95), then --epochs (4) passes of PPO (clip --clip 0.2)
// over minibatches of --minibatch_envs (32) envs' whole sequences, the GRU run from
// the memory each env had at the start of the rollout (BPTT). Actor Adam lr --lr
// (2e-5), gradient norm clipped at --max_grad (0.5); a pass stops early when the
// mean KL passes 1.5 --target_kl (0.02). The critic: its own Adam (lr --critic_lr
// 3e-4), the same passes over all decisions. The first --critic_warmup (10)
// iterations train only the critic (the student does not move until its advantages
// make sense).
// Prints each iteration: episodes ended in the rollout, how many done (the task's
// rule, as distill_app), mean return, the reward terms per episode, losses, KL, std. Every
// --eval_every (25) iterations and at the end: the means in --eval_envs (500) envs
// of the test starts (--eval_seed 1000, as distill_app and vla_app); --out is written
// then, and <out>_best.bin when the eval is the best so far.
// Options: load, out, envs, horizon, iters, epochs, minibatch_envs, lr, critic_lr,
// gamma, lambda, clip, target_kl, max_grad, ent_coef, init_std, critic_warmup,
// eval_every, eval_envs, eval_seed, seed, start_mix, backend (gpu | cpu env),
// timing (true: a line per iteration with where the time went).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "../collect_app/openarm_camera_rig.h"
#include "../collect_app/stack_camera_rig.h"
#include "forge/env/env.h"
#include "forge/env/tasks/openarm_pick_place.h"
#include "forge/env/tasks/openarm_stack.h"
#include "forge/util/config.h"
#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"
#include "temper/algos/ppo/ppo_math.h"
#include "temper/algos/student/student_policy.h"
#include "temper/common/checkpoint.h"
#include "temper/nn/mlp.h"
#include "temper/optim/adam.h"

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
using PickPlace = forge::OpenArmPickPlaceTask;
using Stack = forge::OpenArmStackTask;

// The task's sizes (set once in main from the checkpoint's task).
struct Spec {
    bool stack = false;
    std::string name = PickPlace::kName;
    int act = 5, state = 24, obs = PickPlace::kObsDim, critic = PickPlace::kObsDim + 1;
    int max_steps = PickPlace::kMaxEpisodeSteps;
};
Spec spec;
void set_task(const std::string& task) {
    if (task == PickPlace::kName) return;
    if (task != Stack::kName) throw std::runtime_error("the student's task must be openarm_pick_place or openarm_stack");
    spec = Spec{true, Stack::kName, 10, 48, Stack::kObsDim, Stack::kObsDim + 1, Stack::kMaxEpisodeSteps};
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

// The new episode's actions until its first decision: hold still, grippers open.
void hold_still(float* x) {
    for (int j = 0; j < spec.act; ++j) x[j] = j % 5 == 4 ? -1.0f : 0.0f;
}

// 2 x 2 box filter: [S][S][3] -> [S/2][S/2][3] (distill_app's)
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

forge::EnvOptions env_options(int envs, uint64_t seed, forge::EnvBackend backend) {
    forge::EnvOptions eo;
    eo.task = spec.name;
    eo.num_envs = envs;
    eo.seed = seed;
    eo.backend = backend;
    eo.physics_dt = 0.002f;  // configs/openarm_pick_place.txt, openarm_stack.txt (the same)
    eo.decimation = 10;
    eo.asset_dir = CRUCIBLE_ASSET_DIR;
    return eo;
}

struct Ctx {
    temper::StudentPolicy* student = nullptr;
    std::vector<int> scams;
    int exec = 3;
    std::vector<float> mean, stdev;  // the student's state normalization
    std::vector<float> aux_mean, aux_std;  // the aux head's (kept as loaded; empty: none)
    forge::EnvBackend backend = forge::EnvBackend::kGpu;
    cudaStream_t stream = nullptr;
};

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
    double fk_seconds() const { return st ? st->fk_seconds : 0.0; }  // stack rig only (--timing)
    double draw_seconds() const { return st ? st->draw_seconds : 0.0; }
    std::unique_ptr<OpenArmCameraRig> pp;
    std::unique_ptr<StackCameraRig> st;
};

// The student's images (render at 2x, halve) and normalized state of all envs. The
// copy out of the atlas and the halving run on all CPU threads (each image is its
// own work: the same pixels as one thread, faster).
struct Eyes {
    Rig rig;
    int TS, SS, SC;
    size_t sone;
    int threads;
    std::vector<std::vector<uint8_t>> full;  // one 2x image per thread
    Eyes(const forge::RobotModel& m, int N, int size, int cams)
        : rig(m, N, 2 * size), TS(2 * size), SS(size), SC(cams), sone(size_t(size) * size * 3),
          threads(std::max(1, std::min({int(std::thread::hardware_concurrency()), 32, N}))),
          full(size_t(threads), std::vector<uint8_t>(size_t(4) * size * size * 3)) {}
    double render_seconds = 0, halve_seconds = 0;  // for --timing
    // images [N][SC][SS][SS][3] into img
    void look(const Ctx& c, const forge::Env::HostCopy& h, int N, int D, uint8_t* img) {
        const auto t0 = Clock::now();
        rig.render(h, D);
        const auto t1 = Clock::now();
        auto work = [&](int k) {  // envs [N k / threads, N (k + 1) / threads)
            uint8_t* buf = full[size_t(k)].data();
            for (int e = int(int64_t(N) * k / threads); e < int(int64_t(N) * (k + 1) / threads); ++e)
                for (int j = 0; j < SC; ++j) {
                    rig.camera().copy_rgb(e, c.scams[j], buf);
                    halve(buf, TS, img + (size_t(e) * SC + j) * sone);
                }
        };
        std::vector<std::thread> pool;
        for (int k = 1; k < threads; ++k) pool.emplace_back(work, k);
        work(0);
        for (auto& t : pool) t.join();
        render_seconds += std::chrono::duration<double>(t1 - t0).count();
        halve_seconds += since(t1);
    }
};

void normalize_state(const Ctx& c, const float* obs, int D, int N, float* out) {
    for (int e = 0; e < N; ++e)
        for (int j = 0; j < spec.state; ++j) out[size_t(e) * spec.state + j] = (obs[size_t(e) * D + j] - c.mean[j]) / c.stdev[j];
}

struct EvalResult {
    int episodes = 0, done = 0;
    double steps_done = 0, seconds = 0;
};

// Every env runs one episode from its start; the student's means drive (distill_app's
// closed loop).
EvalResult evaluate(Ctx& c, uint64_t seed, int envs) {
    auto env = forge::make_env(env_options(envs, seed, c.backend));
    const int N = env->num_envs(), D = env->obs_dim(), H = c.student->config().hidden;
    const int CA = c.student->config().act_dim;
    const bool gpu = c.backend == forge::EnvBackend::kGpu;
    Eyes eyes(env->model(), N, c.student->config().size, int(c.scams.size()));
    std::vector<uint8_t> img(size_t(N) * eyes.SC * eyes.sone);
    std::vector<float> st(size_t(N) * spec.state), plan(size_t(N) * CA), act(size_t(N) * spec.act);
    forge::DeviceBuffer<float> dact(gpu ? size_t(N) * spec.act : 0), dstate(st.size()), hid(size_t(N) * H);
    forge::DeviceBuffer<uint8_t> dimg(img.size());
    std::vector<char> ended(N, 0);
    EvalResult r;
    const auto t0 = Clock::now();
    forge::Env::HostCopy h;
    env->download(h);
    for (int t = 0, d = 0; r.episodes < N && t < 2 * spec.max_steps; ++t) {
        const int k = t % c.exec;
        if (k == 0) {
            eyes.look(c, h, N, D, img.data());
            normalize_state(c, h.obs.data(), D, N, st.data());
            CUDA_CHECK(cudaMemcpyAsync(dstate.data(), st.data(), st.size() * 4, cudaMemcpyHostToDevice, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(dimg.data(), img.data(), img.size(), cudaMemcpyHostToDevice, c.stream));
            const float* a = c.student->forward(dstate.data(), dimg.data(), nullptr, nullptr, d == 0 ? nullptr : hid.data(),
                                                1, N, c.stream);
            CUDA_CHECK(cudaMemcpyAsync(plan.data(), a, plan.size() * 4, cudaMemcpyDeviceToHost, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(hid.data(), c.student->last_hidden(), hid.size() * 4, cudaMemcpyDeviceToDevice,
                                       c.stream));
            CUDA_CHECK(cudaStreamSynchronize(c.stream));
            ++d;
        }
        for (int e = 0; e < N; ++e)
            std::copy(plan.begin() + size_t(e) * CA + size_t(k) * spec.act, plan.begin() + size_t(e) * CA + size_t(k + 1) * spec.act,
                      act.begin() + size_t(e) * spec.act);
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
        }
    }
    r.seconds = since(t0);
    return r;
}

// One rollout's decisions, time major: row t * N + e.
struct Rollout {
    int T = 0, N = 0, A = 0;
    std::vector<float> state, act, logp, cobs, value, reward, adv, ret, reset, last_cobs, last_value;
    std::vector<int> done;
    std::vector<uint8_t> img;
    forge::DeviceBuffer<float> h0;  // [N][hidden]: each env's memory before decision 0
    void resize(int T_, int N_, int A_, size_t frame, int H) {
        T = T_, N = N_, A = A_;
        const size_t n = size_t(T) * N;
        state.resize(n * spec.state), act.resize(n * A), logp.resize(n), cobs.resize(n * spec.critic), value.resize(n);
        reward.resize(n), adv.resize(n), ret.resize(n), reset.resize(n), done.resize(n), img.resize(n * frame);
        last_cobs.resize(size_t(N) * spec.critic), last_value.resize(N);
        h0.resize(size_t(N) * H);
    }
};

// Host Adam for the log std vector.
struct SmallAdam {
    std::vector<double> m, v;
    long long t = 0;
    void step(std::vector<float>& p, const std::vector<double>& g, float lr) {
        if (m.empty()) m.assign(p.size(), 0.0), v.assign(p.size(), 0.0);
        ++t;
        for (size_t i = 0; i < p.size(); ++i) {
            m[i] = 0.9 * m[i] + 0.1 * g[i];
            v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
            const double mh = m[i] / (1 - std::pow(0.9, double(t))), vh = v[i] / (1 - std::pow(0.999, double(t)));
            p[i] = float(p[i] - lr * mh / (std::sqrt(vh) + 1e-8));
        }
    }
};

void save(const Ctx& c, const std::string& path, int iter, const std::vector<float>& log_std) {
    temper::Checkpoint k;
    k.algo = "student";
    k.task = spec.name;
    k.obs_dim = spec.obs;
    k.act_dim = spec.act;
    k.iteration = iter;
    c.student->save(k);
    k.layer_sizes["student_run"] = c.scams;  // the cameras, then exec (distill_app's layout)
    k.layer_sizes["student_run"].push_back(c.exec);
    k.arrays["state_mean"] = c.mean;
    k.arrays["state_std"] = c.stdev;
    k.arrays["rl_log_std"] = log_std;
    if (!c.aux_mean.empty()) {  // the aux head's normalization (distill_app's), kept for --load there
        k.arrays["aux_mean"] = c.aux_mean;
        k.arrays["aux_std"] = c.aux_std;
    }
    k.save(path);
}

void print_eval(const char* who, const EvalResult& r) {
    std::printf("%s: done (%s) %d / %d (%.1f%%), mean %.0f steps (%.0f s)\n", who,
                spec.stack ? "tower built, hands home" : "cube in the tray, arm home", r.done,
                r.episodes, 100.0 * r.done / std::max(1, r.episodes), r.steps_done / std::max(1, r.done), r.seconds);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        forge::Config cfg = forge::Config::from_cli(argc, argv);
        const std::string load = cfg.get_string("load", "runs/student.bin");
        const std::string out = cfg.get_string("out", "runs/student_rl.bin");
        const int N = cfg.get_int("envs", 256), T = cfg.get_int("horizon", 32), iters = cfg.get_int("iters", 300);
        const int epochs = cfg.get_int("epochs", 4), Bm = cfg.get_int("minibatch_envs", 32);
        const float lr = cfg.get_float("lr", 2e-5f), critic_lr = cfg.get_float("critic_lr", 3e-4f);
        const float gamma_step = cfg.get_float("gamma", 0.99f), lambda = cfg.get_float("lambda", 0.95f);
        const float clip_eps = cfg.get_float("clip", 0.2f), target_kl = cfg.get_float("target_kl", 0.02f);
        const float max_grad = cfg.get_float("max_grad", 0.5f), ent_coef = cfg.get_float("ent_coef", 0.0f);
        const float init_std = cfg.get_float("init_std", 0.1f);
        const int critic_warmup = cfg.get_int("critic_warmup", 10), eval_every = cfg.get_int("eval_every", 25);
        const int eval_envs = cfg.get_int("eval_envs", 500), eval_seed = cfg.get_int("eval_seed", 1000);
        const int seed = cfg.get_int("seed", 1);
        const bool timing = cfg.get_bool("timing", false);
        const std::string backend = cfg.get_string("backend", "gpu");
        const std::string start_mix = cfg.get_string("start_mix", "");
        for (const auto& k : cfg.unused_keys()) std::fprintf(stderr, "warning: unknown option '%s' (typo?)\n", k.c_str());
        if (backend != "gpu" && backend != "cpu") throw std::runtime_error("backend must be gpu or cpu");
        if (N <= 0 || T <= 0 || epochs <= 0 || Bm <= 0 || N % Bm != 0)
            throw std::runtime_error("bad sizes (envs, horizon, epochs > 0; envs a multiple of minibatch_envs)");
        if (!(init_std > 0.0f)) throw std::runtime_error("init_std must be > 0");
        std::vector<float> mix;  // the start curriculum's 3 chances (empty: normal starts)
        if (!start_mix.empty()) {
            std::string tok;
            for (size_t i = 0; i <= start_mix.size(); ++i) {
                if (i == start_mix.size() || start_mix[i] == ',') {
                    mix.push_back(std::stof(tok));
                    tok.clear();
                } else {
                    tok += start_mix[i];
                }
            }
            float sum = 0.0f;
            for (float x : mix) sum += x;
            if (mix.size() != 3 || sum > 1.0f + 1e-4f || *std::min_element(mix.begin(), mix.end()) < 0.0f)
                throw std::runtime_error("start_mix: 3 chances >= 0, sum <= 1 (red placed, red + green, edge cube)");
        }

        Ctx c;
        c.backend = backend == "gpu" ? forge::EnvBackend::kGpu : forge::EnvBackend::kCpu;
        CUDA_CHECK(cudaStreamCreate(&c.stream));
        const temper::Checkpoint ck = temper::Checkpoint::load(load);
        const temper::StudentConfig sc = temper::StudentPolicy::config_from(ck);
        auto it = ck.layer_sizes.find("student_run");
        if (it == ck.layer_sizes.end() || it->second.size() < 2) throw std::runtime_error(load + ": no student_run");
        c.scams.assign(it->second.begin(), it->second.end() - 1);
        c.exec = it->second.back();
        set_task(ck.task);
        if (!mix.empty() && !spec.stack) throw std::runtime_error("start_mix: only for openarm_stack");
        c.mean = ck.array("state_mean");
        c.stdev = ck.array("state_std");
        if (ck.arrays.count("aux_mean")) {
            c.aux_mean = ck.array("aux_mean");
            c.aux_std = ck.array("aux_std");
        }
        if (sc.state_dim != spec.state || sc.act_dim % spec.act != 0 || c.exec * spec.act > sc.act_dim ||
            int(c.mean.size()) != spec.state)
            throw std::runtime_error(load + ": not an " + spec.name + " student (state " + std::to_string(spec.state) +
                                     ", a chunk of " + std::to_string(spec.act) + "-number actions)");
        const int A = c.exec * spec.act;  // the executed numbers of a chunk: the RL action
        const int max_tb = std::max({N, Bm * T, eval_envs});
        temper::StudentPolicy student(sc, max_tb, uint64_t(seed));
        student.load(ck);
        c.student = &student;
        std::vector<float> log_std(size_t(A), std::log(init_std));
        if (ck.arrays.count("rl_log_std") && ck.array("rl_log_std").size() == size_t(A)) log_std = ck.array("rl_log_std");
        const int H = sc.hidden, CA = sc.act_dim;
        std::string cams_txt;
        for (int k : c.scams) cams_txt += (cams_txt.empty() ? "" : ",") + std::to_string(k);
        std::printf("rl_app: %s student %s (%.2f M parameters, cameras %s at %dx%d, GRU %d, chunk %d, a decision every %d "
                    "steps); PPO on %d envs x %d decisions, %d epochs of %d-env minibatches, lr %.1e, critic lr %.1e, "
                    "std %.3f, critic warm-up %d iterations\n",
                    spec.name.c_str(), load.c_str(), double(student.num_params()) / 1e6, cams_txt.c_str(), sc.size, sc.size, H,
                    CA / spec.act, c.exec, N, T, epochs, Bm, double(lr), double(critic_lr), double(std::exp(log_std[0])),
                    critic_warmup);
        std::fflush(stdout);

        double best = -1.0;
        auto run_eval = [&](int iter) {
            if (eval_envs <= 0) return;
            const EvalResult r = evaluate(c, uint64_t(eval_seed), eval_envs);
            char who[64];
            std::snprintf(who, sizeof(who), "eval after %d iterations", iter);
            print_eval(who, r);
            if (iters == 0) return;
            save(c, out, iter, log_std);
            const double rate = double(r.done) / std::max(1, r.episodes);
            if (rate > best) {
                best = rate;
                const size_t dot = out.rfind(".bin");
                const std::string bp = (dot == std::string::npos ? out : out.substr(0, dot)) + "_best.bin";
                save(c, bp, iter, log_std);
                std::printf("wrote %s (best so far)\n", bp.c_str());
            }
        };
        run_eval(0);
        if (iters == 0) {
            cudaStreamDestroy(c.stream);
            return 0;
        }

        // training state
        auto env = forge::make_env(env_options(N, uint64_t(seed) * 7919 + 17, c.backend));
        if (!mix.empty()) {  // the start curriculum: from the first episodes on
            env->set_task_input(mix.data());
            env->reset_all();
            std::printf("rl_app: training starts: %.0f%% red placed, %.0f%% red + green placed, %.0f%% a cube at the "
                        "edge, the rest normal (the eval: normal)\n",
                        100.0 * mix[0], 100.0 * mix[1], 100.0 * mix[2]);
        }
        const int D = env->obs_dim();
        const bool gpu = c.backend == forge::EnvBackend::kGpu;
        Eyes eyes(env->model(), N, sc.size, int(c.scams.size()));
        const size_t frame = size_t(eyes.SC) * eyes.sone;
        const int cmb = std::max(1, N / Bm);  // critic minibatches per pass
        const size_t cbatch = (size_t(T) * N + cmb - 1) / cmb;
        temper::Mlp critic({spec.critic, 256, 256, 1}, std::max(N, int(cbatch)), uint64_t(seed) + 101);
        critic.init_orthogonal(1.0f, uint64_t(seed) + 102);
        std::vector<temper::Tensor*> P, G, CP, CG;
        student.params(P, G);
        critic.params(CP, CG);
        temper::Adam adam(P, G, temper::AdamOptions{lr});
        temper::Adam cadam(CP, CG, temper::AdamOptions{critic_lr});
        temper::GradClip clip(G), cclip(CG);
        SmallAdam std_adam;
        std::vector<float> cmean(spec.critic, 0.0f), cstd(spec.critic, 1.0f);  // critic input normalization (first rollout)
        bool cnorm = false;

        Rollout ro;
        ro.resize(T, N, A, frame, H);
        std::vector<uint8_t> img(size_t(N) * frame);
        std::vector<float> st(size_t(N) * spec.state), plan(size_t(N) * CA), act(size_t(N) * spec.act), chunk(size_t(N) * A);
        forge::DeviceBuffer<float> dact(gpu ? size_t(N) * spec.act : 0), dstate(st.size()), hid(size_t(N) * H);
        forge::DeviceBuffer<uint8_t> dimg(img.size());
        CUDA_CHECK(cudaMemset(hid.data(), 0, hid.size() * 4));
        std::vector<float> rs(N);
        forge::DeviceBuffer<float> drs{size_t(N)};
        std::vector<char> new_ep(N, 1), ended(N, 0);
        std::vector<int> ep_step(N, 0);
        std::vector<double> ep_ret(N, 0.0);
        std::mt19937_64 rng{uint64_t(seed)};
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        const float gamma = std::pow(gamma_step, float(c.exec));
        // training batch buffers
        const size_t nb = size_t(T) * Bm;
        std::vector<float> bstate(nb * spec.state), breset(nb), bmu(nb * CA), bdy(nb * CA, 0.0f);
        std::vector<uint8_t> bimg(nb * frame);
        forge::DeviceBuffer<float> dbstate(bstate.size()), dbreset(nb), dbdy(bdy.size()), dbh0(size_t(Bm) * H);
        forge::DeviceBuffer<uint8_t> dbimg(bimg.size());
        std::vector<float> ccobs(cbatch * spec.critic), cret(cbatch), cdv(cbatch), cval(cbatch);
        forge::DeviceBuffer<float> dccobs(ccobs.size()), dcdv(cbatch);
        auto critic_values = [&](const float* cobs_raw, size_t n, float* outv) {
            for (size_t s = 0; s < n; s += cbatch) {
                const size_t m = std::min(cbatch, n - s);
                for (size_t i = 0; i < m; ++i)
                    for (int j = 0; j < spec.critic; ++j)
                        ccobs[i * spec.critic + j] = (cobs_raw[(s + i) * spec.critic + j] - cmean[j]) / cstd[j];
                CUDA_CHECK(cudaMemcpyAsync(dccobs.data(), ccobs.data(), m * spec.critic * 4, cudaMemcpyHostToDevice, c.stream));
                const float* v = critic.forward(dccobs.data(), int(m), c.stream);
                CUDA_CHECK(cudaMemcpyAsync(outv + s, v, m * 4, cudaMemcpyDeviceToHost, c.stream));
                CUDA_CHECK(cudaStreamSynchronize(c.stream));
            }
        };

        forge::Env::HostCopy h;
        env->download(h);
        const auto& names = env->reward_term_names();
        env->read_reward_terms();
        for (int iter = 1; iter <= iters; ++iter) {
            const auto t0 = Clock::now();
            int episodes = 0, dones = 0;
            double ret_sum = 0;
            long long env_steps = 0;
            double t_policy = 0, t_physics = 0;  // --timing
            const double fk0 = eyes.rig.fk_seconds(), draw0 = eyes.rig.draw_seconds();
            const double render0 = eyes.render_seconds, halve0 = eyes.halve_seconds;
            CUDA_CHECK(cudaMemcpyAsync(ro.h0.data(), hid.data(), hid.size() * 4, cudaMemcpyDeviceToDevice, c.stream));
            // ---- rollout ----
            for (int t = 0; t < T; ++t) {
                const size_t row0 = size_t(t) * N;
                eyes.look(c, h, N, D, img.data());
                const auto tp = Clock::now();
                normalize_state(c, h.obs.data(), D, N, st.data());
                for (int e = 0; e < N; ++e) rs[e] = new_ep[e] ? 1.0f : 0.0f;
                drs.upload(rs);
                CUDA_CHECK(cudaMemcpyAsync(dstate.data(), st.data(), st.size() * 4, cudaMemcpyHostToDevice, c.stream));
                CUDA_CHECK(cudaMemcpyAsync(dimg.data(), img.data(), img.size(), cudaMemcpyHostToDevice, c.stream));
                const float* mu = student.forward(dstate.data(), dimg.data(), nullptr, drs.data(), hid.data(), 1, N, c.stream);
                CUDA_CHECK(cudaMemcpyAsync(plan.data(), mu, plan.size() * 4, cudaMemcpyDeviceToHost, c.stream));
                CUDA_CHECK(cudaMemcpyAsync(hid.data(), student.last_hidden(), hid.size() * 4, cudaMemcpyDeviceToDevice,
                                           c.stream));
                CUDA_CHECK(cudaStreamSynchronize(c.stream));
                for (int e = 0; e < N; ++e) {
                    const size_t row = row0 + size_t(e);
                    float* a = &ro.act[row * A];
                    const float* m = &plan[size_t(e) * CA];
                    for (int k = 0; k < A; ++k) a[k] = m[k] + std::exp(log_std[k]) * gauss(rng);
                    ro.logp[row] = temper::gaussian_logp(a, m, log_std.data(), A);
                    std::copy(a, a + A, chunk.begin() + size_t(e) * A);
                    std::copy(st.begin() + size_t(e) * spec.state, st.begin() + size_t(e + 1) * spec.state,
                              ro.state.begin() + row * spec.state);
                    for (int j = 0; j < spec.obs; ++j) ro.cobs[row * spec.critic + j] = h.obs[size_t(e) * D + j];
                    ro.cobs[row * spec.critic + spec.obs] = float(ep_step[e]) / float(spec.max_steps);
                    ro.reset[row] = rs[e];
                    ro.reward[row] = 0.0f;
                    ro.done[row] = 0;
                    new_ep[e] = 0;
                    ended[e] = 0;
                }
                std::memcpy(&ro.img[row0 * frame], img.data(), img.size());
                t_policy += since(tp);
                for (int k = 0; k < c.exec; ++k) {
                    for (int e = 0; e < N; ++e) {
                        float* x = &act[size_t(e) * spec.act];
                        if (ended[e]) {  // the new episode waits for its first decision: still, open
                            hold_still(x);
                        } else {
                            for (int j = 0; j < spec.act; ++j)
                                x[j] = std::fmin(std::fmax(chunk[size_t(e) * A + size_t(k) * spec.act + j], -1.0f), 1.0f);
                        }
                    }
                    const auto ts = Clock::now();
                    if (gpu) {
                        dact.upload(act);
                        env->step(dact.data());
                    } else {
                        env->step(act.data());
                    }
                    env->download(h);
                    t_physics += since(ts);
                    ++env_steps;
                    for (int e = 0; e < N; ++e) {
                        ++ep_step[e];
                        if (ended[e]) continue;
                        const size_t row = row0 + size_t(e);
                        ro.reward[row] += h.reward[e];
                        ep_ret[e] += h.reward[e];
                        if (h.terminated[e] || h.truncated[e]) {
                            ended[e] = 1;
                            new_ep[e] = 1;
                            ro.done[row] = 1;
                            ep_step[e] = 0;
                            ++episodes;
                            dones += done_ok(h.final_obs.data() + size_t(e) * D, h.terminated[e] != 0);
                            ret_sum += ep_ret[e];
                            ep_ret[e] = 0.0;
                        }
                    }
                }
            }
            for (int e = 0; e < N; ++e) {
                for (int j = 0; j < spec.obs; ++j) ro.last_cobs[size_t(e) * spec.critic + j] = h.obs[size_t(e) * D + j];
                ro.last_cobs[size_t(e) * spec.critic + spec.obs] = float(ep_step[e]) / float(spec.max_steps);
            }
            const double t_roll = since(t0);
            const std::vector<double> terms = env->read_reward_terms();  // mean per env-step since the last read
            if (!cnorm) {  // critic input normalization, from the first rollout
                const size_t n = size_t(T) * N;
                for (int j = 0; j < spec.critic; ++j) {
                    double m = 0, v = 0;
                    for (size_t i = 0; i < n; ++i) m += ro.cobs[i * spec.critic + j];
                    m /= double(n);
                    for (size_t i = 0; i < n; ++i) v += (ro.cobs[i * spec.critic + j] - m) * (ro.cobs[i * spec.critic + j] - m);
                    cmean[j] = float(m);
                    cstd[j] = float(std::max(std::sqrt(v / double(n)), 1e-2));
                }
                cnorm = true;
            }
            // ---- values, GAE ----
            critic_values(ro.cobs.data(), size_t(T) * N, ro.value.data());
            critic_values(ro.last_cobs.data(), size_t(N), ro.last_value.data());
            double adv_m = 0, adv_v = 0;
            for (int e = 0; e < N; ++e)
                temper::gae_env(&ro.reward[size_t(e)], &ro.done[size_t(e)], &ro.value[size_t(e)], ro.last_value[e], T, N,
                                gamma, lambda, &ro.adv[size_t(e)], &ro.ret[size_t(e)]);
            for (float a : ro.adv) adv_m += a;
            adv_m /= double(ro.adv.size());
            for (float a : ro.adv) adv_v += (a - adv_m) * (a - adv_m);
            const float adv_s = float(std::sqrt(adv_v / double(ro.adv.size())) + 1e-8);
            // ---- updates ----
            const bool train_actor = iter > critic_warmup;
            double pl_sum = 0, kl_sum = 0, cf_sum = 0, vl_sum = 0;
            int pl_n = 0, vl_n = 0, epochs_done = 0;
            std::vector<int> order(N);
            std::iota(order.begin(), order.end(), 0);
            std::vector<size_t> sorder(size_t(T) * N);
            std::iota(sorder.begin(), sorder.end(), size_t(0));
            for (int ep = 0; ep < epochs; ++ep) {
                double kl_ep = 0;
                int kl_n = 0;
                if (train_actor) {
                    std::shuffle(order.begin(), order.end(), rng);
                    for (int mb = 0; mb < N / Bm; ++mb) {
                        // batch: Bm envs' whole sequences, time major
                        for (int b = 0; b < Bm; ++b) {
                            const int e = order[size_t(mb) * Bm + b];
                            CUDA_CHECK(cudaMemcpyAsync(dbh0.data() + size_t(b) * H, ro.h0.data() + size_t(e) * H, size_t(H) * 4,
                                                       cudaMemcpyDeviceToDevice, c.stream));
                            for (int t = 0; t < T; ++t) {
                                const size_t src = size_t(t) * N + e, dst = size_t(t) * Bm + b;
                                std::memcpy(&bstate[dst * spec.state], &ro.state[src * spec.state], spec.state * 4);
                                std::memcpy(&bimg[dst * frame], &ro.img[src * frame], frame);
                                breset[dst] = ro.reset[src];
                            }
                        }
                        CUDA_CHECK(cudaMemcpyAsync(dbstate.data(), bstate.data(), bstate.size() * 4, cudaMemcpyHostToDevice,
                                                   c.stream));
                        CUDA_CHECK(cudaMemcpyAsync(dbimg.data(), bimg.data(), bimg.size(), cudaMemcpyHostToDevice, c.stream));
                        CUDA_CHECK(cudaMemcpyAsync(dbreset.data(), breset.data(), nb * 4, cudaMemcpyHostToDevice, c.stream));
                        const float* mu = student.forward(dbstate.data(), dbimg.data(), nullptr, dbreset.data(), dbh0.data(),
                                                          T, Bm, c.stream);
                        CUDA_CHECK(cudaMemcpyAsync(bmu.data(), mu, bmu.size() * 4, cudaMemcpyDeviceToHost, c.stream));
                        CUDA_CHECK(cudaStreamSynchronize(c.stream));
                        std::vector<double> gls(size_t(A), 0.0);
                        const float inv = 1.0f / float(nb);
                        for (int t = 0; t < T; ++t)
                            for (int b = 0; b < Bm; ++b) {
                                const int e = order[size_t(mb) * Bm + b];
                                const size_t src = size_t(t) * N + e, dst = size_t(t) * Bm + b;
                                const float* a = &ro.act[src * A];
                                const float* m = &bmu[dst * CA];
                                const float lp = temper::gaussian_logp(a, m, log_std.data(), A);
                                const float adv = float((ro.adv[src] - adv_m) / adv_s);
                                const temper::PpoSampleOut o =
                                    temper::ppo_sample(lp, ro.logp[src], adv, 0.0f, 0.0f, clip_eps, 0.0f, inv);
                                pl_sum += o.policy_loss, kl_ep += o.approx_kl, cf_sum += o.clipped;
                                ++pl_n, ++kl_n;
                                float* dy = &bdy[dst * CA];
                                for (int k = 0; k < A; ++k) {
                                    const float s2 = std::exp(2.0f * log_std[k]), z = a[k] - m[k];
                                    dy[k] = o.g_logp * z / s2;
                                    gls[k] += double(o.g_logp) * (z * z / s2 - 1.0f);
                                }
                            }
                        for (int k = 0; k < A; ++k) gls[k] -= double(ent_coef);  // d(-ent_coef H)/dlog_std
                        CUDA_CHECK(cudaMemcpyAsync(dbdy.data(), bdy.data(), bdy.size() * 4, cudaMemcpyHostToDevice, c.stream));
                        student.backward(dbdy.data(), c.stream);
                        clip.compute(max_grad, c.stream);
                        adam.step(c.stream, clip.scale());
                        std_adam.step(log_std, gls, lr * 50.0f);
                        for (auto& x : log_std) x = std::fmin(std::fmax(x, std::log(0.02f)), std::log(0.5f));
                        CUDA_CHECK(cudaStreamSynchronize(c.stream));
                    }
                }
                // critic: all decisions, in random minibatches
                std::shuffle(sorder.begin(), sorder.end(), rng);
                const size_t ns = sorder.size();
                for (size_t s = 0; s < ns; s += cbatch) {
                    const size_t m = std::min(cbatch, ns - s);
                    for (size_t i = 0; i < m; ++i) {
                        const size_t r = sorder[s + i];
                        for (int j = 0; j < spec.critic; ++j) ccobs[i * spec.critic + j] = (ro.cobs[r * spec.critic + j] - cmean[j]) / cstd[j];
                        cret[i] = ro.ret[r];
                    }
                    CUDA_CHECK(cudaMemcpyAsync(dccobs.data(), ccobs.data(), m * spec.critic * 4, cudaMemcpyHostToDevice, c.stream));
                    const float* v = critic.forward(dccobs.data(), int(m), c.stream);
                    CUDA_CHECK(cudaMemcpyAsync(cval.data(), v, m * 4, cudaMemcpyDeviceToHost, c.stream));
                    CUDA_CHECK(cudaStreamSynchronize(c.stream));
                    for (size_t i = 0; i < m; ++i) {
                        cdv[i] = (cval[i] - cret[i]) / float(m);
                        vl_sum += 0.5 * (cval[i] - cret[i]) * (cval[i] - cret[i]);
                        ++vl_n;
                    }
                    CUDA_CHECK(cudaMemcpyAsync(dcdv.data(), cdv.data(), m * 4, cudaMemcpyHostToDevice, c.stream));
                    critic.backward(dcdv.data(), int(m), c.stream);
                    cclip.compute(1.0f, c.stream);
                    cadam.step(c.stream, cclip.scale());
                }
                CUDA_CHECK(cudaStreamSynchronize(c.stream));
                ++epochs_done;
                kl_sum += kl_ep;
                if (train_actor && kl_n > 0 && kl_ep / kl_n > 1.5 * double(target_kl)) break;
            }
            // ---- log ----
            double std_mean = 0;
            for (float x : log_std) std_mean += std::exp(x);
            std_mean /= double(A);
            std::printf("iter %d%s: %d episodes ended, done %d (%.1f%%), return %.2f | per episode:", iter,
                        train_actor ? "" : " (critic only)", episodes, dones, 100.0 * dones / std::max(1, episodes),
                        ret_sum / std::max(1, episodes));
            for (size_t k = 0; k < names.size() && k < terms.size(); ++k)
                std::printf(" %s %.2f", names[k].c_str(), terms[k] * double(env_steps) * N / std::max(1, episodes));
            std::printf(" | policy %.4f kl %.4f clip %.2f value %.3f std %.3f, %d epochs (%.0f s rollout, %.0f s total)\n",
                        pl_n ? pl_sum / pl_n : 0.0, pl_n ? kl_sum / pl_n : 0.0, pl_n ? cf_sum / pl_n : 0.0,
                        vl_n ? vl_sum / vl_n : 0.0, std_mean, epochs_done, t_roll, since(t0));
            if (timing) {
                // Where the iteration's time went. render = forward kinematics + quads on the host,
                // then the Vulkan draw and the atlas copy to the host; halve = the CPU's 2x -> 1x
                // copy of the student's cameras; policy = state, the image upload, the student's
                // forward, noise; physics = env step + download; update = critic values, GAE, PPO.
                const double r = eyes.render_seconds - render0, hv = eyes.halve_seconds - halve0;
                const double fk = eyes.rig.fk_seconds() - fk0, dr = eyes.rig.draw_seconds() - draw0;
                std::printf("  time: render %.1f s (fk %.1f, vulkan %.1f), halve %.1f s, policy %.1f s, physics %.1f s, "
                            "other %.1f s | update %.1f s\n",
                            r, fk, dr, hv, t_policy, t_physics, t_roll - r - hv - t_policy - t_physics, since(t0) - t_roll);
            }
            std::fflush(stdout);
            if (eval_every > 0 && (iter % eval_every == 0 || iter == iters)) run_eval(iter);
        }
        if (eval_envs <= 0) save(c, out, iters, log_std);
        std::printf("wrote %s\n", out.c_str());
        cudaStreamDestroy(c.stream);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
