#pragma once
// Env API (doc 3.6): many copies of one task, all stepped together.
//
//   auto env = make_env(EnvOptions::from_config(cfg));   // cpu or gpu backend
//   env->reset_all();
//   loop: env->step(actions);  read obs(), reward(), terminated(), truncated(), final_obs()
//
// Backend gpu: every pointer below is device memory and every call is
// asynchronous on env->stream() (no sync inside step). Backend cpu: host memory;
// this is the answer key for the GPU, the same per-env code (env_step.h).
//
// step() runs 'decimation' physics steps per action. An env that ends (terminated
// or truncated) is reset inside step(): obs() then already belongs to the new
// episode and final_obs() holds the real last obs of the old one.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "forge/env/terrain.h"
#include "forge/robots/model.h"
#include "forge/util/config.h"

namespace forge {

enum class EnvBackend { kCpu, kGpu };

struct EnvOptions {
    std::string task = "cartpole";
    int num_envs = 1024;
    uint64_t seed = 1;
    float physics_dt = 0.005f;
    int decimation = 4;
    std::string solver = "newton";
    EnvBackend backend = EnvBackend::kGpu;
    int max_episode_steps = 0;  // 0 = the task's default
    std::string asset_dir;      // folder with robots/*.urdf

    // Keys: task, num_envs, seed, physics_dt, decimation, solver, backend (cpu|gpu),
    // max_episode_steps. asset_dir is not a config key (the app knows it).
    static EnvOptions from_config(const Config& cfg, const std::string& asset_dir);
};

class Env {
public:
    virtual ~Env() = default;

    virtual void reset_all() = 0;                      // new episode in every env
    virtual void reset(const int* env_ids, int n) = 0; // ids in the backend's memory
    virtual void step(const float* actions) = 0;       // [num_envs * act_dim], in [-1, 1] (clipped)

    virtual const float* obs() const = 0;          // [num_envs * obs_dim]
    virtual const float* reward() const = 0;       // [num_envs]
    virtual const int* terminated() const = 0;     // [num_envs]
    virtual const int* truncated() const = 0;      // [num_envs]
    virtual const float* final_obs() const = 0;    // [num_envs * obs_dim], valid where an env just ended
    virtual const float* last_return() const = 0;  // [num_envs] return of each env's last finished episode
    virtual const int* last_length() const = 0;    // [num_envs] its length in steps

    // Same deterministic pseudo-random actions on both backends (for tests and
    // benchmarks): uniform in [-1, 1], stream (seed, step).
    virtual void random_actions(float* actions, uint64_t step) const = 0;

    // Copy obs, reward, flags and qpos to the host (syncs). For tests and drawing.
    struct HostCopy {
        std::vector<float> obs, reward, final_obs, qpos;
        std::vector<int> terminated, truncated;
    };
    virtual void download(HostCopy& h) const = 0;

    virtual void* stream() const { return nullptr; }  // cudaStream_t of the gpu backend

    // Task input: numbers a task takes from outside instead of drawing them (g1_walk:
    // the velocity command vx, vy, wz), the same for every env. task_input_dim() is 0
    // for tasks without one. set_task_input(values) uses them from the next step;
    // set_task_input(nullptr) goes back to the task's own (random) values.
    int task_input_dim() const { return task_input_dim_; }
    virtual void set_task_input(const float* values) { (void)values; }

    // Reward terms (doc 7.4), for tasks that name them (else empty): the mean of
    // each term per env-step since the last call, then the sums restart. Syncs.
    const std::vector<std::string>& reward_term_names() const { return reward_term_names_; }

    // The task's heightfield terrain (host copy, for drawing); empty h = flat ground.
    const Terrain& terrain() const { return terrain_; }
    virtual std::vector<double> read_reward_terms() { return {}; }

    const EnvOptions& options() const { return opt_; }
    const RobotModel& model() const { return model_; }
    int num_envs() const { return opt_.num_envs; }
    int obs_dim() const { return obs_dim_; }
    int act_dim() const { return act_dim_; }
    int max_episode_steps() const { return max_episode_steps_; }
    EnvBackend backend() const { return opt_.backend; }
    const char* solver_name() const { return solver_name_.c_str(); }

protected:
    EnvOptions opt_;
    RobotModel model_;
    int obs_dim_ = 0, act_dim_ = 0, max_episode_steps_ = 0, task_input_dim_ = 0;
    std::vector<std::string> reward_term_names_;
    Terrain terrain_;
    std::string solver_name_;
};

// Names of all tasks (forge/env/tasks/).
std::vector<std::string> task_names();

// The backend from opt.backend. Throws for an unknown task or solver, or for the
// gpu backend in a build without CUDA.
std::unique_ptr<Env> make_env(const EnvOptions& opt);

// The two backends (make_env calls one of them).
std::unique_ptr<Env> make_env_cpu(const EnvOptions& opt);
std::unique_ptr<Env> make_env_gpu(const EnvOptions& opt);  // only in builds with CUDA

}  // namespace forge
