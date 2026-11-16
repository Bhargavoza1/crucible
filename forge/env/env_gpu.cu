// Env, GPU backend: one thread per env for the task code (obs, reward, reset),
// the Solver for the physics. Everything on one stream; no sync inside step().
#include <stdexcept>

#include "forge/dynamics/robot_gpu.h"
#include "forge/env/env_setup.h"
#include "forge/env/env_step.h"
#include "forge/env/tasks/all_tasks.h"
#include "forge/solver/solver_gpu.h"
#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"
#include "forge/util/nvtx.h"

namespace forge {

namespace {

constexpr int kBlock = 128;

int grid_for(int n) { return (n + kBlock - 1) / kBlock; }

template <class T>
__global__ void apply_actions_kernel(typename T::Params p, EnvView v, const float* actions) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < v.num_envs) env_apply_actions<T>(p, v, e, actions);
}

template <class T>
__global__ void after_physics_kernel(typename T::Params p, EnvView v) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < v.num_envs) env_after_physics<T>(p, v, e);
}

// ids == nullptr: all envs.
template <class T>
__global__ void reset_kernel(typename T::Params p, EnvView v, const int* ids, int n) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int e = ids ? ids[k] : k;
    if (e < 0 || e >= v.num_envs) return;
    ++v.episode_index[e];
    env_reset_one<T>(p, v, e);
}

__global__ void random_actions_kernel(float* actions, int num_envs, int act_dim, uint64_t seed, uint64_t step) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_envs) return;
    Rng rng(seed ^ 0xA5A5A5A5ull, step, uint64_t(e));
    for (int k = 0; k < act_dim; ++k) actions[size_t(e) * act_dim + k] = rng.uniform(-1.0f, 1.0f);
}

template <typename X>
void zero(DeviceBuffer<X>& b, size_t n) {
    b.resize(n);
    if (n) CUDA_CHECK(cudaMemset(b.data(), 0, n * sizeof(X)));
}

template <typename X>
void copy_down(const DeviceBuffer<X>& d, std::vector<X>& h, cudaStream_t s) {
    h.resize(d.size());
    if (!h.empty()) CUDA_CHECK(cudaMemcpyAsync(h.data(), d.data(), h.size() * sizeof(X), cudaMemcpyDeviceToHost, s));
}

template <class T>
class EnvGpu : public Env {
public:
    explicit EnvGpu(const EnvOptions& opt)
        : setup_(opt), dmodel_(setup_.model), dconstraints_(setup_.constraints), dactuators_(setup_.actuators),
          batch_(setup_.model, opt.num_envs), cbatch_(setup_.constraints, setup_.actuators.nu(), opt.num_envs) {
        opt_ = opt;
        model_ = setup_.model;
        obs_dim_ = T::kObsDim;
        act_dim_ = T::kActDim;
        max_episode_steps_ = setup_.max_episode_steps;
        CUDA_CHECK(cudaStreamCreate(&stream_));
        solver_ = make_solver_gpu(opt.solver, stream_);
        solver_name_ = solver_->name();
        {
            RobotBatchHost h(setup_.model, opt.num_envs);  // qpos0 everywhere, zero velocity and tau
            batch_.upload_state(h);
        }
        const size_t n = size_t(opt.num_envs);
        zero(obs_, n * obs_dim_);
        zero(final_obs_, n * obs_dim_);
        zero(reward_, n);
        zero(terminated_, n);
        zero(truncated_, n);
        zero(episode_step_, n);
        zero(episode_index_, n);
        zero(episode_return_, n);
        zero(last_return_, n);
        zero(last_length_, n);
        zero(last_action_, n * act_dim_);
        zero(prev_action_, n * act_dim_);
        reset_all();
    }
    ~EnvGpu() override {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }

    void reset_all() override {
        reset_kernel<T><<<grid_for(opt_.num_envs), kBlock, 0, stream_>>>(setup_.params, view(), nullptr, opt_.num_envs);
        KERNEL_CHECK();
    }
    void reset(const int* ids, int n) override {
        if (n <= 0) return;
        reset_kernel<T><<<grid_for(n), kBlock, 0, stream_>>>(setup_.params, view(), ids, n);
        KERNEL_CHECK();
    }
    void step(const float* actions) override {
        FORGE_NVTX_RANGE("env step");
        const EnvView v = view();
        apply_actions_kernel<T><<<grid_for(v.num_envs), kBlock, 0, stream_>>>(setup_.params, v, actions);
        KERNEL_CHECK();
        solver_->step(physics(), opt_.physics_dt, opt_.decimation);
        after_physics_kernel<T><<<grid_for(v.num_envs), kBlock, 0, stream_>>>(setup_.params, v);
        KERNEL_CHECK();
    }
    void random_actions(float* actions, uint64_t step) const override {
        random_actions_kernel<<<grid_for(opt_.num_envs), kBlock, 0, stream_>>>(actions, opt_.num_envs, act_dim_,
                                                                                 opt_.seed, step);
        KERNEL_CHECK();
    }
    void download(HostCopy& h) const override {
        copy_down(obs_, h.obs, stream_);
        copy_down(reward_, h.reward, stream_);
        copy_down(final_obs_, h.final_obs, stream_);
        copy_down(batch_.qpos, h.qpos, stream_);
        copy_down(terminated_, h.terminated, stream_);
        copy_down(truncated_, h.truncated, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    void* stream() const override { return stream_; }

    const float* obs() const override { return obs_.data(); }
    const float* reward() const override { return reward_.data(); }
    const int* terminated() const override { return terminated_.data(); }
    const int* truncated() const override { return truncated_.data(); }
    const float* final_obs() const override { return final_obs_.data(); }
    const float* last_return() const override { return last_return_.data(); }
    const int* last_length() const override { return last_length_.data(); }

private:
    EnvView view() {
        EnvView v;
        v.num_envs = opt_.num_envs;
        v.obs_dim = obs_dim_;
        v.act_dim = act_dim_;
        v.max_episode_steps = max_episode_steps_;
        v.seed = opt_.seed;
        v.model = dmodel_.view();
        v.state = batch_.view();
        v.nu = setup_.actuators.nu();
        v.ctrl = v.nu > 0 ? cbatch_.ctrl.data() : nullptr;
        v.obs = obs_.data();
        v.final_obs = final_obs_.data();
        v.reward = reward_.data();
        v.terminated = terminated_.data();
        v.truncated = truncated_.data();
        v.episode_step = episode_step_.data();
        v.episode_index = episode_index_.data();
        v.episode_return = episode_return_.data();
        v.last_return = last_return_.data();
        v.last_length = last_length_.data();
        v.last_action = last_action_.data();
        v.prev_action = prev_action_.data();
    v.rand_friction = cbatch_.rand_friction.data();
    v.rand_body_scale = cbatch_.rand_body_scale.data();
    v.rand_kp = cbatch_.rand_kp.data();
    v.rand_kv = cbatch_.rand_kv.data();
    v.rand_strength = cbatch_.rand_strength.data();
        return v;
    }
    PhysicsViews physics() {
        return {dmodel_.view(), dconstraints_.view(), dactuators_.view(), batch_.view(), batch_.work(), cbatch_.view()};
    }

    TaskSetup<T> setup_;
    RobotModelDevice dmodel_;
    ConstraintModelDevice dconstraints_;
    ActuatorsDevice dactuators_;
    RobotBatchDevice batch_;
    ConstraintBatchDevice cbatch_;
    cudaStream_t stream_ = nullptr;
    std::unique_ptr<Solver> solver_;
    DeviceBuffer<float> obs_, final_obs_, reward_, episode_return_, last_return_, last_action_, prev_action_;
    DeviceBuffer<int> terminated_, truncated_, episode_step_, last_length_;
    DeviceBuffer<uint32_t> episode_index_;
};

}  // namespace

std::unique_ptr<Env> make_env_gpu(const EnvOptions& opt) {
    check_env_options(opt);
#define FORGE_TASK_GPU(T) \
    if (opt.task == T::kName) return std::make_unique<EnvGpu<T>>(opt);
    FORGE_FOR_EACH_TASK(FORGE_TASK_GPU)
#undef FORGE_TASK_GPU
    return make_env_cpu(opt);  // unreachable for a known task; gives the "unknown task" error
}

}  // namespace forge
