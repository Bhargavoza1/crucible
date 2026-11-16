// Env, CPU backend: the answer key for the GPU backend (same per-env code).
#include <stdexcept>

#include "forge/dynamics/robot_batch_host.h"
#include "forge/env/env_setup.h"
#include "forge/env/env_step.h"
#include "forge/env/tasks/all_tasks.h"
#include "forge/solver/constraint_batch_host.h"
#include "forge/solver/solver.h"

namespace forge {

namespace {

template <class T>
class EnvCpu : public Env {
public:
    explicit EnvCpu(const EnvOptions& opt)
        : setup_(opt), batch_(setup_.model, opt.num_envs),
          cbatch_(setup_.constraints, setup_.actuators.nu(), opt.num_envs),
          solver_(make_solver_cpu(opt.solver)) {
        opt_ = opt;
        model_ = setup_.model;
        obs_dim_ = T::kObsDim;
        act_dim_ = T::kActDim;
        max_episode_steps_ = setup_.max_episode_steps;
        task_input_dim_ = TaskInputDim<T>::value;
        task_input_.assign(size_t(task_input_dim_), 0.0f);
        reward_term_names_ = forge::reward_term_names<T>();
        term_sum_.assign(size_t(opt.num_envs) * reward_term_names_.size(), 0.0f);
        solver_name_ = solver_->name();
        const size_t n = size_t(opt.num_envs);
        obs_.assign(n * obs_dim_, 0.0f);
        final_obs_.assign(n * obs_dim_, 0.0f);
        reward_.assign(n, 0.0f);
        terminated_.assign(n, 0);
        truncated_.assign(n, 0);
        episode_step_.assign(n, 0);
        episode_index_.assign(n, 0);
        episode_return_.assign(n, 0.0f);
        last_return_.assign(n, 0.0f);
        last_length_.assign(n, 0);
        last_action_.assign(n * act_dim_, 0.0f);
        prev_action_.assign(n * act_dim_, 0.0f);
        reset_all();
    }

    void reset_all() override {
        const EnvView v = view();
        for (int e = 0; e < v.num_envs; ++e) {
            ++episode_index_[e];
            env_reset_one<T>(setup_.params, v, e);
        }
    }
    void reset(const int* ids, int n) override {
        const EnvView v = view();
        for (int k = 0; k < n; ++k) {
            const int e = ids[k];
            if (e < 0 || e >= v.num_envs) throw std::out_of_range("Env::reset: env id out of range");
            ++episode_index_[e];
            env_reset_one<T>(setup_.params, v, e);
        }
    }
    void step(const float* actions) override {
        const EnvView v = view();
        for (int e = 0; e < v.num_envs; ++e) env_apply_actions<T>(setup_.params, v, e, actions);
        solver_->step(physics(), opt_.physics_dt, opt_.decimation);
        for (int e = 0; e < v.num_envs; ++e) env_after_physics<T>(setup_.params, v, e);
        ++steps_since_read_;
    }
    void random_actions(float* actions, uint64_t step) const override {
        for (int e = 0; e < opt_.num_envs; ++e) {
            Rng rng(opt_.seed ^ 0xA5A5A5A5ull, step, uint64_t(e));
            for (int k = 0; k < act_dim_; ++k) actions[size_t(e) * act_dim_ + k] = rng.uniform(-1.0f, 1.0f);
        }
    }
    void download(HostCopy& h) const override {
        h.obs = obs_;
        h.reward = reward_;
        h.final_obs = final_obs_;
        h.qpos = batch_.qpos;
        h.terminated = terminated_;
        h.truncated = truncated_;
    }

    std::vector<double> read_reward_terms() override {
        const size_t K = reward_term_names_.size();
        std::vector<double> mean(K, 0.0);
        if (K == 0 || steps_since_read_ == 0) return mean;
        for (int e = 0; e < opt_.num_envs; ++e)
            for (size_t k = 0; k < K; ++k) mean[k] += term_sum_[size_t(e) * K + k];
        for (size_t k = 0; k < K; ++k) mean[k] /= double(opt_.num_envs) * steps_since_read_;
        std::fill(term_sum_.begin(), term_sum_.end(), 0.0f);
        steps_since_read_ = 0;
        return mean;
    }

    void set_task_input(const float* values) override {
        task_input_on_ = values != nullptr && task_input_dim_ > 0;
        if (task_input_on_) std::copy(values, values + task_input_dim_, task_input_.begin());
    }

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
        v.model = setup_.model.view();
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
        v.task_input = task_input_on_ ? task_input_.data() : nullptr;
        v.term_sum = term_sum_.empty() ? nullptr : term_sum_.data();
        v.rand_friction = cbatch_.rand_friction.data();
        v.rand_body_scale = cbatch_.rand_body_scale.data();
        v.rand_kp = cbatch_.rand_kp.data();
        v.rand_kv = cbatch_.rand_kv.data();
        v.rand_strength = cbatch_.rand_strength.data();
        return v;
    }
    PhysicsViews physics() {
        return {setup_.model.view(), setup_.constraints.view(), setup_.actuators.view(),
                batch_.view(),       batch_.work(),             cbatch_.view()};
    }

    TaskSetup<T> setup_;
    RobotBatchHost batch_;
    ConstraintBatchHost cbatch_;
    std::unique_ptr<Solver> solver_;
    std::vector<float> obs_, final_obs_, reward_, episode_return_, last_return_, last_action_, prev_action_;
    std::vector<int> terminated_, truncated_, episode_step_, last_length_;
    std::vector<uint32_t> episode_index_;
    std::vector<float> task_input_;
    bool task_input_on_ = false;
    std::vector<float> term_sum_;
    long long steps_since_read_ = 0;
};

}  // namespace

void check_env_options(const EnvOptions& opt) {
    if (opt.num_envs <= 0) throw std::runtime_error("num_envs must be > 0");
    if (!(opt.physics_dt > 0.0f)) throw std::runtime_error("physics_dt must be > 0");
    if (opt.decimation <= 0) throw std::runtime_error("decimation must be > 0");
    if (opt.asset_dir.empty()) throw std::runtime_error("EnvOptions.asset_dir is empty");
}

std::vector<std::string> task_names() {
    std::vector<std::string> names;
#define FORGE_TASK_NAME(T) names.push_back(T::kName);
    FORGE_FOR_EACH_TASK(FORGE_TASK_NAME)
#undef FORGE_TASK_NAME
    return names;
}

std::unique_ptr<Env> make_env_cpu(const EnvOptions& opt) {
    check_env_options(opt);
#define FORGE_TASK_CPU(T) \
    if (opt.task == T::kName) return std::make_unique<EnvCpu<T>>(opt);
    FORGE_FOR_EACH_TASK(FORGE_TASK_CPU)
#undef FORGE_TASK_CPU
    std::string known;
    for (const auto& n : task_names()) known += (known.empty() ? "" : ", ") + n;
    throw std::runtime_error("unknown task '" + opt.task + "' (known: " + known + ")");
}

EnvOptions EnvOptions::from_config(const Config& cfg, const std::string& asset_dir) {
    EnvOptions o;
    o.task = cfg.get_string("task", o.task);
    o.num_envs = cfg.get_int("num_envs", o.num_envs);
    o.seed = static_cast<uint64_t>(cfg.get_int("seed", static_cast<int>(o.seed)));
    o.physics_dt = cfg.get_float("physics_dt", o.physics_dt);
    o.decimation = cfg.get_int("decimation", o.decimation);
    o.solver = cfg.get_string("solver", o.solver);
    const std::string b = cfg.get_string("backend", "gpu");
    if (b == "gpu") o.backend = EnvBackend::kGpu;
    else if (b == "cpu") o.backend = EnvBackend::kCpu;
    else throw std::runtime_error("backend must be cpu or gpu, not '" + b + "'");
    o.max_episode_steps = cfg.get_int("max_episode_steps", 0);
    o.asset_dir = asset_dir;
    return o;
}

}  // namespace forge
