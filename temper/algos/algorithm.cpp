#include "temper/algos/algorithm.h"

#include <stdexcept>

#include "temper/algos/ppo/ppo.h"
#include "temper/algos/sac/sac.h"

namespace temper {

namespace {

// PPO behind the Algorithm interface.
class PpoAlgorithm : public Algorithm {
public:
    PpoAlgorithm(int D, int A, int N, const PpoConfig& cfg, cudaStream_t s) : ppo_(D, A, N, cfg, s) {}
    const char* name() const override { return "ppo"; }
    int steps_per_iteration() const override { return ppo_.config().horizon; }
    void act(const float* obs, float* actions) override { ppo_.act(obs, actions); }
    void act_deterministic(const float* obs, float* actions, int n) override { ppo_.act_deterministic(obs, actions, n); }
    void observe(const TransitionView& t) override {
        ppo_.observe(t.reward, t.terminated, t.truncated, t.final_obs);
    }
    bool ready_to_update() const override { return ppo_.rollout_full(); }
    void update(const float* obs) override { last_ = ppo_.update(obs); }
    std::vector<std::string> stat_names() const override {
        return {"value_loss", "policy_loss", "entropy", "kl", "clip_fraction", "grad_norm", "action_std"};
    }
    std::vector<double> read_stats() override {
        return {last_.value_loss, last_.policy_loss, last_.entropy, last_.approx_kl, last_.clip_fraction,
                last_.grad_norm, last_.std};
    }
    double learning_rate() const override { return ppo_.config().lr; }
    Checkpoint checkpoint(const std::string& task) const override { return ppo_.checkpoint(task); }
    void load(const Checkpoint& c, bool with_optimizer) override { ppo_.load(c, with_optimizer); }
    int64_t iteration() const override { return ppo_.iteration(); }

private:
    Ppo ppo_;
    PpoStats last_;
};

}  // namespace

std::vector<std::string> algorithm_names() { return {"ppo", "sac"}; }

std::unique_ptr<Algorithm> make_algorithm(const std::string& algo, int D, int A, int N, const forge::Config* cfg,
                                          cudaStream_t s) {
    if (algo == "ppo") {
        const PpoConfig pc = cfg ? PpoConfig::from_config(*cfg) : PpoConfig{};
        return std::make_unique<PpoAlgorithm>(D, A, N, pc, s);
    }
    if (algo == "sac") return std::make_unique<Sac>(D, A, N, SacConfig::from_config(cfg), s);
    throw std::runtime_error("unknown algo '" + algo + "' (known: ppo, sac)");
}

}  // namespace temper
