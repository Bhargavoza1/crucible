#pragma once
// Host (CPU) storage for a batch of robots: state + scratch for every env.
#include <vector>

#include "forge/dynamics/robot_batch.h"
#include "forge/robots/model.h"

namespace forge {

class RobotBatchHost {
    // Declared first: the constructor below initializes them first.
    int num_envs_, nq_, nv_, nbody_;

public:
    RobotBatchHost(const RobotModel& m, int num_envs)
        : num_envs_(num_envs), nq_(m.nq), nv_(m.nv), nbody_(m.nbody()),
          qpos(size_t(num_envs) * m.nq), qvel(size_t(num_envs) * m.nv, 0.0f), tau(size_t(num_envs) * m.nv, 0.0f),
          qacc(size_t(num_envs) * m.nv, 0.0f), status(num_envs, 0),
          xpos(size_t(num_envs) * 3 * m.nbody()), xquat(size_t(num_envs) * 4 * m.nbody()),
          S(size_t(num_envs) * m.nv), I(size_t(num_envs) * m.nbody()), Ic(size_t(num_envs) * m.nbody()),
          V(size_t(num_envs) * m.nbody()), A(size_t(num_envs) * m.nbody()), F(size_t(num_envs) * m.nbody()),
          M(size_t(num_envs) * m.nv * m.nv), L(size_t(num_envs) * m.nv * m.nv), bias(size_t(num_envs) * m.nv) {
        const std::vector<float> q0 = m.qpos0();
        for (int e = 0; e < num_envs; ++e)
            for (int i = 0; i < m.nq; ++i) qpos[size_t(e) * m.nq + i] = q0[i];
    }

    RobotBatchView view() {
        return {num_envs_, qpos.data(), qvel.data(), tau.data(), qacc.data(), status.data()};
    }
    WorkBatchView work() {
        return {nbody_, nv_, xpos.data(), xquat.data(), S.data(), I.data(), Ic.data(),
                V.data(), A.data(), F.data(), M.data(), L.data(), bias.data()};
    }

    int num_envs() const { return num_envs_; }
    float* env_qpos(int e) { return qpos.data() + size_t(e) * nq_; }
    float* env_qvel(int e) { return qvel.data() + size_t(e) * nv_; }

    // State
    std::vector<float> qpos, qvel, tau, qacc;
    std::vector<int> status;
    // Scratch / results of the last dynamics call
    std::vector<float> xpos, xquat;
    std::vector<SpatialVec> S;
    std::vector<SpatialInertia> I, Ic;
    std::vector<SpatialVec> V, A, F;
    std::vector<float> M, L, bias;
};

// CPU loop over all envs: 'steps' steps of dt each.
void robot_step_cpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w, float dt, int steps);

// CPU: forward dynamics for every env without moving the state (fills the
// scratch: xpos, xquat, M, bias, and qacc). For tests.
void robot_dynamics_cpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w);

}  // namespace forge
