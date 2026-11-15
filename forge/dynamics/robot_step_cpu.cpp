#include "forge/dynamics/robot_batch_host.h"

namespace forge {

void robot_step_cpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w, float dt, int steps) {
    for (int s = 0; s < steps; ++s)
        for (int e = 0; e < b.num_envs; ++e) robot_step_env(m, b, w, e, dt);
}

void robot_dynamics_cpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w) {
    for (int e = 0; e < b.num_envs; ++e) {
        DynamicsWork dw = w.at(e);
        const float* tau = b.tau ? b.tau + e * m.nv : nullptr;
        const bool ok = forward_dynamics(m, b.qpos + e * m.nq, b.qvel + e * m.nv, tau, dw, b.qacc + e * m.nv);
        if (b.status) b.status[e] = ok ? 0 : 1;
    }
}

}  // namespace forge
