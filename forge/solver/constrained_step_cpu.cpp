#include "forge/solver/constraint_batch_host.h"

namespace forge {

void robot_step_constrained_cpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb,
                                float dt, int steps) {
    for (int s = 0; s < steps; ++s)
        for (int e = 0; e < b.num_envs; ++e) robot_step_constrained_env(m, c, a, b, w, cb, e, dt);
}

void robot_dynamics_constrained_cpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                    const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb) {
    for (int e = 0; e < b.num_envs; ++e) robot_dynamics_constrained_env(m, c, a, b, w, cb, e);
}

}  // namespace forge
