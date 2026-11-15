#include "forge/solver/solver.h"

#include <stdexcept>

#include "forge/solver/constraint_batch_host.h"

namespace forge {

void NewtonSolverCpu::step(const PhysicsViews& v, float dt, int steps) {
    robot_step_constrained_cpu(v.model, v.constraints, v.actuators, v.state, v.work, v.constraint_work, dt, steps);
}

std::unique_ptr<Solver> make_solver_cpu(const std::string& name) {
    if (name == "newton" || name == "newton_thread") return std::make_unique<NewtonSolverCpu>();
    throw std::runtime_error("unknown solver '" + name + "' (known: newton)");
}

}  // namespace forge
