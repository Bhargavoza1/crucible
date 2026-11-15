#pragma once
// Solver interface (doc 3.8): one physics step for all envs. Every solver reads
// and writes the same arrays (the views below), so solvers can be swapped on the
// same scene. Pick one with the config key "solver" (default newton).
//
// Only NewtonSolver exists now (CPU answer key and GPU). CG and Soft Step come
// later, if needed (doc: only after walking works).
#include <memory>
#include <string>

#include "forge/dynamics/robot_batch.h"
#include "forge/robots/actuators.h"
#include "forge/solver/constraint_batch.h"
#include "forge/solver/constraint_model.h"

namespace forge {

// Everything one physics step touches. Host pointers for a CPU solver, device
// pointers for a GPU solver.
struct PhysicsViews {
    ModelView model;
    ConstraintView constraints;
    ActuatorView actuators;
    RobotBatchView state;
    WorkBatchView work;
    ConstraintBatchView constraint_work;
};

class Solver {
public:
    virtual ~Solver() = default;
    virtual const char* name() const = 0;
    // 'steps' steps of dt for every env. A GPU solver is asynchronous (stream).
    virtual void step(const PhysicsViews& v, float dt, int steps) = 0;
};

// Newton on the CPU: the answer key.
class NewtonSolverCpu : public Solver {
public:
    const char* name() const override { return "newton (cpu)"; }
    void step(const PhysicsViews& v, float dt, int steps) override;
};

// "newton" -> NewtonSolverCpu. Throws for an unknown name (and lists the known ones).
std::unique_ptr<Solver> make_solver_cpu(const std::string& name);

}  // namespace forge
