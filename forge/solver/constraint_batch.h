#pragma once
// Constraint scratch for many envs (Phase 6b), like WorkBatchView for the
// dynamics: one block of each field per env (env-major), pointers to host or
// device memory. at(e) gives the ConstraintWork of env e, so the GPU kernel
// runs the same FORGE_HD code as the CPU answer key.
#include "forge/dynamics/robot_batch.h"
#include "forge/math/hd.h"
#include "forge/robots/actuators.h"
#include "forge/solver/constrained_dynamics.h"
#include "forge/solver/constraints.h"

namespace forge {

struct ConstraintBatchView {
    int nv = 0, max_rows = 0, nu = 0;
    float* ctrl = nullptr;     // [num_envs * nu] actuator targets (null = no actuators)
    int* nefc = nullptr;       // [num_envs] rows of the last step
    int* iterations = nullptr; // [num_envs] Newton iterations of the last step
    int* type = nullptr;       // the rest: [num_envs * size], sizes as in ConstraintWork
    int* id = nullptr;
    float* J = nullptr;
    float* pos = nullptr;
    float* aref = nullptr;
    float* R = nullptr;
    float* D = nullptr;
    float* jar = nullptr;
    float* Jd = nullptr;
    float* force = nullptr;
    int* state = nullptr;
    float* qfrc_smooth = nullptr;
    float* qacc_smooth = nullptr;
    float* H = nullptr;
    float* LH = nullptr;
    float* grad = nullptr;
    float* dir = nullptr;
    float* Ma = nullptr;
    float* Mdir = nullptr;

    FORGE_HD ConstraintWork at(int e) const {
        const size_t r = size_t(max_rows) * e, v = size_t(nv) * e, rv = size_t(max_rows) * nv * e,
                     vv = size_t(nv) * nv * e;
        return {0,          0,          type + r,        id + r,          J + rv,  pos + r,  aref + r,
                R + r,      D + r,      jar + r,         Jd + r,          force + r, state + r, qfrc_smooth + v,
                qacc_smooth + v, H + vv, LH + vv,       grad + v,        dir + v,  Ma + v,   Mdir + v};
    }
};

// Forward dynamics with constraints for env e, state not moved. Writes
// b.qacc, b.status, cb.nefc, cb.iterations.
FORGE_HD int robot_dynamics_constrained_env(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                            const RobotBatchView& b, const WorkBatchView& wb,
                                            const ConstraintBatchView& cb, int e) {
    DynamicsWork w = wb.at(e);
    ConstraintWork cw = cb.at(e);
    const float* tau = b.tau ? b.tau + size_t(e) * m.nv : nullptr;
    const float* ctrl = (cb.ctrl && a.nu > 0) ? cb.ctrl + size_t(e) * a.nu : nullptr;
    float* qacc = b.qacc + size_t(e) * m.nv;
    const int status = forward_dynamics_constrained(m, c, a, b.qpos + size_t(e) * m.nq, b.qvel + size_t(e) * m.nv,
                                                    tau, ctrl, w, cw, qacc);
    if (status != kDynamicsOk)
        for (int i = 0; i < m.nv; ++i) qacc[i] = 0.0f;
    if (b.status) b.status[e] = status;
    if (cb.nefc) cb.nefc[e] = cw.nefc;
    if (cb.iterations) cb.iterations[e] = cw.iterations;
    return status;
}

// One constrained step of env e (dynamics, then Euler). On an error the state
// is not moved (status != 0).
FORGE_HD void robot_step_constrained_env(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                         const RobotBatchView& b, const WorkBatchView& wb,
                                         const ConstraintBatchView& cb, int e, float dt) {
    if (robot_dynamics_constrained_env(m, c, a, b, wb, cb, e) != kDynamicsOk) return;
    integrate_euler(m, b.qpos + size_t(e) * m.nq, b.qvel + size_t(e) * m.nv, b.qacc + size_t(e) * m.nv, dt);
}

}  // namespace forge
