#pragma once
// Forward dynamics with constraints (joint limits, contacts) and PD actuators,
// then one Euler step. FORGE_HD: the CPU answer key now, the GPU in Phase 6b.
//
//   qfrc_smooth = tau + actuator forces - bias          (smooth forces)
//   qacc_smooth = M^-1 qfrc_smooth
//   rows        = make_constraints(state)               (constraints.h)
//   qacc        = newton_solve(...)                     (newton.h)
//   qvel += dt qacc, qpos from the new qvel             (integrate_euler)
// Same order and formulas as MuJoCo's mj_step with the Euler integrator.
#include "forge/dynamics/articulated.h"
#include "forge/math/hd.h"
#include "forge/robots/actuators.h"
#include "forge/solver/constraints.h"
#include "forge/solver/newton.h"

namespace forge {

enum DynamicsStatus : int { kDynamicsOk = 0, kMassNotSpd = 1, kHessianNotSpd = 2 };

// tau and ctrl may be null (zero). Leaves the state unchanged; fills qacc and cw.
FORGE_HD int forward_dynamics_constrained(const ModelView& m, const ConstraintView& c, const ActuatorView& act,
                                          const float* qpos, const float* qvel, const float* tau,
                                          const float* ctrl, DynamicsWork& w, ConstraintWork& cw, float* qacc) {
    const int nv = m.nv;
    forward_kinematics(m, qpos, w);
    body_inertias(m, w);
    mass_matrix(m, w);
    bias_forces(m, qvel, w);
    for (int i = 0; i < nv; ++i) cw.qfrc_smooth[i] = (tau ? tau[i] : 0.0f) - w.bias[i];
    if (ctrl) apply_actuators(act, ctrl, qpos, qvel, cw.qfrc_smooth);
    if (!cholesky_solve(w.M, nv, cw.qfrc_smooth, cw.qacc_smooth, w.L)) return kMassNotSpd;
    make_constraints(m, c, qpos, qvel, w, cw);
    if (!newton_solve(nv, w.M, cw, c.max_iterations, qacc)) return kHessianNotSpd;
    return kDynamicsOk;
}

// One step. On an error the state is not moved and qacc is zero.
FORGE_HD int step_constrained(const ModelView& m, const ConstraintView& c, const ActuatorView& act, float* qpos,
                              float* qvel, const float* tau, const float* ctrl, DynamicsWork& w, ConstraintWork& cw,
                              float* qacc, float dt) {
    const int status = forward_dynamics_constrained(m, c, act, qpos, qvel, tau, ctrl, w, cw, qacc);
    if (status != kDynamicsOk) {
        for (int i = 0; i < m.nv; ++i) qacc[i] = 0.0f;
        return status;
    }
    integrate_euler(m, qpos, qvel, qacc, dt);
    return kDynamicsOk;
}

}  // namespace forge
