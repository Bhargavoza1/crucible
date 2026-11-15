#pragma once
// Many copies (envs) of one robot. The model is shared; each env has its own
// state and scratch memory.
//
// Layout: each env's numbers sit together (env-major):
//   qpos[e * nq + i], qvel[e * nv + i], M[e * nv * nv + i * nv + j], ...
// Simple and matches DynamicsWork (pointers to one robot's arrays). With one
// thread per env, neighbor threads read memory far apart (not coalesced).
// That is a known cost: measure first, change the layout later if it matters.
#include "forge/dynamics/articulated.h"
#include "forge/math/hd.h"
#include "forge/robots/model.h"

namespace forge {

// State of all envs. Pointers to host or device memory.
struct RobotBatchView {
    int num_envs = 0;
    float* qpos = nullptr;  // [num_envs * nq]
    float* qvel = nullptr;  // [num_envs * nv]
    float* tau = nullptr;   // [num_envs * nv], applied joint forces (may be null = zero)
    float* qacc = nullptr;  // [num_envs * nv], result of the last step
    int* status = nullptr;  // [num_envs], 0 = ok, 1 = mass matrix not positive definite
};

// Scratch memory of all envs, one block of each field per env.
struct WorkBatchView {
    int nbody = 0, nv = 0;
    float* xpos = nullptr;
    float* xquat = nullptr;
    SpatialVec* S = nullptr;
    SpatialInertia* I = nullptr;
    SpatialInertia* Ic = nullptr;
    SpatialVec* V = nullptr;
    SpatialVec* A = nullptr;
    SpatialVec* F = nullptr;
    float* M = nullptr;
    float* L = nullptr;
    float* bias = nullptr;

    // The DynamicsWork of env e: the same struct the single-robot code uses.
    FORGE_HD DynamicsWork at(int e) const {
        const int nb = nbody, n2 = nv * nv;
        return {xpos + 3 * nb * e, xquat + 4 * nb * e, S + nv * e, I + nb * e, Ic + nb * e, V + nb * e,
                A + nb * e,        F + nb * e,         M + n2 * e, L + n2 * e, bias + nv * e};
    }
};

// One full step of env e: forward dynamics, then Euler. Used by the CPU loop
// and the GPU kernel. If M is not positive definite (broken model), qacc is set
// to zero, the state is not moved, and status[e] = 1.
FORGE_HD void robot_step_env(const ModelView& m, const RobotBatchView& b, const WorkBatchView& wb, int e, float dt) {
    DynamicsWork w = wb.at(e);
    float* qpos = b.qpos + e * m.nq;
    float* qvel = b.qvel + e * m.nv;
    float* qacc = b.qacc + e * m.nv;
    const float* tau = b.tau ? b.tau + e * m.nv : nullptr;
    if (!forward_dynamics(m, qpos, qvel, tau, w, qacc)) {
        for (int i = 0; i < m.nv; ++i) qacc[i] = 0.0f;
        if (b.status) b.status[e] = 1;
        return;
    }
    integrate_euler(m, qpos, qvel, qacc, dt);
}

}  // namespace forge
