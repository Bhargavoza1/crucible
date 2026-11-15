#pragma once
// Newton solver for the constraint forces (doc 3.4), the same problem as
// MuJoCo's default solver (mjSOL_NEWTON, engine_solver.c):
//
//   qacc = argmin_a  1/2 (a - a0)^T M (a - a0)  +  sum_i s_i(J_i a - aref_i)
//          s_i(x) = 1/2 D_i x^2  if x < 0  (row active, pushes)
//                   0            if x >= 0 (row inactive)
//   a0 = M^-1 qfrc_smooth,   force_i = -D_i (J_i qacc - aref_i) on active rows
//
// The cost is convex and piecewise quadratic. Newton step: H = M + sum_active
// D_i J_i^T J_i, dir = -H^-1 grad. Line search: exact. Along dir the cost is a
// 1D piecewise quadratic; we walk its pieces in order (a row switches on or
// off where J_i (a + t dir) = aref_i) and stop in the piece that holds the
// minimum. When a step crosses no switch, the active set did not change, so the
// new qacc is the exact minimum: we stop. Usually 1-5 iterations.
//
// Because the minimum is unique, any solver that converges gives the same qacc
// as MuJoCo, given the same rows. That is what the tests check.
#include "forge/dynamics/articulated.h"
#include "forge/math/hd.h"
#include "forge/solver/constraints.h"

namespace forge {

// M: [nv * nv] mass matrix (row major). Fills qacc and cw.force, cw.jar, cw.state,
// cw.iterations. Returns false if H is not positive definite (a broken model).
FORGE_HD bool newton_solve(int nv, const float* M, ConstraintWork& cw, int max_iterations, float* qacc) {
    const int ne = cw.nefc;
    for (int i = 0; i < nv; ++i) qacc[i] = cw.qacc_smooth[i];
    cw.iterations = 0;

    for (int it = 0; it < max_iterations && ne > 0; ++it) {
        cw.iterations = it + 1;
        // Row residuals and active set at the current qacc.
        for (int r = 0; r < ne; ++r) {
            const float* J = cw.J + r * nv;
            float s = -cw.aref[r];
            for (int i = 0; i < nv; ++i) s += J[i] * qacc[i];
            cw.jar[r] = s;
            cw.state[r] = s < 0.0f ? 1 : 0;
        }
        // grad = M a - qfrc_smooth + sum_active J^T D jar;  H = M + sum_active D J^T J
        for (int i = 0; i < nv; ++i) {
            float s = 0.0f;
            for (int j = 0; j < nv; ++j) s += M[i * nv + j] * qacc[j];
            cw.Ma[i] = s;
            cw.grad[i] = s - cw.qfrc_smooth[i];
        }
        for (int k = 0; k < nv * nv; ++k) cw.H[k] = M[k];
        for (int r = 0; r < ne; ++r) {
            if (!cw.state[r]) continue;
            const float* J = cw.J + r * nv;
            const float Dr = cw.D[r], g = Dr * cw.jar[r];
            for (int i = 0; i < nv; ++i) {
                if (J[i] == 0.0f) continue;
                cw.grad[i] += J[i] * g;
                const float DJi = Dr * J[i];
                for (int j = 0; j < nv; ++j) cw.H[i * nv + j] += DJi * J[j];
            }
        }
        for (int i = 0; i < nv; ++i) cw.grad[i] = -cw.grad[i];
        if (!cholesky_solve(cw.H, nv, cw.grad, cw.dir, cw.LH)) return false;  // dir = -H^-1 grad

        // ---- exact line search along dir ----
        // f'(t) = g0 + t dMd + sum_active D Jd (jar + t Jd)
        float dMd = 0.0f, g0 = 0.0f;
        for (int i = 0; i < nv; ++i) {
            float s = 0.0f;
            for (int j = 0; j < nv; ++j) s += M[i * nv + j] * cw.dir[j];
            cw.Mdir[i] = s;
            dMd += cw.dir[i] * s;
            g0 += cw.dir[i] * (cw.Ma[i] - cw.qfrc_smooth[i]);
        }
        if (!(dMd > 0.0f)) break;  // dir == 0: already at the minimum
        float s1 = 0.0f, s2 = 0.0f;  // sums over the active rows: D Jd jar, D Jd^2
        for (int r = 0; r < ne; ++r) {
            const float* J = cw.J + r * nv;
            float s = 0.0f;
            for (int i = 0; i < nv; ++i) s += J[i] * cw.dir[i];
            cw.Jd[r] = s;
            if (cw.state[r]) {
                s1 += cw.D[r] * s * cw.jar[r];
                s2 += cw.D[r] * s * s;
            }
        }
        // A row switches where jar + t Jd = 0: active rows (jar < 0) when Jd > 0,
        // inactive rows (jar >= 0) when Jd < 0. cw.state[r] |= 2 marks "already switched".
        float t = 0.0f;
        int switches = 0;
        for (;;) {
            const float t_min_piece = -(g0 + s1) / (dMd + s2);  // minimum of the current piece
            int next = -1;
            float t_next = 0.0f;
            for (int r = 0; r < ne; ++r) {
                if (cw.state[r] & 2) continue;
                const bool on = cw.state[r] & 1;
                if (on ? !(cw.Jd[r] > 0.0f) : !(cw.Jd[r] < 0.0f)) continue;
                const float tr = -cw.jar[r] / cw.Jd[r];
                if (next < 0 || tr < t_next) {
                    next = r;
                    t_next = tr;
                }
            }
            if (next < 0 || t_min_piece <= t_next) {
                t = t_min_piece;
                break;
            }
            // Cross the switch of row 'next' and continue in the next piece.
            const float Dj = cw.D[next] * cw.Jd[next];
            if (cw.state[next] & 1) {
                s1 -= Dj * cw.jar[next];
                s2 -= Dj * cw.Jd[next];
            } else {
                s1 += Dj * cw.jar[next];
                s2 += Dj * cw.Jd[next];
            }
            cw.state[next] = ((cw.state[next] & 1) ^ 1) | 2;
            t = t_next;
            ++switches;
        }
        for (int i = 0; i < nv; ++i) qacc[i] += t * cw.dir[i];
        if (switches == 0) break;  // same active set: qacc is the exact minimum
    }

    // Final forces.
    for (int r = 0; r < ne; ++r) {
        const float* J = cw.J + r * nv;
        float s = -cw.aref[r];
        for (int i = 0; i < nv; ++i) s += J[i] * qacc[i];
        cw.jar[r] = s;
        cw.state[r] = s < 0.0f ? 1 : 0;
        cw.force[r] = s < 0.0f ? -cw.D[r] * s : 0.0f;
    }
    return true;
}

}  // namespace forge
