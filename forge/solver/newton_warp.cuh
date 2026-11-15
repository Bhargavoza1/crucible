#pragma once
// Newton solver, one WARP per env (Phase 6b). Device code only.
//
// Same math as newton.h (the CPU answer key); only the work is split over the
// 32 lanes of a warp:
//   lanes in parallel: J qacc (one row per lane), M a and gradient (one dof per
//                      lane), H = M + sum D J^T J (one entry per lane),
//                      the Cholesky updates, J dir, the final forces
//   lane 0 alone:      the two triangular solves and the line search (short
//                      serial loops over nv and the rows)
// All arrays must be in shared memory (the caller sets that up); __syncwarp()
// separates the steps. Every lane must call this function.
//
// Results can differ from the CPU in the last bits: the Cholesky here updates
// column by column (right-looking), the CPU version row by row.
#include "forge/solver/constraints.h"

namespace forge {

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return __shfl_sync(0xffffffffu, v, 0);  // the same value in every lane
}

// In-place Cholesky of the lower triangle of A (n x n, row major). Returns false
// (in every lane) if A is not positive definite.
__device__ inline bool cholesky_warp(int lane, float* A, int n) {
    for (int j = 0; j < n; ++j) {
        int ok = 1;
        if (lane == 0) {
            const float d = A[j * n + j];
            ok = d > 0.0f;
            A[j * n + j] = ok ? sqrtf(d) : 0.0f;
        }
        ok = __shfl_sync(0xffffffffu, ok, 0);
        __syncwarp();
        if (!ok) return false;
        const float djj = A[j * n + j];
        for (int i = j + 1 + lane; i < n; i += 32) A[i * n + j] /= djj;
        __syncwarp();
        const int m = n - j - 1;  // trailing block, lower triangle
        for (int q = lane; q < m * m; q += 32) {
            const int ii = q / m, kk = q % m;
            if (kk > ii) continue;
            const int i = j + 1 + ii, k = j + 1 + kk;
            A[i * n + k] -= A[i * n + j] * A[k * n + j];
        }
        __syncwarp();
    }
    return true;
}

// Lane 0: x = A^-1 b with A = L L^T, L in the lower triangle. b and x may alias.
__device__ inline void cholesky_substitute(const float* L, int n, const float* b, float* x) {
    for (int i = 0; i < n; ++i) {
        float s = b[i];
        for (int k = 0; k < i; ++k) s -= L[i * n + k] * x[k];
        x[i] = s / L[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        float s = x[i];
        for (int k = i + 1; k < n; ++k) s -= L[k * n + i] * x[k];
        x[i] = s / L[i * n + i];
    }
}

// cw.nefc must be set (same value in every lane). Fills qacc, cw.force, cw.jar,
// cw.state, cw.iterations (every lane). Returns false if H is not SPD.
__device__ inline bool newton_solve_warp(int lane, int nv, const float* M, ConstraintWork& cw, int max_iterations,
                                         float* qacc) {
    const int ne = cw.nefc;
    for (int i = lane; i < nv; i += 32) qacc[i] = cw.qacc_smooth[i];
    __syncwarp();
    cw.iterations = 0;

    for (int it = 0; it < max_iterations && ne > 0; ++it) {
        cw.iterations = it + 1;
        // Residuals and active set (one row per lane).
        for (int r = lane; r < ne; r += 32) {
            const float* J = cw.J + r * nv;
            float s = -cw.aref[r];
            for (int i = 0; i < nv; ++i) s += J[i] * qacc[i];
            cw.jar[r] = s;
            cw.state[r] = s < 0.0f ? 1 : 0;
        }
        __syncwarp();
        // M a and the gradient (one dof per lane); grad holds -gradient.
        for (int i = lane; i < nv; i += 32) {
            float s = 0.0f;
            for (int j = 0; j < nv; ++j) s += M[i * nv + j] * qacc[j];
            cw.Ma[i] = s;
            float g = s - cw.qfrc_smooth[i];
            for (int r = 0; r < ne; ++r)
                if (cw.state[r]) g += cw.J[r * nv + i] * (cw.D[r] * cw.jar[r]);
            cw.grad[i] = -g;
        }
        // H, lower triangle (one entry per lane).
        for (int k = lane; k < nv * nv; k += 32) {
            const int i = k / nv, j = k % nv;
            if (j > i) continue;
            float h = M[k];
            for (int r = 0; r < ne; ++r)
                if (cw.state[r]) h += cw.D[r] * cw.J[r * nv + i] * cw.J[r * nv + j];
            cw.H[k] = h;
        }
        __syncwarp();
        if (!cholesky_warp(lane, cw.H, nv)) return false;
        if (lane == 0) cholesky_substitute(cw.H, nv, cw.grad, cw.dir);  // dir = -H^-1 grad
        __syncwarp();

        // Line search inputs.
        float pdMd = 0.0f, pg0 = 0.0f;
        for (int i = lane; i < nv; i += 32) {
            float s = 0.0f;
            for (int j = 0; j < nv; ++j) s += M[i * nv + j] * cw.dir[j];
            cw.Mdir[i] = s;
            pdMd += cw.dir[i] * s;
            pg0 += cw.dir[i] * (cw.Ma[i] - cw.qfrc_smooth[i]);
        }
        const float dMd = warp_sum(pdMd), g0 = warp_sum(pg0);
        if (!(dMd > 0.0f)) break;  // dir == 0: already at the minimum (same in every lane)
        for (int r = lane; r < ne; r += 32) {
            const float* J = cw.J + r * nv;
            float s = 0.0f;
            for (int i = 0; i < nv; ++i) s += J[i] * cw.dir[i];
            cw.Jd[r] = s;
        }
        __syncwarp();

        // Exact line search on lane 0 (same walk over the pieces as newton.h).
        float t = 0.0f;
        int switches = 0;
        if (lane == 0) {
            float s1 = 0.0f, s2 = 0.0f;
            for (int r = 0; r < ne; ++r)
                if (cw.state[r]) {
                    s1 += cw.D[r] * cw.Jd[r] * cw.jar[r];
                    s2 += cw.D[r] * cw.Jd[r] * cw.Jd[r];
                }
            for (;;) {
                const float t_min_piece = -(g0 + s1) / (dMd + s2);
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
        }
        t = __shfl_sync(0xffffffffu, t, 0);
        switches = __shfl_sync(0xffffffffu, switches, 0);
        for (int i = lane; i < nv; i += 32) qacc[i] += t * cw.dir[i];
        __syncwarp();
        if (switches == 0) break;
    }

    for (int r = lane; r < ne; r += 32) {
        const float* J = cw.J + r * nv;
        float s = -cw.aref[r];
        for (int i = 0; i < nv; ++i) s += J[i] * qacc[i];
        cw.jar[r] = s;
        cw.state[r] = s < 0.0f ? 1 : 0;
        cw.force[r] = s < 0.0f ? -cw.D[r] * s : 0.0f;
    }
    __syncwarp();
    return true;
}

}  // namespace forge
