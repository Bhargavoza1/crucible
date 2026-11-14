#pragma once
// One physics step for one particle. The CPU loop (integrate_cpu) and the CUDA
// kernel (integrate_gpu) both call this function, so they run the same math.
//
// Semi-implicit (symplectic) Euler, doc 3.2:
//     v = v + g * dt        velocity first
//     p = p + v * dt        then position, with the NEW velocity
//
// fmaf() is written out on purpose. nvcc fuses "a * b + c" into one FMA on its
// own, but g++ (without -mfma) does not. Then CPU and GPU round differently and
// the error grows every step. With explicit fmaf() both sides compute the same
// correctly rounded result, so CPU and GPU match bit for bit.
#include "forge/math/hd.h"
#include "forge/particles/particles.h"

namespace forge {

FORGE_HD void integrate_particle(const ParticleView& s, int i, const ParticleParams& p) {
    if (s.inv_mass[i] == 0.0f) return;  // static particle

    const float vx = fmaf(p.gravity.x, p.dt, s.vx[i]);
    const float vy = fmaf(p.gravity.y, p.dt, s.vy[i]);
    const float vz = fmaf(p.gravity.z, p.dt, s.vz[i]);
    s.vx[i] = vx;
    s.vy[i] = vy;
    s.vz[i] = vz;
    s.px[i] = fmaf(vx, p.dt, s.px[i]);
    s.py[i] = fmaf(vy, p.dt, s.py[i]);
    s.pz[i] = fmaf(vz, p.dt, s.pz[i]);
}

}  // namespace forge
