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
#include "forge/collision/contact.h"
#include "forge/math/hd.h"
#include "forge/particles/particles.h"

namespace forge {

// Integrates particle i with acceleration a (gravity + contact forces / mass).
FORGE_HD void integrate_particle_accel(const ParticleView& s, int i, Vec3 a, float dt) {
    if (s.inv_mass[i] == 0.0f) return;  // static particle

    const float vx = fmaf(a.x, dt, s.vx[i]);
    const float vy = fmaf(a.y, dt, s.vy[i]);
    const float vz = fmaf(a.z, dt, s.vz[i]);
    s.vx[i] = vx;
    s.vy[i] = vy;
    s.vz[i] = vz;
    s.px[i] = fmaf(vx, dt, s.px[i]);
    s.py[i] = fmaf(vy, dt, s.py[i]);
    s.pz[i] = fmaf(vz, dt, s.pz[i]);
}

// Gravity only (Phase 2 free flight).
FORGE_HD void integrate_particle(const ParticleView& s, int i, const ParticleParams& p) {
    integrate_particle_accel(s, i, p.gravity, p.dt);
}

FORGE_HD Vec3 particle_position(const ParticleView& s, int i) { return {s.px[i], s.py[i], s.pz[i]}; }
FORGE_HD Vec3 particle_velocity(const ParticleView& s, int i) { return {s.vx[i], s.vy[i], s.vz[i]}; }

// Acceleration of particle i from gravity and all planes. Sphere-sphere forces
// are added separately (brute force on the CPU, grid on the GPU).
FORGE_HD Vec3 accel_gravity_and_planes(const ParticleView& s, int i, const ParticleParams& p,
                                       const ContactCoeffs& cc) {
    Vec3 a = p.gravity;
    const float inv_m = s.inv_mass[i];
    if (inv_m == 0.0f) return a;  // static: never integrated anyway
    const Vec3 pos = particle_position(s, i);
    const Vec3 vel = particle_velocity(s, i);
    const float m = 1.0f / inv_m;
    for (int k = 0; k < p.num_planes; ++k)
        a += plane_contact_force(p.planes[k], pos, s.radius[i], vel, m, cc) * inv_m;
    return a;
}

// Acceleration on particle i from particle j. Zero if j is static and i is static,
// if they do not touch, or if they are in different envs (doc 3.4: second guard).
FORGE_HD Vec3 accel_from_particle(const ParticleView& s, int i, int j, const ContactCoeffs& cc) {
    const float inv_mi = s.inv_mass[i];
    if (inv_mi == 0.0f || s.env_id[i] != s.env_id[j]) return {0.0f, 0.0f, 0.0f};
    const Vec3 f = sphere_contact_force(particle_position(s, i), s.radius[i], particle_velocity(s, i), inv_mi,
                                        particle_position(s, j), s.radius[j], particle_velocity(s, j),
                                        s.inv_mass[j], cc);
    return f * inv_mi;
}

// True if particles i and j overlap (same env). Used for contact counting in tests.
FORGE_HD bool particles_touch(const ParticleView& s, int i, int j) {
    if (i == j || s.env_id[i] != s.env_id[j]) return false;
    const Vec3 d = particle_position(s, i) - particle_position(s, j);
    const float r = s.radius[i] + s.radius[j];
    return dot(d, d) < r * r;
}

}  // namespace forge
