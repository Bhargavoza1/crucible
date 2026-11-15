#include "forge/particles/step_cpu.h"

#include "forge/particles/particle_step.h"

namespace forge {

void step_particles_cpu(const ParticleView& s, const ParticleParams& p, int steps) {
    const int n = s.count();
    const int per_env = s.particles_per_env;
    const ContactCoeffs cc = contact_coeffs(p.contact);
    std::vector<Vec3> accel(static_cast<size_t>(n));

    for (int step = 0; step < steps; ++step) {
        // 1. All accelerations from the current state. Nothing moves yet, so the
        //    order of particles does not matter.
        for (int i = 0; i < n; ++i) {
            Vec3 a = accel_gravity_and_planes(s, i, p, cc);
            if (p.particle_collisions && per_env > 0) {
                // Particles of one env are stored next to each other (doc 3.1).
                const int first = (i / per_env) * per_env;
                for (int j = first; j < first + per_env; ++j)
                    if (j != i) a += accel_from_particle(s, i, j, cc);
            }
            accel[i] = a;
        }
        // 2. Integrate everyone.
        for (int i = 0; i < n; ++i) integrate_particle_accel(s, i, accel[i], p.dt);
    }
}

std::vector<int> contact_counts_cpu(const ParticleView& s) {
    const int n = s.count();
    const int per_env = s.particles_per_env;
    std::vector<int> counts(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        const int first = (i / per_env) * per_env;
        for (int j = first; j < first + per_env; ++j)
            if (particles_touch(s, i, j)) ++counts[i];
    }
    return counts;
}

}  // namespace forge
