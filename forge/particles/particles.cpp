#include "forge/particles/particles.h"

#include <stdexcept>

namespace forge {

void ParticlesHost::resize(int num_envs, int particles_per_env) {
    if (num_envs < 0 || particles_per_env < 0)
        throw std::invalid_argument("ParticlesHost::resize: negative size");
    num_envs_ = num_envs;
    particles_per_env_ = particles_per_env;
    const size_t n = static_cast<size_t>(num_envs) * static_cast<size_t>(particles_per_env);

    for (auto* v : {&px, &py, &pz, &vx, &vy, &vz}) v->assign(n, 0.0f);
    inv_mass.assign(n, 1.0f);
    radius.assign(n, 0.05f);
    env_id.resize(n);
    for (int e = 0; e < num_envs; ++e)
        for (int b = 0; b < particles_per_env; ++b) env_id[env_index(e, b, particles_per_env)] = e;
}

void ParticleParams::add_plane(Vec3 normal, float offset) {
    if (num_planes >= kMaxPlanes) throw std::length_error("ParticleParams: too many planes");
    planes[num_planes++] = Plane{normalize(normal), offset};
}

ParticleView ParticlesHost::view() {
    ParticleView v;
    v.px = px.data();
    v.py = py.data();
    v.pz = pz.data();
    v.vx = vx.data();
    v.vy = vy.data();
    v.vz = vz.data();
    v.inv_mass = inv_mass.data();
    v.radius = radius.data();
    v.env_id = env_id.data();
    v.num_envs = num_envs_;
    v.particles_per_env = particles_per_env_;
    return v;
}

}  // namespace forge
