#pragma once
// Particle state in GPU memory. Owns one DeviceBuffer per SoA field.
// Host code only (uses the CUDA runtime API, compiles with g++ or nvcc).
#include "forge/particles/particles.h"
#include "forge/util/device_buffer.h"

namespace forge {

class ParticlesDevice {
public:
    // Allocates (if needed) and copies every field from the host.
    void upload(const ParticlesHost& h) {
        num_envs_ = h.num_envs();
        particles_per_env_ = h.particles_per_env();
        px.upload(h.px);
        py.upload(h.py);
        pz.upload(h.pz);
        vx.upload(h.vx);
        vy.upload(h.vy);
        vz.upload(h.vz);
        inv_mass.upload(h.inv_mass);
        radius.upload(h.radius);
        env_id.upload(h.env_id);
    }

    // Copies every field back. 'h' must have the same size (call resize first).
    void download(ParticlesHost& h) const {
        download_positions(h);
        copy_down(vx, h.vx);
        copy_down(vy, h.vy);
        copy_down(vz, h.vz);
        copy_down(inv_mass, h.inv_mass);
        copy_down(radius, h.radius);
        copy_down(env_id, h.env_id);
    }

    // Stage A drawing (doc 4.2): only positions go to the host each frame.
    // Phase 3 removes this copy (zero-copy).
    void download_positions(ParticlesHost& h) const {
        copy_down(px, h.px);
        copy_down(py, h.py);
        copy_down(pz, h.pz);
    }

    ParticleView view() {
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

    int count() const { return num_envs_ * particles_per_env_; }

    DeviceBuffer<float> px, py, pz, vx, vy, vz, inv_mass, radius;
    DeviceBuffer<int> env_id;

private:
    template <typename T>
    static void copy_down(const DeviceBuffer<T>& d, std::vector<T>& h) {
        if (h.size() != d.size()) h.resize(d.size());
        if (!h.empty())
            CUDA_CHECK(cudaMemcpy(h.data(), d.data(), h.size() * sizeof(T), cudaMemcpyDeviceToHost));
    }
    int num_envs_ = 0;
    int particles_per_env_ = 0;
};

}  // namespace forge
