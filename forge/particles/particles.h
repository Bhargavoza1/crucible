#pragma once
// Particle state, stored as Structure of Arrays (doc 3.1).
//
// ParticleView   plain struct of pointers. Passed by value to kernels and CPU code.
//                The pointers are host memory or device memory: the view does not care.
// ParticlesHost  owns host arrays (std::vector). Used for setup, CPU reference and tests.
//
// Index of particle 'b' in env 'e':  i = e * particles_per_env + b   (env_index()).
// World is Z-up, like MuJoCo. Positions are local to each env; the renderer adds
// the env grid offset when it draws.
#include <vector>

#include "forge/collision/contact.h"
#include "forge/math/hd.h"
#include "forge/math/vec3.h"

namespace forge {

struct ParticleView {
    float* px = nullptr;
    float* py = nullptr;
    float* pz = nullptr;
    float* vx = nullptr;
    float* vy = nullptr;
    float* vz = nullptr;
    float* inv_mass = nullptr;  // 0 = static, never moves
    float* radius = nullptr;    // used for drawing now, for collisions in Phase 4
    int* env_id = nullptr;
    int num_envs = 0;
    int particles_per_env = 0;

    FORGE_HD int count() const { return num_envs * particles_per_env; }
};

FORGE_HD int env_index(int env, int particle, int particles_per_env) {
    return env * particles_per_env + particle;
}

constexpr int kMaxPlanes = 8;

// Passed by value to kernels (stays well under the 4 KB kernel parameter limit).
struct ParticleParams {
    Vec3 gravity{0.0f, 0.0f, -9.81f};  // m/s^2, Z-up
    float dt = 0.005f;                 // s, one step (with collisions: one substep)

    // Collisions (Phase 4). integrate_cpu/integrate_gpu ignore these: they only
    // apply gravity. step_particles_cpu/ParticleStepperGpu use them.
    ContactParams contact;
    bool particle_collisions = true;  // sphere-sphere on/off (planes always on)
    float cell_size = 0.1f;           // m, grid cell edge; must be >= 2 * largest radius
    int num_planes = 0;
    Plane planes[kMaxPlanes];

    // Adds a plane. Throws if there are already kMaxPlanes.
    void add_plane(Vec3 normal, float offset);
};

class ParticlesHost {
public:
    // Allocates num_envs * particles_per_env particles. All fields zero, except
    // inv_mass = 1, radius = 0.05 and env_id filled in.
    void resize(int num_envs, int particles_per_env);

    int count() const { return num_envs_ * particles_per_env_; }
    int num_envs() const { return num_envs_; }
    int particles_per_env() const { return particles_per_env_; }

    // Pointers into the vectors. Invalid after resize().
    ParticleView view();

    std::vector<float> px, py, pz;
    std::vector<float> vx, vy, vz;
    std::vector<float> inv_mass, radius;
    std::vector<int> env_id;

private:
    int num_envs_ = 0;
    int particles_per_env_ = 0;
};

}  // namespace forge
