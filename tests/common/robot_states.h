#pragma once
// Random but valid robot states for tests (fixed seed inside each test).
#include <random>

#include "forge/dynamics/robot_batch_host.h"
#include "forge/robots/model.h"

namespace testutil {

// Fills every env with a different random state: free base somewhere near the
// origin with a random rotation, revolute in [-1.5, 1.5], prismatic in
// [-0.4, 0.4], all velocities in [-2, 2], applied forces in [-5, 5].
inline void randomize(const forge::RobotModel& m, forge::RobotBatchHost& b, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (int e = 0; e < b.num_envs(); ++e) {
        float* q = b.env_qpos(e);
        for (int k = 0; k < m.nbody(); ++k) {
            const int a = m.qadr[k];
            switch (m.jtype[k]) {
                case forge::kJointFree: {
                    q[a] = 0.5f * u(g);
                    q[a + 1] = 0.5f * u(g);
                    q[a + 2] = 0.8f + 0.5f * u(g);
                    forge::Quat r = forge::normalize(forge::Quat{n(g), n(g), n(g), n(g)});
                    q[a + 3] = r.w;  // w x y z, like MuJoCo
                    q[a + 4] = r.x;
                    q[a + 5] = r.y;
                    q[a + 6] = r.z;
                    break;
                }
                case forge::kJointRevolute: q[a] = 1.5f * u(g); break;
                case forge::kJointPrismatic: q[a] = 0.4f * u(g); break;
                default: break;
            }
        }
        for (int i = 0; i < m.nv; ++i) {
            b.env_qvel(e)[i] = 2.0f * u(g);
            b.tau[size_t(e) * m.nv + i] = 5.0f * u(g);
        }
    }
}

}  // namespace testutil
