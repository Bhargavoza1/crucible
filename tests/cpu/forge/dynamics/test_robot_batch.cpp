// The batch loop (many envs) must give exactly what the single-robot code gives.
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/robot_states.h"
#include "forge/dynamics/robot_batch_host.h"
#include "forge/dynamics/robot_work.h"
#include "forge/robots/urdf.h"

using namespace forge;

namespace {
RobotModel quadruped() {
    return load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/quadruped.urdf", {true});
}
}  // namespace

TEST(RobotBatch, StartsAtZeroPose) {
    const RobotModel m = quadruped();
    RobotBatchHost b(m, 3);
    EXPECT_FLOAT_EQ(b.env_qpos(2)[3], 1.0f);  // qw of env 2
    EXPECT_FLOAT_EQ(b.env_qvel(2)[0], 0.0f);
}

TEST(RobotBatch, EachEnvEqualsSingleRobot) {
    const RobotModel m = quadruped();
    RobotBatchHost b(m, 16);
    testutil::randomize(m, b, 7);
    RobotBatchHost start = b;
    robot_step_cpu(m.view(), b.view(), b.work(), 0.002f, 20);

    RobotWork work(m);
    DynamicsWork w = work.view();
    for (int e = 0; e < 16; ++e) {
        std::vector<float> q(start.env_qpos(e), start.env_qpos(e) + m.nq);
        std::vector<float> v(start.env_qvel(e), start.env_qvel(e) + m.nv);
        std::vector<float> a(m.nv);
        const float* tau = start.tau.data() + size_t(e) * m.nv;
        for (int s = 0; s < 20; ++s) {
            ASSERT_TRUE(forward_dynamics(m.view(), q.data(), v.data(), tau, w, a.data()));
            integrate_euler(m.view(), q.data(), v.data(), a.data(), 0.002f);
        }
        for (int i = 0; i < m.nq; ++i) EXPECT_EQ(b.env_qpos(e)[i], q[i]) << "env " << e;  // bit for bit
        for (int i = 0; i < m.nv; ++i) EXPECT_EQ(b.env_qvel(e)[i], v[i]) << "env " << e;
    }
}

// A model with a massless moving link has a singular mass matrix. The step must
// report it and leave the state alone, not fill it with NaN.
TEST(RobotBatch, BrokenModelReportsStatus) {
    RobotModel m = quadruped();
    m.mass[m.body_index("fl_shin")] = 0.0f;
    for (int k = 0; k < 9; ++k) m.inertia[9 * m.body_index("fl_shin") + k] = 0.0f;
    RobotBatchHost b(m, 2);
    const std::vector<float> q_before = b.qpos;
    robot_step_cpu(m.view(), b.view(), b.work(), 0.002f, 1);
    EXPECT_EQ(b.status[0], 1);
    EXPECT_EQ(b.status[1], 1);
    EXPECT_EQ(b.qpos, q_before);
    for (float a : b.qacc) EXPECT_EQ(a, 0.0f);
}
