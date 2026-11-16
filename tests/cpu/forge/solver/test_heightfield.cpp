// Phase 8d: heightfield terrain (forge/solver/heightfield.h).
//   - triangle layout as MuJoCo's hfield (probed: diagonal (0,0)-(1,1))
//   - a flat heightfield gives exactly the rows and qacc of the ground plane
//   - a planar slope gives the rows and qacc of the same inclined plane
// The contact geometry was also compared with MuJoCo on a rough field (one-contact
// states: distance within 1.4e-9 m, normal within 4e-8; see the Phase 8d commit).
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/dynamics/robot_work.h"
#include "forge/env/tasks/g1_stand.h"
#include "forge/robots/urdf.h"
#include "forge/solver/constrained_dynamics.h"
#include "forge/solver/constraint_model.h"
#include "forge/solver/constraint_work.h"
#include "forge/solver/heightfield.h"

using namespace forge;

namespace {

TEST(Heightfield, TrianglesAsMujocoAndOutsideIsFlat) {
    // One cell of size 1, corner (0, 0) raised to 0.5 (the MuJoCo probe).
    std::vector<float> h = {0.5f, 0.0f, 0.0f, 0.0f};
    HeightfieldView f{2, 2, 1.0f, 0.0f, 0.0f, h.data()};
    EXPECT_NEAR(hfield_height(f, 0.2f, 0.8f), 0.1f, 1e-6f) << "v > u: triangle (00, 11, 01), h = 0.5 (1 - y)";
    EXPECT_NEAR(hfield_height(f, 0.8f, 0.2f), 0.1f, 1e-6f) << "u > v: triangle (00, 10, 11), h = 0.5 (1 - x)";
    EXPECT_NEAR(hfield_height(f, 0.3f, 0.1f), 0.35f, 1e-6f);
    EXPECT_NEAR(hfield_height(f, 0.0f, 0.0f), 0.5f, 1e-6f);
    Vec3 p, n;
    hfield_plane(f, 0.2f, 0.8f, p, n);
    EXPECT_NEAR(n.x, 0.0f, 1e-6f);
    EXPECT_NEAR(n.y, 0.4472136f, 1e-6f) << "MuJoCo's contact normal there: (0, 0.447, 0.894)";
    EXPECT_NEAR(n.z, 0.8944272f, 1e-6f);
    hfield_plane(f, -3.0f, -2.0f, p, n);  // outside: flat at the corner's height
    EXPECT_EQ(n.z, 1.0f);
    EXPECT_NEAR(hfield_height(f, -3.0f, -2.0f), 0.5f, 1e-6f);
    EXPECT_NEAR(hfield_height(f, 5.0f, 0.5f), 0.0f, 1e-6f);
}

struct G1Scene {
    RobotModel m;
    Actuators act;
    G1Scene() {
        UrdfOptions u;
        u.floating_base = true;
        G1StandTask::urdf_options(u);
        m = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/g1/g1_12dof.urdf", u);
        act = G1StandTask::make_actuators(m);
    }
    // Standing state on the ground z(x, y) = a x + b y (pelvis lifted by the ground
    // height at its xy, feet slightly inside), sliding: all row kinds matter.
    void state(float a, float b, std::vector<float>& q, std::vector<float>& qd, std::vector<float>& ctrl) const {
        const auto p = G1StandTask::make_params(m);
        q = m.qpos0();
        qd.assign(m.nv, 0.0f);
        ctrl.assign(act.nu(), 0.0f);
        q[p.base_q + 0] = 0.3f;
        q[p.base_q + 1] = -0.2f;
        q[p.base_q + 2] = p.stand_height - 0.002f + a * 0.3f + b * -0.2f;
        for (int k = 0; k < 12; ++k) {
            q[p.q[k]] = p.q0[k];
            ctrl[k] = p.q0[k] + 0.05f * float(k % 3 - 1);
            qd[p.d[k]] = 0.2f * float(k % 5 - 2);
        }
        qd[p.base_d] = 0.15f;
    }
    void solve(const ConstraintModel& cm, const std::vector<float>& q, const std::vector<float>& qd,
               const std::vector<float>& ctrl, std::vector<float>& qacc, ConstraintWorkHost& cwh, int& nefc) const {
        RobotWork work(m);
        DynamicsWork w = work.view();
        ConstraintWork cw = cwh.view();
        qacc.assign(m.nv, 0.0f);
        ASSERT_EQ(forward_dynamics_constrained(m.view(), cm.view(), act.view(), q.data(), qd.data(), nullptr,
                                               ctrl.data(), w, cw, qacc.data()),
                  kDynamicsOk);
        nefc = cw.nefc;
    }
};

// heights z = a x + b y on a 4 m x 4 m grid of 5 cm cells around the robot.
std::vector<float> planar(int n, float cell, float x0, float y0, float a, float b) {
    std::vector<float> h(size_t(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) h[size_t(j) * n + i] = a * (x0 + i * cell) + b * (y0 + j * cell);
    return h;
}

TEST(Heightfield, FlatEqualsGroundPlaneExactly) {
    G1Scene g;
    ConstraintModel plane(g.m), hf(g.m);
    plane.add_ground();
    hf.set_heightfield(81, 81, 0.05f, -2.0f, -2.0f, planar(81, 0.05f, -2.0f, -2.0f, 0.0f, 0.0f));
    EXPECT_EQ(plane.max_rows(), hf.max_rows());
    std::vector<float> q, qd, ctrl, a1, a2;
    g.state(0.0f, 0.0f, q, qd, ctrl);
    ConstraintWorkHost w1(plane), w2(hf);
    int n1 = 0, n2 = 0;
    g.solve(plane, q, qd, ctrl, a1, w1, n1);
    g.solve(hf, q, qd, ctrl, a2, w2, n2);
    ASSERT_GT(n1, 0);
    ASSERT_EQ(n1, n2);
    for (int r = 0; r < n1; ++r) {
        EXPECT_EQ(w1.pos[r], w2.pos[r]) << "row " << r;
        EXPECT_EQ(w1.R[r], w2.R[r]) << "row " << r;
        EXPECT_EQ(w1.aref[r], w2.aref[r]) << "row " << r;
    }
    for (int i = 0; i < g.m.nv; ++i) EXPECT_EQ(a1[i], a2[i]) << "dof " << i;
}

TEST(Heightfield, PlanarSlopeEqualsInclinedPlane) {
    G1Scene g;
    const float a = 0.12f, b = -0.07f;  // about 7 and 4 degrees
    ConstraintModel plane(g.m), hf(g.m);
    plane.add_plane({-a, -b, 1.0f}, {0.0f, 0.0f, 0.0f});
    hf.set_heightfield(81, 81, 0.05f, -2.0f, -2.0f, planar(81, 0.05f, -2.0f, -2.0f, a, b));
    std::vector<float> q, qd, ctrl, a1, a2;
    g.state(a, b, q, qd, ctrl);
    ConstraintWorkHost w1(plane), w2(hf);
    int n1 = 0, n2 = 0;
    g.solve(plane, q, qd, ctrl, a1, w1, n1);
    g.solve(hf, q, qd, ctrl, a2, w2, n2);
    ASSERT_GT(n1, 0);
    ASSERT_EQ(n1, n2);
    for (int r = 0; r < n1; ++r) EXPECT_NEAR(w1.pos[r], w2.pos[r], 2e-6f) << "row " << r;
    double worst = 0;
    for (int i = 0; i < g.m.nv; ++i)
        worst = std::max(worst, double(std::fabs(a1[i] - a2[i])) / std::max(1.0, double(std::fabs(a1[i]))));
    std::printf("planar slope heightfield vs inclined plane: %d rows, worst relative qacc difference %.2e\n", n1, worst);
    EXPECT_LT(worst, 1e-3);
}

}  // namespace
