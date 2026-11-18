// Phase 10d: sphere-sphere, sphere-capsule and capsule-capsule contacts in the
// physics (make_constraints + the solver), and the contype / conaffinity filter.
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "forge/dynamics/robot_work.h"
#include "forge/robots/actuators.h"
#include "forge/solver/constrained_dynamics.h"
#include "forge/solver/constraint_model.h"
#include "forge/solver/constraint_work.h"

using namespace forge;

namespace {

// A free body with one sphere (half_length < 0) or one capsule along body x.
int add_round_body(RobotModel& m, const std::string& name, float r, float half_length, int contype = 1,
                   int conaffinity = 1) {
    const int b = add_free_box(m, name, {r, r, r}, 0.2f);  // the box gives the body and its inertia
    m.boxes.pop_back();                                    // ... but it is not a shape
    if (half_length < 0.0f) {
        m.spheres.push_back({b, {0, 0, 0}, r, contype, conaffinity});
    } else {
        const float s = 0.70710678f;  // capsule z -> body x: 90 degrees about y
        m.capsules.push_back({b, {0, 0, 0}, Quat{0, s, 0, s}, r, half_length, contype, conaffinity});
    }
    return b;
}

struct Scene {
    RobotModel m;
    std::unique_ptr<ConstraintModel> cm;
    std::unique_ptr<RobotWork> work;
    std::unique_ptr<ConstraintWorkHost> cwh;
    std::vector<float> q, qd, qacc;
    void build() {
        ConstraintOptions o;
        o.timestep = 0.002f;
        cm = std::make_unique<ConstraintModel>(m, o);
        cm->add_ground();
        work = std::make_unique<RobotWork>(m);
        cwh = std::make_unique<ConstraintWorkHost>(*cm);
        q = m.qpos0();
        qd.assign(size_t(m.nv), 0.0f);
        qacc.assign(size_t(m.nv), 0.0f);
    }
    void run(int steps) {
        DynamicsWork w = work->view();
        ConstraintWork cw = cwh->view();
        for (int t = 0; t < steps; ++t)
            ASSERT_EQ(step_constrained(m.view(), cm->view(), ActuatorView{}, q.data(), qd.data(), nullptr, nullptr, w, cw,
                                       qacc.data(), 0.002f),
                      kDynamicsOk);
    }
    int rows() {
        DynamicsWork w = work->view();
        ConstraintWork cw = cwh->view();
        EXPECT_EQ(forward_dynamics_constrained(m.view(), cm->view(), ActuatorView{}, q.data(), qd.data(), nullptr,
                                               nullptr, w, cw, qacc.data()),
                  kDynamicsOk);
        return cw.nefc;
    }
};

// Capsule B dropped 1 cm across capsule A (on the floor): B lands on A (balanced on
// it, or tipped over it with one end on the floor: an unstable balance), never into it. With filters that do not match (A 1 / 2,
// B 1 / 4: both still touch the floor), B falls through A to the floor.
TEST(RoundContacts, CapsuleLandsOnCapsule) {
    const float r = 0.02f, hl = 0.1f;
    for (int filtered = 0; filtered < 2; ++filtered) {
        Scene s;
        const int a = add_round_body(s.m, "a", r, hl, 1, filtered ? 2 : 1);
        const int b = add_round_body(s.m, "b", r, hl, 1, filtered ? 4 : 1);
        s.build();
        EXPECT_EQ(s.cm->pair_cc.size(), filtered ? 0u : 2u);
        s.q[2] = r;                       // A on the floor, along x
        s.q[7 + 2] = 3.0f * r + 0.01f;    // B above it ...
        const float t = 0.70710678f;      // ... turned 90 degrees about z: along y
        s.q[7 + 3] = t, s.q[7 + 6] = t;  // w x y z: (cos 45, 0, 0, sin 45)
        float closest = 1.0f;  // the smallest distance between A and B on the way
        for (int step = 0; step < 1500; ++step) {
            s.run(1);
            Vec3 c[2], ax[2];
            for (int k = 0; k < 2; ++k) {
                const float* q = s.q.data() + s.m.qadr[k == 0 ? a : b];
                const Quat qb = Quat{q[4], q[5], q[6], q[3]};
                c[k] = Vec3{q[0], q[1], q[2]};
                ax[k] = rotate(qb * s.m.capsules[size_t(k)].rot, Vec3{0, 0, 1});
            }
            PrimContact pc[2];
            const int n = capsule_capsule(c[0], ax[0], r, hl, c[1], ax[1], r, hl, pc);
            for (int i = 0; i < n; ++i) closest = std::fmin(closest, pc[i].dist);
        }
        const float zb = s.q[size_t(s.m.qadr[b]) + 2];
        if (!filtered) {
            // MuJoCo 4.0.1, the same drop (default solref / solimp): closest -3.4 mm
            EXPECT_GT(closest, -0.004f) << "B never sinks into A (soft contact: MuJoCo -3.4 mm)";
            EXPECT_GT(zb, 1.5f * r) << "B lies on A";
        } else {
            EXPECT_NEAR(zb, r, 2e-3f) << "the filter lets B fall through A to the floor";
            EXPECT_LT(closest, -0.02f) << "B went through A";
        }
        EXPECT_NEAR(s.q[2], r, 2e-3f) << "A stays on the floor";
        std::printf("capsule on capsule%s: B at %.4f m, closest %.4f m\n", filtered ? " (filtered)" : "", zb, closest);
    }
}

// Overlapping pairs give one contact each (4 pyramid rows); far apart, none.
TEST(RoundContacts, RowsForEachPair) {
    Scene s;
    add_round_body(s.m, "s1", 0.03f, -1.0f);
    add_round_body(s.m, "s2", 0.03f, -1.0f);
    add_round_body(s.m, "c", 0.02f, 0.05f);
    s.build();
    EXPECT_EQ(s.cm->pair_ss.size(), 2u);  // one pair (2 indices)
    EXPECT_EQ(s.cm->pair_sc.size(), 4u);  // two pairs
    // all high above the floor; s1 and s2 overlap by 1 cm; c far away
    s.q[2] = 1.0f, s.q[7 + 0] = 0.05f, s.q[7 + 2] = 1.0f, s.q[14 + 0] = 1.0f, s.q[14 + 2] = 1.0f;
    EXPECT_EQ(s.rows(), 4) << "s1-s2";
    s.q[14 + 0] = 0.1f, s.q[14 + 1] = 0.045f;  // capsule c (x 0.05-0.15) next to s2, overlapping by 5 mm
    EXPECT_EQ(s.rows(), 8) << "s1-s2 and s2-c";
}

}  // namespace
