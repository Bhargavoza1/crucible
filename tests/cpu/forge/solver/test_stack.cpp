// Phase 11: 3 cubes stacked on the floor (box-box and box-plane contacts, the solver,
// friction 1) vs MuJoCo (tests/data/mujoco_stack.txt, tools/gen_mujoco_refs.py): an
// aligned stack, a stack with each cube 4-5 mm off and turned 12 degrees, and one
// dropped with 3 mm gaps. Every 0.1 s for 3 s each cube is within 1.5 mm of MuJoCo's,
// and the stack stands (the top cube rests above 9.8 cm and moves under 2 mm in x y).
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/dynamics/robot_work.h"
#include "forge/robots/actuators.h"
#include "forge/solver/constrained_dynamics.h"
#include "forge/solver/constraint_model.h"
#include "forge/solver/constraint_work.h"

using namespace forge;

namespace {

struct Start {
    std::string name;
    float q0[21];
    std::vector<int> at;
    std::vector<std::vector<float>> poses;  // 21 per sample
};

std::vector<Start> read_starts() {
    std::ifstream f(std::string(CRUCIBLE_SOURCE_DIR) + "/tests/data/mujoco_stack.txt");
    std::vector<Start> out;
    std::string line;
    auto read7 = [&](float* d) {
        std::getline(f, line);
        std::istringstream ss(line);
        std::string w;
        if (line.rfind("cube ", 0) == 0) ss >> w;
        for (int i = 0; i < 7; ++i) ss >> d[i];
    };
    while (std::getline(f, line)) {
        if (line.rfind("start ", 0) == 0) {
            out.push_back({});
            out.back().name = line.substr(6);
            for (int k = 0; k < 3; ++k) read7(out.back().q0 + 7 * k);
        } else if (line.rfind("at ", 0) == 0) {
            out.back().at.push_back(std::stoi(line.substr(3)));
            out.back().poses.emplace_back(21);
            for (int k = 0; k < 3; ++k) read7(out.back().poses.back().data() + 7 * k);
        }
    }
    return out;
}

TEST(Stack, ThreeCubesStandAsInMujoco) {
    const std::vector<Start> starts = read_starts();
    ASSERT_EQ(starts.size(), 3u) << "tests/data/mujoco_stack.txt (tools/gen_mujoco_refs.py)";
    for (const Start& s : starts) {
        RobotModel m;
        m.name = "stack";
        for (int k = 0; k < 3; ++k) add_free_box(m, "cube" + std::to_string(k), {0.02f, 0.02f, 0.02f}, 0.1f);
        m.validate();
        ConstraintOptions o;
        o.timestep = 0.002f;
        ConstraintModel cm(m, o);
        cm.add_ground();
        RobotWork work(m);
        ConstraintWorkHost cwh(cm);
        DynamicsWork w = work.view();
        ConstraintWork cw = cwh.view();
        std::vector<float> q(s.q0, s.q0 + 21), qd(size_t(m.nv), 0.0f), qacc(size_t(m.nv), 0.0f);
        double worst = 0;
        size_t next = 0;
        for (int t = 1; t <= s.at.back(); ++t) {
            ASSERT_EQ(step_constrained(m.view(), cm.view(), ActuatorView{}, q.data(), qd.data(), nullptr, nullptr, w, cw,
                                       qacc.data(), 0.002f),
                      kDynamicsOk);
            if (t != s.at[next]) continue;
            for (int k = 0; k < 3; ++k) {
                const float* a = q.data() + 7 * k;
                const float* b = s.poses[next].data() + 7 * k;
                const double e = std::hypot(std::hypot(a[0] - b[0], a[1] - b[1]), a[2] - b[2]);
                worst = std::max(worst, e);
                EXPECT_LT(e, 1.5e-3) << s.name << " cube " << k << " at step " << t;
            }
            ++next;
        }
        const float* top = q.data() + 14;
        const float sink = s.q0[16] - top[2], slide = std::hypot(top[0] - s.q0[14], top[1] - s.q0[15]);
        std::printf("stack %-8s after 3 s: top cube sank %.2f mm (MuJoCo %.2f), slid %.2f mm; worst vs MuJoCo %.2f mm\n",
                    s.name.c_str(), 1000 * sink, 1000 * (s.q0[16] - s.poses.back()[16]), 1000 * slide, 1000 * worst);
        EXPECT_GT(top[2], 0.098f) << s.name << ": the stack stands (top cube resting 5 cube halves up)";
        EXPECT_LT(slide, 0.002f) << s.name << ": the top cube stays on the stack";
    }
}

}  // namespace
