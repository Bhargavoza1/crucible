#pragma once
// Shared setup for tests against MuJoCo reference files: loads our model of
// the same URDF and maps qpos / dof / body indices to MuJoCo's by name.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/mujoco_ref.h"
#include "forge/robots/urdf.h"

namespace testutil {

using forge::RobotModel;
using forge::UrdfOptions;

constexpr double kMujocoTol = 1e-4;  // doc: positions, mass matrix and bias forces within 1e-4

// |a - b| <= tol * max(1, |b|): absolute for small numbers, relative for big ones.
inline bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }

// Our model + the index maps between our qpos/qvel and MuJoCo's (by joint name).
struct Setup {
    MujocoRef ref;
    RobotModel model;
    std::vector<int> q_mj;  // our qpos index -> MuJoCo qpos index
    std::vector<int> d_mj;  // our dof index -> MuJoCo dof index
    std::vector<int> b_mj;  // our body index -> MuJoCo body index (without the world)
};

inline Setup load_mujoco_setup(const std::string& robot) {
    Setup s;
    s.ref = testutil::load_mujoco_ref(std::string(CRUCIBLE_SOURCE_DIR) + "/tests/data/mujoco_" + robot + ".txt");
    UrdfOptions opt;
    opt.floating_base = s.ref.floating;
    s.model = forge::load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/" + s.ref.urdf, opt);
    EXPECT_EQ(s.model.nq, s.ref.nq);
    EXPECT_EQ(s.model.nv, s.ref.nv);
    s.q_mj.assign(s.model.nq, -1);
    s.d_mj.assign(s.model.nv, -1);
    for (const auto& j : s.ref.joints) {
        const int b = s.model.joint_index(j.name);
        EXPECT_GE(b, 0) << "joint " << j.name << " missing in our model";
        if (b < 0) continue;
        for (int k = 0; k < forge::joint_nq(s.model.jtype[b]); ++k) s.q_mj[s.model.qadr[b] + k] = j.qadr + k;
        for (int k = 0; k < forge::joint_nv(s.model.jtype[b]); ++k) s.d_mj[s.model.dadr[b] + k] = j.dadr + k;
    }
    for (int b = 0; b < s.model.nbody(); ++b) {
        int found = -1;
        for (size_t k = 0; k < s.ref.bodies.size(); ++k)
            if (s.ref.bodies[k] == s.model.body_name[b]) found = static_cast<int>(k);
        EXPECT_GE(found, 0) << "body " << s.model.body_name[b] << " missing in MuJoCo";
        s.b_mj.push_back(found);
    }
    return s;
}

inline std::vector<float> to_ours(const std::vector<double>& mj, const std::vector<int>& map) {
    std::vector<float> v(map.size());
    for (size_t i = 0; i < map.size(); ++i) v[i] = static_cast<float>(mj[map[i]]);
    return v;
}

}  // namespace testutil
