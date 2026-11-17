// Phase 10d: sphere-sphere, sphere-capsule and capsule-capsule contacts vs MuJoCo
// (mjraw_SphereSphere, mjraw_SphereCapsule, mjraw_CapsuleCapsule), 300 random pairs of
// each kind (tests/data/mujoco_round_pairs.txt, tools/gen_mujoco_refs.py), a third
// of the capsule pairs parallel (2 contacts). Checks the number of contacts, then
// each contact's distance, point and normal, in MuJoCo's order.
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "forge/math/quat.h"
#include "forge/solver/constraints.h"

using namespace forge;

namespace {

Quat wxyz(const double* q) { return Quat{float(q[1]), float(q[2]), float(q[3]), float(q[0])}; }
Vec3 v3(const double* p) { return Vec3{float(p[0]), float(p[1]), float(p[2])}; }

TEST(RoundPairs, MatchMujoco) {
    std::ifstream f(std::string(CRUCIBLE_SOURCE_DIR) + "/tests/data/mujoco_round_pairs.txt");
    ASSERT_TRUE(f) << "missing tests/data/mujoco_round_pairs.txt (tools/gen_mujoco_refs.py)";
    std::string line;
    std::map<std::string, int> cases, contacts;
    int two = 0;
    double worst_dist = 0, worst_pos = 0, worst_n = 0;
    while (std::getline(f, line)) {
        if (line.rfind("case ", 0) != 0) continue;
        std::istringstream ss(line.substr(5));
        std::string kind;
        ss >> kind;
        double v[19];
        for (double& x : v) ss >> x;
        const int ncon = int(v[18]);
        const Vec3 c1 = v3(v), c2 = v3(v + 9);
        const Vec3 z1 = rotate(wxyz(v + 3), Vec3{0, 0, 1}), z2 = rotate(wxyz(v + 12), Vec3{0, 0, 1});
        const float r1 = float(v[7]), h1 = float(v[8]), r2 = float(v[16]), h2 = float(v[17]);
        PrimContact o[2];
        int n = 0;
        if (kind == "ss") n = sphere_sphere(c1, r1, z1, c2, r2, z2, o[0]) ? 1 : 0;
        else if (kind == "sc") n = sphere_capsule(c1, r1, z1, c2, z2, r2, h2, o[0]) ? 1 : 0;
        else n = capsule_capsule(c1, z1, r1, h1, c2, z2, r2, h2, o);
        SCOPED_TRACE(kind + " case " + std::to_string(cases[kind]));
        ASSERT_EQ(n, ncon) << "number of contacts";
        for (int c = 0; c < ncon; ++c) {
            ASSERT_TRUE(std::getline(f, line));
            std::istringstream cs(line);
            double m[7];
            for (double& x : m) cs >> x;
            const double ed = std::fabs(o[c].dist - m[0]);
            const double ep = std::hypot(std::hypot(o[c].pos.x - m[1], o[c].pos.y - m[2]), o[c].pos.z - m[3]);
            const double en = std::hypot(std::hypot(o[c].n.x - m[4], o[c].n.y - m[5]), o[c].n.z - m[6]);
            worst_dist = std::max(worst_dist, ed);
            worst_pos = std::max(worst_pos, ep);
            worst_n = std::max(worst_n, en);
            EXPECT_LT(ed, 2e-5) << "contact " << c << " dist " << o[c].dist << " vs " << m[0];
            EXPECT_LT(ep, 2e-5) << "contact " << c << " point";
            EXPECT_LT(en, 1e-4) << "contact " << c << " normal";
        }
        ++cases[kind];
        contacts[kind] += ncon;
        two += ncon == 2;
    }
    std::printf("round pairs vs MuJoCo: sphere-sphere %d pairs %d contacts, sphere-capsule %d / %d, capsule-capsule "
                "%d / %d (%d pairs with 2); worst dist %.1e m, point %.1e m, normal %.1e\n",
                cases["ss"], contacts["ss"], cases["sc"], contacts["sc"], cases["cc"], contacts["cc"], two, worst_dist,
                worst_pos, worst_n);
    EXPECT_EQ(cases["ss"], 300);
    EXPECT_EQ(cases["sc"], 300);
    EXPECT_EQ(cases["cc"], 300);
    EXPECT_GT(two, 20) << "parallel capsules";
}

}  // namespace
