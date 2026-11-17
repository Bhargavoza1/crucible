// Phase 10: capsule-box contacts vs MuJoCo's mjc_CapsuleBox, 400 random pairs
// (tests/data/mujoco_capsule_box.txt, tools/gen_mujoco_refs.py): about half give one
// contact, half two (capsules lying along a face or an edge), a few none. Checks
// the number of contacts, and each contact's distance, point and normal, as
// make_constraints builds them (capsule_box_points, then sphere_box per point).
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/math/mat3.h"
#include "forge/math/quat.h"
#include "forge/solver/constraints.h"

using namespace forge;

namespace {

Quat wxyz(const double* q) { return Quat{float(q[1]), float(q[2]), float(q[3]), float(q[0])}; }
Vec3 v3(const double* p) { return Vec3{float(p[0]), float(p[1]), float(p[2])}; }

TEST(CapsuleBox, MatchesMujoco) {
    std::ifstream f(std::string(CRUCIBLE_SOURCE_DIR) + "/tests/data/mujoco_capsule_box.txt");
    ASSERT_TRUE(f) << "missing tests/data/mujoco_capsule_box.txt (tools/gen_mujoco_refs.py)";
    std::string line;
    int cases = 0, contacts = 0, two = 0;
    double worst_dist = 0, worst_pos = 0, worst_n = 0;
    while (std::getline(f, line)) {
        if (line.rfind("case ", 0) != 0) continue;
        std::istringstream ss(line.substr(5));
        double v[20];
        for (double& x : v) ss >> x;
        const int ncon = int(v[19]);
        const Vec3 pc = v3(v), pb = v3(v + 9), h = v3(v + 16);
        const float r = float(v[7]), hl = float(v[8]);
        const Vec3 a = rotate(wxyz(v + 3), Vec3{0, 0, 1});
        const Mat3 R = quat_to_mat3(wxyz(v + 12));
        float sp[2];
        const int np = capsule_box_points(pc, a, hl, r, pb, R, h, sp);
        std::vector<std::vector<float>> ours;  // dist, pos, normal
        for (int i = 0; i < np; ++i) {
            const Vec3 center = pc + a * (hl * sp[i]);
            float dist;
            Vec3 n_out;
            if (!sphere_box(center, r, pb, R, h, dist, n_out)) continue;
            const Vec3 n = -n_out, p = center + n * (0.5f * dist + r);
            ours.push_back({dist, p.x, p.y, p.z, n.x, n.y, n.z});
        }
        SCOPED_TRACE("case " + std::to_string(cases));
        ASSERT_EQ(int(ours.size()), ncon) << "number of contacts";
        for (int c = 0; c < ncon; ++c) {
            ASSERT_TRUE(std::getline(f, line));
            std::istringstream cs(line);
            double m[7];
            for (double& x : m) cs >> x;
            const auto& o = ours[c];
            const double ed = std::fabs(o[0] - m[0]);
            const double ep = std::hypot(std::hypot(o[1] - m[1], o[2] - m[2]), o[3] - m[3]);
            const double en = std::hypot(std::hypot(o[4] - m[4], o[5] - m[5]), o[6] - m[6]);
            worst_dist = std::max(worst_dist, ed);
            worst_pos = std::max(worst_pos, ep);
            worst_n = std::max(worst_n, en);
            EXPECT_LT(ed, 1e-5) << "contact " << c << " dist " << o[0] << " vs " << m[0];
            EXPECT_LT(ep, 1e-5) << "contact " << c << " point";
            EXPECT_LT(en, 1e-4) << "contact " << c << " normal";
        }
        ++cases;
        contacts += ncon;
        two += ncon == 2;
    }
    std::printf("capsule-box vs MuJoCo: %d pairs, %d contacts (%d pairs with 2), worst dist %.1e m, point %.1e m, "
                "normal %.1e\n", cases, contacts, two, worst_dist, worst_pos, worst_n);
    EXPECT_EQ(cases, 400);
}

}  // namespace
