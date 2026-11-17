// Phase 10b: box-box contacts vs MuJoCo's mjraw_BoxBox, 600 random pairs
// (tests/data/mujoco_box_box.txt, tools/gen_mujoco_refs.py): any turn and place,
// boxes resting on boxes (face contacts, up to 8 points), small tilts, and pairs
// apart. Checks the number of contacts, then each contact's distance, point and
// normal, in MuJoCo's order.
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "forge/math/mat3.h"
#include "forge/math/quat.h"
#include "forge/solver/constraints.h"

using namespace forge;

namespace {

Quat wxyz(const double* q) { return Quat{float(q[1]), float(q[2]), float(q[3]), float(q[0])}; }
Vec3 v3(const double* p) { return Vec3{float(p[0]), float(p[1]), float(p[2])}; }

TEST(BoxBox, MatchesMujoco) {
    std::ifstream f(std::string(CRUCIBLE_SOURCE_DIR) + "/tests/data/mujoco_box_box.txt");
    ASSERT_TRUE(f) << "missing tests/data/mujoco_box_box.txt (tools/gen_mujoco_refs.py)";
    std::string line;
    int cases = 0, contacts = 0, most = 0;
    double worst_dist = 0, worst_pos = 0, worst_n = 0;
    while (std::getline(f, line)) {
        if (line.rfind("case ", 0) != 0) continue;
        std::istringstream ss(line.substr(5));
        double v[21];
        for (double& x : v) ss >> x;
        const int ncon = int(v[20]);
        BoxBoxContacts o;
        const int nc = box_box(v3(v), quat_to_mat3(wxyz(v + 3)), v3(v + 7), v3(v + 10), quat_to_mat3(wxyz(v + 13)),
                               v3(v + 17), o);
        SCOPED_TRACE("case " + std::to_string(cases));
        ASSERT_EQ(nc, ncon) << "number of contacts";
        for (int c = 0; c < ncon; ++c) {
            ASSERT_TRUE(std::getline(f, line));
            std::istringstream cs(line);
            double m[7];
            for (double& x : m) cs >> x;
            const double ed = std::fabs(o.dist[c] - m[0]);
            const double ep = std::hypot(std::hypot(o.pos[c].x - m[1], o.pos[c].y - m[2]), o.pos[c].z - m[3]);
            const double en = std::hypot(std::hypot(o.n.x - m[4], o.n.y - m[5]), o.n.z - m[6]);
            worst_dist = std::max(worst_dist, ed);
            worst_pos = std::max(worst_pos, ep);
            worst_n = std::max(worst_n, en);
            EXPECT_LT(ed, 2e-5) << "contact " << c << " dist " << o.dist[c] << " vs " << m[0];
            EXPECT_LT(ep, 2e-5) << "contact " << c << " point";
            EXPECT_LT(en, 1e-4) << "contact " << c << " normal";
        }
        ++cases;
        contacts += ncon;
        most = std::max(most, ncon);
    }
    std::printf("box-box vs MuJoCo: %d pairs, %d contacts (most in one pair %d), worst dist %.1e m, point %.1e m, "
                "normal %.1e\n", cases, contacts, most, worst_dist, worst_pos, worst_n);
    EXPECT_EQ(cases, 600);
}

}  // namespace
