#include "forge/robots/skeleton.h"

#include <algorithm>
#include <cmath>

#include "forge/dynamics/robot_work.h"

namespace forge {

std::vector<SkeletonPoint> make_skeleton(const RobotModel& m, int beads_per_link) {
    std::vector<SkeletonPoint> pts;
    for (int b = 0; b < m.nbody(); ++b) {
        // Joint marker.
        pts.push_back({b, {0, 0, 0}, m.parent[b] < 0 ? 0.04f : 0.03f});
        // Center of mass, radius grows with the cube root of the mass.
        const float r = std::clamp(0.045f * std::cbrt(std::max(m.mass[b], 1e-6f)), 0.02f, 0.12f);
        pts.push_back({b, {m.com[3 * b], m.com[3 * b + 1], m.com[3 * b + 2]}, r});
        // Beads from the parent's origin to this body's origin, fixed in the parent
        // frame. Exact for revolute and fixed joints; a prismatic joint slides the
        // child away from the beads, which is fine for a picture.
        const int p = m.parent[b];
        if (p >= 0) {
            const Vec3 d{m.body_pos[3 * b], m.body_pos[3 * b + 1], m.body_pos[3 * b + 2]};
            if (length(d) > 1e-4f)
                for (int k = 1; k <= beads_per_link; ++k) {
                    const float t = float(k) / float(beads_per_link + 1);
                    pts.push_back({p, {d.x * t, d.y * t, d.z * t}, 0.018f});
                }
        }
    }
    for (const auto& s : m.spheres) pts.push_back({s.body, {s.pos.x, s.pos.y, s.pos.z}, s.radius});
    return pts;
}

float skeleton_extent(const RobotModel& m, const std::vector<SkeletonPoint>& pts) {
    RobotWork work(m);
    DynamicsWork w = work.view();
    const std::vector<float> q0 = m.qpos0();
    forward_kinematics(m.view(), q0.data(), w);
    std::vector<float> px(pts.size()), py(pts.size()), pz(pts.size());
    skeleton_world(pts.data(), static_cast<int>(pts.size()), work.xpos.data(), work.xquat.data(), px.data(),
                   py.data(), pz.data());
    const Vec3 root{work.xpos[0], work.xpos[1], work.xpos[2]};
    float ext = 0.1f;
    for (size_t k = 0; k < pts.size(); ++k)
        ext = std::max(ext, length(Vec3{px[k], py[k], pz[k]} - root) + pts[k].radius);
    return ext;
}

}  // namespace forge
