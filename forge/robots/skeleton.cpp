#include "forge/robots/skeleton.h"

#include <algorithm>
#include <cmath>

#include "forge/dynamics/robot_work.h"

namespace forge {

std::vector<SkeletonPoint> make_skeleton(const RobotModel& m, int beads_per_link, float rail_half_length) {
    std::vector<SkeletonPoint> pts;
    for (int b = 0; b < m.nbody(); ++b) {
        // Joint marker.
        pts.push_back({b, {0, 0, 0}, m.parent[b] < 0 ? 0.04f : 0.03f});
        // Center of mass, radius grows with the cube root of the mass.
        const float r = std::clamp(0.045f * std::cbrt(std::max(m.mass[b], 1e-6f)), 0.02f, 0.12f);
        pts.push_back({b, {m.com[3 * b], m.com[3 * b + 1], m.com[3 * b + 2]}, r});
        // Beads from the parent's origin to this body's origin (at q = 0), fixed in
        // the parent frame. Exact for revolute and fixed joints.
        const int p = m.parent[b];
        if (p >= 0) {
            const Vec3 d{m.body_pos[3 * b], m.body_pos[3 * b + 1], m.body_pos[3 * b + 2]};
            if (length(d) > 1e-4f)
                for (int k = 1; k <= beads_per_link; ++k) {
                    const float t = float(k) / float(beads_per_link + 1);
                    pts.push_back({p, {d.x * t, d.y * t, d.z * t}, 0.018f});
                }
            // A prismatic joint moves the child along its axis: draw that rail in the
            // parent frame, so the child is seen sliding on it (not floating away).
            if (m.jtype[b] == kJointPrismatic) {
                const Quat qb{m.body_quat[4 * b], m.body_quat[4 * b + 1], m.body_quat[4 * b + 2], m.body_quat[4 * b + 3]};
                const Vec3 axis = rotate(qb, Vec3{m.jaxis[3 * b], m.jaxis[3 * b + 1], m.jaxis[3 * b + 2]});
                const bool lim = m.joint_limited[b];
                const float lo = lim ? m.joint_lower[b] : -rail_half_length, hi = lim ? m.joint_upper[b] : rail_half_length;
                const int n = std::max(2, static_cast<int>(std::ceil((hi - lo) / 0.04f)) + 1);
                for (int k = 0; k < n; ++k) {
                    const Vec3 at = d + axis * (lo + (hi - lo) * float(k) / float(n - 1));
                    pts.push_back({p, {at.x, at.y, at.z}, 0.012f});
                }
            }
        }
    }
    for (const auto& s : m.spheres) {
        const float len = length(s.pos);
        const int n = static_cast<int>(len / 0.06f);
        for (int k = 1; k < n; ++k) {  // stick from the body origin to the sphere
            const Vec3 at = s.pos * (float(k) / float(n));
            pts.push_back({s.body, {at.x, at.y, at.z}, 0.015f});
        }
        pts.push_back({s.body, {s.pos.x, s.pos.y, s.pos.z}, s.radius});
    }
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
