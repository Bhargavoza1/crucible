#pragma once
// Points that draw a robot as a "skeleton" with the sphere renderer:
//   - a sphere at every joint (body origin)
//   - a sphere at every body's center of mass, sized by its mass
//   - small beads along each link, from the parent's origin to the child's
//   - for a prismatic joint: a rail of small beads along its axis, over its
//     range (limits, or +-0.3 m without limits), fixed to the parent
//   - the collision spheres from the URDF
// Each point is fixed in one body's frame, so its world position is
//   xpos[body] + rotate(xquat[body], local)
// after forward kinematics. Visual only: no physics uses these points.
#include <vector>

#include "forge/dynamics/articulated.h"
#include "forge/math/hd.h"
#include "forge/robots/model.h"

namespace forge {

struct SkeletonPoint {
    int body;
    float local[3];  // in the body frame
    float radius;    // m
};

// beads_per_link small spheres on each parent -> child segment.
std::vector<SkeletonPoint> make_skeleton(const RobotModel& m, int beads_per_link = 6);

// World positions of all points of one robot, from its FK result (xpos, xquat).
FORGE_HD void skeleton_world(const SkeletonPoint* pts, int npts, const float* xpos, const float* xquat, float* px,
                             float* py, float* pz) {
    for (int k = 0; k < npts; ++k) {
        const int b = pts[k].body;
        const Vec3 p = load3(xpos + 3 * b) + rotate(load_quat(xquat + 4 * b), load3(pts[k].local));
        px[k] = p.x;
        py[k] = p.y;
        pz[k] = p.z;
    }
}

// Size of the robot at its zero pose: largest distance of a skeleton point from
// the root body origin. Used to place the camera and space the envs.
float skeleton_extent(const RobotModel& m, const std::vector<SkeletonPoint>& pts);

}  // namespace forge
