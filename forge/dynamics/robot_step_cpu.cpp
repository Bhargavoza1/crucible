#include "forge/dynamics/robot_batch_host.h"

namespace forge {

void robot_step_cpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w, float dt, int steps) {
    for (int s = 0; s < steps; ++s)
        for (int e = 0; e < b.num_envs; ++e) robot_step_env(m, b, w, e, dt);
}

void skeleton_points_cpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w,
                         const SkeletonPoint* pts, int npts, float* px, float* py, float* pz) {
    for (int e = 0; e < b.num_envs; ++e) {
        DynamicsWork dw = w.at(e);
        forward_kinematics(m, b.qpos + e * m.nq, dw);
        const size_t o = size_t(e) * npts;
        skeleton_world(pts, npts, dw.xpos, dw.xquat, px + o, py + o, pz + o);
    }
}

void body_poses_cpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w, const int* bodies,
                    int nbodies, float* out) {
    for (int e = 0; e < b.num_envs; ++e) {
        DynamicsWork dw = w.at(e);
        forward_kinematics(m, b.qpos + e * m.nq, dw);
        for (int k = 0; k < nbodies; ++k) {
            const int body = bodies[k];
            const Mat3 R = quat_to_mat3(load_quat(dw.xquat + 4 * body));
            float* o = out + (size_t(e) * nbodies + k) * 12;
            for (int r = 0; r < 3; ++r) {
                o[4 * r + 0] = R(r, 0);
                o[4 * r + 1] = R(r, 1);
                o[4 * r + 2] = R(r, 2);
                o[4 * r + 3] = dw.xpos[3 * body + r];
            }
        }
    }
}

void robot_dynamics_cpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w) {
    for (int e = 0; e < b.num_envs; ++e) {
        DynamicsWork dw = w.at(e);
        const float* tau = b.tau ? b.tau + e * m.nv : nullptr;
        const bool ok = forward_dynamics(m, b.qpos + e * m.nq, b.qvel + e * m.nv, tau, dw, b.qacc + e * m.nv);
        if (b.status) b.status[e] = ok ? 0 : 1;
    }
}

}  // namespace forge
