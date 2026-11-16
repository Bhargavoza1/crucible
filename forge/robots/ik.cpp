#include "forge/robots/ik.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "forge/dynamics/articulated.h"
#include "forge/dynamics/robot_work.h"

namespace forge {

namespace {

// Solves the n x n system A x = b in place (Gaussian elimination, partial pivoting).
void solve_dense(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int c = 0; c < n; ++c) {
        int p = c;
        for (int r = c + 1; r < n; ++r)
            if (std::fabs(A[r * n + c]) > std::fabs(A[p * n + c])) p = r;
        if (p != c) {
            for (int k = 0; k < n; ++k) std::swap(A[c * n + k], A[p * n + k]);
            std::swap(b[c], b[p]);
        }
        for (int r = c + 1; r < n; ++r) {
            const double f = A[r * n + c] / A[c * n + c];
            for (int k = c; k < n; ++k) A[r * n + k] -= f * A[c * n + k];
            b[r] -= f * b[c];
        }
    }
    for (int c = n - 1; c >= 0; --c) {
        for (int k = c + 1; k < n; ++k) b[c] -= A[c * n + k] * b[k];
        b[c] /= A[c * n + c];
    }
}

}  // namespace

IkResult solve_ik(const RobotModel& m, int body, Vec3 target_pos, Quat target_rot, const std::vector<int>& joints,
                  std::vector<float>& qpos, const IkOptions& opt) {
    if (body < 0 || body >= m.nbody()) throw std::runtime_error("solve_ik: no such body");
    for (int j : joints)
        if (j < 0 || j >= m.nbody() || (m.jtype[j] != kJointRevolute && m.jtype[j] != kJointPrismatic))
            throw std::runtime_error("solve_ik: joints must be revolute or prismatic");
    const ModelView mv = m.view();
    RobotWork work(m);
    DynamicsWork w = work.view();
    const int rows = opt.rotation ? 6 : 3, n = static_cast<int>(joints.size());
    const Mat3 R0 = quat_to_mat3(target_rot);
    std::vector<double> J(size_t(rows) * n), A(size_t(rows) * rows), e(rows);
    IkResult res;
    for (int it = 0; it <= opt.max_iterations; ++it) {
        forward_kinematics(mv, qpos.data(), w);
        const Vec3 p = load3(w.xpos + 3 * body);
        const Mat3 R = quat_to_mat3(load_quat(w.xquat + 4 * body));
        const Vec3 ep = target_pos - p;
        // rotation error: 0.5 sum_i (current axis i) x (target axis i)
        const Vec3 er = (cross(col(R, 0), col(R0, 0)) + cross(col(R, 1), col(R0, 1)) + cross(col(R, 2), col(R0, 2))) * 0.5f;
        res.iterations = it;
        res.pos_error = length(ep);
        res.rot_error = length(er);
        if (res.pos_error + (opt.rotation ? res.rot_error : 0.0f) < opt.tolerance || it == opt.max_iterations) break;
        e[0] = ep.x, e[1] = ep.y, e[2] = ep.z;
        if (opt.rotation) e[3] = er.x, e[4] = er.y, e[5] = er.z;
        // columns: the body's linear velocity at p and angular velocity for each listed joint;
        // a joint that is not above 'body' gives a zero column
        for (int c = 0; c < n; ++c) {
            const int jb = joints[c];
            bool above = false;
            for (int k = body; k >= 0; k = m.parent[k]) above = above || k == jb;
            const SpatialVec& S = w.S[m.dadr[jb]];
            const Vec3 v = above ? S.lin + cross(S.ang, p) : Vec3{0, 0, 0};
            const Vec3 a = above ? S.ang : Vec3{0, 0, 0};
            J[0 * n + c] = v.x, J[1 * n + c] = v.y, J[2 * n + c] = v.z;
            if (opt.rotation) J[3 * n + c] = a.x, J[4 * n + c] = a.y, J[5 * n + c] = a.z;
        }
        for (int r = 0; r < rows; ++r)
            for (int s = 0; s < rows; ++s) {
                double sum = r == s ? opt.damping : 0.0;
                for (int c = 0; c < n; ++c) sum += J[r * n + c] * J[s * n + c];
                A[r * rows + s] = sum;
            }
        std::vector<double> y = e;
        solve_dense(A, y, rows);
        std::vector<double> dq(n, 0.0);
        double biggest = 0.0;
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < rows; ++r) dq[c] += J[r * n + c] * y[r];
            biggest = std::max(biggest, std::fabs(dq[c]));
        }
        const double scale = biggest > opt.max_step ? opt.max_step / biggest : 1.0;
        for (int c = 0; c < n; ++c) {
            const int jb = joints[c];
            float& q = qpos[m.qadr[jb]];
            q = static_cast<float>(q + scale * dq[c]);
            if (m.joint_limited[jb]) q = std::clamp(q, m.joint_lower[jb], m.joint_upper[jb]);
        }
    }
    return res;
}

}  // namespace forge
