#include "forge/solver/constraint_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "forge/dynamics/articulated.h"
#include "forge/dynamics/robot_work.h"

namespace forge {

namespace {

// True if body b or one of its ancestors has a moving joint.
bool body_moves(const RobotModel& m, int b) {
    for (int k = b; k >= 0; k = m.parent[k])
        if (m.jtype[k] != kJointFixed) return true;
    return false;
}

// In-place Cholesky solve in double (host only, small n).
void solve_spd(std::vector<double> A, int n, std::vector<double>& x) {
    for (int j = 0; j < n; ++j) {
        double s = A[j * n + j];
        for (int k = 0; k < j; ++k) s -= A[j * n + k] * A[j * n + k];
        if (s <= 0) throw std::runtime_error("compute_invweight0: mass matrix not positive definite");
        A[j * n + j] = std::sqrt(s);
        for (int i = j + 1; i < n; ++i) {
            double t = A[i * n + j];
            for (int k = 0; k < j; ++k) t -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = t / A[j * n + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < i; ++k) x[i] -= A[i * n + k] * x[k];
        x[i] /= A[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int k = i + 1; k < n; ++k) x[i] -= A[k * n + i] * x[k];
        x[i] /= A[i * n + i];
    }
}

}  // namespace

void compute_invweight0(const RobotModel& m, std::vector<float>& bw, std::vector<float>& dw) {
    const int nv = m.nv, nb = m.nbody();
    bw.assign(2 * nb, 0.0f);
    dw.assign(nv, 0.0f);
    if (nv == 0) return;
    const ModelView mv = m.view();
    RobotWork work(m);
    DynamicsWork w = work.view();
    const std::vector<float> q0 = m.qpos0();
    forward_kinematics(mv, q0.data(), w);
    body_inertias(mv, w);
    mass_matrix(mv, w);
    std::vector<double> M(work.M.begin(), work.M.end());

    // a^T M^-1 a for a row a of length nv.
    auto quad = [&](const std::vector<double>& a) {
        std::vector<double> x = a;
        solve_spd(M, nv, x);
        double s = 0;
        for (int i = 0; i < nv; ++i) s += a[i] * x[i];
        return s;
    };

    for (int b = 0; b < nb; ++b) {
        if (!body_moves(m, b)) continue;
        const Mat3 R = quat_to_mat3(load_quat(w.xquat + 4 * b));
        const Vec3 c = load3(w.xpos + 3 * b) + R * load3(m.com.data() + 3 * b);  // center of mass, world
        double tran = 0, rot = 0;
        for (int axis = 0; axis < 6; ++axis) {
            std::vector<double> a(nv, 0.0);
            for (int k = b; k >= 0; k = m.parent[k]) {
                if (m.jtype[k] == kJointFixed) continue;
                for (int i = m.dadr[k]; i < m.dadr[k] + joint_nv(m.jtype[k]); ++i) {
                    const SpatialVec& S = w.S[i];
                    const Vec3 v = S.lin + cross(S.ang, c);  // velocity of point c
                    const Vec3 u = axis < 3 ? v : S.ang;
                    const int k3 = axis % 3;
                    a[i] = k3 == 0 ? u.x : (k3 == 1 ? u.y : u.z);
                }
            }
            (axis < 3 ? tran : rot) += quad(a) / 3.0;
        }
        bw[2 * b] = static_cast<float>(tran);
        bw[2 * b + 1] = static_cast<float>(rot);
    }

    for (int b = 0; b < nb; ++b) {
        const int t = m.jtype[b];
        if (t == kJointFixed) continue;
        const int d0 = m.dadr[b];
        auto unit = [&](int i) {
            std::vector<double> a(nv, 0.0);
            a[i] = 1.0;
            return quad(a);
        };
        if (t == kJointFree) {
            const double tr = (unit(d0) + unit(d0 + 1) + unit(d0 + 2)) / 3.0;
            const double ro = (unit(d0 + 3) + unit(d0 + 4) + unit(d0 + 5)) / 3.0;
            for (int k = 0; k < 3; ++k) {
                dw[d0 + k] = static_cast<float>(tr);
                dw[d0 + 3 + k] = static_cast<float>(ro);
            }
        } else {
            dw[d0] = static_cast<float>(unit(d0));
        }
    }
}

ConstraintModel::ConstraintModel(const RobotModel& m, const ConstraintOptions& o) : opt(o), nbody(m.nbody()), nv(m.nv) {
    for (int b = 0; b < m.nbody(); ++b) {
        const int t = m.jtype[b];
        if ((t == kJointRevolute || t == kJointPrismatic) && m.joint_limited[b]) {
            if (!(m.joint_lower[b] < m.joint_upper[b]))
                throw std::runtime_error("joint '" + m.joint_name[b] + "': limit lower must be below upper");
            limit_qadr.push_back(m.qadr[b]);
            limit_dadr.push_back(m.dadr[b]);
            limit_lower.push_back(m.joint_lower[b]);
            limit_upper.push_back(m.joint_upper[b]);
        }
    }
    for (const auto& s : m.spheres) {
        if (!body_moves(m, s.body)) continue;  // welded to the world: MuJoCo skips it too
        sphere_body.push_back(s.body);
        sphere_pos.insert(sphere_pos.end(), {s.pos.x, s.pos.y, s.pos.z});
        sphere_radius.push_back(s.radius);
    }
    compute_invweight0(m, body_invweight0, dof_invweight0);
}

void ConstraintModel::add_plane(Vec3 n, Vec3 p) {
    const float len = length(n);
    if (!(len > 0.0f)) throw std::runtime_error("add_plane: zero normal");
    n = n / len;
    plane_normal.insert(plane_normal.end(), {n.x, n.y, n.z});
    plane_point.insert(plane_point.end(), {p.x, p.y, p.z});
}

void ConstraintModel::set_heightfield(int nx, int ny, float cell, float x0, float y0, std::vector<float> heights) {
    if (nx < 2 || ny < 2 || !(cell > 0.0f)) throw std::runtime_error("set_heightfield: needs nx, ny >= 2 and cell > 0");
    if (heights.size() != size_t(nx) * ny) throw std::runtime_error("set_heightfield: heights must have nx * ny values");
    hf_nx = nx;
    hf_ny = ny;
    hf_cell = cell;
    hf_x0 = x0;
    hf_y0 = y0;
    hf_heights = std::move(heights);
}

int ConstraintModel::max_rows() const {
    const int rows_per_contact = opt.friction > 0.0f ? 4 : 1;
    const int nground = static_cast<int>(plane_point.size() / 3) + (hf_heights.empty() ? 0 : 1);
    return 2 * static_cast<int>(limit_qadr.size()) + rows_per_contact * static_cast<int>(sphere_radius.size()) * nground;
}

ConstraintView ConstraintModel::view() const {
    ConstraintView v;
    v.nbody = nbody;
    v.nv = nv;
    v.nlimit = static_cast<int>(limit_qadr.size());
    v.limit_qadr = limit_qadr.data();
    v.limit_dadr = limit_dadr.data();
    v.limit_lower = limit_lower.data();
    v.limit_upper = limit_upper.data();
    v.nsphere = static_cast<int>(sphere_radius.size());
    v.sphere_body = sphere_body.data();
    v.sphere_pos = sphere_pos.data();
    v.sphere_radius = sphere_radius.data();
    v.nplane = static_cast<int>(plane_point.size() / 3);
    v.plane_normal = plane_normal.data();
    v.plane_point = plane_point.data();
    if (!hf_heights.empty()) v.hfield = HeightfieldView{hf_nx, hf_ny, hf_cell, hf_x0, hf_y0, hf_heights.data()};
    v.body_invweight0 = body_invweight0.data();
    v.dof_invweight0 = dof_invweight0.data();

    // MuJoCo getsolparam + mj_makeImpedance, "standard" (positive) solref.
    const float timeconst = std::max(opt.solref[0], 2.0f * opt.timestep);
    const float dampratio = opt.solref[1];
    for (int i = 0; i < 5; ++i) v.solimp[i] = opt.solimp[i];
    // MuJoCo clamps solimp: dmin, dmax in [0.0001, 0.9999], width >= 0, mid in [0.0001, 0.9999], power >= 1.
    v.solimp[0] = std::min(0.9999f, std::max(0.0001f, v.solimp[0]));
    v.solimp[1] = std::min(0.9999f, std::max(0.0001f, v.solimp[1]));
    v.solimp[2] = std::max(0.0f, v.solimp[2]);
    v.solimp[3] = std::min(0.9999f, std::max(0.0001f, v.solimp[3]));
    v.solimp[4] = std::max(1.0f, v.solimp[4]);
    const float dmax = v.solimp[1];
    v.K = 1.0f / std::max(1e-15f, dmax * dmax * timeconst * timeconst * dampratio * dampratio);
    v.B = 2.0f / std::max(1e-15f, dmax * timeconst);
    v.friction = opt.friction;
    v.impratio = opt.impratio;
    v.max_rows = max_rows();
    v.max_iterations = opt.max_iterations;
    return v;
}

}  // namespace forge
