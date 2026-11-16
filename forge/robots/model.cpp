#include "forge/robots/model.h"

#include <stdexcept>

namespace forge {

int RobotModel::body_index(const std::string& n) const {
    for (int b = 0; b < nbody(); ++b)
        if (body_name[b] == n) return b;
    return -1;
}

int RobotModel::joint_index(const std::string& n) const {
    for (int b = 0; b < nbody(); ++b)
        if (joint_name[b] == n) return b;
    return -1;
}

std::vector<float> RobotModel::qpos0() const {
    std::vector<float> q(static_cast<size_t>(nq), 0.0f);
    for (int b = 0; b < nbody(); ++b)
        if (jtype[b] == kJointFree) q[qadr[b] + 3] = 1.0f;  // qw = 1
    return q;
}

ModelView RobotModel::view() const {
    ModelView v;
    v.nbody = nbody();
    v.nq = nq;
    v.nv = nv;
    v.parent = parent.data();
    v.jtype = jtype.data();
    v.qadr = qadr.data();
    v.dadr = dadr.data();
    v.jaxis = jaxis.data();
    v.body_pos = body_pos.data();
    v.body_quat = body_quat.data();
    v.mass = mass.data();
    v.com = com.data();
    v.inertia = inertia.data();
    v.dof_body = dof_body.data();
    v.dof_armature = dof_armature.empty() ? nullptr : dof_armature.data();
    v.dof_damping = dof_damping.empty() ? nullptr : dof_damping.data();
    v.gravity = gravity;
    return v;
}

void RobotModel::validate() const {
    const int n = nbody();
    auto fail = [&](const std::string& what) { throw std::runtime_error("RobotModel '" + name + "': " + what); };
    if (n == 0) fail("no bodies");
    if (n > kMaxBodies) fail("more than kMaxBodies bodies");
    if (!dof_armature.empty() && static_cast<int>(dof_armature.size()) != nv) fail("dof_armature must have nv entries");
    if (!dof_damping.empty() && static_cast<int>(dof_damping.size()) != nv) fail("dof_damping must have nv entries");
    for (float a : dof_armature) if (!(a >= 0.0f)) fail("armature must be >= 0");
    for (float d : dof_damping) if (!(d >= 0.0f)) fail("damping must be >= 0");
    if (nv > kMaxDofs) fail("more than kMaxDofs dofs");
    const size_t un = static_cast<size_t>(n);
    if (jtype.size() != un || qadr.size() != un || dadr.size() != un || jaxis.size() != 3 * un ||
        body_pos.size() != 3 * un || body_quat.size() != 4 * un || mass.size() != un || com.size() != 3 * un ||
        inertia.size() != 9 * un || body_name.size() != un || joint_name.size() != un)
        fail("array sizes do not match nbody");
    if (dof_body.size() != static_cast<size_t>(nv)) fail("dof_body size != nv");
    int q = 0, d = 0;
    for (int b = 0; b < n; ++b) {
        if (parent[b] >= b) fail("bodies are not in topological order (parent must come first)");
        if (jtype[b] == kJointFree && parent[b] != -1) fail("a free joint is only allowed on the root body");
        if (jtype[b] == kJointFixed) {
            if (qadr[b] != -1 || dadr[b] != -1) fail("fixed joint with qpos/dof address");
            continue;
        }
        if (qadr[b] != q || dadr[b] != d) fail("qpos/dof addresses are not consecutive");
        for (int k = 0; k < joint_nv(jtype[b]); ++k)
            if (dof_body[d + k] != b) fail("dof_body does not match dadr");
        q += joint_nq(jtype[b]);
        d += joint_nv(jtype[b]);
    }
    if (q != nq || d != nv) fail("nq/nv do not match the joints");
    for (const auto& s : spheres)
        if (s.body < 0 || s.body >= n) fail("sphere on a body that does not exist");
    for (const auto& bx : boxes) {
        if (bx.body < 0 || bx.body >= n) fail("box on a body that does not exist");
        if (!(bx.half.x > 0.0f && bx.half.y > 0.0f && bx.half.z > 0.0f)) fail("box half sizes must be > 0");
    }
}

int add_free_box(RobotModel& m, const std::string& name, Vec3 half, float mass) {
    if (!(mass > 0.0f)) throw std::runtime_error("add_free_box: mass must be > 0");
    const int b = m.nbody();
    m.body_name.push_back(name);
    m.joint_name.push_back(name);
    m.parent.push_back(-1);
    m.jtype.push_back(kJointFree);
    m.qadr.push_back(m.nq);
    m.dadr.push_back(m.nv);
    for (int k = 0; k < 6; ++k) m.dof_body.push_back(b);
    m.jaxis.insert(m.jaxis.end(), {0.0f, 0.0f, 1.0f});
    m.body_pos.insert(m.body_pos.end(), {0.0f, 0.0f, 0.0f});
    m.body_quat.insert(m.body_quat.end(), {0.0f, 0.0f, 0.0f, 1.0f});
    m.mass.push_back(mass);
    m.com.insert(m.com.end(), {0.0f, 0.0f, 0.0f});
    // solid box about its center: I_xx = m (b^2 + c^2) / 12 with full sizes 2 * half
    const float x2 = half.x * half.x, y2 = half.y * half.y, z2 = half.z * half.z;
    m.inertia.insert(m.inertia.end(), {mass * (y2 + z2) / 3.0f, 0.0f, 0.0f, 0.0f, mass * (x2 + z2) / 3.0f, 0.0f, 0.0f,
                                       0.0f, mass * (x2 + y2) / 3.0f});
    m.joint_lower.push_back(0.0f);
    m.joint_upper.push_back(0.0f);
    m.joint_effort.push_back(0.0f);
    m.joint_damping.push_back(0.0f);
    m.joint_limited.push_back(false);
    if (!m.dof_armature.empty()) m.dof_armature.insert(m.dof_armature.end(), 6, 0.0f);
    if (!m.dof_damping.empty()) m.dof_damping.insert(m.dof_damping.end(), 6, 0.0f);
    m.nq += 7;
    m.nv += 6;
    m.boxes.push_back({b, {0, 0, 0}, quat_identity(), half});
    m.validate();
    return b;
}

}  // namespace forge
