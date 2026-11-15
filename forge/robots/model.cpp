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
    v.gravity = gravity;
    return v;
}

void RobotModel::validate() const {
    const int n = nbody();
    auto fail = [&](const std::string& what) { throw std::runtime_error("RobotModel '" + name + "': " + what); };
    if (n == 0) fail("no bodies");
    if (n > kMaxBodies) fail("more than kMaxBodies bodies");
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
        if (b > 0 && parent[b] < 0) fail("only the first body may hang from the world");
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
}

}  // namespace forge
