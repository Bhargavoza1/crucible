#include "forge/robots/ik.h"

#include <stdexcept>

#include "forge/dynamics/robot_work.h"

namespace forge {

IkJoints make_ik_joints(const RobotModel& m, const std::vector<int>& joints, const std::vector<float>& posture) {
    if (joints.size() > size_t(kIkMaxJoints)) throw std::runtime_error("ik: more than kIkMaxJoints joints");
    if (!posture.empty() && posture.size() != joints.size()) throw std::runtime_error("ik: posture needs one value per joint");
    IkJoints jt;
    jt.n = static_cast<int>(joints.size());
    for (int c = 0; c < jt.n; ++c) {
        const int j = joints[c];
        if (j < 0 || j >= m.nbody() || (m.jtype[j] != kJointRevolute && m.jtype[j] != kJointPrismatic))
            throw std::runtime_error("ik: joints must be revolute or prismatic");
        jt.body[c] = j;
        jt.lower[c] = m.joint_limited[j] ? m.joint_lower[j] : 0.0f;
        jt.upper[c] = m.joint_limited[j] ? m.joint_upper[j] : 0.0f;
        jt.posture[c] = posture.empty() ? 0.0f : posture[c];
    }
    return jt;
}

IkResult solve_ik(const RobotModel& m, int body, Vec3 target_pos, Quat target_rot, const std::vector<int>& joints,
                  std::vector<float>& qpos, const IkOptions& opt, const std::vector<float>& posture) {
    if (body < 0 || body >= m.nbody()) throw std::runtime_error("solve_ik: no such body");
    const IkJoints jt = make_ik_joints(m, joints, posture);
    const ModelView mv = m.view();
    RobotWork work(m);
    DynamicsWork w = work.view();
    IkResult res;
    for (int it = 0;; ++it) {
        forward_kinematics(mv, qpos.data(), w);
        IkResult now;
        if (it == opt.max_iterations) {  // errors only
            IkJoints none = jt;
            none.n = 0;
            ik_step(mv, w, body, target_pos, target_rot, none, qpos.data(), opt, now);
        } else {
            ik_step(mv, w, body, target_pos, target_rot, jt, qpos.data(), opt, now);
        }
        res.pos_error = now.pos_error;
        res.rot_error = now.rot_error;
        res.iterations = it;
        // with a posture term the joints keep moving along the free motion: run all iterations
        if (it == opt.max_iterations || (opt.posture_gain <= 0.0f && now.pos_error + now.rot_error < opt.tolerance)) break;
    }
    return res;
}

}  // namespace forge
