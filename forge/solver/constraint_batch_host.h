#pragma once
// Host (CPU) storage for the constraint scratch of a batch of envs.
#include <vector>

#include "forge/dynamics/robot_batch_host.h"
#include "forge/solver/constraint_batch.h"
#include "forge/solver/constraint_model.h"

namespace forge {

class ConstraintBatchHost {
public:
    ConstraintBatchHost(const ConstraintModel& c, int nu, int num_envs)
        : nv_(c.nv), max_rows_(c.max_rows()), nu_(nu), ctrl(size_t(num_envs) * nu, 0.0f), nefc(num_envs, 0),
          iterations(num_envs, 0) {
        const size_t N = size_t(num_envs), r = N * max_rows_, v = N * nv_;
        nbody_ = c.nbody;
        rand_friction.assign(N, c.opt.friction);
        rand_body_scale.assign(N * size_t(c.nbody), 1.0f);
        rand_kp.assign(N, 1.0f);
        rand_kv.assign(N, 1.0f);
        rand_strength.assign(N, 1.0f);
        type.resize(r);
        id.resize(r);
        J.resize(r * nv_);
        pos.resize(r);
        aref.resize(r);
        R.resize(r);
        D.resize(r);
        jar.resize(r);
        Jd.resize(r);
        force.resize(r);
        state.resize(r);
        qfrc_smooth.resize(v);
        qacc_smooth.resize(v);
        H.resize(v * nv_);
        LH.resize(v * nv_);
        grad.resize(v);
        dir.resize(v);
        Ma.resize(v);
        Mdir.resize(v);
    }

    ConstraintBatchView view() {
        ConstraintBatchView b;
        b.nv = nv_;
        b.max_rows = max_rows_;
        b.nu = nu_;
        b.ctrl = nu_ > 0 ? ctrl.data() : nullptr;
        b.nefc = nefc.data();
        b.iterations = iterations.data();
        b.type = type.data();
        b.id = id.data();
        b.J = J.data();
        b.pos = pos.data();
        b.aref = aref.data();
        b.R = R.data();
        b.D = D.data();
        b.jar = jar.data();
        b.Jd = Jd.data();
        b.force = force.data();
        b.state = state.data();
        b.qfrc_smooth = qfrc_smooth.data();
        b.qacc_smooth = qacc_smooth.data();
        b.H = H.data();
        b.LH = LH.data();
        b.grad = grad.data();
        b.dir = dir.data();
        b.Ma = Ma.data();
        b.Mdir = Mdir.data();
        b.nbody = nbody_;
        b.rand_friction = rand_friction.data();
        b.rand_body_scale = rand_body_scale.data();
        b.rand_kp = rand_kp.data();
        b.rand_kv = rand_kv.data();
        b.rand_strength = rand_strength.data();
        return b;
    }

    int nu() const { return nu_; }
    int max_rows() const { return max_rows_; }

private:
    int nv_, max_rows_, nu_, nbody_ = 0;

public:
    std::vector<float> ctrl;
    std::vector<int> nefc, iterations;
    std::vector<int> type, id;
    std::vector<float> J, pos, aref, R, D, jar, Jd, force;
    std::vector<int> state;
    std::vector<float> qfrc_smooth, qacc_smooth, H, LH, grad, dir, Ma, Mdir;
    std::vector<float> rand_friction, rand_body_scale, rand_kp, rand_kv, rand_strength;
};

// CPU loops over all envs (the answer key for the GPU kernels).
void robot_step_constrained_cpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb,
                                float dt, int steps);
void robot_dynamics_constrained_cpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                    const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb);

}  // namespace forge
