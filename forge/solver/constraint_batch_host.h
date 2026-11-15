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
        return b;
    }

    int nu() const { return nu_; }
    int max_rows() const { return max_rows_; }

private:
    int nv_, max_rows_, nu_;

public:
    std::vector<float> ctrl;
    std::vector<int> nefc, iterations;
    std::vector<int> type, id;
    std::vector<float> J, pos, aref, R, D, jar, Jd, force;
    std::vector<int> state;
    std::vector<float> qfrc_smooth, qacc_smooth, H, LH, grad, dir, Ma, Mdir;
};

// CPU loops over all envs (the answer key for the GPU kernels).
void robot_step_constrained_cpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb,
                                float dt, int steps);
void robot_dynamics_constrained_cpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                    const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb);

}  // namespace forge
