#pragma once
// GPU storage and kernels for constrained robot steps (Phase 6b).
// Version 1: one thread per env runs the same FORGE_HD code as the CPU answer
// key (constraint_batch.h). Host code only (CUDA runtime API).
#include <cuda_runtime.h>

#include "forge/robots/actuators.h"
#include "forge/solver/constraint_batch.h"
#include "forge/solver/constraint_batch_host.h"
#include "forge/solver/constraint_model.h"
#include "forge/util/device_buffer.h"

namespace forge {

// ConstraintModel in GPU memory (limits, spheres, planes, invweight0). Copied once.
class ConstraintModelDevice {
public:
    explicit ConstraintModelDevice(const ConstraintModel& c);
    ConstraintView view() const { return view_; }

private:
    DeviceBuffer<int> limit_qadr_, limit_dadr_, sphere_body_;
    DeviceBuffer<float> limit_lower_, limit_upper_, sphere_pos_, sphere_radius_, plane_normal_, plane_point_,
        body_invweight0_, dof_invweight0_;
    ConstraintView view_;
};

// Actuators in GPU memory. Copied once.
class ActuatorsDevice {
public:
    explicit ActuatorsDevice(const Actuators& a);
    ActuatorView view() const { return view_; }

private:
    DeviceBuffer<int> qadr_, dadr_;
    DeviceBuffer<float> kp_, kv_, force_lo_, force_hi_;
    ActuatorView view_;
};

// Constraint scratch + ctrl of all envs in GPU memory.
class ConstraintBatchDevice {
public:
    ConstraintBatchDevice(const ConstraintModel& c, int nu, int num_envs);
    ConstraintBatchView view();

    void upload_ctrl(const ConstraintBatchHost& h);
    // nefc, iterations, and the rows (J, pos, R, D, aref, force). For tests.
    void download_all(ConstraintBatchHost& h) const;

    DeviceBuffer<float> ctrl;
    DeviceBuffer<int> nefc, iterations, type, id, state;
    DeviceBuffer<float> J, pos, aref, R, D, jar, Jd, force, qfrc_smooth, qacc_smooth, H, LH, grad, dir, Ma, Mdir;

private:
    int nv_, max_rows_, nu_;
};

// 'steps' constrained steps of dt for every env. Asynchronous, one launch per step.
void robot_step_constrained_gpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb,
                                float dt, int steps, cudaStream_t stream = nullptr);

// Constrained forward dynamics for every env, state not moved. For tests.
void robot_dynamics_constrained_gpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                    const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb,
                                    cudaStream_t stream = nullptr);

}  // namespace forge
