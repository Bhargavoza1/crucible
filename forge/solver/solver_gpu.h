#pragma once
// GPU storage and kernels for constrained robot steps (Phase 6b).
// Host code only (CUDA runtime API). Two kernels:
//   kThreadPerEnv  one thread per env runs the same FORGE_HD code as the CPU
//                  answer key (constraint_batch.h), scratch in global memory
//   kWarpPerEnv    one warp per env, the solver scratch (M, H, J, rows) in
//                  shared memory, the Newton loops split over the 32 lanes
//                  (newton_warp.cuh). Kinematics and dynamics run on lane 0.
//                  Used when one env's scratch fits in shared memory; else
//                  the thread kernel runs instead.
#include <cuda_runtime.h>

#include "forge/robots/actuators.h"
#include "forge/solver/constraint_batch.h"
#include "forge/solver/constraint_batch_host.h"
#include "forge/solver/constraint_model.h"
#include "forge/solver/solver.h"
#include "forge/util/device_buffer.h"

namespace forge {

// ConstraintModel in GPU memory (limits, spheres, planes, boxes, terrain, invweight0). Copied once.
class ConstraintModelDevice {
public:
    explicit ConstraintModelDevice(const ConstraintModel& c);
    ConstraintView view() const { return view_; }

private:
    DeviceBuffer<int> limit_qadr_, limit_dadr_, sphere_body_, box_body_, capsule_body_;
    DeviceBuffer<float> limit_lower_, limit_upper_, sphere_pos_, sphere_radius_, plane_normal_, plane_point_,
        body_invweight0_, dof_invweight0_, hf_heights_, box_pos_, box_quat_, box_half_, capsule_pos_,
        capsule_quat_, capsule_size_;
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
    DeviceBuffer<float> rand_friction, rand_body_scale, rand_kp, rand_kv, rand_strength;  // see ConstraintBatchView

private:
    int nv_, max_rows_, nu_, nbody_ = 0;
};

enum class GpuSolverKernel { kThreadPerEnv, kWarpPerEnv };

// Bytes of shared memory one env needs in the warp kernel (0 if it does not fit,
// then the thread kernel is used).
size_t warp_solver_shared_bytes(int nv, int max_rows);

// 'steps' constrained steps of dt for every env. Asynchronous, one launch per step.
// After a step, cb holds nefc, iterations and force of every env; the other row
// fields (J, pos, R, D, aref) only with kThreadPerEnv.
void robot_step_constrained_gpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb,
                                float dt, int steps, cudaStream_t stream = nullptr,
                                GpuSolverKernel kernel = GpuSolverKernel::kWarpPerEnv);

// Constrained forward dynamics for every env, state not moved. Writes all row
// fields of cb. For tests.
void robot_dynamics_constrained_gpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                    const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb,
                                    cudaStream_t stream = nullptr,
                                    GpuSolverKernel kernel = GpuSolverKernel::kWarpPerEnv);

// Newton on the GPU (doc 3.8). Uses the warp-per-env kernel when it fits.
class NewtonSolverGpu : public Solver {
public:
    explicit NewtonSolverGpu(cudaStream_t stream = nullptr, GpuSolverKernel kernel = GpuSolverKernel::kWarpPerEnv)
        : stream_(stream), kernel_(kernel) {}
    const char* name() const override { return "newton (gpu)"; }
    void step(const PhysicsViews& v, float dt, int steps) override {
        robot_step_constrained_gpu(v.model, v.constraints, v.actuators, v.state, v.work, v.constraint_work, dt,
                                   steps, stream_, kernel_);
    }

private:
    cudaStream_t stream_;
    GpuSolverKernel kernel_;
};

// "newton" -> NewtonSolverGpu. Throws for an unknown name.
std::unique_ptr<Solver> make_solver_gpu(const std::string& name, cudaStream_t stream = nullptr);

}  // namespace forge
