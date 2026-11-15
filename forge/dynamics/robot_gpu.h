#pragma once
// GPU storage and kernels for a batch of robots (Phase 5b).
// One thread per env runs the same FORGE_HD code as the CPU answer key.
// Host code only (CUDA runtime API).
#include <cuda_runtime.h>

#include "forge/dynamics/robot_batch.h"
#include "forge/dynamics/robot_batch_host.h"
#include "forge/robots/model.h"
#include "forge/util/device_buffer.h"

namespace forge {

// The robot model in GPU memory, shared by all envs. Copied once.
class RobotModelDevice {
public:
    explicit RobotModelDevice(const RobotModel& m);
    ModelView view() const { return view_; }

private:
    DeviceBuffer<int> parent_, jtype_, qadr_, dadr_, dof_body_;
    DeviceBuffer<float> jaxis_, body_pos_, body_quat_, mass_, com_, inertia_;
    ModelView view_;
};

// State and scratch of all envs in GPU memory.
class RobotBatchDevice {
public:
    RobotBatchDevice(const RobotModel& m, int num_envs);

    RobotBatchView view();
    WorkBatchView work();

    // Copy the state (qpos, qvel, tau) from / to a host batch of the same size.
    void upload_state(const RobotBatchHost& h);
    void download_state(RobotBatchHost& h) const;
    // Also copies the dynamics results (xpos, xquat, M, bias, qacc, status). For tests.
    void download_all(RobotBatchHost& h) const;

    int num_envs() const { return num_envs_; }

    DeviceBuffer<float> qpos, qvel, tau, qacc;
    DeviceBuffer<int> status;
    DeviceBuffer<float> xpos, xquat;
    DeviceBuffer<SpatialVec> S, V, A, F;
    DeviceBuffer<SpatialInertia> I, Ic;
    DeviceBuffer<float> M, L, bias;

private:
    int num_envs_, nbody_, nv_;
};

// 'steps' steps of dt for every env. Asynchronous, one kernel launch per step.
void robot_step_gpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w, float dt, int steps,
                    cudaStream_t stream = nullptr);

// Skeleton points of every env (runs FK on the current qpos). 'pts' must be in
// device memory. px/py/pz are device pointers with num_envs * npts entries
// (for example the shared Vulkan buffer: zero-copy drawing).
void skeleton_points_gpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w,
                         const SkeletonPoint* pts, int npts, float* px, float* py, float* pz,
                         cudaStream_t stream = nullptr);

// Forward dynamics for every env without moving the state. For tests.
void robot_dynamics_gpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w,
                        cudaStream_t stream = nullptr);

}  // namespace forge
