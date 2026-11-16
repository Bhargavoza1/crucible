#include "forge/dynamics/robot_gpu.h"

#include "forge/util/cuda_check.h"

namespace forge {

namespace {

// One thread per env. Small blocks: each thread does a lot of serial work and
// uses many registers, and there are often only a few thousand envs.
constexpr int kBlock = 64;

__global__ void robot_step_kernel(ModelView m, RobotBatchView b, WorkBatchView w, float dt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= b.num_envs) return;
    robot_step_env(m, b, w, e, dt);
}

__global__ void robot_dynamics_kernel(ModelView m, RobotBatchView b, WorkBatchView w) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= b.num_envs) return;
    DynamicsWork dw = w.at(e);
    const float* tau = b.tau ? b.tau + e * m.nv : nullptr;
    const bool ok = forward_dynamics(m, b.qpos + e * m.nq, b.qvel + e * m.nv, tau, dw, b.qacc + e * m.nv);
    if (b.status) b.status[e] = ok ? 0 : 1;
}

__global__ void skeleton_kernel(ModelView m, RobotBatchView b, WorkBatchView w, const SkeletonPoint* pts, int npts,
                                float* px, float* py, float* pz) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= b.num_envs) return;
    DynamicsWork dw = w.at(e);
    forward_kinematics(m, b.qpos + e * m.nq, dw);
    const size_t o = size_t(e) * npts;
    skeleton_world(pts, npts, dw.xpos, dw.xquat, px + o, py + o, pz + o);
}

template <typename T>
void upload(DeviceBuffer<T>& d, const std::vector<T>& h) {
    d.upload(h);
}

template <typename T>
void copy_down(const DeviceBuffer<T>& d, std::vector<T>& h) {
    if (h.size() != d.size()) h.resize(d.size());
    if (!h.empty()) CUDA_CHECK(cudaMemcpy(h.data(), d.data(), h.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

}  // namespace

RobotModelDevice::RobotModelDevice(const RobotModel& m) {
    m.validate();
    upload(parent_, m.parent);
    upload(jtype_, m.jtype);
    upload(qadr_, m.qadr);
    upload(dadr_, m.dadr);
    upload(dof_body_, m.dof_body);
    upload(jaxis_, m.jaxis);
    upload(body_pos_, m.body_pos);
    upload(body_quat_, m.body_quat);
    upload(mass_, m.mass);
    upload(com_, m.com);
    upload(inertia_, m.inertia);
    if (!m.dof_armature.empty()) upload(armature_, m.dof_armature);
    if (!m.dof_damping.empty()) upload(damping_, m.dof_damping);
    view_.nbody = m.nbody();
    view_.nq = m.nq;
    view_.nv = m.nv;
    view_.parent = parent_.data();
    view_.jtype = jtype_.data();
    view_.qadr = qadr_.data();
    view_.dadr = dadr_.data();
    view_.jaxis = jaxis_.data();
    view_.body_pos = body_pos_.data();
    view_.body_quat = body_quat_.data();
    view_.mass = mass_.data();
    view_.com = com_.data();
    view_.inertia = inertia_.data();
    view_.dof_body = dof_body_.data();
    view_.dof_armature = m.dof_armature.empty() ? nullptr : armature_.data();
    view_.dof_damping = m.dof_damping.empty() ? nullptr : damping_.data();
    view_.gravity = m.gravity;
}

RobotBatchDevice::RobotBatchDevice(const RobotModel& m, int n)
    : num_envs_(n), nbody_(m.nbody()), nv_(m.nv) {
    const size_t N = size_t(n), nb = size_t(m.nbody()), nv = size_t(m.nv);
    qpos.resize(N * m.nq);
    qvel.resize(N * nv);
    tau.resize(N * nv);
    qacc.resize(N * nv);
    status.resize(N);
    xpos.resize(N * 3 * nb);
    xquat.resize(N * 4 * nb);
    S.resize(N * nv);
    I.resize(N * nb);
    Ic.resize(N * nb);
    V.resize(N * nb);
    A.resize(N * nb);
    F.resize(N * nb);
    M.resize(N * nv * nv);
    L.resize(N * nv * nv);
    bias.resize(N * nv);
    CUDA_CHECK(cudaMemset(tau.data(), 0, tau.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(status.data(), 0, status.size() * sizeof(int)));
}

RobotBatchView RobotBatchDevice::view() { return {num_envs_, qpos.data(), qvel.data(), tau.data(), qacc.data(), status.data()}; }

WorkBatchView RobotBatchDevice::work() {
    return {nbody_, nv_, xpos.data(), xquat.data(), S.data(), I.data(), Ic.data(),
            V.data(), A.data(), F.data(), M.data(), L.data(), bias.data()};
}

void RobotBatchDevice::upload_state(const RobotBatchHost& h) {
    upload(qpos, h.qpos);
    upload(qvel, h.qvel);
    upload(tau, h.tau);
    upload(status, h.status);
}

void RobotBatchDevice::download_state(RobotBatchHost& h) const {
    copy_down(qpos, h.qpos);
    copy_down(qvel, h.qvel);
    copy_down(qacc, h.qacc);
    copy_down(status, h.status);
}

void RobotBatchDevice::download_all(RobotBatchHost& h) const {
    download_state(h);
    copy_down(xpos, h.xpos);
    copy_down(xquat, h.xquat);
    copy_down(M, h.M);
    copy_down(bias, h.bias);
}

void robot_step_gpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w, float dt, int steps,
                    cudaStream_t stream) {
    if (b.num_envs == 0 || steps <= 0) return;
    const int grid = (b.num_envs + kBlock - 1) / kBlock;
    for (int s = 0; s < steps; ++s) {
        robot_step_kernel<<<grid, kBlock, 0, stream>>>(m, b, w, dt);
        KERNEL_CHECK();
    }
}

void skeleton_points_gpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w,
                         const SkeletonPoint* pts, int npts, float* px, float* py, float* pz, cudaStream_t stream) {
    if (b.num_envs == 0 || npts == 0) return;
    skeleton_kernel<<<(b.num_envs + kBlock - 1) / kBlock, kBlock, 0, stream>>>(m, b, w, pts, npts, px, py, pz);
    KERNEL_CHECK();
}

void robot_dynamics_gpu(const ModelView& m, const RobotBatchView& b, const WorkBatchView& w, cudaStream_t stream) {
    if (b.num_envs == 0) return;
    robot_dynamics_kernel<<<(b.num_envs + kBlock - 1) / kBlock, kBlock, 0, stream>>>(m, b, w);
    KERNEL_CHECK();
}

}  // namespace forge
