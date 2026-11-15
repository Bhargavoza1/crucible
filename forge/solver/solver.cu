#include "forge/solver/solver_gpu.h"

#include "forge/util/cuda_check.h"
#include "forge/util/nvtx.h"

namespace forge {

namespace {

constexpr int kBlock = 64;  // same as the Phase 5 robot kernel; to be tuned with ncu

__global__ void step_constrained_kernel(ModelView m, ConstraintView c, ActuatorView a, RobotBatchView b,
                                        WorkBatchView w, ConstraintBatchView cb, float dt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= b.num_envs) return;
    robot_step_constrained_env(m, c, a, b, w, cb, e, dt);
}

__global__ void dynamics_constrained_kernel(ModelView m, ConstraintView c, ActuatorView a, RobotBatchView b,
                                            WorkBatchView w, ConstraintBatchView cb) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= b.num_envs) return;
    robot_dynamics_constrained_env(m, c, a, b, w, cb, e);
}

template <typename T>
void copy_down(const DeviceBuffer<T>& d, std::vector<T>& h) {
    if (h.size() != d.size()) h.resize(d.size());
    if (!h.empty()) CUDA_CHECK(cudaMemcpy(h.data(), d.data(), h.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

}  // namespace

ConstraintModelDevice::ConstraintModelDevice(const ConstraintModel& c) {
    limit_qadr_.upload(c.limit_qadr);
    limit_dadr_.upload(c.limit_dadr);
    limit_lower_.upload(c.limit_lower);
    limit_upper_.upload(c.limit_upper);
    sphere_body_.upload(c.sphere_body);
    sphere_pos_.upload(c.sphere_pos);
    sphere_radius_.upload(c.sphere_radius);
    plane_normal_.upload(c.plane_normal);
    plane_point_.upload(c.plane_point);
    body_invweight0_.upload(c.body_invweight0);
    dof_invweight0_.upload(c.dof_invweight0);
    view_ = c.view();  // sizes and constants; now point it at device memory
    view_.limit_qadr = limit_qadr_.data();
    view_.limit_dadr = limit_dadr_.data();
    view_.limit_lower = limit_lower_.data();
    view_.limit_upper = limit_upper_.data();
    view_.sphere_body = sphere_body_.data();
    view_.sphere_pos = sphere_pos_.data();
    view_.sphere_radius = sphere_radius_.data();
    view_.plane_normal = plane_normal_.data();
    view_.plane_point = plane_point_.data();
    view_.body_invweight0 = body_invweight0_.data();
    view_.dof_invweight0 = dof_invweight0_.data();
}

ActuatorsDevice::ActuatorsDevice(const Actuators& a) {
    qadr_.upload(a.qadr);
    dadr_.upload(a.dadr);
    kp_.upload(a.kp);
    kv_.upload(a.kv);
    force_lo_.upload(a.force_lo);
    force_hi_.upload(a.force_hi);
    view_ = {a.nu(), qadr_.data(), dadr_.data(), kp_.data(), kv_.data(), force_lo_.data(), force_hi_.data()};
}

ConstraintBatchDevice::ConstraintBatchDevice(const ConstraintModel& c, int nu, int n)
    : nv_(c.nv), max_rows_(c.max_rows()), nu_(nu) {
    const size_t N = size_t(n), r = N * max_rows_, v = N * nv_;
    ctrl.resize(N * nu);
    nefc.resize(N);
    iterations.resize(N);
    type.resize(r);
    id.resize(r);
    state.resize(r);
    J.resize(r * nv_);
    pos.resize(r);
    aref.resize(r);
    R.resize(r);
    D.resize(r);
    jar.resize(r);
    Jd.resize(r);
    force.resize(r);
    qfrc_smooth.resize(v);
    qacc_smooth.resize(v);
    H.resize(v * nv_);
    LH.resize(v * nv_);
    grad.resize(v);
    dir.resize(v);
    Ma.resize(v);
    Mdir.resize(v);
    if (ctrl.size()) CUDA_CHECK(cudaMemset(ctrl.data(), 0, ctrl.size() * sizeof(float)));
}

ConstraintBatchView ConstraintBatchDevice::view() {
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

void ConstraintBatchDevice::upload_ctrl(const ConstraintBatchHost& h) {
    if (!h.ctrl.empty()) ctrl.upload(h.ctrl);
}

void ConstraintBatchDevice::download_all(ConstraintBatchHost& h) const {
    copy_down(nefc, h.nefc);
    copy_down(iterations, h.iterations);
    copy_down(type, h.type);
    copy_down(J, h.J);
    copy_down(pos, h.pos);
    copy_down(aref, h.aref);
    copy_down(R, h.R);
    copy_down(D, h.D);
    copy_down(force, h.force);
}

void robot_step_constrained_gpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb,
                                float dt, int steps, cudaStream_t stream) {
    if (b.num_envs == 0 || steps <= 0) return;
    FORGE_NVTX_RANGE("robot_step_constrained_gpu");
    const int grid = (b.num_envs + kBlock - 1) / kBlock;
    for (int s = 0; s < steps; ++s) {
        step_constrained_kernel<<<grid, kBlock, 0, stream>>>(m, c, a, b, w, cb, dt);
        KERNEL_CHECK();
    }
}

void robot_dynamics_constrained_gpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                    const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb,
                                    cudaStream_t stream) {
    if (b.num_envs == 0) return;
    dynamics_constrained_kernel<<<(b.num_envs + kBlock - 1) / kBlock, kBlock, 0, stream>>>(m, c, a, b, w, cb);
    KERNEL_CHECK();
}

}  // namespace forge
