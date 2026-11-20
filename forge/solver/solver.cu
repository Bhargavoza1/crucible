#include <algorithm>
#include "forge/solver/solver_gpu.h"

#include <stdexcept>

#include "forge/solver/newton_warp.cuh"

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

// ---- one warp per env ----------------------------------------------------------------

// make_constraints with the contact tests spread over the warp (Phase 11): lane 0
// writes the limit rows, then the contact slots (constraints.h) run 32 at a time, one
// per lane. Each lane counts its contacts, a prefix sum over the warp gives each one
// its place, then each lane builds its own contacts' rows. Same rows, in the same
// order, as make_constraints (also when max_rows cuts them off). Every lane must call
// it; cw.nefc is set in every lane.
__device__ inline void make_constraints_warp(int lane, const ModelView& m, const ConstraintView& c, const float* qpos,
                                             const float* qvel, const DynamicsWork& w, ConstraintWork& cw) {
    constexpr unsigned kAll = 0xffffffffu;
    int r0 = 0;
    if (lane == 0) r0 = limit_rows(c, m.nv, qpos, qvel, cw);
    r0 = __shfl_sync(kAll, r0, 0);
    const int nrow = contact_rows_per(c);
    const ContactSlots S = contact_slots(c);
    const int room = c.max_rows > r0 ? (c.max_rows - r0) / nrow : 0;  // contacts that fit
    int done = 0;  // contacts found so far (also those past room)
    for (int base = 0; base < S.total; base += 32) {
        const int slot = base + lane;
        int n = 0;
        if (slot < S.total)
            contact_slot(m, c, w, S, slot, [&](int, int, Vec3, Vec3, float dist, int, Vec3) { n += !(dist > 0.0f); });
        int inc = n;  // inclusive prefix sum of n over the lanes
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(kAll, inc, o);
            if (lane >= o) inc += v;
        }
        int k = done + inc - n;  // index of this lane's first contact
        if (n > 0)
            contact_slot(m, c, w, S, slot, [&](int b1, int b2, Vec3 p, Vec3 nn, float dist, int id, Vec3 t) {
                if (dist > 0.0f) return;
                if (k < room) contact_rows(m, c, qvel, w, cw, r0 + k * nrow, b1, b2, p, nn, dist, id, t);
                ++k;
            });
        done += __shfl_sync(kAll, inc, 31);
    }
    cw.nefc = r0 + nrow * min(done, room);
    __syncwarp();
}
constexpr int kWarpEnvsPerBlock = 4;           // 128 threads per block
constexpr size_t kMaxSharedPerBlock = 48 * 1024;  // default limit, no opt-in needed

// Where each array of one env sits in its shared-memory slice (in 4-byte words).
struct WarpLayout {
    int M, L, H, J, pos, aref, R, D, jar, Jd, force, qfrc, qacc_s, grad, dir, Ma, Mdir, qacc;  // float arrays
    int type, id, state, scal;  // int arrays; scal: [0] status, [1] nefc
    int words;
};

__host__ __device__ inline WarpLayout make_layout(int nv, int rows) {
    WarpLayout L{};
    int o = 0;
    auto take = [&o](int n) {
        const int at = o;
        o += n;
        return at;
    };
    L.M = take(nv * nv);
    L.L = take(nv * nv);
    L.H = take(nv * nv);
    L.J = take(rows * nv);
    L.pos = take(rows);
    L.aref = take(rows);
    L.R = take(rows);
    L.D = take(rows);
    L.jar = take(rows);
    L.Jd = take(rows);
    L.force = take(rows);
    L.qfrc = take(nv);
    L.qacc_s = take(nv);
    L.grad = take(nv);
    L.dir = take(nv);
    L.Ma = take(nv);
    L.Mdir = take(nv);
    L.qacc = take(nv);
    L.type = take(rows);
    L.id = take(rows);
    L.state = take(rows);
    L.scal = take(2);
    L.words = o;
    return L;
}

// kRowsGlobal (Phase 11): the rows (J and the per-row arrays) stay in the env's slice
// of global memory (cb), only the nv-sized arrays in shared memory: for models whose
// rows do not fit (many contact pairs, or many dofs: openarm_stack's two arms), which
// otherwise fell back to the thread-per-env kernel (32 envs in a warp, diverging).
template <bool kWriteRows, bool kIntegrate, bool kRowsGlobal = false>
__global__ void warp_constrained_kernel(ModelView m, ConstraintView c, ActuatorView a, RobotBatchView b,
                                        WorkBatchView wb, ConstraintBatchView cb, float dt, WarpLayout lay) {
    extern __shared__ float smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int e = blockIdx.x * int(blockDim.x >> 5) + warp;
    if (e >= b.num_envs) return;  // the whole warp leaves together
    float* f = smem + size_t(warp) * lay.words;
    int* fi = reinterpret_cast<int*>(f);
    const int nv = m.nv;

    DynamicsWork w = wb.at(e);
    w.M = f + lay.M;
    w.L = f + lay.L;
    ConstraintWork cw{0,          0,           fi + lay.type, fi + lay.id,   f + lay.J,     f + lay.pos,
                      f + lay.aref, f + lay.R,  f + lay.D,     f + lay.jar,   f + lay.Jd,    f + lay.force,
                      fi + lay.state, f + lay.qfrc, f + lay.qacc_s, f + lay.H, f + lay.H, f + lay.grad,
                      f + lay.dir, f + lay.Ma, f + lay.Mdir};
    if (kRowsGlobal) {  // the rows in global memory; the rest stays in shared memory
        const ConstraintWork g = cb.at(e);
        cw.type = g.type, cw.id = g.id, cw.J = g.J, cw.pos = g.pos, cw.aref = g.aref, cw.R = g.R, cw.D = g.D;
        cw.jar = g.jar, cw.Jd = g.Jd, cw.force = g.force, cw.state = g.state;
    }
    // (LH = H: the warp Cholesky factors H in place.)
    const float* qpos = b.qpos + size_t(e) * m.nq;
    const float* qvel = b.qvel + size_t(e) * nv;

    // Lane 0: kinematics, mass matrix, bias, smooth forces; then the whole warp: the rows.
    const EnvRandom rnd = cb.random_at(e);  // domain randomization (same as forward_dynamics_constrained)
    ConstraintView ce = c;
    if (rnd.friction >= 0.0f) ce.friction = rnd.friction;
    if (lane == 0) {
        forward_kinematics(m, qpos, w);
        body_inertias(m, w);
        if (rnd.body_scale)
            for (int k = 0; k < m.nbody; ++k) w.I[k] = scaled(w.I[k], rnd.body_scale[k]);
        mass_matrix(m, w);
        bias_forces(m, qvel, w);
        const float* tau = b.tau ? b.tau + size_t(e) * nv : nullptr;
        for (int i = 0; i < nv; ++i) cw.qfrc_smooth[i] = (tau ? tau[i] : 0.0f) - w.bias[i];
        add_passive_forces(m, qvel, cw.qfrc_smooth);
        if (cb.ctrl && a.nu > 0)
            apply_actuators(a, cb.ctrl + size_t(e) * a.nu, qpos, qvel, cw.qfrc_smooth, rnd.kp, rnd.kv, rnd.strength);
        fi[lay.scal] = cholesky_solve(w.M, nv, cw.qfrc_smooth, cw.qacc_smooth, w.L) ? kDynamicsOk : kMassNotSpd;
    }
    __syncwarp();
    int status = fi[lay.scal];
    if (status == kDynamicsOk) make_constraints_warp(lane, m, ce, qpos, qvel, w, cw);
    else cw.nefc = 0;
    float* qacc = f + lay.qacc;
    if (status == kDynamicsOk && !newton_solve_warp(lane, nv, w.M, cw, c.max_iterations, qacc))
        status = kHessianNotSpd;

    // Results to global memory.
    float* qacc_out = b.qacc + size_t(e) * nv;
    for (int i = lane; i < nv; i += 32) qacc_out[i] = status == kDynamicsOk ? qacc[i] : 0.0f;
    const ConstraintWork g = cb.at(e);
    const int ne = cw.nefc;
    if (!kRowsGlobal) for (int r = lane; r < ne; r += 32) g.force[r] = cw.force[r];
    if (kWriteRows && !kRowsGlobal) {
        for (int k = lane; k < ne * nv; k += 32) g.J[k] = cw.J[k];
        for (int r = lane; r < ne; r += 32) {
            g.type[r] = cw.type[r];
            g.pos[r] = cw.pos[r];
            g.aref[r] = cw.aref[r];
            g.R[r] = cw.R[r];
            g.D[r] = cw.D[r];
        }
    }
    if (lane == 0) {
        if (b.status) b.status[e] = status;
        if (cb.nefc) cb.nefc[e] = ne;
        if (cb.iterations) cb.iterations[e] = cw.iterations;
    }
    __syncwarp();
    if (kIntegrate && lane == 0 && status == kDynamicsOk)
        integrate_euler(m, b.qpos + size_t(e) * m.nq, b.qvel + size_t(e) * nv, qacc_out, dt);
}

// Launches the warp kernel if it fits; returns false if it does not.
template <bool kWriteRows, bool kIntegrate>
bool launch_warp(const ModelView& m, const ConstraintView& c, const ActuatorView& a, const RobotBatchView& b,
                 const WorkBatchView& w, const ConstraintBatchView& cb, float dt, cudaStream_t stream) {
    const size_t bytes = warp_solver_shared_bytes(m.nv, c.max_rows);
    if (bytes > 0) {
        const WarpLayout lay = make_layout(m.nv, c.max_rows);
        const int grid = (b.num_envs + kWarpEnvsPerBlock - 1) / kWarpEnvsPerBlock;
        warp_constrained_kernel<kWriteRows, kIntegrate>
            <<<grid, 32 * kWarpEnvsPerBlock, bytes * kWarpEnvsPerBlock, stream>>>(m, c, a, b, w, cb, dt, lay);
        KERNEL_CHECK();
        return true;
    }
    // the rows do not fit in shared memory: keep them in global memory (Phase 11)
    const WarpLayout lay = make_layout(m.nv, 0);
    const size_t small = size_t(lay.words) * 4;
    const int per_block = int(std::min<size_t>(kWarpEnvsPerBlock, kMaxSharedPerBlock / small));
    if (per_block < 1) return false;
    const int grid = (b.num_envs + per_block - 1) / per_block;
    warp_constrained_kernel<kWriteRows, kIntegrate, true>
        <<<grid, 32 * per_block, small * per_block, stream>>>(m, c, a, b, w, cb, dt, lay);
    KERNEL_CHECK();
    return true;
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
    if (!c.hf_heights.empty()) hf_heights_.upload(c.hf_heights);
    box_body_.upload(c.box_body);
    box_pos_.upload(c.box_pos);
    box_quat_.upload(c.box_quat);
    box_half_.upload(c.box_half);
    capsule_body_.upload(c.capsule_body);
    capsule_pos_.upload(c.capsule_pos);
    capsule_quat_.upload(c.capsule_quat);
    capsule_size_.upload(c.capsule_size);
    sphere_mask_.upload(c.sphere_mask);
    box_mask_.upload(c.box_mask);
    capsule_mask_.upload(c.capsule_mask);
    pair_ss_.upload(c.pair_ss);
    pair_sc_.upload(c.pair_sc);
    pair_cc_.upload(c.pair_cc);
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
    if (!c.hf_heights.empty()) view_.hfield.h = hf_heights_.data();
    view_.box_body = box_body_.data();
    view_.box_pos = box_pos_.data();
    view_.box_quat = box_quat_.data();
    view_.box_half = box_half_.data();
    view_.capsule_body = capsule_body_.data();
    view_.capsule_pos = capsule_pos_.data();
    view_.capsule_quat = capsule_quat_.data();
    view_.capsule_size = capsule_size_.data();
    view_.sphere_mask = sphere_mask_.data();
    view_.box_mask = box_mask_.data();
    view_.capsule_mask = capsule_mask_.data();
    view_.pair_ss = pair_ss_.data();
    view_.pair_sc = pair_sc_.data();
    view_.pair_cc = pair_cc_.data();
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
    nbody_ = c.nbody;
    rand_friction.upload(std::vector<float>(N, c.opt.friction));
    rand_body_scale.upload(std::vector<float>(N * size_t(c.nbody), 1.0f));
    rand_kp.upload(std::vector<float>(N, 1.0f));
    rand_kv.upload(std::vector<float>(N, 1.0f));
    rand_strength.upload(std::vector<float>(N, 1.0f));
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
    b.nbody = nbody_;
    b.rand_friction = rand_friction.data();
    b.rand_body_scale = rand_body_scale.data();
    b.rand_kp = rand_kp.data();
    b.rand_kv = rand_kv.data();
    b.rand_strength = rand_strength.data();
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

size_t warp_solver_shared_bytes(int nv, int max_rows) {
    const size_t bytes = size_t(make_layout(nv, max_rows).words) * 4;
    return bytes * kWarpEnvsPerBlock <= kMaxSharedPerBlock ? bytes : 0;
}

void robot_step_constrained_gpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb,
                                float dt, int steps, cudaStream_t stream, GpuSolverKernel kernel) {
    if (b.num_envs == 0 || steps <= 0) return;
    FORGE_NVTX_RANGE("robot_step_constrained_gpu");
    const int grid = (b.num_envs + kBlock - 1) / kBlock;
    for (int s = 0; s < steps; ++s) {
        if (kernel == GpuSolverKernel::kWarpPerEnv && launch_warp<false, true>(m, c, a, b, w, cb, dt, stream))
            continue;
        step_constrained_kernel<<<grid, kBlock, 0, stream>>>(m, c, a, b, w, cb, dt);
        KERNEL_CHECK();
    }
}

void robot_dynamics_constrained_gpu(const ModelView& m, const ConstraintView& c, const ActuatorView& a,
                                    const RobotBatchView& b, const WorkBatchView& w, const ConstraintBatchView& cb,
                                    cudaStream_t stream, GpuSolverKernel kernel) {
    if (b.num_envs == 0) return;
    if (kernel == GpuSolverKernel::kWarpPerEnv && launch_warp<true, false>(m, c, a, b, w, cb, 0.0f, stream)) return;
    dynamics_constrained_kernel<<<(b.num_envs + kBlock - 1) / kBlock, kBlock, 0, stream>>>(m, c, a, b, w, cb);
    KERNEL_CHECK();
}

std::unique_ptr<Solver> make_solver_gpu(const std::string& name, cudaStream_t stream) {
    if (name == "newton") return std::make_unique<NewtonSolverGpu>(stream);
    // Same solver, thread-per-env kernel (for comparisons and debugging).
    if (name == "newton_thread") return std::make_unique<NewtonSolverGpu>(stream, GpuSolverKernel::kThreadPerEnv);
    throw std::runtime_error("unknown solver '" + name + "' (known: newton, newton_thread)");
}

}  // namespace forge
