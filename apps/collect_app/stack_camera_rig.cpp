#include "stack_camera_rig.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "flux/robot_mesh.h"
#include "forge/dynamics/articulated.h"

namespace {
// The cameras need only the two hands (OpenArmStackTask::cameras): any OpenArm cell
// model works (openarm_stack's 3 cubes, openarm_handover's block)
forge::OpenArmStackTask::Params camera_params(const forge::RobotModel& m) {
    forge::OpenArmStackTask::Params p;
    p.hand = m.body_index("openarm_left_ee_base_link");
    p.hand_r = m.body_index("openarm_right_ee_base_link");
    if (p.hand < 0 || p.hand_r < 0) throw std::runtime_error("StackCameraRig: needs both ee_base_links");
    return p;
}
}  // namespace

StackCameraRig::StackCameraRig(const forge::RobotModel& m, int num_envs, int size, const std::string& shader_dir,
                               const std::string& cmesh)
    : m_(m), p_(camera_params(m)), n_(num_envs), work_(m), w_(work_.view()) {
    const flux::RobotMesh mesh = flux::load_robot_mesh(cmesh);
    for (const std::string& name : mesh.links) {
        const auto it = std::find(m.body_name.begin(), m.body_name.end(), name);
        if (it == m.body_name.end()) throw std::runtime_error(cmesh + ": link '" + name + "' is not a body of the model");
        mesh_bodies_.push_back(int(it - m.body_name.begin()));
    }
    for (size_t k = 0; k < m.boxes.size(); ++k)
        if (m.boxes[k].drawn) boxes_.push_back(int(k));
    flux::BatchCameraOptions o;
    o.num_envs = num_envs;
    o.num_cams = kCameras;
    o.width = o.height = size;
    o.points_per_env = 0;
    quads_per_env_ = 6 * int(boxes_.size());
    o.quads_per_env = quads_per_env_;
    o.shader_dir = shader_dir;
    o.clear_color[0] = o.clear_color[1] = o.clear_color[2] = 0.25f;  // the cell's black walls (as openarm_pick_place)
    cam_ = std::make_unique<flux::BatchCamera>(o);
    cam_->set_robot_mesh(mesh, 1);
    std::vector<flux::Quad> fixed;  // the drawn-only boxes: the same in every env
    for (const auto& vb : m.visuals) {
        flux::Quad q[6];
        flux::box_quads({vb.center.x, vb.center.y, vb.center.z}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                        {vb.half.x, vb.half.y, vb.half.z}, vb.rgb, q);
        fixed.insert(fixed.end(), q, q + 6);
    }
    cam_->set_static_quads(fixed);
    poses_.resize(size_t(num_envs) * mesh_bodies_.size() * 12);
    quads_.resize(size_t(num_envs) * quads_per_env_);
    views_.resize(size_t(num_envs) * kCameras);
}

void StackCameraRig::render(const float* qpos) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    const forge::ModelView mv = m_.view();
    const size_t nl = mesh_bodies_.size();
    for (int e = 0; e < n_; ++e) {
        const float* q = qpos + size_t(e) * m_.nq;
        forge::OpenArmStackTask::Camera cams[kCameras];
        forge::OpenArmStackTask::cameras(p_, mv, q, w_, cams);  // runs forward kinematics
        for (int c = 0; c < kCameras; ++c)
            views_[size_t(e) * kCameras + c] = flux::CameraView{cams[c].eye, cams[c].target, cams[c].up, cams[c].fovy};
        for (size_t k = 0; k < nl; ++k) {
            const int b = mesh_bodies_[k];
            const forge::Mat3 R = forge::quat_to_mat3(forge::load_quat(w_.xquat + 4 * b));
            float* o = poses_.data() + (size_t(e) * nl + k) * 12;
            for (int r = 0; r < 3; ++r) {
                o[4 * r + 0] = R(r, 0), o[4 * r + 1] = R(r, 1), o[4 * r + 2] = R(r, 2);
                o[4 * r + 3] = w_.xpos[3 * b + r];
            }
        }
        flux::Quad* cq = &quads_[size_t(e) * quads_per_env_];
        for (size_t k = 0; k < boxes_.size(); ++k) {
            const auto& bx = m_.boxes[size_t(boxes_[k])];
            const forge::Quat qb = forge::load_quat(w_.xquat + 4 * bx.body);
            const forge::Vec3 c = forge::load3(w_.xpos + 3 * bx.body) + forge::rotate(qb, bx.pos);
            const forge::Mat3 R = forge::quat_to_mat3(qb * bx.rot);
            flux::box_quads(c, forge::col(R, 0), forge::col(R, 1), forge::col(R, 2), {bx.half.x, bx.half.y, bx.half.z},
                            bx.rgb, cq + 6 * k);
        }
    }
    const auto t1 = Clock::now();
    cam_->render(nullptr, nullptr, nullptr, nullptr, quads_.data(), views_.data(), poses_.data());
    fk_seconds += std::chrono::duration<double>(t1 - t0).count();
    draw_seconds += std::chrono::duration<double>(Clock::now() - t1).count();
}
