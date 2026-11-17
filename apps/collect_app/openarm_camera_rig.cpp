#include "openarm_camera_rig.h"

#include <algorithm>
#include <stdexcept>

#include "flux/robot_mesh.h"
#include "forge/dynamics/articulated.h"

OpenArmCameraRig::OpenArmCameraRig(const forge::RobotModel& m, int num_envs, int size, const std::string& shader_dir,
                                   const std::string& cmesh)
    : m_(m), p_(forge::OpenArmPickPlaceTask::make_params(m)), n_(num_envs), work_(m), w_(work_.view()) {
    const flux::RobotMesh mesh = flux::load_robot_mesh(cmesh);
    for (const std::string& name : mesh.links) {
        const auto it = std::find(m.body_name.begin(), m.body_name.end(), name);
        if (it == m.body_name.end()) throw std::runtime_error(cmesh + ": link '" + name + "' is not a body of the model");
        mesh_bodies_.push_back(int(it - m.body_name.begin()));
    }
    flux::BatchCameraOptions o;
    o.num_envs = num_envs;
    o.num_cams = kCameras;
    o.width = o.height = size;
    o.points_per_env = 0;
    for (const auto& vb : m.visuals)
        if (vb.obs_xy >= 0) moving_.push_back(vb);
    quads_per_env_ = 6 * int(1 + moving_.size());  // the cube, the tray's boxes
    o.quads_per_env = quads_per_env_;
    o.shader_dir = shader_dir;
    // behind everything: the cell's matte black walls (cut away from the mesh so the
    // viewer can look in; the real cameras see them)
    o.clear_color[0] = o.clear_color[1] = o.clear_color[2] = 0.25f;
    cam_ = std::make_unique<flux::BatchCamera>(o);
    cam_->set_robot_mesh(mesh, 1);
    // the task's other drawn-only boxes (room floor): static, the same in every env
    std::vector<flux::Quad> fixed;
    for (const auto& vb : m.visuals) {
        if (vb.obs_xy >= 0) continue;
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

void OpenArmCameraRig::render(const float* qpos, const float* obs, int obs_dim) {
    const float orange[3] = {1.0f, 0.4f, 0.0f};  // the cell's cube
    const float h = forge::OpenArmPickPlaceTask::kHalf;
    const forge::ModelView mv = m_.view();
    const size_t nl = mesh_bodies_.size();
    for (int e = 0; e < n_; ++e) {
        const float* q = qpos + size_t(e) * m_.nq;
        forge::OpenArmPickPlaceTask::Camera cams[kCameras];
        forge::OpenArmPickPlaceTask::cameras(p_, mv, q, w_, cams);  // runs forward kinematics
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
        const forge::Mat3 Rc = forge::quat_to_mat3(forge::load_quat(w_.xquat + 4 * p_.cube));
        flux::Quad* cq = &quads_[size_t(e) * quads_per_env_];
        flux::box_quads(forge::load3(w_.xpos + 3 * p_.cube), forge::col(Rc, 0), forge::col(Rc, 1), forge::col(Rc, 2),
                        {h, h, h}, orange, cq);
        const float* o = obs + size_t(e) * obs_dim;
        for (size_t k = 0; k < moving_.size(); ++k) {
            const auto& vb = moving_[k];
            flux::box_quads({vb.center.x + o[vb.obs_xy], vb.center.y + o[vb.obs_xy + 1], vb.center.z}, {1, 0, 0},
                            {0, 1, 0}, {0, 0, 1}, {vb.half.x, vb.half.y, vb.half.z}, vb.rgb, cq + 6 * (k + 1));
        }
    }
    cam_->render(nullptr, nullptr, nullptr, nullptr, quads_.data(), views_.data(), poses_.data());
}
