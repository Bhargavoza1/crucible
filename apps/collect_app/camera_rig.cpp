#include "camera_rig.h"

#include <algorithm>

#include "forge/dynamics/articulated.h"

PickCameraRig::PickCameraRig(const forge::RobotModel& m, int num_envs, int size, const std::string& shader_dir)
    : m_(m), p_(forge::PandaPickTask::make_params(m)), n_(num_envs), work_(m), w_(work_.view()) {
    // The cube is drawn as a box. On the hand and fingers the skeleton's joint and
    // mass markers (2-4 cm, for the viewer) would fill the wrist camera's view and
    // are bigger than the real fingers: capped at 1 cm (the 4 mm pad spheres stay).
    const int hand = m.body_index("hand"), lf = m.body_index("left_finger"), rf = m.body_index("right_finger");
    for (auto pt : forge::make_skeleton(m)) {
        if (pt.body == p_.cube) continue;
        if (pt.body == hand || pt.body == lf || pt.body == rf) pt.radius = std::min(pt.radius, 0.01f);
        skel_.push_back(pt);
    }
    npts_ = static_cast<int>(skel_.size());
    flux::BatchCameraOptions o;
    o.num_envs = num_envs;
    o.num_cams = kCameras;
    o.width = o.height = size;
    o.points_per_env = npts_;
    o.quads_per_env = 6;
    o.color_height = 1.0f;
    o.shader_dir = shader_dir;
    cam_ = std::make_unique<flux::BatchCamera>(o);
    // floor: 3 x 3 m, 10 cm grid (gives the cameras a sense of depth and motion)
    cam_->set_static_quads({flux::Quad{{0.5f, 0.0f, 0.0f}, {1.5f, 0, 0}, {0, 1.5f, 0}, {0.30f, 0.32f, 0.35f, 1.0f}, 0.1f}});
    const size_t total = size_t(num_envs) * npts_;
    px_.resize(total), py_.resize(total), pz_.resize(total), radius_.resize(total);
    for (int e = 0; e < num_envs; ++e)
        for (int k = 0; k < npts_; ++k) radius_[size_t(e) * npts_ + k] = skel_[k].radius;
    quads_.resize(size_t(num_envs) * 6);
    views_.resize(size_t(num_envs) * kCameras);
}

void PickCameraRig::render(const float* qpos) {
    const float cube_rgb[3] = {0.95f, 0.45f, 0.1f};
    const forge::ModelView mv = m_.view();
    for (int e = 0; e < n_; ++e) {
        const float* q = qpos + size_t(e) * m_.nq;
        forge::PandaPickTask::Camera cams[kCameras];
        forge::PandaPickTask::cameras(p_, mv, q, w_, cams);  // runs forward kinematics
        for (int c = 0; c < kCameras; ++c)
            views_[size_t(e) * kCameras + c] = flux::CameraView{cams[c].eye, cams[c].target, cams[c].up, cams[c].fovy};
        const size_t o = size_t(e) * npts_;
        forge::skeleton_world(skel_.data(), npts_, w_.xpos, w_.xquat, px_.data() + o, py_.data() + o, pz_.data() + o);
        const forge::Mat3 R = forge::quat_to_mat3(forge::load_quat(w_.xquat + 4 * p_.cube));
        const float h = forge::PandaPickTask::kHalf;
        flux::box_quads(forge::load3(w_.xpos + 3 * p_.cube), forge::col(R, 0), forge::col(R, 1), forge::col(R, 2),
                        {h, h, h}, cube_rgb, &quads_[size_t(e) * 6]);
    }
    cam_->render(px_.data(), py_.data(), pz_.data(), radius_.data(), quads_.data(), views_.data());
}
