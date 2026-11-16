#include "camera_rig.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "forge/dynamics/articulated.h"

PickCameraRig::PickCameraRig(const forge::RobotModel& m, int num_envs, int size, const std::string& shader_dir)
    : m_(m), p_(forge::PandaPickTask::make_params(m)), n_(num_envs), work_(m), w_(work_.view()) {
    // The cube is drawn as a box. On the hand and fingers the skeleton's joint and
    // mass markers (2-4 cm, for the viewer) would fill the wrist camera's view and
    // are bigger than the real fingers: capped at 1 cm (the 4 mm pad spheres stay).
    const int hand = m.body_index("hand"), lf = m.body_index("left_finger"), rf = m.body_index("right_finger");
    for (int k = 1;; ++k) {
        const int b = m.body_index("cube_" + std::to_string(k));
        if (b < 0) break;
        others_.push_back(b);
    }
    for (auto pt : forge::make_skeleton(m)) {
        if (pt.body == p_.cube || std::find(others_.begin(), others_.end(), pt.body) != others_.end()) continue;
        if (pt.body == hand || pt.body == lf || pt.body == rf) pt.radius = std::min(pt.radius, 0.01f);
        skel_.push_back(pt);
    }
    npts_ = static_cast<int>(skel_.size());
    flux::BatchCameraOptions o;
    o.num_envs = num_envs;
    o.num_cams = kCameras;
    o.width = o.height = size;
    o.points_per_env = npts_;
    o.quads_per_env = 6 * (1 + int(others_.size()));
    o.color_height = 1.0f;
    o.shader_dir = shader_dir;
    cam_ = std::make_unique<flux::BatchCamera>(o);
    // floor: 3 x 3 m, 10 cm grid (gives the cameras a sense of depth and motion)
    cam_->set_static_quads({flux::Quad{{0.5f, 0.0f, 0.0f}, {1.5f, 0, 0}, {0, 1.5f, 0}, {0.30f, 0.32f, 0.35f, 1.0f}, 0.1f}});
    const size_t total = size_t(num_envs) * npts_;
    px_.resize(total), py_.resize(total), pz_.resize(total), radius_.resize(total);
    for (int e = 0; e < num_envs; ++e)
        for (int k = 0; k < npts_; ++k) radius_[size_t(e) * npts_ + k] = skel_[k].radius;
    quads_.resize(size_t(num_envs) * 6 * (1 + others_.size()));
    views_.resize(size_t(num_envs) * kCameras);
}

void PickCameraRig::render(const float* qpos, const float* obs, int obs_dim) {
    const float orange[3] = {0.95f, 0.45f, 0.1f};
    // panda_pick_color's colors 0 red, 1 green, 2 yellow (not blue: the robot is drawn blue)
    const float color_rgb[3][3] = {{0.85f, 0.12f, 0.10f}, {0.15f, 0.70f, 0.20f}, {0.92f, 0.80f, 0.10f}};
    const int no = int(others_.size()), per = 6 * (1 + no);
    if (no > 0 && (!obs || obs_dim < 39 + 3 + 4 * no))
        throw std::runtime_error("PickCameraRig: panda_pick_color needs the obs (cube colors)");
    const forge::ModelView mv = m_.view();
    const float h = forge::PandaPickTask::kHalf;
    auto box = [&](int body, const float* rgb, flux::Quad* out) {
        const forge::Mat3 R = forge::quat_to_mat3(forge::load_quat(w_.xquat + 4 * body));
        flux::box_quads(forge::load3(w_.xpos + 3 * body), forge::col(R, 0), forge::col(R, 1), forge::col(R, 2),
                        {h, h, h}, rgb, out);
    };
    auto color = [&](float id) { return color_rgb[std::min(2, std::max(0, int(id + 0.5f)))]; };
    for (int e = 0; e < n_; ++e) {
        const float* q = qpos + size_t(e) * m_.nq;
        forge::PandaPickTask::Camera cams[kCameras];
        forge::PandaPickTask::cameras(p_, mv, q, w_, cams);  // runs forward kinematics
        for (int c = 0; c < kCameras; ++c)
            views_[size_t(e) * kCameras + c] = flux::CameraView{cams[c].eye, cams[c].target, cams[c].up, cams[c].fovy};
        const size_t o = size_t(e) * npts_;
        forge::skeleton_world(skel_.data(), npts_, w_.xpos, w_.xquat, px_.data() + o, py_.data() + o, pz_.data() + o);
        flux::Quad* qd = &quads_[size_t(e) * per];
        if (no == 0) {
            box(p_.cube, orange, qd);
        } else {
            const float* ob = obs + size_t(e) * obs_dim;
            int target = 0;  // obs 39-41: the target's color, one-hot
            for (int k = 1; k < 3; ++k)
                if (ob[39 + k] > ob[39 + target]) target = k;
            box(p_.cube, color_rgb[target], qd);
            for (int k = 0; k < no; ++k) box(others_[k], color(ob[42 + 3 * no + k]), qd + 6 * (k + 1));
        }
    }
    cam_->render(px_.data(), py_.data(), pz_.data(), radius_.data(), quads_.data(), views_.data());
}
