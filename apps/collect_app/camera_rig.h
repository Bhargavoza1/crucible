#pragma once
// Camera images of the panda_pick envs (Phase 9c): each env's robot (skeleton
// spheres), cube (a box; panda_pick_color: three colored boxes) and floor, from the task's two cameras (front, wrist;
// PandaPickTask::cameras), drawn by flux::BatchCamera. Forward kinematics on the
// host from qpos, so it works with both env backends.
#include <memory>
#include <string>
#include <vector>

#include "flux/batch_camera.h"
#include "forge/dynamics/robot_work.h"
#include "forge/env/tasks/panda_pick.h"
#include "forge/robots/skeleton.h"

class PickCameraRig {
public:
    PickCameraRig(const forge::RobotModel& m, int num_envs, int size, const std::string& shader_dir);
    // qpos: num_envs * nq (host). After this, images are in camera(). obs (num_envs *
    // obs_dim, host) is needed for panda_pick_color only: the cube colors.
    void render(const float* qpos, const float* obs = nullptr, int obs_dim = 0);
    const flux::BatchCamera& camera() const { return *cam_; }
    static constexpr int kCameras = forge::PandaPickTask::kCameras;

private:
    const forge::RobotModel& m_;
    forge::PandaPickTask::Params p_;
    int n_ = 0, npts_ = 0;
    std::vector<forge::SkeletonPoint> skel_;
    std::vector<int> others_;  // panda_pick_color: bodies cube_1, cube_2
    std::unique_ptr<flux::BatchCamera> cam_;
    forge::RobotWork work_;
    forge::DynamicsWork w_;
    std::vector<float> px_, py_, pz_, radius_;
    std::vector<flux::Quad> quads_;
    std::vector<flux::CameraView> views_;
};
