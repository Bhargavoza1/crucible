#pragma once
// Camera images of the openarm_pick_place envs (Phase 10): the OpenArm Cell drawn from
// its meshes (openarm_cell.cmesh, the camera level of detail), the cube (a box) and the
// tray, from the task's four cameras (head, left wrist, ceiling, right wrist;
// OpenArmPickPlaceTask::cameras), by flux::BatchCamera. Forward kinematics on the
// host from qpos, so it works with both env backends.
#include <memory>
#include <string>
#include <vector>

#include "flux/batch_camera.h"
#include "forge/dynamics/robot_work.h"
#include "forge/env/tasks/openarm_pick_place.h"

class OpenArmCameraRig {
public:
    // cmesh: the robot's mesh file (assets/robots/openarm/openarm_cell.cmesh)
    OpenArmCameraRig(const forge::RobotModel& m, int num_envs, int size, const std::string& shader_dir,
                     const std::string& cmesh);
    void render(const float* qpos);  // num_envs * nq (host); then the images are in camera()
    const flux::BatchCamera& camera() const { return *cam_; }
    static constexpr int kCameras = forge::OpenArmPickPlaceTask::kCameras;

private:
    const forge::RobotModel& m_;
    forge::OpenArmPickPlaceTask::Params p_;
    int n_ = 0;
    std::vector<int> mesh_bodies_;
    std::unique_ptr<flux::BatchCamera> cam_;
    forge::RobotWork work_;
    forge::DynamicsWork w_;
    std::vector<float> poses_;
    std::vector<flux::Quad> quads_;
    std::vector<flux::CameraView> views_;
};
