#pragma once
// Camera images of the two-arm OpenArm Cell envs (Phase 11 openarm_stack, Phase 12
// openarm_handover): the cell (both arms) from its meshes (openarm_cell.cmesh, the camera
// level of detail), the task's boxes in their colors (the stack's cubes, the handover's
// block), the drawn-only boxes (room floor, the gray spot), from the four cameras (head,
// left wrist, ceiling, right wrist; OpenArmStackTask::cameras, the same in both tasks),
// by flux::BatchCamera. Forward kinematics on the host from
// qpos, so it works with both env backends.
#include <memory>
#include <string>
#include <vector>

#include "flux/batch_camera.h"
#include "forge/dynamics/robot_work.h"
#include "forge/env/tasks/openarm_stack.h"

class StackCameraRig {
public:
    StackCameraRig(const forge::RobotModel& m, int num_envs, int size, const std::string& shader_dir,
                   const std::string& cmesh);
    // qpos: num_envs * nq (host); then the images are in camera()
    void render(const float* qpos);
    const flux::BatchCamera& camera() const { return *cam_; }
    static constexpr int kCameras = forge::OpenArmStackTask::kCameras;
    // Seconds spent in render() so far: forward kinematics + quads (host), and the Vulkan
    // draw + copy of the atlas to the host. For timing only (rl_app --timing).
    double fk_seconds = 0, draw_seconds = 0;

private:
    const forge::RobotModel& m_;
    forge::OpenArmStackTask::Params p_;
    int n_ = 0;
    std::vector<int> mesh_bodies_, boxes_;  // the drawn boxes (m.boxes indices)
    std::unique_ptr<flux::BatchCamera> cam_;
    forge::RobotWork work_;
    forge::DynamicsWork w_;
    std::vector<float> poses_;
    int quads_per_env_ = 0;
    std::vector<flux::Quad> quads_;
    std::vector<flux::CameraView> views_;
};
