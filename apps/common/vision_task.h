#pragma once
// The vision tasks of the learning pipeline (refactor after Phase 11): everything the
// pipeline apps need to know about one task, in ONE place.
//
//   collect_app (demos) -> VLA (vla_openarm.py) -> vla_app (closed loop, DAgger)
//   -> distill_app (CNN + GRU student) -> rl_app (PPO fine-tune); sandbox_app
//   --vla / --student shows the policies
//
// A new vision task: its task file in forge/env/tasks/ (physics, obs, reward, expert),
// a camera rig if it needs a new scene (apps/collect_app/), and one entry in
// make_vision_tasks() (vision_task.cpp). The apps take everything else from here.
// (panda_pick and panda_pick_color, Phase 9, are older tasks with their own code in
// collect_app and vla_app; they are not pipeline tasks.)
//
// Builds without flux too (collect_app headless): then make_rig is null
// (vision_task.cpp compiled with CRUCIBLE_NO_CAMERAS).
#include <memory>
#include <string>
#include <vector>

#include "forge/env/env.h"
#ifndef CRUCIBLE_NO_CAMERAS
#include "flux/batch_camera.h"
#else
namespace flux {
class BatchCamera;
}
#endif

namespace pipeline {

// The cameras of a task: draws all envs' images from the host copy of the state.
class CameraRig {
public:
    virtual ~CameraRig() = default;
    virtual void render(const forge::Env::HostCopy& h, int obs_dim) = 0;
    virtual const flux::BatchCamera& camera() const = 0;
    // For rl_app --timing: seconds in forward kinematics / the Vulkan draw so far
    // (0 when the rig does not count them).
    virtual double fk_seconds() const { return 0.0; }
    virtual double draw_seconds() const { return 0.0; }
};

struct VisionTask {
    std::string name;            // forge's task name (EnvOptions::task)
    int act = 0;                 // action numbers per step
    int state = 0;               // the policy's state: the first 'state' numbers of obs (VLA and student)
    int obs = 0;                 // the full obs (the critic sees it, with the episode time)
    int max_steps = 0;           // episode limit
    int cameras = 0;             // camera ids 0 .. cameras - 1
    std::string student_cameras; // distill_app's default --student_cameras ("0,1,3")
    float physics_dt = 0.002f;   // as the task's config file
    int decimation = 10;
    std::string done_text;       // what "done" means, for the eval lines
    // true when the episode ended in success: final obs, terminated (not a time limit)
    bool (*done_ok)(const float* final_obs, bool terminated) = nullptr;
    // The new episode's actions before its first decision (rl_app, sandbox): hold still.
    void (*hold_still)(float* act) = nullptr;
    // The aux head's targets from obs (distill_app; aux_dim numbers). 0 / null: none.
    int aux_dim = 0;
    void (*aux)(const float* obs, float* out) = nullptr;
    // Env::set_task_input starts (rl_app --start_mix): how many numbers (0 = none) and
    // what they are, in order.
    int start_mix_dim = 0;
    std::string start_mix_names;
    // null in a build without cameras (CRUCIBLE_NO_CAMERAS)
    std::unique_ptr<CameraRig> (*make_rig)(const forge::RobotModel& m, int num_envs, int size) = nullptr;

    // collect_app: the gripper actions (--push_prob keeps them, --slip_prob opens one);
    // stream_frames: each env's frames go into its own zlib stream as they are rendered
    // (long episodes: keeping them raw would not fit in memory)
    std::vector<int> grips;
    bool stream_frames = false;

    // vla_app: where a failed episode stopped. 'progress' holds progress_dim numbers per env:
    // progress_reset() at the episode start, progress_step() with each obs before a step;
    // at a failure fail_kind() names the stage it reached and fail_detail() prints the
    // numbers (without a newline). Strips: one column per strip_every steps, envs 0 ..
    // strip_envs - 1 only (0 = all).
    int progress_dim = 0;
    void (*progress_reset)(float* progress) = nullptr;
    void (*progress_step)(const float* obs, float* progress) = nullptr;
    const char* (*fail_kind)(const float* progress, const float* final_obs) = nullptr;
    std::string (*fail_detail)(const forge::RobotModel& m, const float* progress, const float* final_obs) = nullptr;
    int strip_every = 10, strip_envs = 0;
    // the VLA's info file (_openarm.txt) must name the task (false: files from before the
    // 'task' line are taken as this task)
    bool info_needs_task = true;

    forge::EnvOptions env_options(int envs, uint64_t seed, forge::EnvBackend backend) const;
    int critic() const { return obs + 1; }  // rl_app's critic input: obs + episode time
};

// The task by name; throws, listing the known tasks, for an unknown one.
const VisionTask& vision_task(const std::string& name);
// The task by name, or null.
const VisionTask* find_vision_task(const std::string& name);
std::vector<std::string> vision_task_names();
std::string vision_task_list();  // "openarm_pick_place, openarm_stack, openarm_handover"

// 2 x 2 box filter: [S][S][3] -> [S/2][S/2][3] (the students' 2x supersampled images)
void halve(const uint8_t* in, int S, uint8_t* out);

}  // namespace pipeline
