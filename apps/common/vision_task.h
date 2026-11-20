#pragma once
// The vision tasks of the learning pipeline (refactor after Phase 11): everything the
// pipeline apps (distill_app, rl_app) need to know about one task, in ONE place.
//
//   collect demos -> VLA (vla_openarm.py) -> DAgger -> distill_app (CNN + GRU student)
//   -> rl_app (PPO fine-tune)
//
// A new vision task: its task file in forge/env/tasks/ (physics, obs, reward, expert),
// a camera rig if it needs a new scene (apps/collect_app/), and one entry in
// make_vision_tasks() (vision_task.cpp). The apps take everything else from here.
#include <memory>
#include <string>
#include <vector>

#include "flux/batch_camera.h"
#include "forge/env/env.h"

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
    int state = 0;               // the student's state: the first 'state' numbers of obs
    int obs = 0;                 // the full obs (the critic sees it, with the episode time)
    int max_steps = 0;           // episode limit
    int cameras = 0;             // camera ids 0 .. cameras - 1
    std::string student_cameras; // distill_app's default --student_cameras ("0,1,3")
    float physics_dt = 0.002f;   // as the task's config file
    int decimation = 10;
    std::string done_text;       // what "done" means, for the eval lines
    // true when the episode ended in success: final obs, terminated (not a time limit)
    bool (*done_ok)(const float* final_obs, bool terminated) = nullptr;
    // The new episode's actions before its first decision (rl_app): hold still.
    void (*hold_still)(float* act) = nullptr;
    // The aux head's targets from obs (distill_app; aux_dim numbers). 0 / null: none.
    int aux_dim = 0;
    void (*aux)(const float* obs, float* out) = nullptr;
    // Env::set_task_input starts (rl_app --start_mix): how many numbers (0 = none) and
    // what they are, in order.
    int start_mix_dim = 0;
    std::string start_mix_names;
    std::unique_ptr<CameraRig> (*make_rig)(const forge::RobotModel& m, int num_envs, int size) = nullptr;

    forge::EnvOptions env_options(int envs, uint64_t seed, forge::EnvBackend backend) const;
    int critic() const { return obs + 1; }  // rl_app's critic input: obs + episode time
};

// The task by name; throws, listing the known tasks, for an unknown one.
const VisionTask& vision_task(const std::string& name);
std::vector<std::string> vision_task_names();

// 2 x 2 box filter: [S][S][3] -> [S/2][S/2][3] (the students' 2x supersampled images)
void halve(const uint8_t* in, int S, uint8_t* out);

}  // namespace pipeline
