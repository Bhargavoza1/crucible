#pragma once
// Camera-driven policies in the sandbox, so a trained policy can be watched:
//   --student FILE   the CNN + GRU student (distill_app's checkpoint; temper, cuDNN):
//                    openarm_pick_place (Phase 10)
//   --vla FILE.onnx  the VLA (tools/vla_openarm.py --onnx; ONNX Runtime), --exec N:
//                    openarm_pick_place (Phase 10) and openarm_stack (Phase 11, both arms)
// All envs decide together every 'exec' steps (one clock, as rl_app and distill_app;
// an env whose episode starts between decisions holds still, gripper open, until
// the next one): at a decision the cell's cameras render (the off-screen camera rig, 128 x 128; the student's
// halved to 64 x 64), the policy plans an action chunk, and the env runs the next
// 'exec' actions of it (the student: its own exec, from the checkpoint; the VLA:
// --exec), then decides again. The student's GRU memory carries from one decision to
// the next and starts from zero when its env starts a new episode. The VLA uses
// fresh flow noise at every decision (as vla_app).
#include <memory>
#include <string>
#include <vector>

#include "forge/env/env.h"

namespace sandbox {

class VisionPolicy {
public:
    virtual ~VisionPolicy() = default;
    // h: the envs after the last step (obs, qpos; host). new_episode[e]: env e has
    // just started an episode (all envs before the first step). actions: host
    // [num_envs][act] (pick-place 5, stack 10).
    virtual void act(const forge::Env::HostCopy& h, const std::vector<char>& new_episode, float* actions) = 0;
    virtual std::string describe() const = 0;
};

// Throw when the build lacks what they need (temper with cuDNN; ONNX Runtime).
std::unique_ptr<VisionPolicy> make_student_policy(const forge::RobotModel& m, int num_envs, const std::string& path);
// task: openarm_pick_place or openarm_stack (the model must be for it: its _openarm.txt);
// noise: fresh (new flow noise at every plan) or zero (noise 0: the same answer for the same input)
std::unique_ptr<VisionPolicy> make_vla_policy(const forge::RobotModel& m, int num_envs, const std::string& path,
                                              int exec, const std::string& task,
                                              const std::string& noise);

}  // namespace sandbox
