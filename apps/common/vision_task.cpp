#include "common/vision_task.h"

#include <cmath>
#include <stdexcept>

#include "collect_app/openarm_camera_rig.h"
#include "collect_app/stack_camera_rig.h"
#include "forge/env/tasks/openarm_pick_place.h"
#include "forge/env/tasks/openarm_stack.h"

namespace pipeline {

namespace {

using PickPlace = forge::OpenArmPickPlaceTask;
using Stack = forge::OpenArmStackTask;

std::string cell_mesh() { return std::string(CRUCIBLE_ASSET_DIR) + "/robots/openarm/openarm_cell.cmesh"; }

// ---- openarm_pick_place (Phase 10) -------------------------------------------------
// obs: TCP z 20, cube z 26, cube - tray center 36-37 (vla_app's rule)
bool pick_place_done(const float* fo, bool terminated) {
    if (!terminated) return false;
    return std::fabs(fo[36]) < 0.06f && std::fabs(fo[37]) < 0.06f && fo[26] < 0.031f && fo[20] > 0.08f;
}

class PickPlaceRig : public CameraRig {
public:
    PickPlaceRig(const forge::RobotModel& m, int n, int size) : rig_(m, n, size, FLUX_SHADER_DIR, cell_mesh()) {}
    void render(const forge::Env::HostCopy& h, int obs_dim) override {
        rig_.render(h.qpos.data(), h.obs.data(), obs_dim);
    }
    const flux::BatchCamera& camera() const override { return rig_.camera(); }

private:
    OpenArmCameraRig rig_;
};

// ---- openarm_stack (Phase 11) ------------------------------------------------------
// obs: red 48, green 55, blue 62 (xyz first); TCP z 38, 44 (vla_app's rule)
bool stack_done(const float* fo, bool terminated) {
    if (!terminated) return false;
    const float *r = fo + 48, *g = fo + 55, *b = fo + 62;
    auto on = [](const float* x, const float* u) {
        return std::hypot(x[0] - u[0], x[1] - u[1]) < 0.018f && std::fabs(x[2] - u[2] - 0.04f) < 0.005f;
    };
    return std::hypot(r[0] - 0.41f, r[1]) < 0.025f && r[2] < 0.025f && on(g, r) && on(b, g) && fo[38] > 0.12f &&
           fo[44] > 0.12f;
}

// The aux targets of one obs: per cube x y z, sin and cos of 4 x yaw (15 numbers).
void stack_aux(const float* o, float* out) {
    for (int c = 0; c < 3; ++c) {
        const float* q = o + 48 + 7 * c;  // x y z, quat w x y z
        const float yaw = std::atan2(2.0f * (q[3] * q[6] + q[4] * q[5]), 1.0f - 2.0f * (q[5] * q[5] + q[6] * q[6]));
        float* a = out + 5 * c;
        a[0] = q[0], a[1] = q[1], a[2] = q[2];
        a[3] = std::sin(4.0f * yaw), a[4] = std::cos(4.0f * yaw);
    }
}

class StackRig : public CameraRig {
public:
    StackRig(const forge::RobotModel& m, int n, int size) : rig_(m, n, size, FLUX_SHADER_DIR, cell_mesh()) {}
    void render(const forge::Env::HostCopy& h, int) override { rig_.render(h.qpos.data()); }
    const flux::BatchCamera& camera() const override { return rig_.camera(); }
    double fk_seconds() const override { return rig_.fk_seconds; }
    double draw_seconds() const override { return rig_.draw_seconds; }

private:
    StackCameraRig rig_;
};

// ---- shared ----------------------------------------------------------------------
// Both OpenArm tasks: 5 numbers per arm (TCP move 3, turn, gripper): still, open.
void openarm_hold_still(float* x, int act) {
    for (int j = 0; j < act; ++j) x[j] = j % 5 == 4 ? -1.0f : 0.0f;
}

template <typename Rig>
std::unique_ptr<CameraRig> make(const forge::RobotModel& m, int n, int size) {
    return std::make_unique<Rig>(m, n, size);
}

// The pipeline's tasks: add a new one here.
std::vector<VisionTask> make_vision_tasks() {
    std::vector<VisionTask> t;
    {
        VisionTask p;
        p.name = PickPlace::kName;
        p.act = 5, p.state = 24, p.obs = PickPlace::kObsDim, p.max_steps = PickPlace::kMaxEpisodeSteps;
        p.cameras = PickPlace::kCameras, p.student_cameras = "0,1,2";
        p.done_text = "cube in the tray, arm home";
        p.done_ok = pick_place_done;
        p.hold_still = [](float* x) { openarm_hold_still(x, 5); };
        p.make_rig = make<PickPlaceRig>;
        t.push_back(std::move(p));
    }
    {
        VisionTask s;
        s.name = Stack::kName;
        s.act = 10, s.state = 48, s.obs = Stack::kObsDim, s.max_steps = Stack::kMaxEpisodeSteps;
        s.cameras = Stack::kCameras, s.student_cameras = "0,1,3";
        s.done_text = "tower built, hands home";
        s.done_ok = stack_done;
        s.hold_still = [](float* x) { openarm_hold_still(x, 10); };
        s.aux_dim = 15, s.aux = stack_aux;
        s.start_mix_dim = Stack::kTaskInputDim;
        s.start_mix_names = "red placed, red + green placed, a cube at the edge";
        s.make_rig = make<StackRig>;
        t.push_back(std::move(s));
    }
    return t;
}

const std::vector<VisionTask>& all() {
    static const std::vector<VisionTask> tasks = make_vision_tasks();
    return tasks;
}

}  // namespace

forge::EnvOptions VisionTask::env_options(int envs, uint64_t seed, forge::EnvBackend backend) const {
    forge::EnvOptions eo;
    eo.task = name;
    eo.num_envs = envs;
    eo.seed = seed;
    eo.backend = backend;
    eo.physics_dt = physics_dt;
    eo.decimation = decimation;
    eo.asset_dir = CRUCIBLE_ASSET_DIR;
    return eo;
}

const VisionTask& vision_task(const std::string& name) {
    for (const VisionTask& t : all())
        if (t.name == name) return t;
    std::string known;
    for (const VisionTask& t : all()) known += (known.empty() ? "" : ", ") + t.name;
    throw std::runtime_error("no vision task '" + name + "' (known: " + known + ")");
}

std::vector<std::string> vision_task_names() {
    std::vector<std::string> n;
    for (const VisionTask& t : all()) n.push_back(t.name);
    return n;
}

void halve(const uint8_t* in, int S, uint8_t* out) {
    const int h = S / 2;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < h; ++x)
            for (int c = 0; c < 3; ++c) {
                const int s = in[((2 * y) * S + 2 * x) * 3 + c] + in[((2 * y) * S + 2 * x + 1) * 3 + c] +
                              in[((2 * y + 1) * S + 2 * x) * 3 + c] + in[((2 * y + 1) * S + 2 * x + 1) * 3 + c];
                out[(y * h + x) * 3 + c] = uint8_t((s + 2) / 4);
            }
}

}  // namespace pipeline
