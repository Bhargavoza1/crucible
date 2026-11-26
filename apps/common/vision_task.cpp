#include "common/vision_task.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>

#ifndef CRUCIBLE_NO_CAMERAS
#include "collect_app/openarm_camera_rig.h"
#include "collect_app/stack_camera_rig.h"
#endif
#include "forge/env/tasks/openarm_handover.h"
#include "forge/env/tasks/openarm_pick_place.h"
#include "forge/env/tasks/openarm_stack.h"

namespace pipeline {

namespace {

using PickPlace = forge::OpenArmPickPlaceTask;
using Stack = forge::OpenArmStackTask;
using Handover = forge::OpenArmHandoverTask;

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

#ifndef CRUCIBLE_NO_CAMERAS
std::string cell_mesh() { return std::string(CRUCIBLE_ASSET_DIR) + "/robots/openarm/openarm_cell.cmesh"; }
#endif

// ---- openarm_pick_place (Phase 10) -------------------------------------------------
// obs: TCP 18-20, cube 24-26, cube - TCP 31-33, cube - tray center 36-37
bool pick_place_placed(const float* o) {
    return std::fabs(o[36]) < 0.06f && std::fabs(o[37]) < 0.06f && o[26] < 0.031f;
}
// the episode finished (terminated: placed and the arm home), the cube in the tray, resting,
// the TCP up (a broken env's final obs is all zeros)
bool pick_place_done(const float* fo, bool terminated) {
    if (!terminated) return false;
    return std::fabs(fo[36]) < 0.06f && std::fabs(fo[37]) < 0.06f && fo[26] < 0.031f && fo[20] > 0.08f;
}

// progress: closest pads - cube (m), the cube's highest z, ever placed (0 / 1)
void pick_place_reset(float* p) { p[0] = 1e9f, p[1] = 0.0f, p[2] = 0.0f; }
void pick_place_step(const float* ob, float* p) {
    p[0] = std::min(p[0], std::sqrt(ob[31] * ob[31] + ob[32] * ob[32] + ob[33] * ob[33]));
    p[1] = std::max(p[1], ob[26]);
    if (pick_place_placed(ob)) p[2] = 1.0f;
}
const char* pick_place_kind(const float* p, const float* fo) {
    const bool lifted = p[1] >= PickPlace::kHalf + 0.02f;
    return !lifted && p[0] > 0.02f    ? "never reached the cube"
           : !lifted                  ? "reached, never lifted it"
           : p[2] == 0.0f             ? "lifted, never put it in the tray"
           : !pick_place_placed(fo)   ? "put it in the tray, then moved it out"
                                      : "placed, did not go home";
}
std::string pick_place_detail(const forge::RobotModel& m, const float* p, const float* fo) {
    const auto prm = PickPlace::make_params(m);
    const forge::Vec3 home{prm.home_tcp[0], prm.home_tcp[1], prm.home_tcp[2]};
    const forge::Vec3 tcp{fo[18], fo[19], fo[20]};
    return fmt("closest pads-cube %.1f cm, cube max %.1f cm up; at the end cube - tray (%.1f, %.1f) cm, cube %.1f cm "
               "up, TCP %.1f cm from home",
               100.0f * p[0], 100.0f * (p[1] - PickPlace::kHalf), 100.0f * fo[36], 100.0f * fo[37],
               100.0f * (fo[26] - PickPlace::kHalf), 100.0f * forge::length(tcp - home));
}

#ifndef CRUCIBLE_NO_CAMERAS
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
#endif

// ---- openarm_stack (Phase 11) ------------------------------------------------------
// obs: TCPs 36-38 and 42-44; red 48, green 55, blue 62 (xyz first); the spot (0.41, 0)
constexpr int kTcpL = 36, kTcpR = 42, kRed = 48, kGreen = 55, kBlue = 62;
constexpr float kSpotX = 0.41f, kSpotY = 0.0f;

bool stack_on(const float* a, const float* u) {  // cube a on cube u: 1.8 cm in x y, 4 cm +- 5 mm up
    return std::hypot(a[0] - u[0], a[1] - u[1]) < 0.018f && std::fabs(a[2] - u[2] - 0.04f) < 0.005f;
}
// tower height from an obs: 0 none, 1 red on the spot, 2 + green on red, 3 + blue on green
int stack_tower(const float* o) {
    const float *r = o + kRed, *g = o + kGreen, *b = o + kBlue;
    if (!(std::hypot(r[0] - kSpotX, r[1] - kSpotY) < 0.025f && r[2] < 0.025f)) return 0;
    if (!stack_on(g, r)) return 1;
    return stack_on(b, g) ? 3 : 2;
}
// the episode finished (terminated: the tower built, both hands home), and the final obs
// agree: the tower red, green, blue on the spot, both TCPs up (a broken env's final obs is
// all zeros)
bool stack_done(const float* fo, bool terminated) {
    return terminated && stack_tower(fo) == 3 && fo[kTcpL + 2] > 0.12f && fo[kTcpR + 2] > 0.12f;
}

// The aux targets of one obs: per cube x y z, sin and cos of 4 x yaw (15 numbers).
void stack_aux(const float* o, float* out) {
    for (int c = 0; c < 3; ++c) {
        const float* q = o + kRed + 7 * c;  // x y z, quat w x y z
        const float yaw = std::atan2(2.0f * (q[3] * q[6] + q[4] * q[5]), 1.0f - 2.0f * (q[5] * q[5] + q[6] * q[6]));
        float* a = out + 5 * c;
        a[0] = q[0], a[1] = q[1], a[2] = q[2];
        a[3] = std::sin(4.0f * yaw), a[4] = std::cos(4.0f * yaw);
    }
}

// progress: the highest tower so far, the highest cube z so far
void stack_reset(float* p) { p[0] = 0.0f, p[1] = 0.0f; }
void stack_step(const float* ob, float* p) {
    p[0] = std::max(p[0], float(stack_tower(ob)));
    for (int c : {kRed, kGreen, kBlue}) p[1] = std::max(p[1], ob[c + 2]);
}
const char* stack_kind(const float* p, const float* fo) {
    const int end_tower = stack_tower(fo), max_tower = int(p[0]);
    const bool lifted = p[1] >= Stack::kHalf + 0.02f;
    return end_tower == 3   ? "tower built, hands not home"
           : max_tower == 3 ? "tower built, then knocked down"
           : max_tower == 2 ? "red + green placed, blue never on top"
           : max_tower == 1 ? "red placed, green never on it"
           : lifted         ? "lifted a cube, red never on the spot"
                            : "never lifted a cube";
}
std::string stack_detail(const forge::RobotModel&, const float* p, const float* fo) {
    return fmt("tower %d at the end (max %d), red - spot (%.1f, %.1f) cm, TCPs %.1f / %.1f cm up", stack_tower(fo),
               int(p[0]), 100.0f * (fo[kRed] - kSpotX), 100.0f * (fo[kRed + 1] - kSpotY), 100.0f * fo[kTcpL + 2],
               100.0f * fo[kTcpR + 2]);
}

#ifndef CRUCIBLE_NO_CAMERAS
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
#endif

// ---- openarm_handover (Phase 12) ---------------------------------------------------
// obs: TCPs 36-38 and 42-44, openings 41 and 47; the block 48 (xyz, then quat w x y z);
// the meeting point M (0.40, 0, 0.10), the place spot (0.40, -0.25)
constexpr int kHoOpenL = 41, kHoOpenR = 47, kHoBlock = 48;
constexpr float kHoMeet[3] = {0.40f, 0.0f, 0.10f}, kHoSpot[2] = {0.40f, -0.25f};

float ho_dist(const float* a, const float* b) { return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])); }
bool ho_closed(float opening) { return opening > 0.14f && opening < 0.30f; }  // the expert's holding range
bool ho_placed(const float* o) {
    const float* b = o + kHoBlock;
    return std::hypot(b[0] - kHoSpot[0], b[1] - kHoSpot[1]) < 0.025f && b[2] < 0.025f;
}
// The stage of an obs (from positions and finger openings: about Handover::stage, which
// uses the pads' contacts): 0 start, 1 the left holds it up, 2 at M, 3 both hold, 4 the
// right holds it alone, 5 placed
int ho_stage(const float* o) {
    if (ho_placed(o)) return 5;
    const float* b = o + kHoBlock;
    const bool l = ho_closed(o[kHoOpenL]) && ho_dist(b, o + kTcpL) < 0.05f && b[2] > 0.035f;
    const bool r = ho_closed(o[kHoOpenR]) && ho_dist(b, o + kTcpR) < 0.05f && b[2] > 0.035f;
    if (r) return l ? 3 : 4;
    if (!l) return 0;
    return ho_dist(b, kHoMeet) < 0.02f ? 2 : 1;
}
// the episode finished (terminated: placed, both hands home), and the final obs agree:
// the block on the spot, both TCPs up (a broken env's final obs is all zeros)
bool ho_done(const float* fo, bool terminated) {
    return terminated && ho_placed(fo) && fo[kTcpL + 2] > 0.12f && fo[kTcpR + 2] > 0.12f;
}
// The aux targets of one obs: the block's x y z, sin and cos of 2 x yaw (it looks the same
// turned by pi): 5 numbers.
void ho_aux(const float* o, float* out) {
    const float* q = o + kHoBlock;  // x y z, quat w x y z
    const float yaw = std::atan2(2.0f * (q[3] * q[6] + q[4] * q[5]), 1.0f - 2.0f * (q[5] * q[5] + q[6] * q[6]));
    out[0] = q[0], out[1] = q[1], out[2] = q[2];
    out[3] = std::sin(2.0f * yaw), out[4] = std::cos(2.0f * yaw);
}
// progress: the highest stage so far, the block's highest z so far
void ho_reset(float* p) { p[0] = 0.0f, p[1] = 0.0f; }
void ho_step(const float* ob, float* p) {
    p[0] = std::max(p[0], float(ho_stage(ob)));
    p[1] = std::max(p[1], ob[kHoBlock + 2]);
}
const char* ho_kind(const float* p, const float*) {
    const int s = int(p[0]);
    return s == 5   ? "placed, hands not home"
           : s == 4 ? "handed over, never placed"
           : s == 3 ? "both held it, the left never let go"
           : s == 2 ? "at the meeting point, the right never took it"
           : s == 1 ? "lifted, never at the meeting point"
           : p[1] >= Handover::kHalfYZ + 0.02f ? "lifted it, dropped it"
                                               : "never lifted the block";
}
std::string ho_detail(const forge::RobotModel&, const float* p, const float* fo) {
    const float* b = fo + kHoBlock;
    return fmt("stage %d at the end (max %d), block - spot (%.1f, %.1f) cm, %.1f cm up, TCPs %.1f / %.1f cm up",
               ho_stage(fo), int(p[0]), 100.0f * (b[0] - kHoSpot[0]), 100.0f * (b[1] - kHoSpot[1]),
               100.0f * (b[2] - Handover::kHalfYZ), 100.0f * fo[kTcpL + 2], 100.0f * fo[kTcpR + 2]);
}

// ---- shared ----------------------------------------------------------------------
// Both OpenArm tasks: 5 numbers per arm (TCP move 3, turn, gripper): still, open.
void openarm_hold_still(float* x, int act) {
    for (int j = 0; j < act; ++j) x[j] = j % 5 == 4 ? -1.0f : 0.0f;
}

#ifndef CRUCIBLE_NO_CAMERAS
template <typename Rig>
std::unique_ptr<CameraRig> make(const forge::RobotModel& m, int n, int size) {
    return std::make_unique<Rig>(m, n, size);
}
#define VISION_RIG(R) make<R>
#else
#define VISION_RIG(R) nullptr
#endif

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
        p.make_rig = VISION_RIG(PickPlaceRig);
        p.grips = {4};
        p.progress_dim = 3, p.progress_reset = pick_place_reset, p.progress_step = pick_place_step;
        p.fail_kind = pick_place_kind, p.fail_detail = pick_place_detail;
        p.strip_every = 10, p.strip_envs = 0;
        p.info_needs_task = false;  // its VLA files are older than the 'task' line
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
        s.make_rig = VISION_RIG(StackRig);
        s.grips = {4, 9};
        s.stream_frames = true;  // up to 800 steps
        s.progress_dim = 2, s.progress_reset = stack_reset, s.progress_step = stack_step;
        s.fail_kind = stack_kind, s.fail_detail = stack_detail;
        s.strip_every = 20, s.strip_envs = 256;  // 800-step episodes: fewer frames and envs
        t.push_back(std::move(s));
    }
    {
        VisionTask h;
        h.name = Handover::kName;
        h.act = 10, h.state = 48, h.obs = Handover::kObsDim, h.max_steps = Handover::kMaxEpisodeSteps;
        h.cameras = Handover::kCameras, h.student_cameras = "0,1,3";
        h.done_text = "block on the spot, hands home";
        h.done_ok = ho_done;
        h.hold_still = [](float* x) { openarm_hold_still(x, 10); };
        h.aux_dim = 5, h.aux = ho_aux;
        h.make_rig = VISION_RIG(StackRig);  // the same cell and cameras
        h.grips = {4, 9};
        h.stream_frames = true;  // up to 800 steps
        h.progress_dim = 2, h.progress_reset = ho_reset, h.progress_step = ho_step;
        h.fail_kind = ho_kind, h.fail_detail = ho_detail;
        h.strip_every = 20, h.strip_envs = 256;
        t.push_back(std::move(h));
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

const VisionTask* find_vision_task(const std::string& name) {
    for (const VisionTask& t : all())
        if (t.name == name) return &t;
    return nullptr;
}

const VisionTask& vision_task(const std::string& name) {
    if (const VisionTask* t = find_vision_task(name)) return *t;
    throw std::runtime_error("no vision task '" + name + "' (known: " + vision_task_list() + ")");
}

std::vector<std::string> vision_task_names() {
    std::vector<std::string> n;
    for (const VisionTask& t : all()) n.push_back(t.name);
    return n;
}

std::string vision_task_list() {
    std::string known;
    for (const VisionTask& t : all()) known += (known.empty() ? "" : ", ") + t.name;
    return known;
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
