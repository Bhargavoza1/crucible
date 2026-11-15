#pragma once
// Cartpole task (doc 6.1). assets/robots/cartpole.urdf.
//   obs (4):    cart position, cart velocity, pole angle, pole angular velocity
//   action (1): force on the cart, action * force_scale (N)
//   reward:     +1 per step alive, minus action_cost * action^2
//   terminated: |pole angle| > 0.5 rad or |cart position| > 2.4 m
//   truncated:  500 steps (10 s at 50 Hz with physics_dt 0.005, decimation 4)
//   reset:      all four numbers uniform in [-0.05, 0.05]
#include <stdexcept>

#include "forge/env/env_view.h"
#include "forge/robots/actuators.h"
#include "forge/solver/constraint_model.h"
#include "forge/math/hd.h"
#include "forge/util/rng.h"

namespace forge {

struct CartpoleTask {
    static constexpr const char* kName = "cartpole";
    static constexpr const char* kUrdf = "robots/cartpole.urdf";
    static constexpr bool kFloating = false;
    static constexpr int kObsDim = 4, kActDim = 1, kMaxEpisodeSteps = 500;

    struct Params {
        int cart_q = 0, cart_d = 0, pole_q = 1, pole_d = 1;  // qpos / dof index of the two joints
        float force_scale = 10.0f;  // N per unit action
        float x_limit = 2.4f;       // m
        float angle_limit = 0.5f;   // rad
        float action_cost = 0.01f;
        float reset_noise = 0.05f;
    };

    // ---- host: set up once ----
    static Params make_params(const RobotModel& m) {
        const int s = m.joint_index("slider"), h = m.joint_index("hinge");
        if (s < 0 || h < 0) throw std::runtime_error("cartpole: robot needs joints 'slider' and 'hinge'");
        Params p;
        p.cart_q = m.qadr[s];
        p.cart_d = m.dadr[s];
        p.pole_q = m.qadr[h];
        p.pole_d = m.dadr[h];
        return p;
    }
    static Actuators make_actuators(const RobotModel&) { return {}; }  // force control: tau, no PD
    static void setup_constraints(ConstraintModel&) {}                 // no floor, no limits

    // ---- per env (CPU and GPU) ----
    FORGE_HD static void apply_action(const Params& p, const EnvView& v, int e, const float* a) {
        v.tau(e)[p.cart_d] = p.force_scale * a[0];
        v.tau(e)[p.pole_d] = 0.0f;
    }
    FORGE_HD static void observe(const Params& p, const EnvView& v, int e, float* o) {
        o[0] = v.qpos(e)[p.cart_q];
        o[1] = v.qvel(e)[p.cart_d];
        o[2] = v.qpos(e)[p.pole_q];
        o[3] = v.qvel(e)[p.pole_d];
    }
    FORGE_HD static float reward(const Params& p, const EnvView&, int, const float* a) {
        return 1.0f - p.action_cost * a[0] * a[0];
    }
    FORGE_HD static bool failed(const Params& p, const EnvView& v, int e) {
        return fabsf(v.qpos(e)[p.pole_q]) > p.angle_limit || fabsf(v.qpos(e)[p.cart_q]) > p.x_limit;
    }
    FORGE_HD static void reset(const Params& p, const EnvView& v, int e, Rng& rng) {
        const float n = p.reset_noise;
        v.qpos(e)[p.cart_q] = rng.uniform(-n, n);
        v.qpos(e)[p.pole_q] = rng.uniform(-n, n);
        v.qvel(e)[p.cart_d] = rng.uniform(-n, n);
        v.qvel(e)[p.pole_d] = rng.uniform(-n, n);
    }
};

}  // namespace forge
