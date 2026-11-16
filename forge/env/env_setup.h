#pragma once
// Host-side setup shared by both Env backends: robot, constraints, actuators and
// task parameters for task T.
#include <string>
#include <type_traits>

#include "forge/env/env.h"
#include "forge/robots/actuators.h"
#include "forge/robots/urdf.h"
#include "forge/solver/constraint_model.h"

namespace forge {

// A task may define static void urdf_options(UrdfOptions&) (G1: armature, damping).
template <class T, class = void>
struct TaskUrdfOptions {
    static void apply(UrdfOptions&) {}
};
template <class T>
struct TaskUrdfOptions<T, std::void_t<decltype(&T::urdf_options)>> {
    static void apply(UrdfOptions& u) { T::urdf_options(u); }
};

template <class T>
struct TaskSetup {
    RobotModel model;
    ConstraintModel constraints;
    Actuators actuators;
    typename T::Params params;
    int max_episode_steps;

    explicit TaskSetup(const EnvOptions& opt)
        : model(load(opt)), constraints(model, constraint_options(opt)), actuators(T::make_actuators(model)),
          params(T::make_params(model)),
          max_episode_steps(opt.max_episode_steps > 0 ? opt.max_episode_steps : T::kMaxEpisodeSteps) {
        T::setup_constraints(constraints);
    }

private:
    static RobotModel load(const EnvOptions& opt) {
        UrdfOptions u;
        u.floating_base = T::kFloating;
        TaskUrdfOptions<T>::apply(u);
        return load_urdf(opt.asset_dir + "/" + T::kUrdf, u);
    }
    static ConstraintOptions constraint_options(const EnvOptions& opt) {
        ConstraintOptions c;
        c.timestep = opt.physics_dt;
        return c;
    }
};

void check_env_options(const EnvOptions& opt);  // throws on bad values

}  // namespace forge
