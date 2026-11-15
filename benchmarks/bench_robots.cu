// Headless robot benchmark: env-steps per second, GPU vs one CPU thread.
//
// Default: dynamics only (FK + mass matrix + bias forces + Cholesky + Euler).
// --constraints: the full Phase 6 step: floor contacts with friction (floating
// robots), joint limits, PD actuators on every joint, Newton solver. The
// quadruped starts standing, so its feet are on the floor the whole time.
// One env per thread in both modes.
//
//   ./benchmarks/bench_robots                               quadruped, 4096 envs
//   ./benchmarks/bench_robots --robot arm3 --envs 16384
//   ./benchmarks/bench_robots --envs_list 256,1024,4096,16384
//   ./benchmarks/bench_robots --constraints --envs_list 1024,4096,16384
#include <chrono>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "forge/dynamics/robot_batch_host.h"
#include "forge/dynamics/robot_gpu.h"
#include "forge/robots/actuators.h"
#include "forge/robots/urdf.h"
#include "forge/solver/constraint_batch_host.h"
#include "forge/solver/solver_gpu.h"
#include "forge/util/config.h"
#include "forge/util/device_info.h"

using namespace forge;

namespace {

void small_random_state(const RobotModel& m, RobotBatchHost& b) {
    for (int e = 0; e < b.num_envs(); ++e) {
        for (int k = 0; k < m.nbody(); ++k)
            if (m.jtype[k] == kJointRevolute) b.env_qpos(e)[m.qadr[k]] = 0.3f * float((e * 7 + k) % 11 - 5) / 5.0f;
        for (int i = 0; i < m.nv; ++i) b.env_qvel(e)[i] = 0.2f * float((e + 3 * i) % 9 - 4) / 4.0f;
    }
}

double gpu_env_steps_per_s(const RobotModel& m, int envs, int steps, float dt) {
    RobotBatchHost h(m, envs);
    small_random_state(m, h);
    RobotModelDevice dm(m);
    RobotBatchDevice d(m, envs);
    d.upload_state(h);
    robot_step_gpu(dm.view(), d.view(), d.work(), dt, 10);  // warm up
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    robot_step_gpu(dm.view(), d.view(), d.work(), dt, steps);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return double(envs) * steps / (ms * 1e-3);
}

double cpu_env_steps_per_s(const RobotModel& m, int envs, int steps, float dt) {
    RobotBatchHost h(m, envs);
    small_random_state(m, h);
    const auto t0 = std::chrono::steady_clock::now();
    robot_step_cpu(m.view(), h.view(), h.work(), dt, steps);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return double(envs) * steps / s;
}

// ---- --constraints mode ----
struct Scene {
    ConstraintModel cm;
    Actuators act;
};

Scene make_scene(const RobotModel& m, bool floating) {
    Scene s{ConstraintModel(m), Actuators::position_all(m, 80.0f, 2.0f, -40.0f, 40.0f)};
    if (floating) s.cm.add_ground();
    return s;
}

// Standing start: base 1 cm above standing height, knees bent 0.3 (the PD
// target), a different small velocity in each env.
void standing_state(const RobotModel& m, const Scene& sc, RobotBatchHost& b, ConstraintBatchHost& cb) {
    for (int e = 0; e < b.num_envs(); ++e) {
        float* q = b.env_qpos(e);
        for (int k = 0; k < m.nbody(); ++k) {
            if (m.jtype[k] == kJointFree) q[m.qadr[k] + 2] = 0.51f;
            if (m.jtype[k] == kJointRevolute && m.joint_name[k].find("knee") != std::string::npos) q[m.qadr[k]] = 0.3f;
        }
        for (int i = 0; i < m.nv; ++i) b.env_qvel(e)[i] = 0.1f * float((e + 3 * i) % 9 - 4) / 4.0f;
        for (int k = 0; k < sc.act.nu(); ++k) cb.ctrl[size_t(e) * sc.act.nu() + k] = q[sc.act.qadr[k]];
    }
}

double gpu_constrained(const RobotModel& m, const Scene& sc, int envs, int steps, float dt, double& rows,
                       double& iters) {
    RobotBatchHost h(m, envs);
    ConstraintBatchHost ch(sc.cm, sc.act.nu(), envs);
    standing_state(m, sc, h, ch);
    RobotModelDevice dm(m);
    ConstraintModelDevice dcm(sc.cm);
    ActuatorsDevice dact(sc.act);
    RobotBatchDevice d(m, envs);
    ConstraintBatchDevice dc(sc.cm, sc.act.nu(), envs);
    d.upload_state(h);
    dc.upload_ctrl(ch);
    robot_step_constrained_gpu(dm.view(), dcm.view(), dact.view(), d.view(), d.work(), dc.view(), dt, 10);  // warm up
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    robot_step_constrained_gpu(dm.view(), dcm.view(), dact.view(), d.view(), d.work(), dc.view(), dt, steps);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    dc.download_all(ch);
    d.download_state(h);
    rows = iters = 0;
    int bad = 0;
    for (int e = 0; e < envs; ++e) {
        rows += ch.nefc[e];
        iters += ch.iterations[e];
        bad += h.status[e] != 0;
    }
    rows /= envs;
    iters /= envs;
    if (bad) std::fprintf(stderr, "warning: %d envs reported a dynamics error\n", bad);
    return double(envs) * steps / (ms * 1e-3);
}

double cpu_constrained(const RobotModel& m, const Scene& sc, int envs, int steps, float dt) {
    RobotBatchHost h(m, envs);
    ConstraintBatchHost ch(sc.cm, sc.act.nu(), envs);
    standing_state(m, sc, h, ch);
    const auto t0 = std::chrono::steady_clock::now();
    robot_step_constrained_cpu(m.view(), sc.cm.view(), sc.act.view(), h.view(), h.work(), ch.view(), dt, steps);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return double(envs) * steps / s;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = Config::from_cli(argc, argv);
        const std::string robot = cfg.get_string("robot", "quadruped");
        const bool floating = cfg.get_bool("floating", robot == "quadruped");
        const std::string list = cfg.get_string("envs_list", "");
        const int envs_one = cfg.get_int("envs", 4096);
        const int steps = cfg.get_int("steps", 500);
        const float dt = cfg.get_float("dt", 0.002f);
        const bool constraints = cfg.get_bool("constraints", false);
        for (const auto& k : cfg.unused_keys()) std::fprintf(stderr, "warning: unknown key '%s'\n", k.c_str());
        if (cuda_device_count() == 0) {
            std::fprintf(stderr, "no CUDA device\n");
            return 1;
        }
        RobotModel m = load_urdf(std::string(CRUCIBLE_ASSET_DIR) + "/robots/" + robot + ".urdf", {floating});
        std::vector<int> envs_list;
        if (list.empty()) {
            envs_list.push_back(envs_one);
        } else {
            std::stringstream ss(list);
            std::string tok;
            while (std::getline(ss, tok, ',')) envs_list.push_back(std::stoi(tok));
        }

        std::printf("GPU: %s\n", cuda_device_description(0).c_str());
        if (constraints) {
            const Scene sc = make_scene(m, floating);
            std::printf("robot %s: %d bodies, nv %d, %d actuators, %d limits, %d spheres, %s. Full step with "
                        "constraints, dt %.3f s\n",
                        robot.c_str(), m.nbody(), m.nv, sc.act.nu(), int(sc.cm.limit_qadr.size()),
                        int(sc.cm.sphere_radius.size()), floating ? "floor" : "no floor", dt);
            const double cpu = cpu_constrained(m, sc, 256, 200, dt);
            std::printf("CPU, 1 thread:      %10.3g env-steps/s\n", cpu);
            for (int envs : envs_list) {
                double rows = 0, iters = 0;
                const double gpu = gpu_constrained(m, sc, envs, steps, dt, rows, iters);
                std::printf("GPU, %6d envs:   %10.3g env-steps/s  (%.0fx CPU; %.2f ms per step of all envs; "
                            "last step: %.1f rows, %.1f Newton iterations per env)\n",
                            envs, gpu, gpu / cpu, 1e3 * envs / gpu, rows, iters);
            }
            return 0;
        }
        std::printf("robot %s: %d bodies, nq %d, nv %d. Dynamics only (no contacts), dt %.3f s\n", robot.c_str(),
                    m.nbody(), m.nq, m.nv, dt);
        const double cpu = cpu_env_steps_per_s(m, 256, 200, dt);
        std::printf("CPU, 1 thread:      %10.3g env-steps/s\n", cpu);
        for (int envs : envs_list) {
            const double gpu = gpu_env_steps_per_s(m, envs, steps, dt);
            std::printf("GPU, %6d envs:   %10.3g env-steps/s  (%.0fx CPU; %.1f ms per step of all envs)\n", envs, gpu,
                        gpu / cpu, 1e3 * envs / gpu);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
