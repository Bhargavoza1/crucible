// sandbox_app: window mode.
//
//   ./sandbox_app                              10,000 particles fall into a bin (Phase 4)
//   ./sandbox_app --scene rain                 free fall, no collisions (Phase 2)
//   ./sandbox_app --scene robot                URDF robots under gravity (Phase 5)
//   ./sandbox_app --envs 16                    16 envs in a grid
//   ./sandbox_app --backend cpu                CPU answer-key code (Stage A drawing)
//   ./sandbox_app --interop off                GPU physics, but copy through the host (Stage A)
//   ./sandbox_app --vsync false                no 60 FPS limit, to compare speed
//   ./sandbox_app --frames 300                 exit after 300 frames (for tests)
//
// Drawing paths (doc 4.2):
//   Stage A  GPU -> host -> Vulkan buffer every frame. CPU backend, fallback.
//   Stage B  CUDA writes into a Vulkan buffer directly; a timeline semaphore
//            syncs both sides. No copy in the loop.
// Scene options are in particle_scene.cpp and robot_scene.cpp.
#include <cstdio>
#include <exception>
#include <string>

#include "sandbox.h"
#include "forge/util/device_info.h"

int main(int argc, char** argv) {
    try {
        forge::Config cfg = forge::Config::from_cli(argc, argv);
        sandbox::Args a;
        a.cfg = &cfg;
        a.width = cfg.get_int("width", 1280);
        a.height = cfg.get_int("height", 720);
        a.max_frames = cfg.get_int("frames", 0);
        a.max_steps_per_frame = cfg.get_int("max_steps_per_frame", 8);
        a.orbit_speed = cfg.get_float("orbit_speed", 0.15f);
        a.ro.validation = cfg.get_bool("validation", true);
        a.ro.vsync = cfg.get_bool("vsync", true);
        a.ro.shader_dir = cfg.get_string("shader_dir", FLUX_SHADER_DIR);
        a.backend = cfg.get_string("backend", "auto");
        a.interop = cfg.get_string("interop", "auto");
        const std::string scene = cfg.get_string("scene", "bin");

        if (a.interop != "auto" && a.interop != "on" && a.interop != "off")
            throw std::runtime_error("interop must be auto, on or off");
        const bool have_gpu = forge::cuda_device_count() > 0;
        if (a.backend == "auto") a.backend = have_gpu ? "gpu" : "cpu";
        if (a.backend != "cpu" && a.backend != "gpu") throw std::runtime_error("backend must be auto, cpu or gpu");
        if (a.backend == "gpu" && !have_gpu)
            throw std::runtime_error(forge::built_with_cuda() ? "backend gpu: no CUDA device found"
                                                              : "backend gpu: built without CUDA");
        if (a.interop == "on" && a.backend != "gpu") throw std::runtime_error("interop on needs backend gpu");
        if (have_gpu) std::printf("cuda: %s\n", forge::cuda_device_description(0).c_str());

        if (scene == "bin" || scene == "rain") return sandbox::run_particle_scene(a, scene);
        if (scene == "robot") return sandbox::run_robot_scene(a);
        throw std::runtime_error("scene must be bin, rain or robot");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
