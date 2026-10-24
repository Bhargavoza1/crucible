// sandbox_app: window mode. Phase 1 opens a window and draws a triangle.
//
//   ./sandbox_app                       run until the window is closed
//   ./sandbox_app --frames 120          draw 120 frames, then exit (for tests)
//   ./sandbox_app --validation false    turn off the Vulkan validation layer
#include <chrono>
#include <cstdio>
#include <exception>

#include "flux/renderer.h"
#include "flux/window/window.h"
#include "forge/util/config.h"
#include "forge/util/device_info.h"

int main(int argc, char** argv) {
    try {
        forge::Config cfg = forge::Config::from_cli(argc, argv);
        const int width = cfg.get_int("width", 1280);
        const int height = cfg.get_int("height", 720);
        const int max_frames = cfg.get_int("frames", 0);  // 0 = until closed

        flux::RendererOptions ro;
        ro.validation = cfg.get_bool("validation", true);
        ro.shader_dir = cfg.get_string("shader_dir", FLUX_SHADER_DIR);

        for (const auto& k : cfg.unused_keys())
            std::fprintf(stderr, "warning: unknown config key '%s' (typo?)\n", k.c_str());

        if (forge::cuda_device_count() > 0)
            std::printf("cuda: %s\n", forge::cuda_device_description(0).c_str());

        flux::Window window(width, height, "crucible sandbox");
        flux::Renderer renderer(window, ro);

        const auto t0 = std::chrono::steady_clock::now();
        while (!window.should_close()) {
            window.poll_events();
            renderer.draw_frame();
            if (max_frames > 0 && renderer.frames_drawn() >= static_cast<uint64_t>(max_frames)) break;
        }
        renderer.wait_idle();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("sandbox: %llu frames in %.2f s\n",
                    static_cast<unsigned long long>(renderer.frames_drawn()), s);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
