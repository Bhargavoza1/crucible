#include <cstdio>

#include "sandbox.h"
#include "forge/util/device_info.h"

namespace sandbox {

Viewer::~Viewer() {
#ifdef FLUX_HAS_INTEROP
    if (shared) {
        cudaDeviceSynchronize();
        shared.reset();
    }
#endif
    renderer.reset();
}

void Viewer::set_shared_source(flux::FrameInput& frame) {
#ifdef FLUX_HAS_INTEROP
    flux::SharedDrawSource src;
    src.buffer = shared->buffer();
    for (int f = 0; f < flux::SharedParticles::kNumFields; ++f)
        src.offsets[f] = shared->offset(static_cast<flux::SharedParticles::Field>(f));
    src.timeline = shared->timeline();
    src.wait_value = shared->vk_wait_value();
    src.signal_value = shared->vk_signal_value();
    frame.shared = src;
#else
    (void)frame;
#endif
}

void Viewer::next_frame() {
#ifdef FLUX_HAS_INTEROP
    if (stage_b) shared->next_frame();
#endif
}

void make_viewer(Viewer& v, flux::Window& window, const Args& a, int capacity) {
#ifdef FLUX_HAS_INTEROP
    if (a.backend == "gpu" && a.interop != "off") {
        try {
            flux::RendererOptions ro_b = a.ro;
            ro_b.interop = true;
            auto uuid = forge::cuda_device_uuid(0);
            if (uuid) ro_b.required_uuid = *uuid;  // Vulkan must pick the CUDA GPU
            v.renderer = std::make_unique<flux::Renderer>(window, ro_b);
            v.shared = std::make_unique<flux::SharedParticles>(v.renderer->context(), capacity);
            v.stage_b = true;
            return;
        } catch (const std::exception& e) {
            if (a.interop == "on") throw;
            std::fprintf(stderr, "sandbox: zero-copy not available, using Stage A copy.\n  reason: %s\n", e.what());
            v.shared.reset();
            v.renderer.reset();
        }
    }
#else
    if (a.interop == "on") throw std::runtime_error("interop on: this build has no CUDA-Vulkan interop");
    (void)capacity;
#endif
    v.renderer = std::make_unique<flux::Renderer>(window, a.ro);
}

void warn_unused_keys(const forge::Config& cfg) {
    for (const auto& k : cfg.unused_keys()) std::fprintf(stderr, "warning: unknown config key '%s' (typo?)\n", k.c_str());
}

}  // namespace sandbox
