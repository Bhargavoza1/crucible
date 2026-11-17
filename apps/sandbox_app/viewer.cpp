#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

#include "sandbox.h"
#include "flux/robot_mesh.h"
#include "forge/dynamics/articulated.h"
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

void MeshView::load(const std::string& cmesh, const forge::RobotModel& m, flux::Renderer& r, int envs) {
    if (cmesh.empty()) return;
    const flux::RobotMesh mesh = flux::load_robot_mesh(cmesh);
    bodies.clear();
    for (const std::string& name : mesh.links) {
        const auto it = std::find(m.body_name.begin(), m.body_name.end(), name);
        if (it == m.body_name.end()) throw std::runtime_error(cmesh + ": link '" + name + "' is not a body of the model");
        bodies.push_back(int(it - m.body_name.begin()));
    }
    r.set_robot_mesh(mesh, 0);
    num_envs = envs;
    poses.assign(size_t(envs) * bodies.size() * 12, 0.0f);
    std::printf("sandbox: mesh %s (%d links, %zu triangles)\n", cmesh.c_str(), mesh.num_links(),
                mesh.lods[0].size() / 3);
}

void MeshView::set_env(int env, const float* xpos, const float* xquat, forge::Vec3 shift) {
    for (size_t k = 0; k < bodies.size(); ++k) {
        const int b = bodies[k];
        const forge::Mat3 R = forge::quat_to_mat3(forge::load_quat(xquat + 4 * b));
        const float t[3] = {xpos[3 * b] + shift.x, xpos[3 * b + 1] + shift.y, xpos[3 * b + 2] + shift.z};
        float* o = poses.data() + (size_t(env) * bodies.size() + k) * 12;
        for (int row = 0; row < 3; ++row) {
            o[4 * row + 0] = R(row, 0);
            o[4 * row + 1] = R(row, 1);
            o[4 * row + 2] = R(row, 2);
            o[4 * row + 3] = t[row];
        }
    }
}

std::string find_cmesh(const std::string& asset_dir, const std::string& robot_name) {
    namespace fs = std::filesystem;
    const fs::path root = fs::path(asset_dir) / "robots";
    std::error_code ec;
    for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file() && it->path().filename() == robot_name + ".cmesh") return it->path().string();
    return "";
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
