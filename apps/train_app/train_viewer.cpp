#include "train_viewer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include <GLFW/glfw3.h>

#include "flux/camera_control.h"
#include "flux/window/window.h"
#include "forge/dynamics/articulated.h"
#include "forge/dynamics/robot_batch_host.h"
#include "forge/dynamics/robot_work.h"
#include "forge/robots/skeleton.h"

using Clock = std::chrono::steady_clock;

struct TrainViewer::Impl {
    forge::Env& env;
    const forge::RobotModel& model;
    int K = 0, npts = 0;
    bool terrain = false, drawing = true, v_was_down = false;
    flux::Window window;
    std::unique_ptr<flux::Renderer> renderer;
    flux::OrbitControl mouse;
    flux::FrameInput frame;
    std::vector<forge::SkeletonPoint> skel;
    forge::RobotBatchHost draw;
    forge::RobotWork work;
    forge::DynamicsWork w;
    std::vector<float> px, py, pz, radius;
    std::vector<int> env_id;
    std::vector<flux::Quad> boxes;
    float env_spacing = 4.0f;
    Clock::time_point last_draw, last_poll;
    long long steps = 0, drawn = 0;

    Impl(forge::Env& e, int max_envs, const flux::RendererOptions& ro, int width, int height)
        : env(e), model(e.model()), K(std::max(1, std::min(max_envs, e.num_envs()))),
          terrain(!e.terrain().h.empty()), window(width, height, "crucible train: " + e.options().task),
          draw(model, K), work(model), w(work.view()) {
        renderer = std::make_unique<flux::Renderer>(window, ro);
        // The keys are read at most 60 (drawing) or 30 (not drawing) times a second: sticky
        // keys keep a short tap of V until it is read, so no tap is missed.
        glfwSetInputMode(window.handle(), GLFW_STICKY_KEYS, GLFW_TRUE);
        std::vector<char> box_root(model.nbody(), 0);  // free boxes (a cube) are drawn as boxes, not skeletons
        for (const auto& bx : model.boxes)
            if (model.parent[bx.body] < 0 && model.jtype[bx.body] == forge::kJointFree) box_root[bx.body] = 1;
        for (const auto& pt : forge::make_skeleton(model, 6, 2.4f))
            if (!box_root[pt.body]) skel.push_back(pt);
        npts = static_cast<int>(skel.size());
        const float extent = forge::skeleton_extent(model, skel);
        const int total = K * npts;
        px.resize(total), py.resize(total), pz.resize(total), radius.resize(total), env_id.resize(total);
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < npts; ++i) {
                radius[size_t(k) * npts + i] = skel[i].radius;
                env_id[size_t(k) * npts + i] = terrain ? 0 : k;  // terrain: true positions, no grid offset
            }
        env_spacing = 1.3f * extent + 0.3f;
        flux::SceneQuads quads;
        if (terrain) {
            const forge::Terrain& t = e.terrain();
            const forge::Vec3 light = forge::normalize(forge::Vec3{0.35f, -0.45f, 0.82f});
            for (int j = 0; j + 2 < t.ny; j += 2)
                for (int i = 0; i + 2 < t.nx; i += 2) {
                    auto hh = [&](int a, int b) { return t.h[size_t(b) * t.nx + a]; };
                    const float c = 2.0f * t.cell;
                    const float h00 = hh(i, j), h10 = hh(i + 2, j), h01 = hh(i, j + 2), h11 = hh(i + 2, j + 2);
                    const forge::Vec3 u{0.5f * c, 0.0f, 0.25f * (h10 + h11 - h00 - h01)};
                    const forge::Vec3 v{0.0f, 0.5f * c, 0.25f * (h01 + h11 - h00 - h10)};
                    const forge::Vec3 n = forge::normalize(forge::cross(u, v));
                    const float lit = (0.55f + 0.45f * std::fmax(0.0f, forge::dot(n, light))) * (((i / 2 + j / 2) & 1) ? 0.92f : 1.0f);
                    quads.before.push_back({{t.x0 + (float(i) + 1.0f) * t.cell, t.y0 + (float(j) + 1.0f) * t.cell,
                                             0.25f * (h00 + h10 + h01 + h11)},
                                            {u.x, u.y, u.z}, {v.x, v.y, v.z},
                                            {0.30f * lit, 0.33f * lit, 0.36f * lit, 1.0f}, 0.0f});
                }
            frame.camera.target = {0.0f, 0.0f, 0.0f};
            frame.camera.distance = 25.0f;
        } else {
            for (int k = 0; k < K; ++k) {
                const flux::Vec3 o = flux::env_grid_offset(k, K, env_spacing);
                const float h = 0.5f * env_spacing - 0.1f;
                quads.before.push_back({{o.x, o.y, 0.0f}, {h, 0, 0}, {0, h, 0}, {0.16f, 0.17f, 0.19f, 1.0f}, 0.25f});
            }
            frame.camera.target = {0.0f, 0.0f, 0.6f * extent};
            frame.camera.distance = 1.6f * extent + (K > 1 ? 0.6f * env_spacing * std::sqrt(float(K)) : 0.0f);
        }
        renderer->set_scene(quads);
        frame.env_spacing = env_spacing;
        frame.color_height = 2.0f * extent;
        frame.particles.count = total;
        frame.particles.num_envs = terrain ? 1 : K;
        last_draw = last_poll = Clock::now();
        std::printf("viewer: %d of %d envs drawn%s; V = drawing on / off (off = headless speed), close the window "
                    "to stop training (final.bin is saved)\n",
                    K, e.num_envs(), terrain ? " at their positions on the terrain" : "");
    }

    void draw_frame() {
        env.download_qpos(0, K, draw.qpos.data());
        forge::skeleton_points_cpu(model.view(), draw.view(), draw.work(), skel.data(), npts, px.data(), py.data(),
                                   pz.data());
        boxes.clear();
        for (int k = 0; k < K && !model.boxes.empty(); ++k) {
            forward_kinematics(model.view(), draw.qpos.data() + size_t(k) * model.nq, w);
            const flux::Vec3 o = terrain ? flux::Vec3{0, 0, 0} : flux::env_grid_offset(k, K, env_spacing);
            for (const auto& bx : model.boxes) {
                const forge::Quat qb = forge::load_quat(w.xquat + 4 * bx.body);
                const forge::Vec3 c = forge::load3(w.xpos + 3 * bx.body) + forge::rotate(qb, bx.pos);
                const forge::Mat3 R = forge::quat_to_mat3(qb * bx.rot);
                const float hs[3] = {bx.half.x, bx.half.y, bx.half.z};
                for (int ax = 0; ax < 3; ++ax)
                    for (int sg = -1; sg <= 1; sg += 2) {
                        const forge::Vec3 n = forge::col(R, ax) * (hs[ax] * float(sg));
                        const forge::Vec3 u = forge::col(R, (ax + 1) % 3) * hs[(ax + 1) % 3];
                        const forge::Vec3 v = forge::col(R, (ax + 2) % 3) * hs[(ax + 2) % 3];
                        const float shade = ax == 2 ? 1.0f : (ax == 0 ? 0.85f : 0.7f);
                        boxes.push_back({{o.x + c.x + n.x, o.y + c.y + n.y, c.z + n.z}, {u.x, u.y, u.z},
                                         {v.x, v.y, v.z}, {0.95f * shade, 0.45f * shade, 0.1f * shade, 1.0f}, 0.0f});
                    }
            }
        }
        frame.solid_quads = boxes.empty() ? nullptr : &boxes;
        frame.particles.px = px.data();
        frame.particles.py = py.data();
        frame.particles.pz = pz.data();
        frame.particles.radius = radius.data();
        frame.particles.env_id = env_id.data();
        mouse.update(frame.camera, window.take_mouse());
        renderer->draw_frame(frame);
        ++drawn;
    }

    bool after_step() {
        ++steps;
        const auto now = Clock::now();
        const double since_draw = std::chrono::duration<double>(now - last_draw).count();
        const double since_poll = std::chrono::duration<double>(now - last_poll).count();
        if (!(drawing ? since_draw >= 1.0 / 60.0 : since_poll >= 1.0 / 30.0)) return true;
        last_poll = now;
        window.poll_events();
        if (window.should_close()) {
            std::printf("viewer: window closed, stopping training\n");
            return false;
        }
        const bool v = glfwGetKey(window.handle(), GLFW_KEY_V) == GLFW_PRESS;
        if (v && !v_was_down) {
            drawing = !drawing;
            std::printf("viewer: drawing %s\n", drawing ? "on" : "off (headless speed)");
        }
        v_was_down = v;
        if (drawing) {
            last_draw = now;
            draw_frame();
        }
        return true;
    }
};

TrainViewer::TrainViewer(forge::Env& env, int max_envs, const flux::RendererOptions& ro, int width, int height)
    : impl_(std::make_unique<Impl>(env, max_envs, ro, width, height)) {}

TrainViewer::~TrainViewer() {
    if (impl_ && impl_->renderer) impl_->renderer->wait_idle();
}

bool TrainViewer::after_step() { return impl_->after_step(); }
