#pragma once
// Live view of training (train_app --viz), like Isaac Gym's viewer: the first
// --viz_envs envs of the training batch (16), drawn while training runs at full
// speed. Training never waits for real time; the picture is redrawn at most 60
// times a second (skipped steps are not drawn), with no vsync wait. Each redraw
// copies only the drawn envs' qpos from the GPU.
//   V          drawing on / off (off: the training loop only polls the window,
//              about 30 times a second: headless speed)
//   mouse      left drag orbit, right / middle drag pan, wheel zoom
//   close      stop training; final.bin is saved
// Flat tasks: the envs in a grid, as in the sandbox. Terrain tasks: the shared
// terrain (every 2nd height sample) with the robots at their true positions.
#include <memory>

#include "flux/renderer.h"
#include "forge/env/env.h"

class TrainViewer {
public:
    TrainViewer(forge::Env& env, int max_envs, const flux::RendererOptions& ro, int width, int height);
    ~TrainViewer();
    // Call after every env step. Returns false when the window was closed.
    bool after_step();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
