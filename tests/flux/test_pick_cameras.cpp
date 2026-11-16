// Phase 9c: the panda_pick cameras frame what the policy needs. Front: every corner
// of the cube's spawn area (on the floor and 15 cm up, the lift) is inside the image
// with a margin. Wrist: at the grasp pose (pads around the cube) the cube is in the
// middle of the image, below the camera.
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "flux/batch_camera.h"
#include "forge/dynamics/robot_work.h"
#include "forge/env/tasks/panda_pick.h"
#include "forge/robots/ik.h"
#include "forge/robots/urdf.h"

using namespace forge;

namespace {

struct Pick {
    RobotModel m;
    PandaPickTask::Params p;
    Pick() {
        UrdfOptions u;
        PandaPickTask::urdf_options(u);
        m = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/panda/panda.urdf", u);
        PandaPickTask::edit_model(m);
        p = PandaPickTask::make_params(m);
    }
};

flux::BatchCamera make_cam() {
    flux::BatchCameraOptions o;
    o.num_envs = 1;
    o.num_cams = 2;
    o.shader_dir = FLUX_SHADER_DIR;
    return flux::BatchCamera(o);
}

TEST(PickCameras, FrontSeesTheWholeCubeArea) {
    Pick k;
    std::vector<float> q(k.m.nq, 0.0f);
    for (int j = 0; j < 7; ++j) q[k.p.q[j]] = k.p.start[j];
    RobotWork work(k.m);
    DynamicsWork w = work.view();
    PandaPickTask::Camera c[2];
    PandaPickTask::cameras(k.p, k.m.view(), q.data(), w, c);
    flux::BatchCamera cam = make_cam();
    const flux::CameraView front{c[0].eye, c[0].target, c[0].up, c[0].fovy};
    for (float x : {k.p.cube_x_lo, k.p.cube_x_hi})
        for (float y : {-k.p.cube_y, k.p.cube_y})
            for (float z : {0.0f, 0.2f}) {
                float px, py;
                ASSERT_TRUE(cam.project(front, {x, y, z}, px, py));
                EXPECT_GT(px, 2.0f) << x << " " << y << " " << z;
                EXPECT_LT(px, 62.0f) << x << " " << y << " " << z;
                EXPECT_GT(py, 2.0f) << x << " " << y << " " << z;
                EXPECT_LT(py, 62.0f) << x << " " << y << " " << z;
            }
}

TEST(PickCameras, WristSeesTheCubeAtTheGrasp) {
    Pick k;
    std::vector<float> q(k.m.nq, 0.0f);
    for (int j = 0; j < 7; ++j) q[k.p.q[j]] = k.p.start[j];
    q[k.p.q[7]] = q[k.p.q[8]] = 0.04f;
    const Vec3 cube{0.55f, -0.1f, PandaPickTask::kHalf};
    std::vector<int> arm(k.p.ik.body, k.p.ik.body + 7);
    const Quat down{k.p.down[0], k.p.down[1], k.p.down[2], k.p.down[3]};
    solve_ik(k.m, k.p.hand, {cube.x, cube.y, cube.z + k.p.pad_z}, down, arm, q);  // pads at the cube center
    RobotWork work(k.m);
    DynamicsWork w = work.view();
    PandaPickTask::Camera c[2];
    PandaPickTask::cameras(k.p, k.m.view(), q.data(), w, c);
    flux::BatchCamera cam = make_cam();
    const flux::CameraView wrist{c[1].eye, c[1].target, c[1].up, c[1].fovy};
    float px, py;
    ASSERT_TRUE(cam.project(wrist, cube + Vec3{0, 0, PandaPickTask::kHalf}, px, py));  // top face center
    EXPECT_NEAR(px, 32.0f, 12.0f);
    EXPECT_NEAR(py, 32.0f, 20.0f);
}

}  // namespace
