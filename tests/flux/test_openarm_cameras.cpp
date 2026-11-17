// Phase 10: the openarm_pick_place camera images (apps/collect_app/openarm_camera_rig).
//   - the cube is drawn where the camera projects it (head and ceiling views)
//   - left and right as on the real cell's cameras (MuJoCo: image x to the camera's
//     right): the left arm's side (+y) is on the image's left in the head and
//     ceiling views
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "openarm_camera_rig.h"
#include "forge/robots/urdf.h"

using namespace forge;

namespace {

TEST(OpenArmCameras, CubeWhereProjectedAndSidesAsTheCell) {
    UrdfOptions u;
    OpenArmPickPlaceTask::urdf_options(u);
    const std::string dir = std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/openarm/";
    RobotModel m = load_urdf(dir + "openarm_cell.urdf", u);
    OpenArmPickPlaceTask::edit_model(m);
    const auto p = OpenArmPickPlaceTask::make_params(m);
    const int S = 128;
    OpenArmCameraRig rig(m, 1, S, FLUX_SHADER_DIR, dir + "openarm_cell.cmesh");
    std::vector<float> q(size_t(m.nq), 0.0f);
    for (int k = 0; k < 9; ++k) q[size_t(p.q[k])] = p.start[k];
    const Vec3 cube{0.34f, 0.28f, 0.02f};  // in the work area, not under the arm at its start pose
    q[size_t(p.cube_q)] = cube.x, q[size_t(p.cube_q) + 1] = cube.y, q[size_t(p.cube_q) + 2] = cube.z;
    q[size_t(p.cube_q) + 3] = 1.0f;
    rig.render(q.data());

    RobotWork work(m);
    DynamicsWork w = work.view();
    OpenArmPickPlaceTask::Camera cams[4];
    OpenArmPickPlaceTask::cameras(p, m.view(), q.data(), w, cams);
    std::vector<uint8_t> img(size_t(S) * S * 3);
    for (int c : {0, 2}) {  // head, ceiling
        const flux::CameraView v{cams[c].eye, cams[c].target, cams[c].up, cams[c].fovy};
        float px, py;
        ASSERT_TRUE(rig.camera().project(v, cube, px, py));
        rig.camera().copy_rgb(0, c, img.data());
        double sx = 0, sy = 0;
        int n = 0;
        for (int y = 0; y < S; ++y)
            for (int x = 0; x < S; ++x) {
                const uint8_t* rgb = img.data() + (size_t(y) * S + x) * 3;
                if (rgb[0] > 150 && rgb[1] > 40 && rgb[1] < 140 && rgb[2] < 60) sx += x, sy += y, ++n;  // orange
            }
        ASSERT_GT(n, 4) << "camera " << c << ": the cube is visible";
        std::printf("camera %d: cube projected at (%.1f, %.1f), drawn at (%.1f, %.1f), %d pixels\n", c, px, py, sx / n,
                    sy / n, n);
        EXPECT_NEAR(sx / n, px, 4.0) << "camera " << c;
        EXPECT_NEAR(sy / n, py, 4.0) << "camera " << c;
        float xl, yl, xr, yr;  // a point on the left arm's side (+y) and one on the right's (-y)
        ASSERT_TRUE(rig.camera().project(v, {0.40f, 0.25f, 0.0f}, xl, yl));
        ASSERT_TRUE(rig.camera().project(v, {0.40f, -0.25f, 0.0f}, xr, yr));
        EXPECT_LT(xl, xr) << "camera " << c << ": +y on the image's left, as the cell's camera";
    }
}

}  // namespace
