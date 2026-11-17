// Phase 10: robot meshes (flux/robot_mesh.h, flux/pipeline/mesh_pipeline.h).
//   - the OpenArm .cmesh files load, and every mesh link is a body of its URDF
//   - a mesh drawn by BatchCamera lands where its link pose puts it (rotation and
//     translation both used), and each tile shows only its own env's pose
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "flux/batch_camera.h"
#include "flux/robot_mesh.h"
#include "forge/robots/urdf.h"

using namespace flux;

namespace {

TEST(RobotMesh, OpenArmMeshLinksAreUrdfBodies) {
    for (const std::string name : {"openarm", "openarm_bimanual"}) {
        const std::string urdf = std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/openarm/" + name + ".urdf";
        const std::string path = mesh_path_for_urdf(urdf);
        ASSERT_FALSE(path.empty()) << name << ".cmesh not found next to the URDF";
        const RobotMesh mesh = load_robot_mesh(path);
        const forge::RobotModel model = forge::load_urdf(urdf, forge::UrdfOptions{});
        ASSERT_EQ(mesh.lods.size(), 2u);
        EXPECT_GT(mesh.lods[0].size(), mesh.lods[1].size()) << "LOD 1 (camera images) has fewer triangles";
        for (const std::string& link : mesh.links)
            EXPECT_NE(std::find(model.body_name.begin(), model.body_name.end(), link), model.body_name.end())
                << name << ": mesh link " << link << " is not a URDF body";
        std::printf("%s: %d links, %zu / %zu triangles\n", name.c_str(), mesh.num_links(), mesh.lods[0].size() / 3,
                    mesh.lods[1].size() / 3);
    }
}

// One link: a red 10 cm square, centered at (0.1, 0, 0) in the link frame, facing +z.
RobotMesh red_square() {
    RobotMesh m;
    m.links = {"square"};
    auto v = [](float x, float y) {
        MeshVertex p{};
        p.pos[0] = 0.1f + x;
        p.pos[1] = y;
        p.normal[2] = 1.0f;
        p.rgba[0] = 230;
        p.rgba[1] = 25;
        p.rgba[2] = 25;
        p.rgba[3] = 255;
        p.link = 0;
        return p;
    };
    const float h = 0.05f;
    m.lods.push_back({v(-h, -h), v(h, -h), v(h, h), v(-h, -h), v(h, h), v(-h, h)});
    return m;
}

bool is_red(const uint8_t* p) { return p[0] > 150 && p[1] < 60 && p[2] < 60; }

TEST(BatchCamera, MeshFollowsItsLinkPose) {
    BatchCameraOptions o;
    o.num_envs = 2;
    o.width = o.height = 64;
    o.shader_dir = FLUX_SHADER_DIR;
    BatchCamera cam(o);
    cam.set_robot_mesh(red_square(), 0);
    // env 0: no turn, at (0.05, 0.1, 0.05)  -> square center (0.15, 0.1, 0.05)
    // env 1: turned +90 deg about z, at (-0.15, 0, 0.05) -> center (-0.15, 0.1, 0.05)
    const float poses[2 * 12] = {1, 0, 0, 0.05f, 0, 1, 0, 0.1f, 0, 0, 1, 0.05f,
                                 0, -1, 0, -0.15f, 1, 0, 0, 0.0f, 0, 0, 1, 0.05f};
    CameraView v;
    v.eye = {0, 0, 1};
    v.target = {0, 0, 0};
    v.up = {0, 1, 0};
    v.fovy = 0.8f;
    const CameraView views[2] = {v, v};
    cam.render(nullptr, nullptr, nullptr, nullptr, nullptr, views, poses);

    auto red_at = [&](int env, Vec3 p) {
        float x, y;
        EXPECT_TRUE(cam.project(v, p, x, y));
        std::vector<uint8_t> img(64 * 64 * 3);
        cam.copy_rgb(env, 0, img.data());
        return is_red(img.data() + (size_t(y) * 64 + size_t(x)) * 3);
    };
    EXPECT_TRUE(red_at(0, {0.15f, 0.1f, 0.05f}));
    EXPECT_FALSE(red_at(0, {-0.15f, 0.1f, 0.05f})) << "env 0's tile must not show env 1's mesh";
    EXPECT_TRUE(red_at(1, {-0.15f, 0.1f, 0.05f})) << "the turn moves the square to +y";
    EXPECT_FALSE(red_at(1, {-0.05f, 0.0f, 0.05f})) << "where it would be without the turn";
    EXPECT_FALSE(red_at(1, {0.15f, 0.1f, 0.05f}));
}

}  // namespace
