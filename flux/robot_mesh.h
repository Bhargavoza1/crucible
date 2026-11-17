#pragma once
// Robot visual meshes (Phase 10): a robot's links drawn as colored triangles.
//
// File <robot>.cmesh (made by tools/cmesh.py, called by tools/openarm_to_urdf.py,
// unitree_to_assets.py and mjcf_to_cmesh.py from the robot's visual meshes), little
// endian: "CRUMESH2", uint32 size of the data, then the data compressed with zlib:
//   uint32 nlinks, nlinks x (uint32 length, name bytes), uint32 nlods,
//   nlods x (uint32 nverts, uint32 nindices, nverts x MeshVertex, nindices x uint32)
// Indexed triangles: every 3 indices are one triangle; vertices are shared (smooth
// normals). A vertex is in the frame of its link (link = index into the names).
// LOD 0: all of the model's triangles (the viewer); LOD 1: simplified (small
// camera images).
//
// To draw: per env and link, the link's world pose as a 3 x 4 matrix (rows of
// [R | t]); flux draws vertex v of link l at R v + t.
#include <cstdint>
#include <string>
#include <vector>

namespace flux {

struct MeshVertex {
    float pos[3];
    float normal[3];
    uint8_t rgba[4];
    uint32_t link;
};
static_assert(sizeof(MeshVertex) == 32, "must match the .cmesh file and mesh.vert");

struct MeshLod {
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t> indices;  // 3 per triangle
    size_t triangles() const { return indices.size() / 3; }
};

struct RobotMesh {
    std::vector<std::string> links;
    std::vector<MeshLod> lods;
    int num_links() const { return static_cast<int>(links.size()); }
};

// Throws if the file is missing or bad.
RobotMesh load_robot_mesh(const std::string& path);

// <urdf path without .urdf>.cmesh if that file exists, else "".
std::string mesh_path_for_urdf(const std::string& urdf);

}  // namespace flux
