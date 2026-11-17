#include "flux/robot_mesh.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace flux {

RobotMesh load_robot_mesh(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open mesh file " + path);
    auto fail = [&](const char* what) { throw std::runtime_error(path + ": " + what); };
    auto u32 = [&]() {
        uint32_t v = 0;
        if (!f.read(reinterpret_cast<char*>(&v), 4)) fail("file ends early");
        return v;
    };
    char magic[8];
    if (!f.read(magic, 8) || std::memcmp(magic, "CRUMESH1", 8) != 0) fail("not a CRUMESH1 file");
    RobotMesh m;
    const uint32_t nlinks = u32();
    if (nlinks > 10000) fail("too many links");
    for (uint32_t i = 0; i < nlinks; ++i) {
        const uint32_t len = u32();
        if (len > 1000) fail("bad link name");
        std::string name(len, '\0');
        if (!f.read(name.data(), len)) fail("file ends early");
        m.links.push_back(name);
    }
    const uint32_t nlods = u32();
    if (nlods == 0 || nlods > 8) fail("bad number of levels of detail");
    for (uint32_t k = 0; k < nlods; ++k) {
        const uint32_t n = u32();
        if (n % 3 != 0 || n > 50000000u) fail("bad vertex count");
        std::vector<MeshVertex> v(n);
        if (!f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n) * sizeof(MeshVertex))) fail("file ends early");
        for (const auto& x : v)
            if (x.link >= nlinks) fail("vertex link out of range");
        m.lods.push_back(std::move(v));
    }
    return m;
}

std::string mesh_path_for_urdf(const std::string& urdf) {
    const size_t dot = urdf.rfind(".urdf");
    const std::string p = (dot == std::string::npos ? urdf : urdf.substr(0, dot)) + ".cmesh";
    std::ifstream f(p, std::ios::binary);
    return f ? p : std::string();
}

}  // namespace flux
