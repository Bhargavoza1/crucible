#include "flux/robot_mesh.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include <zlib.h>

namespace flux {

RobotMesh load_robot_mesh(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open mesh file " + path);
    const std::vector<char> file((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto fail = [&](const char* what) { throw std::runtime_error(path + ": " + what); };
    if (file.size() < 12 || std::memcmp(file.data(), "CRUMESH2", 8) != 0)
        fail("not a CRUMESH2 file (old files: make them again with the tools/ scripts)");
    uint32_t raw_size = 0;
    std::memcpy(&raw_size, file.data() + 8, 4);
    if (raw_size > (1u << 30)) fail("data too large");
    std::vector<unsigned char> data(raw_size);
    uLongf out_len = raw_size;
    if (uncompress(data.data(), &out_len, reinterpret_cast<const Bytef*>(file.data() + 12), uLong(file.size() - 12)) !=
            Z_OK ||
        out_len != raw_size)
        fail("bad compressed data");

    size_t pos = 0;
    auto take = [&](void* dst, size_t n) {
        if (pos + n > data.size()) fail("data ends early");
        std::memcpy(dst, data.data() + pos, n);
        pos += n;
    };
    auto u32 = [&]() {
        uint32_t v = 0;
        take(&v, 4);
        return v;
    };
    RobotMesh m;
    const uint32_t nlinks = u32();
    if (nlinks > 10000) fail("too many links");
    for (uint32_t i = 0; i < nlinks; ++i) {
        const uint32_t len = u32();
        if (len > 1000) fail("bad link name");
        std::string name(len, '\0');
        take(name.data(), len);
        m.links.push_back(name);
    }
    const uint32_t nlods = u32();
    if (nlods == 0 || nlods > 8) fail("bad number of levels of detail");
    for (uint32_t k = 0; k < nlods; ++k) {
        MeshLod lod;
        const uint32_t nv = u32(), ni = u32();
        if (ni % 3 != 0 || nv > 100000000u || ni > 300000000u) fail("bad vertex or index count");
        lod.vertices.resize(nv);
        lod.indices.resize(ni);
        take(lod.vertices.data(), size_t(nv) * sizeof(MeshVertex));
        take(lod.indices.data(), size_t(ni) * sizeof(uint32_t));
        for (const auto& x : lod.vertices)
            if (x.link >= nlinks) fail("vertex link out of range");
        for (uint32_t i : lod.indices)
            if (i >= nv) fail("index out of range");
        m.lods.push_back(std::move(lod));
    }
    if (pos != data.size()) fail("extra data at the end");
    return m;
}

std::string mesh_path_for_urdf(const std::string& urdf) {
    const size_t dot = urdf.rfind(".urdf");
    const std::string p = (dot == std::string::npos ? urdf : urdf.substr(0, dot)) + ".cmesh";
    std::ifstream f(p, std::ios::binary);
    return f ? p : std::string();
}

}  // namespace flux
