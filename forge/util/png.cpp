#include "forge/util/png.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

#include <zlib.h>

namespace forge {

namespace {

void put_u32(std::vector<uint8_t>& o, uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) o.push_back(uint8_t(v >> s));
}

void chunk(std::vector<uint8_t>& o, const char* tag, const std::vector<uint8_t>& body) {
    put_u32(o, uint32_t(body.size()));
    const size_t start = o.size();
    o.insert(o.end(), tag, tag + 4);
    o.insert(o.end(), body.begin(), body.end());
    put_u32(o, uint32_t(crc32(0L, o.data() + start, uInt(o.size() - start))));
}

}  // namespace

void write_png(const std::string& path, const uint8_t* rgb, int width, int height) {
    if (width <= 0 || height <= 0) throw std::runtime_error("write_png: empty image");
    std::vector<uint8_t> raw;  // each row: filter byte 0, then the pixels
    raw.reserve(size_t(height) * (size_t(width) * 3 + 1));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb + size_t(y) * width * 3, rgb + size_t(y + 1) * width * 3);
    }
    uLongf n = compressBound(uLong(raw.size()));
    std::vector<uint8_t> z(n);
    if (compress2(z.data(), &n, raw.data(), uLong(raw.size()), 6) != Z_OK) throw std::runtime_error("write_png: zlib");
    z.resize(n);
    std::vector<uint8_t> o = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    std::vector<uint8_t> ihdr;
    put_u32(ihdr, uint32_t(width));
    put_u32(ihdr, uint32_t(height));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // 8 bits, RGB, deflate, no filter, no interlace
    chunk(o, "IHDR", ihdr);
    chunk(o, "IDAT", z);
    chunk(o, "IEND", {});
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("write_png: cannot write " + path);
    const bool ok = std::fwrite(o.data(), 1, o.size(), f) == o.size();
    std::fclose(f);
    if (!ok) throw std::runtime_error("write_png: write failed for " + path);
}

}  // namespace forge
