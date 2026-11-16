#include "forge/env/demo_file.h"

#include <cstring>
#include <stdexcept>

#include <zlib.h>

namespace forge {

namespace {

constexpr char kMagic[8] = {'C', 'R', 'U', 'C', 'D', 'E', 'M', 'O'};
constexpr uint32_t kVersionPlain = 1, kVersionImages = 2;

void put(std::FILE* f, const void* p, size_t n, const std::string& path) {
    if (n && std::fwrite(p, 1, n, f) != n) throw std::runtime_error("demo file: write failed: " + path);
}
void get(std::FILE* f, void* p, size_t n, const std::string& path) {
    if (n && std::fread(p, 1, n, f) != n) throw std::runtime_error("demo file: truncated: " + path);
}

}  // namespace

DemoWriter::DemoWriter(const std::string& path, const std::string& task, int obs_dim, int act_dim,
                       const DemoImages& images)
    : obs_dim_(obs_dim), act_dim_(act_dim), images_(images), path_(path) {
    if (images.cameras < 0 || (images.cameras > 0 && (images.width <= 0 || images.height <= 0 || images.every <= 0)))
        throw std::runtime_error("demo file: bad image settings");
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) throw std::runtime_error("demo file: cannot open " + path);
    char name[32] = {};
    std::strncpy(name, task.c_str(), sizeof(name) - 1);
    const bool with_images = images.cameras > 0;
    const uint32_t v[3] = {with_images ? kVersionImages : kVersionPlain, uint32_t(obs_dim), uint32_t(act_dim)};
    put(f_, kMagic, 8, path_);
    put(f_, v, sizeof(v), path_);
    put(f_, name, sizeof(name), path_);
    if (with_images) {
        const uint32_t im[4] = {uint32_t(images.cameras), uint32_t(images.width), uint32_t(images.height),
                                uint32_t(images.every)};
        put(f_, im, sizeof(im), path_);
    }
}

DemoWriter::~DemoWriter() {
    if (f_) std::fclose(f_);
}

void DemoWriter::write(const DemoEpisode& ep) {
    const size_t T = ep.reward.size();
    if (ep.obs.size() != T * obs_dim_ || ep.expert.size() != T * act_dim_ || ep.executed.size() != T * act_dim_ ||
        ep.final_obs.size() != size_t(obs_dim_))
        throw std::runtime_error("demo file: episode sizes do not match obs_dim / act_dim");
    const size_t want_frames = images_.cameras > 0 ? (T + size_t(images_.every) - 1) / size_t(images_.every) : 0;
    if (images_.cameras > 0 && ep.images.size() != want_frames * images_.frame_bytes())  // check before writing anything
        throw std::runtime_error("demo file: an episode of " + std::to_string(T) + " steps needs " +
                                 std::to_string(want_frames) + " image frames");
    const uint32_t steps = uint32_t(T);
    const uint8_t flags[4] = {uint8_t(ep.success ? 1 : 0), 0, 0, 0};
    put(f_, &steps, 4, path_);
    put(f_, flags, 4, path_);
    put(f_, &ep.noise, 4, path_);
    put(f_, ep.obs.data(), ep.obs.size() * 4, path_);
    put(f_, ep.expert.data(), ep.expert.size() * 4, path_);
    put(f_, ep.executed.data(), ep.executed.size() * 4, path_);
    put(f_, ep.reward.data(), T * 4, path_);
    put(f_, ep.final_obs.data(), ep.final_obs.size() * 4, path_);
    if (images_.cameras > 0) {
        uLongf packed = compressBound(uLong(ep.images.size()));
        std::vector<uint8_t> buf(packed);
        if (compress2(buf.data(), &packed, ep.images.data(), uLong(ep.images.size()), 6) != Z_OK)
            throw std::runtime_error("demo file: zlib compress failed");
        const uint32_t hdr[2] = {uint32_t(want_frames), uint32_t(packed)};
        put(f_, hdr, sizeof(hdr), path_);
        put(f_, buf.data(), packed, path_);
        image_bytes_raw_ += ep.images.size();
        image_bytes_stored_ += packed;
    }
    if (std::fflush(f_) != 0) throw std::runtime_error("demo file: write failed: " + path_);
    ++episodes_;
}

DemoFile read_demo_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("demo file: cannot open " + path);
    DemoFile d;
    try {
        char magic[8];
        uint32_t v[3];
        char name[33] = {};
        get(f, magic, 8, path);
        if (std::memcmp(magic, kMagic, 8) != 0) throw std::runtime_error("demo file: not a CRUCDEMO file: " + path);
        get(f, v, sizeof(v), path);
        if (v[0] != kVersionPlain && v[0] != kVersionImages) throw std::runtime_error("demo file: unknown version in " + path);
        get(f, name, 32, path);
        d.task = name;
        d.version = int(v[0]);
        d.obs_dim = int(v[1]);
        d.act_dim = int(v[2]);
        if (d.version == int(kVersionImages)) {
            uint32_t im[4];
            get(f, im, sizeof(im), path);
            d.images = DemoImages{int(im[0]), int(im[1]), int(im[2]), int(im[3])};
        }
        for (;;) {
            uint32_t steps;
            if (std::fread(&steps, 1, 4, f) != 4) break;  // end of file
            uint8_t flags[4];
            DemoEpisode ep;
            get(f, flags, 4, path);
            get(f, &ep.noise, 4, path);
            ep.success = flags[0] != 0;
            ep.obs.resize(size_t(steps) * d.obs_dim);
            ep.expert.resize(size_t(steps) * d.act_dim);
            ep.executed.resize(size_t(steps) * d.act_dim);
            ep.reward.resize(steps);
            ep.final_obs.resize(d.obs_dim);
            get(f, ep.obs.data(), ep.obs.size() * 4, path);
            get(f, ep.expert.data(), ep.expert.size() * 4, path);
            get(f, ep.executed.data(), ep.executed.size() * 4, path);
            get(f, ep.reward.data(), ep.reward.size() * 4, path);
            get(f, ep.final_obs.data(), ep.final_obs.size() * 4, path);
            if (d.images.cameras > 0) {
                uint32_t hdr[2];
                get(f, hdr, sizeof(hdr), path);
                std::vector<uint8_t> packed(hdr[1]);
                get(f, packed.data(), packed.size(), path);
                ep.images.resize(size_t(hdr[0]) * d.images.frame_bytes());
                uLongf n = uLongf(ep.images.size());
                if (uncompress(ep.images.data(), &n, packed.data(), uLong(packed.size())) != Z_OK || n != ep.images.size())
                    throw std::runtime_error("demo file: bad image data in " + path);
            }
            d.episodes.push_back(std::move(ep));
        }
    } catch (...) {
        std::fclose(f);
        throw;
    }
    std::fclose(f);
    return d;
}

}  // namespace forge
