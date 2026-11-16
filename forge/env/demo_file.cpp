#include "forge/env/demo_file.h"

#include <cstring>
#include <stdexcept>

namespace forge {

namespace {

constexpr char kMagic[8] = {'C', 'R', 'U', 'C', 'D', 'E', 'M', 'O'};
constexpr uint32_t kVersion = 1;

void put(std::FILE* f, const void* p, size_t n, const std::string& path) {
    if (n && std::fwrite(p, 1, n, f) != n) throw std::runtime_error("demo file: write failed: " + path);
}
void get(std::FILE* f, void* p, size_t n, const std::string& path) {
    if (n && std::fread(p, 1, n, f) != n) throw std::runtime_error("demo file: truncated: " + path);
}

}  // namespace

DemoWriter::DemoWriter(const std::string& path, const std::string& task, int obs_dim, int act_dim)
    : obs_dim_(obs_dim), act_dim_(act_dim), path_(path) {
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) throw std::runtime_error("demo file: cannot open " + path);
    char name[32] = {};
    std::strncpy(name, task.c_str(), sizeof(name) - 1);
    const uint32_t v[3] = {kVersion, uint32_t(obs_dim), uint32_t(act_dim)};
    put(f_, kMagic, 8, path_);
    put(f_, v, sizeof(v), path_);
    put(f_, name, sizeof(name), path_);
}

DemoWriter::~DemoWriter() {
    if (f_) std::fclose(f_);
}

void DemoWriter::write(const DemoEpisode& ep) {
    const size_t T = ep.reward.size();
    if (ep.obs.size() != T * obs_dim_ || ep.expert.size() != T * act_dim_ || ep.executed.size() != T * act_dim_ ||
        ep.final_obs.size() != size_t(obs_dim_))
        throw std::runtime_error("demo file: episode sizes do not match obs_dim / act_dim");
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
        if (v[0] != kVersion) throw std::runtime_error("demo file: unknown version in " + path);
        get(f, name, 32, path);
        d.task = name;
        d.obs_dim = int(v[1]);
        d.act_dim = int(v[2]);
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
