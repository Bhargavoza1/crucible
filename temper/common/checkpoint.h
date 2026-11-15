#pragma once
// Checkpoint file (doc 7.3): one binary file.
//   header: magic "CRUCKPT1", format version, algorithm, task, obs_dim, act_dim,
//           layer sizes of each network, iteration, seed, optimizer step
//   then named float arrays: weights, observation normalizer, Adam m and v
// Host code only, no CUDA: tests_cpu checks the save/load round trip.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace temper {

struct Checkpoint {
    static constexpr uint32_t kVersion = 1;
    std::string algo, task;
    int obs_dim = 0, act_dim = 0;
    std::map<std::string, std::vector<int>> layer_sizes;  // e.g. "actor" -> {33, 256, 256, 256, 8}
    int64_t iteration = 0;
    uint64_t seed = 0;
    int64_t optimizer_steps = 0;
    std::map<std::string, std::vector<float>> arrays;     // name -> numbers

    const std::vector<float>& array(const std::string& name) const {
        auto it = arrays.find(name);
        if (it == arrays.end()) throw std::runtime_error("checkpoint: no array '" + name + "'");
        return it->second;
    }

    void save(const std::string& path) const {
        const std::string tmp = path + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary);
            if (!f) throw std::runtime_error("checkpoint: cannot write " + tmp);
            f.write("CRUCKPT1", 8);
            put(f, kVersion);
            put_str(f, algo);
            put_str(f, task);
            put(f, int32_t(obs_dim));
            put(f, int32_t(act_dim));
            put(f, uint32_t(layer_sizes.size()));
            for (const auto& [name, dims] : layer_sizes) {
                put_str(f, name);
                put(f, uint32_t(dims.size()));
                for (int d : dims) put(f, int32_t(d));
            }
            put(f, iteration);
            put(f, seed);
            put(f, optimizer_steps);
            put(f, uint32_t(arrays.size()));
            for (const auto& [name, v] : arrays) {
                put_str(f, name);
                put(f, uint64_t(v.size()));
                f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(float)));
            }
            if (!f) throw std::runtime_error("checkpoint: write failed for " + tmp);
        }
        // Write then rename: a crash never leaves a half-written checkpoint.
        if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("checkpoint: cannot rename to " + path);
    }

    static Checkpoint load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("checkpoint: cannot open " + path);
        char magic[8];
        f.read(magic, 8);
        if (!f || std::string(magic, 8) != "CRUCKPT1") throw std::runtime_error(path + ": not a crucible checkpoint");
        Checkpoint c;
        const uint32_t version = get<uint32_t>(f);
        if (version != kVersion)
            throw std::runtime_error(path + ": checkpoint format " + std::to_string(version) + ", this build reads " +
                                     std::to_string(kVersion));
        c.algo = get_str(f);
        c.task = get_str(f);
        c.obs_dim = get<int32_t>(f);
        c.act_dim = get<int32_t>(f);
        const uint32_t nl = get<uint32_t>(f);
        for (uint32_t i = 0; i < nl; ++i) {
            const std::string name = get_str(f);
            std::vector<int> dims(get<uint32_t>(f));
            for (int& d : dims) d = get<int32_t>(f);
            c.layer_sizes[name] = dims;
        }
        c.iteration = get<int64_t>(f);
        c.seed = get<uint64_t>(f);
        c.optimizer_steps = get<int64_t>(f);
        const uint32_t na = get<uint32_t>(f);
        for (uint32_t i = 0; i < na; ++i) {
            const std::string name = get_str(f);
            const uint64_t n = get<uint64_t>(f);
            if (n > (uint64_t(1) << 32)) throw std::runtime_error(path + ": array '" + name + "' too big (bad file?)");
            std::vector<float> v(n);
            f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(float)));
            c.arrays[name] = std::move(v);
        }
        if (!f) throw std::runtime_error(path + ": file is cut short");
        return c;
    }

private:
    template <typename T>
    static void put(std::ofstream& f, T v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(T));
    }
    static void put_str(std::ofstream& f, const std::string& s) {
        put(f, uint32_t(s.size()));
        f.write(s.data(), std::streamsize(s.size()));
    }
    template <typename T>
    static T get(std::ifstream& f) {
        T v{};
        f.read(reinterpret_cast<char*>(&v), sizeof(T));
        if (!f) throw std::runtime_error("checkpoint: file is cut short");
        return v;
    }
    static std::string get_str(std::ifstream& f) {
        const uint32_t n = get<uint32_t>(f);
        if (n > 4096) throw std::runtime_error("checkpoint: bad string length (bad file?)");
        std::string s(n, '\0');
        f.read(s.data(), n);
        return s;
    }
};

}  // namespace temper
