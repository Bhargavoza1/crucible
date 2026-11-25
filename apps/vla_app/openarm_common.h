#pragma once
// vla_app's runner for the vision tasks (vision.cpp): the model's info file, the per-env
// zlib frame stream for DAgger recording, the camera strips of failed episodes.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <zlib.h>

#include "forge/util/png.h"

namespace vla {

// <onnx>_openarm.txt (vla_openarm.py --onnx): "cameras 0 1", "task NAME" and "text v1 v2 ..."
struct Info {
    std::vector<int> cameras;
    std::vector<float> text;
    std::string task;  // empty in files from before openarm_stack (then openarm_pick_place)
};
inline Info read_info(const std::string& path, int num_cameras, const char* task) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path + " (written by vla_openarm.py --onnx)");
    Info info;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "cameras") {
            int c;
            while (ss >> c) info.cameras.push_back(c);
        } else if (key == "task") {
            ss >> info.task;
        } else if (key == "text") {
            float v;
            while (ss >> v) info.text.push_back(v);
        }
    }
    if (info.cameras.empty() || info.text.empty()) throw std::runtime_error(path + ": needs 'cameras' and 'text' lines");
    for (int c : info.cameras)
        if (c < 0 || c >= num_cameras) throw std::runtime_error(path + ": camera ids are 0-" + std::to_string(num_cameras - 1));
    if (!info.task.empty() && info.task != task)
        throw std::runtime_error(path + ": a model for " + info.task + ", not " + task + " (use --task " + info.task + ")");
    return info;
}

// One episode's camera frames as one zlib stream (what DemoWriter::compress_images
// makes of them all at once), compressed as they come.
class FrameStream {
public:
    FrameStream() {
        z_ = z_stream{};
        if (deflateInit(&z_, 1) != Z_OK) throw std::runtime_error("zlib deflateInit failed");
    }
    ~FrameStream() { deflateEnd(&z_); }
    FrameStream(const FrameStream&) = delete;
    FrameStream& operator=(const FrameStream&) = delete;
    void add(const uint8_t* data, size_t n) { pump(data, n, Z_NO_FLUSH); }
    std::vector<uint8_t> finish() {  // the stream; then ready for the next episode
        pump(nullptr, 0, Z_FINISH);
        std::vector<uint8_t> done(out_.begin(), out_.end());
        out_.clear();
        deflateReset(&z_);
        return done;
    }

private:
    void pump(const uint8_t* data, size_t n, int flush) {
        z_.next_in = const_cast<Bytef*>(data);
        z_.avail_in = uInt(n);
        uint8_t buf[1 << 16];
        int ret;
        do {
            z_.next_out = buf;
            z_.avail_out = sizeof(buf);
            ret = deflate(&z_, flush);
            if (ret == Z_STREAM_ERROR) throw std::runtime_error("zlib deflate failed");
            out_.insert(out_.end(), buf, buf + (sizeof(buf) - z_.avail_out));
        } while (z_.avail_out == 0 || (flush == Z_FINISH && ret != Z_STREAM_END));
    }
    z_stream z_;
    std::vector<uint8_t> out_;
};

// Camera strip: rows = cameras, one column per stored frame.
inline void save_strip(const std::string& path, const std::vector<uint8_t>& frames, int cams, int size) {
    const size_t one = size_t(size) * size * 3, frame = one * cams;
    const int nf = int(frames.size() / frame);
    if (nf == 0) return;
    const int W = nf * size, H = cams * size;
    std::vector<uint8_t> img(size_t(W) * H * 3);
    for (int f = 0; f < nf; ++f)
        for (int c = 0; c < cams; ++c)
            for (int y = 0; y < size; ++y)
                std::copy_n(frames.data() + f * frame + c * one + size_t(y) * size * 3, size_t(size) * 3,
                            img.data() + ((size_t(c) * size + y) * W + size_t(f) * size) * 3);
    forge::write_png(path, img.data(), W, H);
}

}  // namespace vla
