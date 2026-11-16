#pragma once
// Demonstration files (Phase 9b): episodes of (obs, expert action, executed action,
// reward), written by collect_app and read by tools/read_demos.py for imitation
// learning. Little-endian, float32:
//
//   header   char[8] "CRUCDEMO", u32 version (1 or 2), u32 obs_dim, u32 act_dim,
//            char[32] task name (zero padded)
//            version 2 only: u32 cameras, u32 width, u32 height, u32 every
//   episode  u32 steps T, u8 success, u8[3] zero, f32 noise (std of the DART noise
//            added to the executed actions in this episode),
//            obs[T * obs_dim]       the obs before each step
//            expert[T * act_dim]    the expert's action (the label)
//            executed[T * act_dim]  the action that was run (expert + noise, clipped)
//            reward[T]
//            final_obs[obs_dim]     the obs after the last step
//            version 2 only: u32 frames F, u32 bytes B, then B bytes: zlib-compressed
//            images, uint8 [F][cameras][height][width][3] (RGB, row 0 = top). Frame
//            f is the camera view before step f * every (with the obs of that step).
// Episodes follow each other until the end of the file. Files without images are
// written as version 1.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace forge {

struct DemoImages {
    int cameras = 0, width = 0, height = 0, every = 0;  // cameras == 0: no images
    size_t frame_bytes() const { return size_t(cameras) * width * height * 3; }
};

struct DemoEpisode {
    std::vector<float> obs, expert, executed, reward, final_obs;
    std::vector<uint8_t> images;  // frames * DemoImages::frame_bytes(), uncompressed
    bool success = false;
    float noise = 0.0f;
    int steps() const { return static_cast<int>(reward.size()); }
};

class DemoWriter {
public:
    DemoWriter(const std::string& path, const std::string& task, int obs_dim, int act_dim,
               const DemoImages& images = {});
    ~DemoWriter();
    DemoWriter(const DemoWriter&) = delete;
    DemoWriter& operator=(const DemoWriter&) = delete;
    // Throws if the sizes do not match. packed: the episode's images already
    // compressed by compress_images (so many episodes can be compressed on several
    // threads at once); null = compress here.
    void write(const DemoEpisode& ep, const std::vector<uint8_t>* packed = nullptr);
    // zlib level 6 (about 7x smaller on the pick task's camera images). Thread safe.
    static std::vector<uint8_t> compress_images(const std::vector<uint8_t>& raw);
    int episodes() const { return episodes_; }

private:
    std::FILE* f_ = nullptr;
    int obs_dim_, act_dim_, episodes_ = 0;
    DemoImages images_;
    std::string path_;
    size_t image_bytes_raw_ = 0, image_bytes_stored_ = 0;

public:
    // Sizes of the images written so far (for the compression ratio in logs).
    size_t image_bytes_raw() const { return image_bytes_raw_; }
    size_t image_bytes_stored() const { return image_bytes_stored_; }
};

struct DemoFile {
    std::string task;
    int obs_dim = 0, act_dim = 0, version = 0;
    DemoImages images;
    std::vector<DemoEpisode> episodes;
};
DemoFile read_demo_file(const std::string& path);  // throws on a bad file

}  // namespace forge
