#pragma once
// Demonstration files (Phase 9b): episodes of (obs, expert action, executed action,
// reward), written by collect_app and read by tools/read_demos.py for imitation
// learning. Little-endian, float32:
//
//   header   char[8] "CRUCDEMO", u32 version (1), u32 obs_dim, u32 act_dim,
//            char[32] task name (zero padded)
//   episode  u32 steps T, u8 success, u8[3] zero, f32 noise (std of the DART noise
//            added to the executed actions in this episode),
//            obs[T * obs_dim]       the obs before each step
//            expert[T * act_dim]    the expert's action (the label)
//            executed[T * act_dim]  the action that was run (expert + noise, clipped)
//            reward[T]
//            final_obs[obs_dim]     the obs after the last step
// Episodes follow each other until the end of the file.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace forge {

struct DemoEpisode {
    std::vector<float> obs, expert, executed, reward, final_obs;
    bool success = false;
    float noise = 0.0f;
    int steps() const { return static_cast<int>(reward.size()); }
};

class DemoWriter {
public:
    DemoWriter(const std::string& path, const std::string& task, int obs_dim, int act_dim);
    ~DemoWriter();
    DemoWriter(const DemoWriter&) = delete;
    DemoWriter& operator=(const DemoWriter&) = delete;
    void write(const DemoEpisode& ep);  // throws if the sizes do not match
    int episodes() const { return episodes_; }

private:
    std::FILE* f_ = nullptr;
    int obs_dim_, act_dim_, episodes_ = 0;
    std::string path_;
};

struct DemoFile {
    std::string task;
    int obs_dim = 0, act_dim = 0;
    std::vector<DemoEpisode> episodes;
};
DemoFile read_demo_file(const std::string& path);  // throws on a bad file

}  // namespace forge
