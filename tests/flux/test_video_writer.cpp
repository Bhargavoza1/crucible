// flux::VideoWriter: frames piped into ffmpeg come out as a playable .mp4 with the
// asked size, frame count and frame rate, in the right colors (B G R A in). Skipped
// without the ffmpeg / ffprobe programs.
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "flux/video_writer.h"

namespace {

std::string run(const std::string& cmd) {  // a command's output
    std::string out;
    if (std::FILE* p = popen(cmd.c_str(), "r")) {
        char buf[256];
        while (std::fgets(buf, sizeof(buf), p)) out += buf;
        pclose(p);
    }
    return out;
}

TEST(VideoWriter, WritesPlayableMp4) {
    if (!flux::VideoWriter::ffmpeg_available() || std::system("ffprobe -version > /dev/null 2>&1") != 0)
        GTEST_SKIP() << "no ffmpeg / ffprobe";
    const std::string path = (std::filesystem::temp_directory_path() / "crucible_test_video.mp4").string();
    std::filesystem::remove(path);
    const int W = 160, H = 90, F = 25;
    flux::VideoWriter v;
    v.open(path, W, H, 25, true);
    std::vector<uint8_t> px(size_t(W) * H * 4);
    for (int f = 0; f < F; ++f) {
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {  // left half red, right half blue (B G R A bytes)
                uint8_t* p = px.data() + (size_t(y) * W + x) * 4;
                const bool left = x < W / 2;
                p[0] = left ? 0 : 255, p[1] = 0, p[2] = left ? 255 : 0, p[3] = 255;
            }
        EXPECT_TRUE(v.write(px.data()));
    }
    EXPECT_EQ(v.frames(), F);
    v.close();
    EXPECT_FALSE(v.is_open());
    EXPECT_FALSE(v.write(px.data())) << "closed: nothing written";

    const std::string info = run("ffprobe -v error -select_streams v:0 -count_frames -show_entries "
                                 "stream=width,height,nb_read_frames,r_frame_rate -of csv=p=0 '" + path + "'");
    std::printf("video_writer: ffprobe: %s", info.c_str());
    EXPECT_EQ(info, "160,90,25/1,25\n");

    // colors: decode the middle frame to RGB, sample the left and right halves
    const std::string raw = (std::filesystem::temp_directory_path() / "crucible_test_video.rgb").string();
    run("ffmpeg -v error -y -i '" + path + "' -vf 'select=eq(n\\,12)' -frames:v 1 -f rawvideo -pix_fmt rgb24 '" + raw +
        "'");
    std::FILE* f = std::fopen(raw.c_str(), "rb");
    ASSERT_NE(f, nullptr);
    std::vector<uint8_t> rgb(size_t(W) * H * 3);
    ASSERT_EQ(std::fread(rgb.data(), 1, rgb.size(), f), rgb.size());
    std::fclose(f);
    const uint8_t* l = rgb.data() + (size_t(H / 2) * W + W / 4) * 3;
    const uint8_t* r = rgb.data() + (size_t(H / 2) * W + 3 * W / 4) * 3;
    std::printf("video_writer: left %d %d %d, right %d %d %d\n", l[0], l[1], l[2], r[0], r[1], r[2]);
    EXPECT_GT(l[0], 200);  // red
    EXPECT_LT(l[2], 60);
    EXPECT_GT(r[2], 200);  // blue
    EXPECT_LT(r[0], 60);
    std::filesystem::remove(path);
    std::filesystem::remove(raw);
}

TEST(VideoWriter, BadArgumentsThrow) {
    flux::VideoWriter v;
    EXPECT_THROW(v.open("x.mp4", 0, 10, 25, true), std::runtime_error);
    EXPECT_THROW(v.open("x.mp4", 10, 10, 0, true), std::runtime_error);
    EXPECT_THROW(v.open("bad'name.mp4", 10, 10, 25, true), std::runtime_error);
    EXPECT_FALSE(v.is_open());
}

}  // namespace
