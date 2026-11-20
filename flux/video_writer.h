#pragma once
// Video file writer: raw frames piped into ffmpeg (H.264 .mp4). No Vulkan; the frames
// come from anywhere (Renderer::captured(), BatchCamera images, tests).
//
//   flux::VideoWriter v;
//   v.open("out.mp4", 1280, 720, 50, true);   // B G R A bytes per pixel
//   v.write(pixels);                           // width * height * 4 bytes
//   v.close();                                 // also done by the destructor
//
// Needs the ffmpeg program (Ubuntu: sudo apt install ffmpeg); open() throws without it.
// Encoding: libx264, -preset veryfast -crf 18 (near lossless), yuv420p (plays
// everywhere; an odd width or height is cut by one pixel).
#include <cstdint>
#include <cstdio>
#include <string>

namespace flux {

class VideoWriter {
public:
    VideoWriter() = default;
    ~VideoWriter() {
        try {
            close();
        } catch (...) {  // a destructor must not throw: call close() to see errors
        }
    }
    VideoWriter(const VideoWriter&) = delete;
    VideoWriter& operator=(const VideoWriter&) = delete;

    static bool ffmpeg_available();
    // bgra: the bytes are B G R A (a B8G8R8A8 swapchain), else R G B A.
    void open(const std::string& path, int width, int height, int fps, bool bgra);
    // One frame, width * height * 4 bytes. Returns false (and writes nothing) when closed.
    bool write(const uint8_t* pixels);
    // Finishes the file (waits for ffmpeg). Throws if ffmpeg failed.
    void close();

    bool is_open() const { return pipe_ != nullptr; }
    int width() const { return w_; }
    int height() const { return h_; }
    long frames() const { return frames_; }
    const std::string& path() const { return path_; }

private:
    std::FILE* pipe_ = nullptr;
    int w_ = 0, h_ = 0;
    long frames_ = 0;
    std::string path_;
};

}  // namespace flux
