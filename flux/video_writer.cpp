#include "flux/video_writer.h"

#include <cstdlib>
#include <stdexcept>

namespace flux {

bool VideoWriter::ffmpeg_available() { return std::system("ffmpeg -version > /dev/null 2>&1") == 0; }

void VideoWriter::open(const std::string& path, int width, int height, int fps, bool bgra) {
    close();
    if (width <= 0 || height <= 0 || fps <= 0) throw std::runtime_error("video: bad size or fps");
    if (path.empty() || path.find('\'') != std::string::npos) throw std::runtime_error("video: bad file name");
    if (!ffmpeg_available()) throw std::runtime_error("video: needs the ffmpeg program (sudo apt install ffmpeg)");
    const std::string cmd = "ffmpeg -y -loglevel error -f rawvideo -pix_fmt " + std::string(bgra ? "bgra" : "rgba") +
                            " -s " + std::to_string(width) + "x" + std::to_string(height) + " -r " +
                            std::to_string(fps) +
                            " -i - -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264 -preset veryfast -crf 18 "
                            "-pix_fmt yuv420p -movflags +faststart '" +
                            path + "'";
    pipe_ = popen(cmd.c_str(), "w");
    if (!pipe_) throw std::runtime_error("video: could not start ffmpeg");
    w_ = width, h_ = height, frames_ = 0, path_ = path;
}

bool VideoWriter::write(const uint8_t* pixels) {
    if (!pipe_) return false;
    const size_t n = size_t(w_) * h_ * 4;
    if (std::fwrite(pixels, 1, n, pipe_) != n) {
        close();
        throw std::runtime_error("video: ffmpeg stopped taking frames (" + path_ + ")");
    }
    ++frames_;
    return true;
}

void VideoWriter::close() {
    if (!pipe_) return;
    const int rc = pclose(pipe_);
    pipe_ = nullptr;
    if (rc != 0) throw std::runtime_error("video: ffmpeg failed writing " + path_);
}

}  // namespace flux
