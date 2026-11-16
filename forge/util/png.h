#pragma once
// Minimal PNG writer (zlib, no other libraries): 8-bit RGB, row 0 = top.
#include <cstdint>
#include <string>

namespace forge {

// rgb: height * width * 3 bytes. Throws if the file cannot be written.
void write_png(const std::string& path, const uint8_t* rgb, int width, int height);

}  // namespace forge
