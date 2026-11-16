// forge::write_png: a valid PNG header with the right size (the pixels were
// checked once with Python's PIL when this was written).
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/util/png.h"

TEST(Png, WritesHeaderAndSize) {
    const int w = 5, h = 3;
    std::vector<uint8_t> rgb(size_t(w) * h * 3);
    for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = uint8_t(i * 7);
    const std::string path = std::string(CRUCIBLE_TEST_TMP) + "/test.png";
    std::filesystem::create_directories(CRUCIBLE_TEST_TMP);
    forge::write_png(path, rgb.data(), w, h);
    std::ifstream f(path, std::ios::binary);
    const std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ASSERT_GT(b.size(), 33u);
    const uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    for (int i = 0; i < 8; ++i) EXPECT_EQ(b[i], sig[i]);
    EXPECT_EQ(std::string(b.begin() + 12, b.begin() + 16), "IHDR");
    EXPECT_EQ(b[19], w);  // width, big endian
    EXPECT_EQ(b[23], h);
    EXPECT_EQ(b[24], 8);  // bit depth
    EXPECT_EQ(b[25], 2);  // RGB
    std::remove(path.c_str());
    EXPECT_THROW(forge::write_png(path, rgb.data(), 0, h), std::runtime_error);
}
