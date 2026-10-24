// main() for tests_gpu. With no CUDA device it returns 77, which CTest reports
// as "skipped" (SKIP_RETURN_CODE 77). Listing tests (--gtest_list_tests) still
// works without a device, so test discovery does not fail.
#include <cstdio>
#include <cstring>

#include <gtest/gtest.h>

#include "forge/util/device_info.h"

int main(int argc, char** argv) {
    bool listing = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--gtest_list_tests") == 0) listing = true;

    ::testing::InitGoogleTest(&argc, argv);

    if (!listing && forge::cuda_device_count() == 0) {
        std::printf("No CUDA device found. GPU tests skipped (exit code 77).\n");
        return 77;
    }
    if (!listing) std::printf("GPU: %s\n", forge::cuda_device_description(0).c_str());
    return RUN_ALL_TESTS();
}
