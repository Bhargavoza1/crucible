# Tests (doc 11.1).
#   tests_cpu  plain C++ (no CUDA): math, parsers. Runs anywhere.
#   tests_gpu  kernel tests, label "gpu". Returns 77 (skipped) with no CUDA device.
#
#   ctest            all tests (GPU tests are skipped without a GPU)
#   ctest -LE gpu    CPU tests only
#   ctest -L gpu     GPU tests only

include(FetchContent)
# Uses an installed GoogleTest (e.g. libgtest-dev) if found, otherwise clones v1.15.2.
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS NAMES GTest)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# ---- CPU tests ----------------------------------------------------------------
add_executable(tests_cpu
  cpu/test_vec3.cpp
  cpu/test_mat3.cpp
  cpu/test_quat.cpp
  cpu/test_config.cpp)
# Link forge_math (headers) + forge (config). Math headers compile with plain g++,
# which proves FORGE_HD is empty without nvcc.
target_link_libraries(tests_cpu PRIVATE forge GTest::gtest_main)
target_compile_options(tests_cpu PRIVATE ${CRUCIBLE_CXX_WARNINGS})
target_compile_definitions(tests_cpu PRIVATE CRUCIBLE_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}/tmp")
gtest_discover_tests(tests_cpu DISCOVERY_MODE PRE_TEST)

# ---- GPU tests ----------------------------------------------------------------
if(CRUCIBLE_HAS_CUDA)
  add_executable(tests_gpu
    gpu/gpu_main.cpp
    gpu/test_math_gpu.cu)
  target_link_libraries(tests_gpu PRIVATE forge GTest::gtest)
  target_compile_options(tests_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CRUCIBLE_CUDA_FLAGS}>)
  # PRE_TEST: list tests when ctest runs, not at build time. --gtest_list_tests
  # works without a device because gpu_main.cpp only checks for a device when
  # tests are really run.
  gtest_discover_tests(tests_gpu
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
endif()
