# Tests (doc 11.1).
#   tests_cpu  plain C++ (no CUDA). Runs anywhere.
#   tests_gpu  kernel tests, label "gpu". Returns 77 (skipped) with no CUDA device.
#
#   ctest            all tests (GPU tests are skipped without a GPU)
#   ctest -LE gpu    CPU tests only
#   ctest -L gpu     GPU tests only
#
# Layout: tests/cpu/ and tests/gpu/ mirror the source folders.
#   forge/math/quat.h  ->  tests/cpu/forge/math/test_quat.cpp
#   GPU test files end in _gpu.cu. Shared helpers live in tests/common/.
# New test files are picked up by the globs below; re-run cmake after adding one.

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
file(GLOB_RECURSE TESTS_CPU_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp)
add_executable(tests_cpu ${TESTS_CPU_SOURCES})
target_include_directories(tests_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})  # for common/
# forge_cpu has no CUDA in it, so tests_cpu builds and runs on any machine.
target_link_libraries(tests_cpu PRIVATE forge_cpu temper_cpu GTest::gtest_main)
target_compile_options(tests_cpu PRIVATE ${CRUCIBLE_CXX_WARNINGS})
target_compile_definitions(tests_cpu PRIVATE CRUCIBLE_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}/tmp"
                                             CRUCIBLE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
gtest_discover_tests(tests_cpu DISCOVERY_MODE PRE_TEST)

# ---- GPU tests ----------------------------------------------------------------
if(CRUCIBLE_HAS_CUDA)
  # flux tests need Vulkan, so they go in their own program (tests_interop below).
  file(GLOB_RECURSE TESTS_GPU_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gpu/*_gpu.cu)
  list(FILTER TESTS_GPU_SOURCES EXCLUDE REGEX "/gpu/flux/")
  add_executable(tests_gpu gpu/gpu_main.cpp ${TESTS_GPU_SOURCES})
  target_include_directories(tests_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(tests_gpu PRIVATE forge temper CUDA::cublas GTest::gtest)
  target_compile_definitions(tests_gpu PRIVATE CRUCIBLE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  target_compile_options(tests_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CRUCIBLE_CUDA_FLAGS}>)
  # PRE_TEST: list tests when ctest runs, not at build time. --gtest_list_tests
  # works without a device because gpu_main.cpp only checks for a device when
  # tests are really run.
  gtest_discover_tests(tests_gpu
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
endif()

# ---- CUDA-Vulkan interop tests ---------------------------------------------------
# Need CUDA + Vulkan (BUILD_FLUX=ON) on Linux. Label gpu, skip code 77 without a GPU.
if(CRUCIBLE_HAS_CUDA AND BUILD_FLUX AND TARGET flux)
  get_target_property(_flux_defs flux INTERFACE_COMPILE_DEFINITIONS)
  if(_flux_defs MATCHES "FLUX_HAS_INTEROP")
    file(GLOB_RECURSE TESTS_INTEROP_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gpu/flux/*_gpu.cu)
    add_executable(tests_interop gpu/gpu_main.cpp ${TESTS_INTEROP_SOURCES})
    target_include_directories(tests_interop PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(tests_interop PRIVATE flux forge GTest::gtest)
    target_compile_options(tests_interop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CRUCIBLE_CUDA_FLAGS}>)
    gtest_discover_tests(tests_interop
      DISCOVERY_MODE PRE_TEST
      PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  endif()
endif()
