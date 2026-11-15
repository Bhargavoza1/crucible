cmake_minimum_required(VERSION 3.24)
project(crucible LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Options
#   BUILD_FLUX   ON  = build the Vulkan renderer and sandbox_app (window mode)
#                OFF = headless only, no Vulkan needed (doc 2.2)
#   FORGE_CUDA   AUTO = use CUDA if nvcc is found, ON = require it, OFF = CPU only
# ---------------------------------------------------------------------------
option(BUILD_FLUX "Build the Vulkan renderer (flux) and sandbox_app" OFF)
option(BUILD_TESTS "Build tests_cpu and tests_gpu" ON)
set(FORGE_CUDA "AUTO" CACHE STRING "Use CUDA: AUTO, ON or OFF")
set_property(CACHE FORGE_CUDA PROPERTY STRINGS AUTO ON OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# ---- CUDA -----------------------------------------------------------------
set(CRUCIBLE_HAS_CUDA OFF)
if(NOT FORGE_CUDA STREQUAL "OFF")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4070 Ti (Ada, sm_89)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(CRUCIBLE_HAS_CUDA ON)
  elseif(FORGE_CUDA STREQUAL "ON")
    message(FATAL_ERROR "FORGE_CUDA=ON but no CUDA compiler was found")
  endif()
endif()
message(STATUS "crucible: CUDA ${CRUCIBLE_HAS_CUDA}, BUILD_FLUX ${BUILD_FLUX}")

# Warnings for our own C++ code. -lineinfo for CUDA so Nsight can map kernels to source.
set(CRUCIBLE_CXX_WARNINGS -Wall -Wextra -Wshadow)
set(CRUCIBLE_CUDA_FLAGS -lineinfo --expt-relaxed-constexpr --extended-lambda)

# ---- Modules ----------------------------------------------------------------
add_subdirectory(forge)
if(BUILD_FLUX)
  add_subdirectory(flux)
endif()
add_subdirectory(apps)
add_subdirectory(benchmarks)

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
