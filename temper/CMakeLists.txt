# temper: RL stack (doc 5). CUDA only; no PyTorch.
#   temper_cpu  header-only CPU reference (double) for tests: tests_cpu uses it
#   temper      nn (Tensor, CUTLASS Linear, ELU, MLP), optim (Adam, grad clip)
add_library(temper_cpu INTERFACE)
target_include_directories(temper_cpu INTERFACE ${PROJECT_SOURCE_DIR})

if(CRUCIBLE_HAS_CUDA)
  # CUTLASS: header-only. Fixed version. SOURCE_SUBDIR points at a folder that
  # does not exist, so CMake only downloads it and does not build CUTLASS's own
  # (large) CMake project. To use a local copy instead of downloading:
  #   cmake .. -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/path/to/cutlass
  include(FetchContent)
  FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v3.5.1
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR do_not_build)
  FetchContent_MakeAvailable(cutlass)

  add_library(temper STATIC
    nn/gemm.cu
    nn/linear.cu
    nn/activations.cu
    nn/mlp.cu
    optim/adam.cu)
  target_include_directories(temper PUBLIC ${PROJECT_SOURCE_DIR})
  target_include_directories(temper SYSTEM PRIVATE ${cutlass_SOURCE_DIR}/include ${cutlass_SOURCE_DIR}/tools/util/include)
  target_link_libraries(temper PUBLIC forge CUDA::cudart)
  target_compile_options(temper PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CRUCIBLE_CUDA_FLAGS}>)
endif()
