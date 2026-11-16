# temper: RL stack (doc 5). CUDA only; no PyTorch.
#   temper_cpu  header-only CPU reference (double) for tests: tests_cpu uses it
#   temper      nn (Tensor, Linear on cuBLAS or CUTLASS, ELU, ReLU, Tanh, MLP, Sequential, LayerNorm,
#               Conv2d with cuDNN), optim, algos (PPO, SAC, TQC, behavior cloning)
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
    nn/layernorm.cu
    optim/adam.cu
    common/obs_normalizer.cu
    algos/ppo/ppo.cu
    algos/sac/sac.cu
    algos/algorithm.cpp
    algos/bc/bc_ops.cu
    buffers/replay/replay_buffer.cu
    common/train_loop.cpp)
  target_include_directories(temper PUBLIC ${PROJECT_SOURCE_DIR})
  target_include_directories(temper SYSTEM PRIVATE ${cutlass_SOURCE_DIR}/include ${cutlass_SOURCE_DIR}/tools/util/include)
  target_link_libraries(temper PUBLIC forge CUDA::cudart CUDA::cublas)
  target_compile_options(temper PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CRUCIBLE_CUDA_FLAGS}>)

  # cuDNN (Phase 9d: Conv2d). Optional: without it everything else builds, only
  # Conv2d (and what uses it) is left out. Ubuntu: apt install libcudnn9-dev-cuda-12
  # (match your CUDA major version), or point -DCUDNN_ROOT=/path at an install.
  find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDAToolkit_INCLUDE_DIRS}
            PATH_SUFFIXES include)
  find_library(CUDNN_LIBRARY cudnn HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDAToolkit_LIBRARY_DIR}
               PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
  if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    target_sources(temper PRIVATE nn/conv2d.cu algos/bc/bc_policy.cpp)
    target_include_directories(temper SYSTEM PUBLIC ${CUDNN_INCLUDE_DIR})
    target_link_libraries(temper PUBLIC ${CUDNN_LIBRARY})
    target_compile_definitions(temper PUBLIC TEMPER_HAS_CUDNN=1)
    message(STATUS "temper: cuDNN ${CUDNN_LIBRARY}: Conv2d ON")
  else()
    message(STATUS "temper: cuDNN not found: Conv2d OFF (apt install libcudnn9-dev-cuda-12, or -DCUDNN_ROOT=...)")
  endif()
endif()
