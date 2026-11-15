# forge: physics. CUDA only, never Vulkan (doc 2, rule 1).
#
#   forge_math  header-only math, host + device
#   forge_cpu   plain C++: config, CPU reference code (the answer keys). No CUDA.
#   forge       forge_cpu + CUDA kernels. What apps link.
#
# Rule: a .cpp file goes in forge_cpu, a .cu file goes in forge.
# tests_cpu links only forge_cpu, so it builds on a machine without CUDA.

add_library(forge_math INTERFACE)
target_include_directories(forge_math INTERFACE ${PROJECT_SOURCE_DIR})

# ---- forge_cpu ------------------------------------------------------------------
add_library(forge_cpu STATIC
  util/config.cpp
  particles/particles.cpp
  particles/integrate_cpu.cpp
  particles/spawn.cpp
  particles/step_cpu.cpp)
target_link_libraries(forge_cpu PUBLIC forge_math)
target_compile_options(forge_cpu PRIVATE ${CRUCIBLE_CXX_WARNINGS})

# ---- forge ----------------------------------------------------------------------
if(CRUCIBLE_HAS_CUDA)
  add_library(forge STATIC
    util/device_info.cu
    particles/integrate.cu)
  target_link_libraries(forge PUBLIC forge_cpu CUDA::cudart)
  target_compile_definitions(forge PUBLIC CRUCIBLE_HAS_CUDA=1)
  target_compile_options(forge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CRUCIBLE_CUDA_FLAGS}>)
else()
  add_library(forge STATIC
    util/device_info_nocuda.cpp)
  target_link_libraries(forge PUBLIC forge_cpu)
  target_compile_options(forge PRIVATE ${CRUCIBLE_CXX_WARNINGS})
endif()
