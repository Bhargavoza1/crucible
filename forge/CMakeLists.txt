# forge: physics. CUDA only, never Vulkan (doc 2, rule 1).

# Header-only math, shared by CPU and GPU code.
add_library(forge_math INTERFACE)
target_include_directories(forge_math INTERFACE ${PROJECT_SOURCE_DIR})

# forge library. Host code (config, device info) + CUDA kernels when CUDA is on.
set(FORGE_SOURCES util/config.cpp)
if(CRUCIBLE_HAS_CUDA)
  list(APPEND FORGE_SOURCES util/device_info.cu)
else()
  list(APPEND FORGE_SOURCES util/device_info_nocuda.cpp)
endif()

add_library(forge STATIC ${FORGE_SOURCES})
target_link_libraries(forge PUBLIC forge_math)
target_compile_options(forge PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${CRUCIBLE_CXX_WARNINGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:${CRUCIBLE_CUDA_FLAGS}>)

if(CRUCIBLE_HAS_CUDA)
  target_link_libraries(forge PUBLIC CUDA::cudart)
  target_compile_definitions(forge PUBLIC CRUCIBLE_HAS_CUDA=1)
  set_target_properties(forge PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()
