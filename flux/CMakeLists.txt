# flux: Vulkan renderer. Only built with -DBUILD_FLUX=ON.
find_package(Vulkan 1.2 REQUIRED COMPONENTS glslc)
find_package(glfw3 3.3 REQUIRED)  # doc says 3.4; 3.3 has everything we use

# ---- Shaders: GLSL -> SPIR-V at build time --------------------------------------
set(FLUX_SHADER_OUT ${CMAKE_BINARY_DIR}/shaders)
set(FLUX_SHADERS shaders/particle.vert shaders/particle.frag shaders/quad.vert shaders/quad.frag shaders/mesh.vert
    shaders/mesh.frag)
set(FLUX_SPV)
foreach(src ${FLUX_SHADERS})
  get_filename_component(name ${src} NAME)
  set(out ${FLUX_SHADER_OUT}/${name}.spv)
  add_custom_command(
    OUTPUT ${out}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${FLUX_SHADER_OUT}
    COMMAND Vulkan::glslc --target-env=vulkan1.2 -Werror -o ${out} ${CMAKE_CURRENT_SOURCE_DIR}/${src}
    DEPENDS ${src}
    COMMENT "glslc ${name}")
  list(APPEND FLUX_SPV ${out})
endforeach()
add_custom_target(flux_shaders DEPENDS ${FLUX_SPV})

# ---- Library -------------------------------------------------------------------
add_library(flux STATIC
  context.cpp
  memory.cpp
  batch_camera.cpp
  renderer.cpp
  window/window.cpp
  window/swapchain.cpp
  pipeline/particle_pipeline.cpp
  pipeline/quad_pipeline.cpp
  pipeline/mesh_pipeline.cpp
  robot_mesh.cpp)
add_dependencies(flux flux_shaders)
target_include_directories(flux PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(flux PUBLIC Vulkan::Vulkan glfw forge_math)
# "VkXxx ci{sType}" zero-fills the other fields on purpose, so this warning is noise.
target_compile_options(flux PRIVATE ${CRUCIBLE_CXX_WARNINGS} -Wno-missing-field-initializers)
# Default shader folder. Apps can override it with --shader_dir.
target_compile_definitions(flux PUBLIC FLUX_SHADER_DIR="${FLUX_SHADER_OUT}")

# ---- CUDA-Vulkan sharing (Stage B, doc 4.2) --------------------------------------
# Only this folder touches both APIs. It needs CUDA and Linux (opaque fd handles).
if(CRUCIBLE_HAS_CUDA AND UNIX AND NOT APPLE)
  target_sources(flux PRIVATE interop/shared_particles.cpp)
  target_link_libraries(flux PUBLIC CUDA::cudart)
  target_compile_definitions(flux PUBLIC FLUX_HAS_INTEROP=1)
  message(STATUS "flux: CUDA-Vulkan interop ON")
else()
  message(STATUS "flux: CUDA-Vulkan interop OFF (needs CUDA on Linux)")
endif()
