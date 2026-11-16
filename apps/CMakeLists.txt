# train_app builds and runs with BUILD_FLUX=OFF (headless only).
add_executable(train_app train_app/main.cpp)
target_link_libraries(train_app PRIVATE forge)
if(TARGET temper)
  target_link_libraries(train_app PRIVATE temper)
  target_compile_definitions(train_app PRIVATE CRUCIBLE_HAS_TEMPER=1)
endif()
target_compile_definitions(train_app PRIVATE CRUCIBLE_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
target_compile_options(train_app PRIVATE ${CRUCIBLE_CXX_WARNINGS})
if(BUILD_FLUX)
  # --viz: live view of training (train_viewer.cpp). Without flux, train_app builds
  # and runs headless only.
  target_sources(train_app PRIVATE train_app/train_viewer.cpp)
  target_link_libraries(train_app PRIVATE flux)
  target_compile_definitions(train_app PRIVATE CRUCIBLE_HAS_FLUX=1)
endif()

# collect_app: records a task's scripted expert into a demonstration file (Phase 9b).
add_executable(collect_app collect_app/main.cpp)
find_package(Threads REQUIRED)
target_link_libraries(collect_app PRIVATE forge Threads::Threads)
target_compile_definitions(collect_app PRIVATE CRUCIBLE_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
target_compile_options(collect_app PRIVATE ${CRUCIBLE_CXX_WARNINGS})
if(BUILD_FLUX)
  # --images: camera images with flux's offscreen BatchCamera (Phase 9c)
  target_sources(collect_app PRIVATE collect_app/camera_rig.cpp)
  target_link_libraries(collect_app PRIVATE flux)
  target_compile_definitions(collect_app PRIVATE CRUCIBLE_HAS_FLUX=1)
  target_compile_options(collect_app PRIVATE -Wno-missing-field-initializers)
endif()

# bc_app: behavior cloning for panda_pick (Phase 9d). Needs temper with cuDNN
# (Conv2d); the closed-loop test with camera images also needs flux.
if(TARGET temper)
  get_target_property(_temper_defs temper INTERFACE_COMPILE_DEFINITIONS)
  if(_temper_defs MATCHES "TEMPER_HAS_CUDNN")
    add_executable(bc_app bc_app/main.cpp)
    target_link_libraries(bc_app PRIVATE forge temper)
    target_compile_definitions(bc_app PRIVATE CRUCIBLE_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
    target_compile_options(bc_app PRIVATE ${CRUCIBLE_CXX_WARNINGS})
    if(BUILD_FLUX)
      target_sources(bc_app PRIVATE collect_app/camera_rig.cpp)
      target_link_libraries(bc_app PRIVATE flux)
      target_compile_definitions(bc_app PRIVATE CRUCIBLE_HAS_FLUX=1)
      target_compile_options(bc_app PRIVATE -Wno-missing-field-initializers)
    endif()
  else()
    message(STATUS "bc_app: off (needs cuDNN for Conv2d)")
  endif()
endif()

if(BUILD_FLUX)
  add_executable(sandbox_app
    sandbox_app/main.cpp
    sandbox_app/viewer.cpp
    sandbox_app/particle_scene.cpp
    sandbox_app/robot_scene.cpp
    sandbox_app/task_scene.cpp
    sandbox_app/grasp_scene.cpp)
  target_compile_definitions(sandbox_app PRIVATE CRUCIBLE_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
  target_link_libraries(sandbox_app PRIVATE forge flux)
  if(TARGET temper)
    target_link_libraries(sandbox_app PRIVATE temper)
    target_compile_definitions(sandbox_app PRIVATE CRUCIBLE_HAS_TEMPER=1)
  endif()
  target_compile_options(sandbox_app PRIVATE ${CRUCIBLE_CXX_WARNINGS} -Wno-missing-field-initializers)
endif()
