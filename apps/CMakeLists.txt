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
find_package(ZLIB REQUIRED)  # openarm_stack: frames compressed as they come
target_link_libraries(collect_app PRIVATE forge Threads::Threads ZLIB::ZLIB)
target_compile_definitions(collect_app PRIVATE CRUCIBLE_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
target_compile_options(collect_app PRIVATE ${CRUCIBLE_CXX_WARNINGS})
if(BUILD_FLUX)
  # --images: camera images with flux's offscreen BatchCamera (Phase 9c)
  target_sources(collect_app PRIVATE collect_app/camera_rig.cpp collect_app/openarm_camera_rig.cpp
                 collect_app/stack_camera_rig.cpp)
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

# vla_app: the small VLA (tools/vla_train.py --onnx) in panda_pick_color, and with --task
# openarm_pick_place tools/vla_openarm.py's in the OpenArm Cell, through ONNX Runtime
# (Phases 9e, 10). Needs flux (cameras) and ONNX Runtime: download a release
# (onnxruntime-linux-x64-<version>.tgz, or -gpu- for --device cuda), unpack it and
# configure with -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-<version>.
set(ONNXRUNTIME_DIR "" CACHE PATH "Unpacked ONNX Runtime release (include/, lib/) for vla_app")
if(BUILD_FLUX AND ONNXRUNTIME_DIR)
  find_path(ORT_INCLUDE onnxruntime_cxx_api.h PATHS "${ONNXRUNTIME_DIR}/include" NO_DEFAULT_PATH)
  find_library(ORT_LIB onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" NO_DEFAULT_PATH)
  if(ORT_INCLUDE AND ORT_LIB)
    add_executable(vla_app vla_app/main.cpp vla_app/openarm.cpp collect_app/camera_rig.cpp
      collect_app/openarm_camera_rig.cpp)
    target_include_directories(vla_app SYSTEM PRIVATE ${ORT_INCLUDE})
    target_link_libraries(vla_app PRIVATE forge flux ${ORT_LIB})
    target_compile_definitions(vla_app PRIVATE CRUCIBLE_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
    target_compile_options(vla_app PRIVATE ${CRUCIBLE_CXX_WARNINGS} -Wno-missing-field-initializers)
    set_target_properties(vla_app PROPERTIES BUILD_RPATH "${ONNXRUNTIME_DIR}/lib")
    message(STATUS "vla_app: on (ONNX Runtime ${ONNXRUNTIME_DIR})")
    # distill_app: the OpenArm VLA (teacher) distilled into the CNN + GRU student
    # (Phase 10). Also needs temper with cuDNN.
    if(TARGET temper)
      get_target_property(_temper_defs2 temper INTERFACE_COMPILE_DEFINITIONS)
      if(_temper_defs2 MATCHES "TEMPER_HAS_CUDNN")
        add_executable(distill_app distill_app/main.cpp collect_app/openarm_camera_rig.cpp)
        target_include_directories(distill_app SYSTEM PRIVATE ${ORT_INCLUDE})
        target_link_libraries(distill_app PRIVATE forge flux temper ${ORT_LIB})
        target_compile_definitions(distill_app PRIVATE CRUCIBLE_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
        target_compile_options(distill_app PRIVATE ${CRUCIBLE_CXX_WARNINGS} -Wno-missing-field-initializers)
        set_target_properties(distill_app PROPERTIES BUILD_RPATH "${ONNXRUNTIME_DIR}/lib")
        message(STATUS "distill_app: on")
      endif()
    endif()
  else()
    message(WARNING "vla_app: no ONNX Runtime in ${ONNXRUNTIME_DIR} (need include/onnxruntime_cxx_api.h, lib/libonnxruntime.so)")
  endif()
elseif(BUILD_FLUX)
  message(STATUS "vla_app: off (configure with -DONNXRUNTIME_DIR=... to build it)")
endif()

if(BUILD_FLUX)
  add_executable(sandbox_app
    sandbox_app/main.cpp
    sandbox_app/viewer.cpp
    sandbox_app/particle_scene.cpp
    sandbox_app/robot_scene.cpp
    sandbox_app/task_scene.cpp
    sandbox_app/grasp_scene.cpp
    sandbox_app/vision_policy.cpp
    collect_app/openarm_camera_rig.cpp)
  target_compile_definitions(sandbox_app PRIVATE CRUCIBLE_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
  target_link_libraries(sandbox_app PRIVATE forge flux)
  if(TARGET temper)
    target_link_libraries(sandbox_app PRIVATE temper)
    target_compile_definitions(sandbox_app PRIVATE CRUCIBLE_HAS_TEMPER=1)
    get_target_property(_temper_defs3 temper INTERFACE_COMPILE_DEFINITIONS)
    if(_temper_defs3 MATCHES "TEMPER_HAS_CUDNN")  # --student: the CNN + GRU student
      target_compile_definitions(sandbox_app PRIVATE CRUCIBLE_HAS_STUDENT=1)
      # rl_app: PPO fine-tuning of the OpenArm student (Phase 10c)
      add_executable(rl_app rl_app/main.cpp collect_app/openarm_camera_rig.cpp)
      target_link_libraries(rl_app PRIVATE forge flux temper)
      target_compile_definitions(rl_app PRIVATE CRUCIBLE_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
      target_compile_options(rl_app PRIVATE ${CRUCIBLE_CXX_WARNINGS} -Wno-missing-field-initializers)
      message(STATUS "rl_app: on")
    endif()
  endif()
  if(ORT_INCLUDE AND ORT_LIB)  # --vla: ONNX Runtime
    target_include_directories(sandbox_app SYSTEM PRIVATE ${ORT_INCLUDE})
    target_link_libraries(sandbox_app PRIVATE ${ORT_LIB})
    target_compile_definitions(sandbox_app PRIVATE CRUCIBLE_HAS_ORT=1)
    set_target_properties(sandbox_app PROPERTIES BUILD_RPATH "${ONNXRUNTIME_DIR}/lib")
  endif()
  target_compile_options(sandbox_app PRIVATE ${CRUCIBLE_CXX_WARNINGS} -Wno-missing-field-initializers)
endif()
