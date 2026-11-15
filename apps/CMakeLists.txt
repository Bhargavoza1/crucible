# train_app never links flux, so it builds and runs with BUILD_FLUX=OFF.
add_executable(train_app train_app/main.cpp)
target_link_libraries(train_app PRIVATE forge)
target_compile_options(train_app PRIVATE ${CRUCIBLE_CXX_WARNINGS})
if(BUILD_FLUX)
  # Only tells train_app that --viz is possible. Real viewer hookup comes in Phase 3.
  target_compile_definitions(train_app PRIVATE CRUCIBLE_HAS_FLUX=1)
endif()

if(BUILD_FLUX)
  add_executable(sandbox_app
    sandbox_app/main.cpp
    sandbox_app/viewer.cpp
    sandbox_app/particle_scene.cpp
    sandbox_app/robot_scene.cpp)
  target_compile_definitions(sandbox_app PRIVATE CRUCIBLE_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
  target_link_libraries(sandbox_app PRIVATE forge flux)
  target_compile_options(sandbox_app PRIVATE ${CRUCIBLE_CXX_WARNINGS} -Wno-missing-field-initializers)
endif()
