# Headless benchmarks (doc Phase 10). GPU only.
if(CRUCIBLE_HAS_CUDA)
  add_executable(bench_particles bench_particles.cu)
  target_link_libraries(bench_particles PRIVATE forge)
  target_compile_options(bench_particles PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CRUCIBLE_CUDA_FLAGS}>)
endif()
