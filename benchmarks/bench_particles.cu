// Headless particle benchmark: no window, no real-time clock.
// Drops particles into a bin, lets the pile settle (not timed), then times
// physics substeps with CUDA events, for the original and the sorted layout.
//
//   ./benchmarks/bench_particles                         100k particles, 2.5 m bin
//   ./benchmarks/bench_particles --particles 10000 --bin_half 1.0
//   ./benchmarks/bench_particles --layout sorted         only one layout (for ncu)
//
// Profile one force kernel of the settled pile with Nsight Compute:
//   sudo ncu --kernel-name regex:accel --launch-skip 4000 --launch-count 1 --set full \
//        -o accel ./benchmarks/bench_particles --layout sorted
#include <cmath>
#include <cstdio>
#include <string>

#include "forge/particles/particles_device.h"
#include "forge/particles/spawn.h"
#include "forge/particles/stepper_gpu.h"
#include "forge/util/config.h"
#include "forge/util/device_info.h"

using namespace forge;

namespace {

struct Result {
    double ms_per_substep;
    ParticlesHost final_state;
};

Result run(const ParticlesHost& start, const ParticleParams& p, bool sorted, int settle_steps, int timed_steps) {
    ParticlesDevice dev;
    dev.upload(start);
    ParticleStepperGpu stepper;
    stepper.set_sorted_layout(sorted);
    stepper.step(dev.view(), p, settle_steps);  // settle the pile first: the dense case is what we measure

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    stepper.step(dev.view(), p, timed_steps);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    Result r{ms / timed_steps, start};
    dev.download(r.final_state);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = Config::from_cli(argc, argv);
        const int particles = cfg.get_int("particles", 100000);
        const float half = cfg.get_float("bin_half", 2.5f);
        const float settle_s = cfg.get_float("settle", 3.0f);  // sim seconds before timing
        const int timed = cfg.get_int("timed_substeps", 2000);
        const std::string layout = cfg.get_string("layout", "both");  // both | sorted | original
        for (const auto& k : cfg.unused_keys()) std::fprintf(stderr, "warning: unknown key '%s'\n", k.c_str());
        if (forge::cuda_device_count() == 0) {
            std::fprintf(stderr, "no CUDA device\n");
            return 1;
        }

        ParticleParams p;
        p.dt = 0.0005f;  // same substep as the sandbox
        add_bin_planes(p, half);
        SpawnLattice lat;
        lat.nx = lat.ny = std::max(1, int((2.0f * half - 2.0f * lat.radius) / lat.spacing));
        const float w = lat.nx * lat.spacing;
        lat.origin = {-0.5f * w, -0.5f * w, 0.3f};
        ParticlesHost start;
        spawn_lattice(start, 1, particles, lat, 1);
        const int settle = static_cast<int>(std::lround(settle_s / p.dt));  // round: 3.0 / 0.0005 is 5999.99 in float

        std::printf("GPU: %s\n", forge::cuda_device_description(0).c_str());
        std::printf("%d particles, bin %.1f m, settle %.1f s (%d substeps), then time %d substeps of %.4f s\n",
                    particles, 2 * half, settle_s, settle, timed, p.dt);

        Result orig{}, sort{};
        if (layout != "sorted") {
            orig = run(start, p, false, settle, timed);
            std::printf("original layout: %.4f ms/substep  (%.2f ms per 5 ms physics step)\n", orig.ms_per_substep,
                        10 * orig.ms_per_substep);
        }
        if (layout != "original") {
            sort = run(start, p, true, settle, timed);
            std::printf("sorted layout:   %.4f ms/substep  (%.2f ms per 5 ms physics step)\n", sort.ms_per_substep,
                        10 * sort.ms_per_substep);
        }
        if (layout == "both") {
            std::printf("speedup: %.2fx\n", orig.ms_per_substep / sort.ms_per_substep);
            const bool same = orig.final_state.px == sort.final_state.px && orig.final_state.vz == sort.final_state.vz;
            std::printf("results bit for bit equal: %s\n", same ? "yes" : "NO");
            std::printf("real time needs <= %.3f ms/substep (200 physics steps/s x 10 substeps)\n", 0.5);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
