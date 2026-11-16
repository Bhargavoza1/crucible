// GEMM speed, cuBLAS vs CUTLASS (temper/nn/gemm.h), on the layer shapes of our
// SAC/TQC networks (hidden 256, ant: obs 36 + act 8 = 44 critic inputs).
//   ./benchmarks/bench_gemm            batches 1024 and 4096
//   ./benchmarks/bench_gemm 512 8192   other batches
// Each shape: 20 warmup calls, then 500 timed calls (CUDA events), one stream.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "forge/util/cuda_check.h"
#include "temper/nn/gemm.h"
#include "temper/nn/tensor.h"

using namespace temper;

namespace {

struct Shape {
    const char* name;
    Op oa, ob;
    int M, N, K;
    bool bias;
};

double time_us(const Shape& s, cudaStream_t st, Tensor& A, Tensor& B, Tensor& C, Tensor& D) {
    const int lda = s.oa == Op::N ? s.K : s.M, ldb = s.ob == Op::N ? s.N : s.K;
    auto call = [&] {
        gemm(s.oa, s.ob, s.M, s.N, s.K, 1.0f, A.data(), lda, B.data(), ldb, s.bias ? 1.0f : 0.0f, C.data(),
             s.bias ? 0 : s.N, s.bias ? D.data() : C.data(), s.N, st);
    };
    for (int i = 0; i < 20; ++i) call();
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    const int n = 500;
    CUDA_CHECK(cudaEventRecord(e0, st));
    for (int i = 0; i < n; ++i) call();
    CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 1000.0 * ms / n;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> batches;
    for (int i = 1; i < argc; ++i) batches.push_back(std::atoi(argv[i]));
    if (batches.empty()) batches = {1024, 4096};
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("bench_gemm on %s, FP32 (no TF32)\n", prop.name);
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreate(&st));
    double total[2] = {0, 0};
    for (int Bt : batches) {
        const Shape shapes[] = {
            {"fwd  critic in   (B x 44 -> 256)", Op::N, Op::T, Bt, 256, 44, true},
            {"fwd  hidden      (B x 256 -> 256)", Op::N, Op::T, Bt, 256, 256, true},
            {"fwd  tqc out     (B x 256 -> 25)", Op::N, Op::T, Bt, 25, 256, true},
            {"bwd  dW hidden   (256 x 256, K=B)", Op::T, Op::N, 256, 256, Bt, false},
            {"bwd  dx hidden   (B x 256 <- 256)", Op::N, Op::N, Bt, 256, 256, false},
        };
        std::printf("\nbatch %d          %28s %14s %14s %8s\n", Bt, "", "cublas us", "cutlass us", "ratio");
        for (const Shape& s : shapes) {
            const size_t na = size_t(s.M) * s.K, nbm = size_t(s.K) * s.N, nc = size_t(s.M) * s.N;
            Tensor A(1, int(na)), B(1, int(nbm)), C(1, int(std::max(nc, size_t(s.N)))), D(1, int(nc));
            A.zero(st);
            B.zero(st);
            C.zero(st);
            double us[2];
            for (int b = 0; b < 2; ++b) {
                set_gemm_backend(b == 0 ? GemmBackend::kCublas : GemmBackend::kCutlass);
                us[b] = time_us(s, st, A, B, C, D);
                total[b] += us[b];
            }
            std::printf("  %-44s %14.2f %14.2f %7.2fx\n", s.name, us[0], us[1], us[1] / us[0]);
        }
    }
    std::printf("\nsum of all shapes: cublas %.1f us, cutlass %.1f us (cutlass / cublas = %.2fx)\n", total[0], total[1],
                total[1] / total[0]);
    set_gemm_backend(GemmBackend::kCublas);
    cudaStreamDestroy(st);
    return 0;
}
