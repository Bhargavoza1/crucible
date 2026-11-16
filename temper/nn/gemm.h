#pragma once
// GEMM for every Linear layer, forward and backward.
//
//   D = alpha * op(A) * op(B) + beta * C        all row-major, D is M x N
//   op(A) is M x K:  Op::N -> A is stored M x K (lda)
//                    Op::T -> A is stored K x M (lda), used transposed
//   op(B) is K x N:  same rule (stored K x N, or N x K for Op::T)
//   C may be D (in place) or a bias row: ldc = 0 repeats row 0 for every row.
//
// Two backends, both exact FP32 (no TF32):
//   cuBLAS  (default): NVIDIA's tuned GEMMs
//   CUTLASS (Phase 7b): one generic SIMT kernel; kept to compare speed
//                       (benchmarks/bench_gemm) and as a second answer in tests
// cuBLAS: one handle per host thread, set to the caller's stream on each call;
// same inputs, same GPU and same stream give the same numbers run to run.
#include <cuda_runtime.h>

namespace temper {

enum class Op { N, T };
enum class GemmBackend { kCublas, kCutlass };

void set_gemm_backend(GemmBackend b);  // for the whole process
GemmBackend gemm_backend();
const char* gemm_backend_name(GemmBackend b);

void gemm(Op op_a, Op op_b, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
          float beta, const float* C, int ldc, float* D, int ldd, cudaStream_t stream);

}  // namespace temper
