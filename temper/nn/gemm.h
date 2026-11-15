#pragma once
// GEMM with CUTLASS (doc 5): every Linear layer, forward and backward.
//
//   D = alpha * op(A) * op(B) + beta * C        all row-major, D is M x N
//   op(A) is M x K:  Op::N -> A is stored M x K (lda)
//                    Op::T -> A is stored K x M (lda), used transposed
//   op(B) is K x N:  same rule (stored K x N, or N x K for Op::T)
//   C may be D (in place) or a bias row: ldc = 0 repeats row 0 for every row.
//
// FP32 in, FP32 accumulate, CUDA cores (SIMT). Exact FP32, any size and
// alignment. Tensor cores (TF32) come later, only if a profile says the GEMMs
// are the slow part.
#include <cuda_runtime.h>

namespace temper {

enum class Op { N, T };

void gemm(Op op_a, Op op_b, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
          float beta, const float* C, int ldc, float* D, int ldd, cudaStream_t stream);

}  // namespace temper
