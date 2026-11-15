#include "temper/nn/gemm.h"

#include <stdexcept>
#include <string>

#include "cutlass/gemm/device/gemm.h"
#include "forge/util/cuda_check.h"

namespace temper {

namespace {

using cutlass::layout::ColumnMajor;
using cutlass::layout::RowMajor;

// A stored K x M row-major is M x K column-major with the same ld: that is how
// Op::T is done without copying.
template <class LayoutA, class LayoutB>
using SimtGemm = cutlass::gemm::device::Gemm<float, LayoutA, float, LayoutB, float, RowMajor, float,
                                             cutlass::arch::OpClassSimt, cutlass::arch::Sm80>;

template <class LayoutA, class LayoutB>
void run(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta,
         const float* C, int ldc, float* D, int ldd, cudaStream_t stream) {
    using G = SimtGemm<LayoutA, LayoutB>;
    typename G::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {D, ldd}, {alpha, beta});
    G op;
    cutlass::Status st = op.can_implement(args);
    if (st != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("gemm: CUTLASS cannot run this problem: ") +
                                 cutlass::cutlassGetStatusString(st));
    st = op(args, nullptr, stream);
    if (st != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("gemm: CUTLASS launch failed: ") + cutlass::cutlassGetStatusString(st));
    KERNEL_CHECK();
}

}  // namespace

void gemm(Op op_a, Op op_b, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
          float beta, const float* C, int ldc, float* D, int ldd, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) throw std::runtime_error("gemm: K must be > 0");
    if (op_a == Op::N && op_b == Op::N) run<RowMajor, RowMajor>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, stream);
    else if (op_a == Op::N && op_b == Op::T) run<RowMajor, ColumnMajor>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, stream);
    else if (op_a == Op::T && op_b == Op::N) run<ColumnMajor, RowMajor>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, stream);
    else run<ColumnMajor, ColumnMajor>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, stream);
}

}  // namespace temper
