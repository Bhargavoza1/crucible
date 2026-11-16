#include "temper/nn/gemm.h"

#include <stdexcept>
#include <string>

#include <atomic>

#include <cublas_v2.h>

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

std::atomic<GemmBackend> g_backend{GemmBackend::kCublas};

void cublas_check(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("gemm: cuBLAS ") + what + " failed (status " + std::to_string(int(st)) + ")");
}

struct CublasHandle {
    cublasHandle_t h = nullptr;
    CublasHandle() {
        cublas_check(cublasCreate(&h), "cublasCreate");
        cublas_check(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");  // FP32, no TF32
    }
    ~CublasHandle() {
        if (h) cublasDestroy(h);
    }
};

cublasHandle_t handle() {
    thread_local CublasHandle hd;
    return hd.h;
}

// D (M x N, ldd) = C, where ldc = 0 repeats C's first row.
__global__ void copy_c_kernel(int M, int N, const float* C, int ldc, float* D, int ldd) {
    const size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= size_t(M) * N) return;
    const size_t i = t / N, j = t % N;
    D[i * ldd + j] = C[i * ldc + j];
}

void run_cublas(Op op_a, Op op_b, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
                float beta, const float* C, int ldc, float* D, int ldd, cudaStream_t stream) {
    // cuBLAS computes D = alpha op(A) op(B) + beta D in place, so first D = C.
    if (beta != 0.0f && C != D) {
        const size_t n = size_t(M) * N;
        copy_c_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(M, N, C, ldc, D, ldd);
        KERNEL_CHECK();
    }
    // Row-major D (M x N) is column-major D^T (N x M):  D^T = op(B)^T op(A)^T.
    // A row-major matrix read as column-major is its transpose, so Op::N stays
    // CUBLAS_OP_N and Op::T stays CUBLAS_OP_T, with A and B swapped.
    cublasHandle_t h = handle();
    cublas_check(cublasSetStream(h, stream), "cublasSetStream");
    const cublasOperation_t tb = op_b == Op::T ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t ta = op_a == Op::T ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublas_check(cublasSgemm(h, tb, ta, N, M, K, &alpha, B, ldb, A, lda, &beta, D, ldd), "cublasSgemm");
}

}  // namespace

void set_gemm_backend(GemmBackend b) { g_backend = b; }
GemmBackend gemm_backend() { return g_backend; }
const char* gemm_backend_name(GemmBackend b) { return b == GemmBackend::kCublas ? "cublas" : "cutlass"; }

void gemm(Op op_a, Op op_b, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
          float beta, const float* C, int ldc, float* D, int ldd, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) throw std::runtime_error("gemm: K must be > 0");
    if (g_backend == GemmBackend::kCublas) {
        run_cublas(op_a, op_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, stream);
        return;
    }
    if (op_a == Op::N && op_b == Op::N) run<RowMajor, RowMajor>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, stream);
    else if (op_a == Op::N && op_b == Op::T) run<RowMajor, ColumnMajor>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, stream);
    else if (op_a == Op::T && op_b == Op::N) run<ColumnMajor, RowMajor>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, stream);
    else run<ColumnMajor, ColumnMajor>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, stream);
}

}  // namespace temper
