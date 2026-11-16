#pragma once
// CPU answer key for the nn code (double precision, plain loops). Same layout as
// the GPU layers: x is batch x in, W is out x in, row major. Tests only.
#include <cmath>
#include <vector>

namespace temper {
namespace ref {

// y = x W^T + b
inline void linear_forward(const std::vector<double>& x, const std::vector<double>& W, const std::vector<double>& b,
                           int batch, int in, int out, std::vector<double>& y) {
    y.assign(size_t(batch) * out, 0.0);
    for (int i = 0; i < batch; ++i)
        for (int o = 0; o < out; ++o) {
            double s = b[o];
            for (int k = 0; k < in; ++k) s += x[size_t(i) * in + k] * W[size_t(o) * in + k];
            y[size_t(i) * out + o] = s;
        }
}

// dW = dy^T x, db = column sums of dy, dx = dy W
inline void linear_backward(const std::vector<double>& x, const std::vector<double>& W, const std::vector<double>& dy,
                            int batch, int in, int out, std::vector<double>& dW, std::vector<double>& db,
                            std::vector<double>& dx) {
    dW.assign(size_t(out) * in, 0.0);
    db.assign(out, 0.0);
    dx.assign(size_t(batch) * in, 0.0);
    for (int i = 0; i < batch; ++i)
        for (int o = 0; o < out; ++o) {
            const double g = dy[size_t(i) * out + o];
            db[o] += g;
            for (int k = 0; k < in; ++k) {
                dW[size_t(o) * in + k] += g * x[size_t(i) * in + k];
                dx[size_t(i) * in + k] += g * W[size_t(o) * in + k];
            }
        }
}

inline double elu(double v, double a = 1.0) { return v > 0 ? v : a * std::expm1(v); }
inline double elu_grad(double v, double a = 1.0) { return v > 0 ? 1.0 : a * std::exp(v); }

// The same MLP as temper::Mlp: Linear, ELU, ..., Linear.
struct Mlp {
    std::vector<int> dims;
    std::vector<std::vector<double>> W, b, dW, db;
    std::vector<std::vector<double>> pre, post;  // per linear layer: output before / after ELU
    std::vector<double> input, dx;
    int batch = 0;

    const std::vector<double>& forward(const std::vector<double>& x, int n) {
        batch = n;
        input = x;
        const size_t L = W.size();
        pre.resize(L);
        post.resize(L);
        const std::vector<double>* in = &input;
        for (size_t l = 0; l < L; ++l) {
            linear_forward(*in, W[l], b[l], n, dims[l], dims[l + 1], pre[l]);
            post[l] = pre[l];
            if (l + 1 < L)
                for (auto& v : post[l]) v = elu(v);
            in = &post[l];
        }
        return post.back();
    }

    void backward(const std::vector<double>& dy) {
        const size_t L = W.size();
        dW.resize(L);
        db.resize(L);
        std::vector<double> g = dy, gx;
        for (size_t l = L; l-- > 0;) {
            if (l + 1 < L)
                for (size_t k = 0; k < g.size(); ++k) g[k] *= elu_grad(pre[l][k]);
            const std::vector<double>& in = l == 0 ? input : post[l - 1];
            linear_backward(in, W[l], g, batch, dims[l], dims[l + 1], dW[l], db[l], gx);
            g.swap(gx);
        }
        dx = g;
    }
};

// ---- Phase 9d: Conv2d and LayerNorm ----
// Conv2d, PyTorch layout: x [batch][C][H][W], W [K][C][k][k], b [K], y [batch][K][Ho][Wo],
// Ho = (H + 2 pad - k) / stride + 1 (zero padding). Cross-correlation, like PyTorch.
struct ConvShape {
    int C, H, W, K, k, stride, pad;
    int Ho() const { return (H + 2 * pad - k) / stride + 1; }
    int Wo() const { return (W + 2 * pad - k) / stride + 1; }
};

inline void conv2d_forward(const std::vector<double>& x, const std::vector<double>& W, const std::vector<double>& b,
                           int batch, const ConvShape& s, std::vector<double>& y) {
    const int Ho = s.Ho(), Wo = s.Wo();
    y.assign(size_t(batch) * s.K * Ho * Wo, 0.0);
    for (int n = 0; n < batch; ++n)
        for (int o = 0; o < s.K; ++o)
            for (int i = 0; i < Ho; ++i)
                for (int j = 0; j < Wo; ++j) {
                    double acc = b[o];
                    for (int c = 0; c < s.C; ++c)
                        for (int u = 0; u < s.k; ++u)
                            for (int v = 0; v < s.k; ++v) {
                                const int r = i * s.stride + u - s.pad, q = j * s.stride + v - s.pad;
                                if (r < 0 || r >= s.H || q < 0 || q >= s.W) continue;
                                acc += x[((size_t(n) * s.C + c) * s.H + r) * s.W + q] *
                                       W[((size_t(o) * s.C + c) * s.k + u) * s.k + v];
                            }
                    y[((size_t(n) * s.K + o) * Ho + i) * Wo + j] = acc;
                }
}

inline void conv2d_backward(const std::vector<double>& x, const std::vector<double>& W, const std::vector<double>& dy,
                            int batch, const ConvShape& s, std::vector<double>& dW, std::vector<double>& db,
                            std::vector<double>& dx) {
    const int Ho = s.Ho(), Wo = s.Wo();
    dW.assign(size_t(s.K) * s.C * s.k * s.k, 0.0);
    db.assign(s.K, 0.0);
    dx.assign(size_t(batch) * s.C * s.H * s.W, 0.0);
    for (int n = 0; n < batch; ++n)
        for (int o = 0; o < s.K; ++o)
            for (int i = 0; i < Ho; ++i)
                for (int j = 0; j < Wo; ++j) {
                    const double g = dy[((size_t(n) * s.K + o) * Ho + i) * Wo + j];
                    db[o] += g;
                    for (int c = 0; c < s.C; ++c)
                        for (int u = 0; u < s.k; ++u)
                            for (int v = 0; v < s.k; ++v) {
                                const int r = i * s.stride + u - s.pad, q = j * s.stride + v - s.pad;
                                if (r < 0 || r >= s.H || q < 0 || q >= s.W) continue;
                                const size_t xi = ((size_t(n) * s.C + c) * s.H + r) * s.W + q;
                                const size_t wi = ((size_t(o) * s.C + c) * s.k + u) * s.k + v;
                                dW[wi] += g * x[xi];
                                dx[xi] += g * W[wi];
                            }
                }
}

// LayerNorm over each row of x [batch][D] (PyTorch: biased variance, eps inside the sqrt):
//   y = (x - mean) / sqrt(var + eps) * gamma + beta
inline void layernorm_forward(const std::vector<double>& x, const std::vector<double>& gamma,
                              const std::vector<double>& beta, int batch, int D, double eps, std::vector<double>& y) {
    y.assign(size_t(batch) * D, 0.0);
    for (int n = 0; n < batch; ++n) {
        const double* r = x.data() + size_t(n) * D;
        double mean = 0, var = 0;
        for (int d = 0; d < D; ++d) mean += r[d];
        mean /= D;
        for (int d = 0; d < D; ++d) var += (r[d] - mean) * (r[d] - mean);
        var /= D;
        const double inv = 1.0 / std::sqrt(var + eps);
        for (int d = 0; d < D; ++d) y[size_t(n) * D + d] = (r[d] - mean) * inv * gamma[d] + beta[d];
    }
}

// dgamma = sum_n dy xhat, dbeta = sum_n dy,
// dx = inv / D (D g - sum(g) - xhat sum(g xhat)), g = dy gamma
inline void layernorm_backward(const std::vector<double>& x, const std::vector<double>& gamma,
                               const std::vector<double>& dy, int batch, int D, double eps, std::vector<double>& dgamma,
                               std::vector<double>& dbeta, std::vector<double>& dx) {
    dgamma.assign(D, 0.0);
    dbeta.assign(D, 0.0);
    dx.assign(size_t(batch) * D, 0.0);
    std::vector<double> xhat(D), g(D);
    for (int n = 0; n < batch; ++n) {
        const double* r = x.data() + size_t(n) * D;
        double mean = 0, var = 0;
        for (int d = 0; d < D; ++d) mean += r[d];
        mean /= D;
        for (int d = 0; d < D; ++d) var += (r[d] - mean) * (r[d] - mean);
        var /= D;
        const double inv = 1.0 / std::sqrt(var + eps);
        double sg = 0, sgx = 0;
        for (int d = 0; d < D; ++d) {
            xhat[d] = (r[d] - mean) * inv;
            const double gy = dy[size_t(n) * D + d];
            dgamma[d] += gy * xhat[d];
            dbeta[d] += gy;
            g[d] = gy * gamma[d];
            sg += g[d];
            sgx += g[d] * xhat[d];
        }
        for (int d = 0; d < D; ++d) dx[size_t(n) * D + d] = inv / D * (D * g[d] - sg - xhat[d] * sgx);
    }
}

}  // namespace ref
}  // namespace temper
