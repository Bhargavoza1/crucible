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

}  // namespace ref
}  // namespace temper
