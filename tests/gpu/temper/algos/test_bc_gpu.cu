// Phase 9d (2b): behavior-cloning pieces on the GPU against plain CPU code:
//   Tanh layer, image preprocessing with DrQ shifts (written here as bc_train.py
//   does it: replicate-pad, then crop), column copy, MSE loss, and the whole
//   BcPolicy (2 cameras + state): output and every parameter gradient against a
//   double-precision CPU version built from temper::ref.
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"
#include "temper/algos/bc/bc_ops.h"
#include "temper/nn/activations.h"
#include "temper/nn/reference_cpu.h"
#ifdef TEMPER_HAS_CUDNN
#include "temper/algos/bc/bc_policy.h"
#include "temper/optim/adam.h"
#endif

using namespace temper;

namespace {

std::vector<float> randf(size_t n, unsigned seed, float scale = 1.0f) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}
std::vector<double> to_d(const std::vector<float>& v) { return {v.begin(), v.end()}; }

double rel_err(const std::vector<float>& a, const std::vector<double>& b) {
    double e = 0, s = 1e-30;
    for (size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, std::fabs(double(a[i]) - b[i]));
        s = std::max(s, std::fabs(b[i]));
    }
    return e / s;
}

std::vector<uint8_t> rand_u8(size_t n, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<uint8_t> v(n);
    for (auto& x : v) x = uint8_t(g() & 255u);
    return v;
}

// bc_train.py's random_shift for one image: replicate-pad by 'pad', crop at (oy, ox).
// in: uint8 [H][W][3]; out: float [3][H][W] in [0, 1].
void shift_like_python(const uint8_t* in, int H, int W, int ox, int oy, int pad, double* out) {
    const int Hp = H + 2 * pad, Wp = W + 2 * pad;
    std::vector<double> padded(size_t(3) * Hp * Wp);
    for (int ch = 0; ch < 3; ++ch)
        for (int y = 0; y < Hp; ++y)
            for (int x = 0; x < Wp; ++x) {
                const int sy = std::min(std::max(y - pad, 0), H - 1), sx = std::min(std::max(x - pad, 0), W - 1);
                padded[(size_t(ch) * Hp + y) * Wp + x] = in[(size_t(sy) * W + sx) * 3 + ch] / 255.0;
            }
    for (int ch = 0; ch < 3; ++ch)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) out[(size_t(ch) * H + y) * W + x] = padded[(size_t(ch) * Hp + y + oy) * Wp + x + ox];
}

TEST(BcGpu, Tanh) {
    const int batch = 7, D = 5;
    const auto x = randf(size_t(batch) * D, 1, 2.0f), dy = randf(size_t(batch) * D, 2);
    forge::DeviceBuffer<float> dx_in, y(x.size()), ddy, dx(x.size());
    dx_in.upload(x);
    ddy.upload(dy);
    Tanh t(D);
    t.forward(dx_in.data(), y.data(), batch, nullptr);
    t.backward(ddy.data(), dx.data(), batch, nullptr);
    std::vector<double> ry(x.size()), rdx(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        ry[i] = std::tanh(double(x[i]));
        rdx[i] = dy[i] * (1.0 - ry[i] * ry[i]);
    }
    EXPECT_LT(rel_err(y.download(), ry), 1e-6);
    EXPECT_LT(rel_err(dx.download(), rdx), 1e-5);
}

TEST(BcGpu, ImagesToInputMatchesPythonShift) {
    const int batch = 3, cams = 2, H = 9, W = 7, pad = 4;
    const auto img = rand_u8(size_t(batch) * cams * H * W * 3, 3);
    std::vector<int> shifts(size_t(batch) * cams * 2);
    std::mt19937 g(4);
    for (auto& s : shifts) s = int(g() % (2 * pad + 1));
    shifts[0] = 0, shifts[1] = 0;              // corner cases: full shift both ways
    shifts[2] = 2 * pad, shifts[3] = 2 * pad;
    forge::DeviceBuffer<uint8_t> dimg;
    forge::DeviceBuffer<int> dsh;
    forge::DeviceBuffer<float> out(size_t(batch) * cams * 3 * H * W);
    dimg.upload(img);
    dsh.upload(shifts);
    for (bool shifted : {false, true}) {
        bc::images_to_input(dimg.data(), batch, cams, H, W, shifted ? dsh.data() : nullptr, pad, out.data(), nullptr);
        std::vector<double> ref(out.size());
        for (int c = 0; c < cams; ++c)
            for (int n = 0; n < batch; ++n) {
                const int* s = shifts.data() + (size_t(n) * cams + c) * 2;
                const int ox = shifted ? s[0] : pad, oy = shifted ? s[1] : pad;  // offset pad = no shift
                shift_like_python(img.data() + (size_t(n) * cams + c) * H * W * 3, H, W, ox, oy, pad,
                                  ref.data() + (size_t(c) * batch + n) * 3 * H * W);
            }
        EXPECT_LT(rel_err(out.download(), ref), 1e-6) << (shifted ? "shifted" : "plain");
    }
}

TEST(BcGpu, CopyColumns) {
    const int rows = 5;
    const auto a = randf(size_t(rows) * 3, 5), b = randf(size_t(rows) * 4, 6);
    forge::DeviceBuffer<float> da, db, cat(size_t(rows) * 7), back(size_t(rows) * 4);
    da.upload(a);
    db.upload(b);
    bc::copy_columns(da.data(), 3, 0, cat.data(), 7, 0, rows, 3, nullptr);
    bc::copy_columns(db.data(), 4, 0, cat.data(), 7, 3, rows, 4, nullptr);
    bc::copy_columns(cat.data(), 7, 3, back.data(), 4, 0, rows, 4, nullptr);
    const auto c = cat.download();
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < 7; ++k)
            EXPECT_EQ(c[size_t(r) * 7 + k], k < 3 ? a[size_t(r) * 3 + k] : b[size_t(r) * 4 + k - 3]);
    EXPECT_EQ(back.download(), b);
}

TEST(BcGpu, MseLoss) {
    const int rows = 300, cols = 5;  // more than one pass of the 256 threads
    const auto p = randf(size_t(rows) * cols, 7), t = randf(size_t(rows) * cols, 8);
    forge::DeviceBuffer<float> dp, dt, dy(p.size()), loss;
    dp.upload(p);
    dt.upload(t);
    loss.upload(std::vector<float>{0.5f});  // adds to what is there
    bc::mse_loss(dp.data(), dt.data(), rows, cols, dy.data(), loss.data(), nullptr);
    double l = 0;
    std::vector<double> rdy(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        const double d = double(p[i]) - t[i];
        l += d * d / double(p.size());
        rdy[i] = 2.0 * d / double(p.size());
    }
    EXPECT_NEAR(loss.download()[0], 0.5 + l, 1e-5);
    EXPECT_LT(rel_err(dy.download(), rdy), 1e-6);
}

#ifdef TEMPER_HAS_CUDNN
// CPU double version of BcPolicy (same weights): forward keeps every activation,
// backward returns the gradients in params() order.
struct RefPolicy {
    BcPolicyConfig cfg;
    std::vector<std::vector<double>> P;  // params() order
    std::vector<ref::ConvShape> conv;
    int n4 = 0;

    RefPolicy(const BcPolicyConfig& c, std::vector<std::vector<double>> params) : cfg(c), P(std::move(params)) {
        int h = c.size;
        const int spec[4][4] = {{3, 32, 5, 2}, {32, 64, 3, 2}, {64, 64, 3, 2}, {64, 64, 3, 1}};
        for (auto& s : spec) {
            conv.push_back(ref::ConvShape{s[0], h, h, s[1], s[2], s[3], 0});
            h = conv.back().Ho();
        }
        n4 = 64 * h * h;
    }

    static void relu(std::vector<double>& v) {
        for (auto& x : v) x = std::max(x, 0.0);
    }
    static void relu_back(const std::vector<double>& y, std::vector<double>& g) {
        for (size_t i = 0; i < g.size(); ++i)
            if (y[i] <= 0) g[i] = 0;
    }

    // per camera: input, conv outputs (after ReLU), fc output, ln output
    struct Enc {
        std::vector<double> in, a[4], fc, ln;
    };
    std::vector<Enc> enc;
    std::vector<double> s_in, s_h, cat, h1, h2, out;
    int B = 0;

    std::vector<double> forward(const std::vector<std::vector<double>>& cam_in, const std::vector<double>& state, int batch) {
        B = batch;
        const int F = BcPolicy::kFeature, SH = BcPolicy::kStateHidden, HID = BcPolicy::kHidden;
        const int cat_dim = F * cfg.cams + SH;
        enc.assign(cfg.cams, {});
        cat.assign(size_t(B) * cat_dim, 0.0);
        for (int c = 0; c < cfg.cams; ++c) {
            const auto* p = &P[size_t(c) * 12];
            Enc& e = enc[c];
            e.in = cam_in[c];
            const std::vector<double>* x = &e.in;
            for (int l = 0; l < 4; ++l) {
                ref::conv2d_forward(*x, p[2 * l], p[2 * l + 1], B, conv[l], e.a[l]);
                relu(e.a[l]);
                x = &e.a[l];
            }
            ref::linear_forward(e.a[3], p[8], p[9], B, n4, F, e.fc);
            ref::layernorm_forward(e.fc, p[10], p[11], B, F, 1e-5, e.ln);
            for (int n = 0; n < B; ++n)
                for (int k = 0; k < F; ++k) cat[size_t(n) * cat_dim + c * F + k] = e.ln[size_t(n) * F + k];
        }
        const size_t o = size_t(cfg.cams) * 12;
        s_in = state;
        ref::linear_forward(s_in, P[o], P[o + 1], B, cfg.state_dim, SH, s_h);
        relu(s_h);
        for (int n = 0; n < B; ++n)
            for (int k = 0; k < SH; ++k) cat[size_t(n) * cat_dim + cfg.cams * F + k] = s_h[size_t(n) * SH + k];
        ref::linear_forward(cat, P[o + 2], P[o + 3], B, cat_dim, HID, h1);
        relu(h1);
        ref::linear_forward(h1, P[o + 4], P[o + 5], B, HID, HID, h2);
        relu(h2);
        ref::linear_forward(h2, P[o + 6], P[o + 7], B, HID, cfg.act_dim, out);
        for (auto& v : out) v = std::tanh(v);
        return out;
    }

    std::vector<std::vector<double>> backward(std::vector<double> dy) {
        const int F = BcPolicy::kFeature, SH = BcPolicy::kStateHidden, HID = BcPolicy::kHidden;
        const int cat_dim = F * cfg.cams + SH;
        std::vector<std::vector<double>> G(P.size());
        const size_t o = size_t(cfg.cams) * 12;
        for (size_t i = 0; i < dy.size(); ++i) dy[i] *= 1.0 - out[i] * out[i];
        std::vector<double> g2, g1, gcat, tmp;
        ref::linear_backward(h2, P[o + 6], dy, B, HID, cfg.act_dim, G[o + 6], G[o + 7], g2);
        relu_back(h2, g2);
        ref::linear_backward(h1, P[o + 4], g2, B, HID, HID, G[o + 4], G[o + 5], g1);
        relu_back(h1, g1);
        ref::linear_backward(cat, P[o + 2], g1, B, cat_dim, HID, G[o + 2], G[o + 3], gcat);
        std::vector<double> gs(size_t(B) * SH);
        for (int n = 0; n < B; ++n)
            for (int k = 0; k < SH; ++k) gs[size_t(n) * SH + k] = gcat[size_t(n) * cat_dim + cfg.cams * F + k];
        relu_back(s_h, gs);
        ref::linear_backward(s_in, P[o], gs, B, cfg.state_dim, SH, G[o], G[o + 1], tmp);
        for (int c = 0; c < cfg.cams; ++c) {
            const size_t q = size_t(c) * 12;
            Enc& e = enc[c];
            std::vector<double> g(size_t(B) * F), gfc, ga;
            for (int n = 0; n < B; ++n)
                for (int k = 0; k < F; ++k) g[size_t(n) * F + k] = gcat[size_t(n) * cat_dim + c * F + k];
            ref::layernorm_backward(e.fc, P[q + 10], g, B, F, 1e-5, G[q + 10], G[q + 11], gfc);
            ref::linear_backward(e.a[3], P[q + 8], gfc, B, n4, F, G[q + 8], G[q + 9], ga);
            for (int l = 3; l >= 0; --l) {
                relu_back(e.a[l], ga);
                std::vector<double> gx;
                ref::conv2d_backward(l > 0 ? e.a[l - 1] : e.in, P[q + 2 * l], ga, B, conv[l], G[q + 2 * l],
                                     G[q + 2 * l + 1], gx);
                ga = std::move(gx);
            }
        }
        return G;
    }
};

TEST(BcGpu, PolicyMatchesCpu) {
    // 36 x 36 is the smallest image this encoder takes (conv outputs 16, 7, 3, 1)
    const BcPolicyConfig cfg{5, 2, 36, 5};
    const int batch = 3, pad = BcPolicy::kShiftPad;
    BcPolicy pol(cfg, 8, 3);
    std::vector<Tensor*> P, G;
    pol.params(P, G);
    ASSERT_EQ(P.size(), size_t(cfg.cams) * 12 + 8);
    // LayerNorm gamma / beta start at 1 / 0: random values test them properly
    for (int c = 0; c < cfg.cams; ++c) {
        auto g = randf(BcPolicy::kFeature, 20 + c, 0.5f), b = randf(BcPolicy::kFeature, 30 + c, 0.5f);
        for (auto& v : g) v += 1.0f;
        P[size_t(c) * 12 + 10]->upload(g);
        P[size_t(c) * 12 + 11]->upload(b);
    }
    std::vector<std::vector<double>> host;
    for (auto* t : P) host.push_back(to_d(t->download()));

    const int S = cfg.size;
    const auto img = rand_u8(size_t(batch) * cfg.cams * S * S * 3, 11);
    const auto state = randf(size_t(batch) * cfg.state_dim, 12);
    const auto dy = randf(size_t(batch) * cfg.act_dim, 13);
    std::vector<int> shifts(size_t(batch) * cfg.cams * 2);
    std::mt19937 g(14);
    for (auto& s : shifts) s = int(g() % (2 * pad + 1));
    forge::DeviceBuffer<uint8_t> dimg;
    forge::DeviceBuffer<int> dsh;
    forge::DeviceBuffer<float> dstate, ddy;
    dimg.upload(img);
    dsh.upload(shifts);
    dstate.upload(state);
    ddy.upload(dy);
    const float* y = pol.forward(dstate.data(), dimg.data(), dsh.data(), batch, nullptr);
    std::vector<float> out(size_t(batch) * cfg.act_dim);
    CUDA_CHECK(cudaMemcpy(out.data(), y, out.size() * 4, cudaMemcpyDeviceToHost));
    pol.backward(ddy.data(), batch, nullptr);

    std::vector<std::vector<double>> cam_in(cfg.cams, std::vector<double>(size_t(batch) * 3 * S * S));
    for (int c = 0; c < cfg.cams; ++c)
        for (int n = 0; n < batch; ++n) {
            const int* s = shifts.data() + (size_t(n) * cfg.cams + c) * 2;
            shift_like_python(img.data() + (size_t(n) * cfg.cams + c) * S * S * 3, S, S, s[0], s[1], pad,
                              cam_in[c].data() + size_t(n) * 3 * S * S);
        }
    RefPolicy r(cfg, host);
    EXPECT_LT(rel_err(out, r.forward(cam_in, to_d(state), batch)), 1e-5) << "actions";
    const auto rg = r.backward(to_d(dy));
    for (size_t k = 0; k < G.size(); ++k) EXPECT_LT(rel_err(G[k]->download(), rg[k]), 1e-4) << "gradient " << k;

    // save / load round trip
    Checkpoint ck;
    pol.save(ck);
    BcPolicy pol2(BcPolicy::config_from(ck), 8, 99);
    pol2.load(ck);
    const float* y2 = pol2.forward(dstate.data(), dimg.data(), dsh.data(), batch, nullptr);
    std::vector<float> out2(out.size());
    CUDA_CHECK(cudaMemcpy(out2.data(), y2, out2.size() * 4, cudaMemcpyDeviceToHost));
    EXPECT_EQ(out, out2);
}

TEST(BcGpu, StateOnlyPolicyLearns) {
    // cams = 0 (bc_app --input state): a few hundred Adam steps fit a fixed batch.
    const BcPolicyConfig cfg{6, 0, 0, 5};
    const int batch = 64;
    BcPolicy pol(cfg, batch, 5);
    std::vector<Tensor*> P, G;
    pol.params(P, G);
    Adam adam(P, G, AdamOptions{1e-3f});
    const auto state = randf(size_t(batch) * cfg.state_dim, 40);
    auto target = randf(size_t(batch) * cfg.act_dim, 41, 0.5f);
    for (auto& t : target) t = std::tanh(t);
    forge::DeviceBuffer<float> ds, dt, dy(target.size()), loss(1);
    ds.upload(state);
    dt.upload(target);
    float first = 0, last = 0;
    for (int it = 0; it < 300; ++it) {
        CUDA_CHECK(cudaMemset(loss.data(), 0, 4));
        const float* y = pol.forward(ds.data(), nullptr, nullptr, batch, nullptr);
        bc::mse_loss(y, dt.data(), batch, cfg.act_dim, dy.data(), loss.data(), nullptr);
        pol.backward(dy.data(), batch, nullptr);
        adam.step(nullptr);
        const float l = loss.download()[0];
        if (it == 0) first = l;
        last = l;
    }
    EXPECT_LT(last, 0.05f * first) << "first " << first << " last " << last;
}
#endif

}  // namespace
