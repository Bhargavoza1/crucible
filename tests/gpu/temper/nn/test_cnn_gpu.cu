// Phase 9d: GPU CNN layers against the CPU answer key (double, checked against
// finite differences and PyTorch in tests_cpu):
//   Conv2d (cuDNN): forward, dW, db, dx for several shapes (stride, padding, not square)
//   LayerNorm: forward, dgamma, dbeta, dx
//   the small CNN of tools/gen_torch_cnn_ref.py, GPU layers stacked: output and
//   every gradient as PyTorch (float32 vs double: relative 1e-5)
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "temper/nn/activations.h"
#include "temper/nn/layernorm.h"
#include "temper/nn/linear.h"
#include "temper/nn/reference_cpu.h"
#ifdef TEMPER_HAS_CUDNN
#include "temper/nn/conv2d.h"
#endif

using namespace temper;

namespace {

std::vector<float> randf(size_t n, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}
std::vector<double> to_d(const std::vector<float>& v) { return {v.begin(), v.end()}; }
std::vector<float> to_f(const std::vector<double>& v) { return {v.begin(), v.end()}; }

double rel_err(const std::vector<float>& a, const std::vector<double>& b) {
    double e = 0, s = 1;
    for (size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, std::fabs(double(a[i]) - b[i]));
        s = std::max(s, std::fabs(b[i]));
    }
    return e / s;
}

struct Dev {
    forge::DeviceBuffer<float> b;
    explicit Dev(const std::vector<float>& h) { b.upload(h); }
    explicit Dev(size_t n) { b.resize(n); }
    float* p() { return b.data(); }
    std::vector<float> get() const { return b.download(); }
};

TEST(LayerNormGpu, MatchesReference) {
    for (int D : {6, 256, 300}) {
        const int batch = 37;
        LayerNorm ln(D);
        const auto x = randf(size_t(batch) * D, 1), dy = randf(size_t(batch) * D, 2);
        const auto g = randf(D, 3), bt = randf(D, 4);
        ln.gamma.upload(g);
        ln.beta.upload(bt);
        Dev dx_(x), y(size_t(batch) * D), ddy(dy), dx(size_t(batch) * D);
        ln.forward(dx_.p(), y.p(), batch, nullptr);
        ln.backward(ddy.p(), dx.p(), batch, nullptr);
        std::vector<double> ry, rdg, rdb, rdx;
        ref::layernorm_forward(to_d(x), to_d(g), to_d(bt), batch, D, 1e-5, ry);
        ref::layernorm_backward(to_d(x), to_d(g), to_d(dy), batch, D, 1e-5, rdg, rdb, rdx);
        EXPECT_LT(rel_err(y.get(), ry), 1e-5) << "D " << D;
        EXPECT_LT(rel_err(ln.dgamma.download(), rdg), 1e-5) << "D " << D;
        EXPECT_LT(rel_err(ln.dbeta.download(), rdb), 1e-5) << "D " << D;
        EXPECT_LT(rel_err(dx.get(), rdx), 1e-5) << "D " << D;
    }
}

#ifdef TEMPER_HAS_CUDNN
TEST(Conv2dGpu, MatchesReference) {
    // the bc_train.py encoder's first layer (3 -> 32, k5 s2 on 64 x 64) and odd shapes
    for (const ref::ConvShape s : {ref::ConvShape{3, 64, 64, 32, 5, 2, 0}, ref::ConvShape{4, 13, 11, 5, 3, 2, 1},
                                   ref::ConvShape{2, 7, 9, 3, 3, 1, 1}, ref::ConvShape{64, 6, 6, 64, 3, 1, 0}}) {
        const int batch = 5;
        Conv2d conv(s.C, s.H, s.W, s.K, s.k, s.stride, s.pad, 9);
        ASSERT_EQ(conv.out_height(), s.Ho());
        ASSERT_EQ(conv.out_width(), s.Wo());
        const auto x = randf(size_t(batch) * conv.in_dim(), 5), dy = randf(size_t(batch) * conv.out_dim(), 6);
        Dev dxin(x), y(size_t(batch) * conv.out_dim()), ddy(dy), dx(size_t(batch) * conv.in_dim());
        conv.forward(dxin.p(), y.p(), batch, nullptr);
        conv.backward(ddy.p(), dx.p(), batch, nullptr);
        const auto W = to_d(conv.W.download()), b = to_d(conv.b.download());
        std::vector<double> ry, rdW, rdb, rdx;
        ref::conv2d_forward(to_d(x), W, b, batch, s, ry);
        ref::conv2d_backward(to_d(x), W, to_d(dy), batch, s, rdW, rdb, rdx);
        const std::string shape = std::to_string(s.C) + "x" + std::to_string(s.H) + "x" + std::to_string(s.W) + " k" +
                                  std::to_string(s.k) + " s" + std::to_string(s.stride) + " p" + std::to_string(s.pad);
        EXPECT_LT(rel_err(y.get(), ry), 1e-5) << shape;
        EXPECT_LT(rel_err(conv.dW.download(), rdW), 1e-5) << shape;
        EXPECT_LT(rel_err(conv.db.download(), rdb), 1e-5) << shape;
        EXPECT_LT(rel_err(dx.get(), rdx), 1e-5) << shape;
    }
}

std::map<std::string, std::vector<double>> load_torch_ref() {
    std::ifstream in(std::string(CRUCIBLE_SOURCE_DIR) + "/tests/data/torch_cnn.txt");
    std::map<std::string, std::vector<double>> m;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        double v;
        while (ss >> v) m[key].push_back(v);
    }
    return m;
}

// The GPU layers stacked as in tools/gen_torch_cnn_ref.py: same output and gradients as PyTorch.
TEST(CnnGpu, MatchesPyTorch) {
    auto t = load_torch_ref();
    ASSERT_FALSE(t.empty());
    const int batch = 2;
    Conv2d c1(3, 13, 11, 4, 5, 2, 0, 1), c2(4, 5, 4, 5, 3, 2, 1, 2);
    Relu r1(c1.out_dim()), r2(c2.out_dim());
    Linear fc(30, 6, 3);
    LayerNorm ln(6);
    c1.W.upload(to_f(t["conv1_w"]));
    c1.b.upload(to_f(t["conv1_b"]));
    c2.W.upload(to_f(t["conv2_w"]));
    c2.b.upload(to_f(t["conv2_b"]));
    fc.W.upload(to_f(t["fc_w"]));
    fc.b.upload(to_f(t["fc_b"]));
    ln.gamma.upload(to_f(t["ln_g"]));
    ln.beta.upload(to_f(t["ln_b"]));
    std::vector<Layer*> net = {&c1, &r1, &c2, &r2, &fc, &ln};
    std::vector<std::unique_ptr<Dev>> act;
    act.push_back(std::make_unique<Dev>(to_f(t["x"])));
    for (Layer* l : net) {
        act.push_back(std::make_unique<Dev>(size_t(batch) * l->out_dim()));
        l->forward(act[act.size() - 2]->p(), act.back()->p(), batch, nullptr);
    }
    EXPECT_LT(rel_err(act.back()->get(), t["y"]), 1e-5) << "y";
    auto g = std::make_unique<Dev>(to_f(t["dy"]));
    for (size_t k = net.size(); k-- > 0;) {
        auto gx = std::make_unique<Dev>(size_t(batch) * net[k]->in_dim());
        net[k]->backward(g->p(), gx->p(), batch, nullptr);
        g = std::move(gx);
    }
    EXPECT_LT(rel_err(g->get(), t["dx"]), 1e-5) << "dx";
    EXPECT_LT(rel_err(c1.dW.download(), t["conv1_dw"]), 1e-5);
    EXPECT_LT(rel_err(c1.db.download(), t["conv1_db"]), 1e-5);
    EXPECT_LT(rel_err(c2.dW.download(), t["conv2_dw"]), 1e-5);
    EXPECT_LT(rel_err(c2.db.download(), t["conv2_db"]), 1e-5);
    EXPECT_LT(rel_err(fc.dW.download(), t["fc_dw"]), 1e-5);
    EXPECT_LT(rel_err(fc.db.download(), t["fc_db"]), 1e-5);
    EXPECT_LT(rel_err(ln.dgamma.download(), t["ln_dg"]), 1e-5);
    EXPECT_LT(rel_err(ln.dbeta.download(), t["ln_db"]), 1e-5);
}
#endif

}  // namespace
