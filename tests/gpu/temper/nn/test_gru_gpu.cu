// Phase 10: temper::Gru against the CPU answer key (reference_cpu.h, double; itself
// checked against finite differences and PyTorch): output, every parameter gradient,
// dx and dh0, for several shapes, with and without an initial state, episode resets
// and a gradient into the last state. And stepping one step at a time (acting: T = 1,
// the last h fed back) gives the same outputs as the whole sequence.
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "temper/nn/gru.h"
#include "temper/nn/reference_cpu.h"

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

TEST(GruGpu, MatchesReference) {
    struct Case {
        ref::GruShape s;
        bool h0, reset, dhT;
    };
    for (const Case c : {Case{{1, 4, 5, 8}, false, false, false}, Case{{6, 3, 7, 16}, true, true, true},
                         Case{{24, 9, 40, 64}, true, true, false}, Case{{16, 32, 300, 128}, false, true, true}}) {
        const ref::GruShape s = c.s;
        const int TB = s.T * s.B;
        Gru gru(s.in, s.H, 3);
        gru.allocate(TB);
        const auto x = randf(size_t(TB) * s.in, 1), h0 = randf(size_t(s.B) * s.H, 2, 0.5f);
        const auto dy = randf(size_t(TB) * s.H, 4), dhT = randf(size_t(s.B) * s.H, 5);
        std::vector<float> reset(TB, 0.0f);
        if (c.reset)
            for (int i = 0; i < TB; i += 7) reset[i] = 1.0f;  // episodes start at scattered (t, b)
        Dev dx_in(x), dh0_in(h0), dreset(reset), ddy(dy), ddhT(dhT), y(size_t(TB) * s.H), dx(size_t(TB) * s.in),
            dh0(size_t(s.B) * s.H);
        gru.forward(dx_in.p(), c.h0 ? dh0_in.p() : nullptr, c.reset ? dreset.p() : nullptr, s.T, s.B, y.p(), nullptr);
        gru.backward(ddy.p(), c.dhT ? ddhT.p() : nullptr, dx.p(), dh0.p(), nullptr);
        const auto Wih = to_d(gru.W_ih.download()), Whh = to_d(gru.W_hh.download());
        const auto bih = to_d(gru.b_ih.download()), bhh = to_d(gru.b_hh.download());
        const std::vector<double> rh0 = c.h0 ? to_d(h0) : std::vector<double>{};
        const std::vector<double> rreset = c.reset ? to_d(reset) : std::vector<double>{};
        const std::vector<double> rdhT = c.dhT ? to_d(dhT) : std::vector<double>{};
        std::vector<double> ry, a, b, cc, d, rdx, rdh0;
        ref::gru_forward(to_d(x), rh0, rreset, Wih, Whh, bih, bhh, s, ry);
        ref::gru_backward(to_d(x), rh0, rreset, Wih, Whh, bih, bhh, s, to_d(dy), rdhT, a, b, cc, d, rdx, rdh0);
        const double tol = 5e-5;
        EXPECT_LT(rel_err(y.get(), ry), tol) << "T " << s.T << " B " << s.B;
        EXPECT_LT(rel_err(gru.dW_ih.download(), a), tol) << "dW_ih T " << s.T;
        EXPECT_LT(rel_err(gru.dW_hh.download(), b), tol) << "dW_hh T " << s.T;
        EXPECT_LT(rel_err(gru.db_ih.download(), cc), tol) << "db_ih T " << s.T;
        EXPECT_LT(rel_err(gru.db_hh.download(), d), tol) << "db_hh T " << s.T;
        EXPECT_LT(rel_err(dx.get(), rdx), tol) << "dx T " << s.T;
        EXPECT_LT(rel_err(dh0.get(), rdh0), tol) << "dh0 T " << s.T;
    }
}

TEST(GruGpu, StepByStepEqualsSequence) {
    const int T = 10, B = 6, in = 12, H = 32;
    Gru gru(in, H, 8);
    gru.allocate(T * B);
    const auto x = randf(size_t(T) * B * in, 9);
    std::vector<float> reset(T * B, 0.0f);
    reset[4 * B + 2] = 1.0f;
    Dev dx(x), dreset(reset), y(size_t(T) * B * H);
    gru.forward(dx.p(), nullptr, dreset.p(), T, B, y.p(), nullptr);
    const auto whole = y.get();
    Dev h(size_t(B) * H), hn(size_t(B) * H);
    double worst = 0;
    for (int t = 0; t < T; ++t) {
        gru.forward(dx.p() + size_t(t) * B * in, t == 0 ? nullptr : h.p(), dreset.p() + size_t(t) * B, 1, B, hn.p(),
                    nullptr);
        const auto step = hn.get();
        for (size_t i = 0; i < step.size(); ++i)
            worst = std::max(worst, double(std::fabs(step[i] - whole[size_t(t) * B * H + i])));
        CUDA_CHECK(cudaMemcpy(h.p(), hn.p(), step.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    EXPECT_LT(worst, 1e-6) << "one step at a time, the last h fed back";
}

}  // namespace
