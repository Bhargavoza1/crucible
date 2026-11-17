// Phase 10: the CNN + GRU student policy (temper/algos/student).
//   - one step at a time (T = 1, its last hidden state fed back) gives the same
//     actions as the whole sequence at once
//   - it learns a task that needs memory and the camera: the image at step 0 is red
//     or blue, later images are gray noise; the label at every step is +1 (red) or -1
//     (blue). Only the GRU can carry the color to the later steps.
#ifdef TEMPER_HAS_CUDNN
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "forge/util/device_buffer.h"
#include "temper/algos/student/student_ops.h"
#include "temper/algos/student/student_policy.h"
#include "temper/optim/adam.h"

using namespace temper;

namespace {

TEST(StudentPolicy, StepByStepEqualsSequence) {
    // as distill_app: 3 cameras, a 12-step chunk of 5 actions
    const int T = 6, B = 4, S = 40, cams = 3, A = 60;
    StudentConfig cfg{8, cams, S, A, 32};
    StudentPolicy pol(cfg, T * B, 3);
    std::mt19937 g(1);
    std::uniform_int_distribution<int> u(0, 255);
    std::normal_distribution<float> n(0.0f, 1.0f);
    const size_t frame = size_t(cams) * S * S * 3;
    std::vector<uint8_t> img(size_t(T) * B * frame);
    for (auto& v : img) v = uint8_t(u(g));
    std::vector<float> st(size_t(T) * B * 8);
    for (auto& v : st) v = n(g);
    forge::DeviceBuffer<uint8_t> dimg;
    forge::DeviceBuffer<float> dst, h(size_t(B) * 32);
    dimg.upload(img);
    dst.upload(st);
    const float* y = pol.forward(dst.data(), dimg.data(), nullptr, nullptr, nullptr, T, B, nullptr);
    forge::DeviceBuffer<float> whole_d(size_t(T) * B * A);
    CUDA_CHECK(cudaMemcpy(whole_d.data(), y, whole_d.size() * 4, cudaMemcpyDeviceToDevice));
    const auto whole = whole_d.download();
    double worst = 0;
    for (int t = 0; t < T; ++t) {
        const float* a = pol.forward(dst.data() + size_t(t) * B * 8, dimg.data() + size_t(t) * B * frame, nullptr,
                                     nullptr, t == 0 ? nullptr : h.data(), 1, B, nullptr);
        std::vector<float> step(size_t(B) * A);
        CUDA_CHECK(cudaMemcpy(step.data(), a, step.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < step.size(); ++i)
            worst = std::max(worst, double(std::fabs(step[i] - whole[size_t(t) * B * A + i])));
        CUDA_CHECK(cudaMemcpy(h.data(), pol.last_hidden(), size_t(B) * 32 * 4, cudaMemcpyDeviceToDevice));
    }
    EXPECT_LT(worst, 1e-5) << "acting one decision at a time = the sequence";
}

TEST(StudentPolicy, LearnsAColorItSawAtTheStart) {
    const int T = 8, B = 32, S = 40;
    StudentConfig cfg{4, 1, S, 5, 64};
    StudentPolicy pol(cfg, T * B, 7);
    std::mt19937 g(2);
    std::uniform_int_distribution<int> u(100, 156);
    std::vector<uint8_t> img(size_t(T) * B * S * S * 3);
    std::vector<float> label(size_t(T) * B * 5, 0.0f), st(size_t(T) * B * 4, 0.0f), mask(size_t(T) * B, 0.0f);
    auto fill = [&]() {
        for (int b = 0; b < B; ++b) {
            const bool red = g() % 2 == 0;
            for (int t = 0; t < T; ++t) {
                uint8_t* p = img.data() + (size_t(t) * B + b) * S * S * 3;
                for (int k = 0; k < S * S; ++k) {
                    if (t == 0) {
                        p[3 * k] = red ? 230 : 20, p[3 * k + 1] = 20, p[3 * k + 2] = red ? 20 : 230;
                    } else {
                        p[3 * k] = p[3 * k + 1] = p[3 * k + 2] = uint8_t(u(g));
                    }
                }
                label[(size_t(t) * B + b) * 5] = red ? 0.8f : -0.8f;
                mask[size_t(t) * B + b] = t >= 2 ? 1.0f : 0.0f;  // scored only where the color is gone
            }
        }
    };
    std::vector<Tensor*> P, G;
    pol.params(P, G);
    Adam adam(P, G, AdamOptions{1e-3f});
    forge::DeviceBuffer<uint8_t> dimg;
    forge::DeviceBuffer<float> dst, dlab, dmask, dy(size_t(T) * B * 5), loss(1), count(1);
    dst.upload(st);
    const float cnt = float(B * (T - 2));
    count.upload(std::vector<float>{cnt});
    float last = 1e9f;
    for (int it = 0; it < 400; ++it) {
        fill();
        dimg.upload(img);
        dlab.upload(label);
        dmask.upload(mask);
        CUDA_CHECK(cudaMemset(loss.data(), 0, 4));
        const float* y = pol.forward(dst.data(), dimg.data(), nullptr, nullptr, nullptr, T, B, nullptr);
        student::masked_mse(y, dlab.data(), dmask.data(), count.data(), T * B, 5, dy.data(), loss.data(), nullptr);
        pol.backward(dy.data(), nullptr);
        adam.step(nullptr);
        last = loss.download()[0];
    }
    std::printf("student memory task: loss after 400 steps %.4f (guessing: %.4f)\n", last, 0.64f / 5.0f);
    EXPECT_LT(last, 0.01f) << "remembers the first frame's color through the GRU";
}

}  // namespace
#endif
