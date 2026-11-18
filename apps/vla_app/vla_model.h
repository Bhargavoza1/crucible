#pragma once
// vla_app's model: the ONNX graph from tools/vla_train.py / vla_openarm.py (the whole
// flow-matching sampler) run by ONNX Runtime. Inputs images uint8 [n, cams, size,
// size, 3], state [n, S] (raw: normalized inside), text [n, E], noise [n, chunk * A];
// output chunk [n, chunk, A]. S and A from the graph: 24 and 5 for panda_pick_color and
// openarm_pick_place, 48 and 10 for openarm_stack.
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <onnxruntime_cxx_api.h>

namespace vla {

using Clock = std::chrono::steady_clock;
inline double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

constexpr int kAct = 5, kState = 24;  // panda_pick_color, openarm_pick_place

struct Model {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "vla_app"};
    std::unique_ptr<Ort::Session> session;
    int cams = 0, size = 0, text_dim = 0, chunk = 0, state_dim = 0, act = 0;
    double seconds = 0;
    int calls = 0;

    Model(const std::string& path, bool cuda) {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(0);
        if (cuda) {
            OrtCUDAProviderOptions o{};
            so.AppendExecutionProvider_CUDA(o);
        }
        session = std::make_unique<Ort::Session>(env, path.c_str(), so);
        auto in_shape = [&](size_t i) { return session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape(); };
        const auto img = in_shape(0), sta = in_shape(1), txt = in_shape(2);
        const auto out = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (img.size() != 5 || sta.size() != 2 || txt.size() != 2 || out.size() != 3)
            throw std::runtime_error("unexpected ONNX inputs");
        cams = int(img[1]), size = int(img[2]), text_dim = int(txt[1]), chunk = int(out[1]);
        state_dim = int(sta[1]), act = int(out[2]);
    }

    // images uint8 [n][cams][size][size][3], state [n][state_dim] (raw), text [n][E], noise
    // [n][chunk * act] -> chunk [n][chunk][act]
    void run(int n, uint8_t* images, float* state, float* text, float* noise, float* out) {
        const auto t0 = Clock::now();
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 5> si{n, cams, size, size, 3};
        const std::array<int64_t, 2> ss{n, state_dim}, st{n, text_dim}, sn{n, int64_t(chunk) * act};
        const std::array<int64_t, 3> so{n, chunk, act};
        std::array<Ort::Value, 4> in = {
            Ort::Value::CreateTensor<uint8_t>(mem, images, size_t(n) * cams * size * size * 3, si.data(), si.size()),
            Ort::Value::CreateTensor<float>(mem, state, size_t(n) * state_dim, ss.data(), ss.size()),
            Ort::Value::CreateTensor<float>(mem, text, size_t(n) * text_dim, st.data(), st.size()),
            Ort::Value::CreateTensor<float>(mem, noise, size_t(n) * chunk * act, sn.data(), sn.size())};
        Ort::Value res = Ort::Value::CreateTensor<float>(mem, out, size_t(n) * chunk * act, so.data(), so.size());
        const char* in_names[4] = {"images", "state", "text", "noise"};
        const char* out_names[1] = {"chunk"};
        Ort::RunOptions ro;
        session->Run(ro, in_names, in.data(), 4, out_names, &res, 1);
        seconds += since(t0);
        ++calls;
    }
};

}  // namespace vla
