#include "vision_policy.h"

#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

#include "../collect_app/openarm_camera_rig.h"
#include "forge/env/tasks/openarm_pick_place.h"
#ifdef CRUCIBLE_HAS_STUDENT
#include <cuda_runtime.h>

#include "forge/util/cuda_check.h"
#include "forge/util/device_buffer.h"
#include "temper/algos/student/student_policy.h"
#include "temper/common/checkpoint.h"
#endif
#ifdef CRUCIBLE_HAS_ORT
#include "../vla_app/vla_model.h"
#endif

namespace sandbox {

namespace {

using Task = forge::OpenArmPickPlaceTask;
constexpr int kAct = 5, kState = 24;

[[maybe_unused]] void halve(const uint8_t* in, int S, uint8_t* out) {  // 2 x 2 box filter, as distill_app
    const int h = S / 2;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < h; ++x)
            for (int c = 0; c < 3; ++c) {
                const int s = in[((2 * y) * S + 2 * x) * 3 + c] + in[((2 * y) * S + 2 * x + 1) * 3 + c] +
                              in[((2 * y + 1) * S + 2 * x) * 3 + c] + in[((2 * y + 1) * S + 2 * x + 1) * 3 + c];
                out[(y * h + x) * 3 + c] = uint8_t((s + 2) / 4);
            }
}

// Shared: one clock for all envs, as rl_app and distill_app run the policy: every
// exec steps all envs decide (one render for all). An env whose episode starts
// between two decisions holds still with the gripper open until the next one (rl_app
// does the same), and starts it with an empty memory. (Per-env clocks drifted apart
// after the resets and rendered every step: 3x the camera work.)
class ChunkPolicy : public VisionPolicy {
public:
    ChunkPolicy(const forge::RobotModel& m, int n, int size, int chunk, int exec)
        : n_(n), chunk_(chunk), exec_(exec), wait_(n, 0), fresh_(n, 1), plan_(size_t(n) * chunk * kAct, 0.0f),
          rig_(std::make_unique<OpenArmCameraRig>(
              m, n, size, FLUX_SHADER_DIR, std::string(CRUCIBLE_ASSET_DIR) + "/robots/openarm/openarm_cell.cmesh")) {
        if (exec < 1 || exec > chunk) throw std::runtime_error("exec must be 1 .. the chunk length");
    }
    void act(const forge::Env::HostCopy& h, const std::vector<char>& new_episode, float* actions) override {
        for (int e = 0; e < n_; ++e)
            if (new_episode[e]) fresh_[e] = 1, wait_[e] = step_ != 0;  // a new episode waits for the next decision
        if (step_ == 0) {
            rig_->render(h.qpos.data(), h.obs.data(), Task::kObsDim);
            const std::vector<char> all(n_, 1);
            plan(h, fresh_, all);
            for (int e = 0; e < n_; ++e) fresh_[e] = 0, wait_[e] = 0;
        }
        for (int e = 0; e < n_; ++e) {
            float* out = actions + size_t(e) * kAct;
            if (wait_[e]) {
                out[0] = out[1] = out[2] = out[3] = 0.0f, out[4] = -1.0f;
            } else {
                const float* a = plan_.data() + (size_t(e) * chunk_ + step_) * kAct;
                std::copy(a, a + kAct, out);
            }
        }
        step_ = (step_ + 1) % exec_;
    }

protected:
    // Fill plan_ rows (chunk x 5) of the envs with decide[e]; the rig has rendered.
    virtual void plan(const forge::Env::HostCopy& h, const std::vector<char>& new_episode,
                      const std::vector<char>& decide) = 0;
    const flux::BatchCamera& cams() const { return rig_->camera(); }
    int n_, chunk_, exec_, step_ = 0;
    std::vector<char> wait_, fresh_;  // waiting for the next decision; starts an episode at it
    std::vector<float> plan_;

private:
    std::unique_ptr<OpenArmCameraRig> rig_;
};

#ifdef CRUCIBLE_HAS_STUDENT
class StudentPolicy : public ChunkPolicy {
public:
    StudentPolicy(const forge::RobotModel& m, int n, const temper::Checkpoint& k, const temper::StudentConfig& sc,
                  std::vector<int> cams, int exec, std::string path)
        : ChunkPolicy(m, n, 2 * sc.size, sc.act_dim / kAct, exec), cfg_(sc), cams_(std::move(cams)),
          path_(std::move(path)), pol_(sc, n, 1), mean_(k.array("state_mean")), std_(k.array("state_std")) {
        pol_.load(k);
        CUDA_CHECK(cudaStreamCreate(&stream_));
        const size_t sone = size_t(sc.size) * sc.size * 3;
        full_.resize(sone * 4);
        simg_.resize(size_t(n) * cams_.size() * sone);
        st_.resize(size_t(n) * kState);
        out_.resize(size_t(n) * sc.act_dim);
        dimg_.resize(simg_.size());
        dst_.resize(st_.size());
        hid_.resize(size_t(n) * sc.hidden);
        CUDA_CHECK(cudaMemset(hid_.data(), 0, hid_.size() * sizeof(float)));
    }
    ~StudentPolicy() override { cudaStreamDestroy(stream_); }
    std::string describe() const override {
        std::string c;
        for (int k : cams_) c += (c.empty() ? "" : ",") + std::to_string(k);
        return "student " + path_ + " (CNN + GRU " + std::to_string(cfg_.hidden) + ", cameras " + c + " at " +
               std::to_string(cfg_.size) + "x" + std::to_string(cfg_.size) + ", chunk " + std::to_string(chunk_) +
               ", runs " + std::to_string(exec_) + " of each)";
    }

protected:
    void plan(const forge::Env::HostCopy& h, const std::vector<char>& new_episode,
              const std::vector<char>& decide) override {
        const int S = cfg_.size, C = int(cams_.size());
        const size_t sone = size_t(S) * S * 3;
        for (int e = 0; e < n_; ++e)
            for (int j = 0; j < C; ++j) {
                cams().copy_rgb(e, cams_[j], full_.data());
                halve(full_.data(), 2 * S, simg_.data() + (size_t(e) * C + j) * sone);
            }
        for (int e = 0; e < n_; ++e)
            for (int j = 0; j < kState; ++j)
                st_[size_t(e) * kState + j] = (h.obs[size_t(e) * Task::kObsDim + j] - mean_[j]) / std_[j];
        const size_t H = size_t(cfg_.hidden);
        for (int e = 0; e < n_; ++e)  // a new episode starts with an empty memory
            if (new_episode[e]) CUDA_CHECK(cudaMemsetAsync(hid_.data() + e * H, 0, H * sizeof(float), stream_));
        CUDA_CHECK(cudaMemcpyAsync(dimg_.data(), simg_.data(), simg_.size(), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(dst_.data(), st_.data(), st_.size() * 4, cudaMemcpyHostToDevice, stream_));
        const float* y = pol_.forward(dst_.data(), dimg_.data(), nullptr, nullptr, hid_.data(), 1, n_, stream_);
        CUDA_CHECK(cudaMemcpyAsync(out_.data(), y, out_.size() * 4, cudaMemcpyDeviceToHost, stream_));
        for (int e = 0; e < n_; ++e)  // only the deciding envs' memory moves on
            if (decide[e])
                CUDA_CHECK(cudaMemcpyAsync(hid_.data() + e * H, pol_.last_hidden() + e * H, H * sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        const size_t row = size_t(chunk_) * kAct;
        for (int e = 0; e < n_; ++e)
            if (decide[e]) std::copy(out_.begin() + e * row, out_.begin() + (e + 1) * row, plan_.begin() + e * row);
    }

private:
    temper::StudentConfig cfg_;
    std::vector<int> cams_;
    std::string path_;
    temper::StudentPolicy pol_;
    std::vector<float> mean_, std_, st_, out_;
    std::vector<uint8_t> full_, simg_;
    forge::DeviceBuffer<uint8_t> dimg_;
    forge::DeviceBuffer<float> dst_, hid_;
    cudaStream_t stream_ = nullptr;
};
#endif

#ifdef CRUCIBLE_HAS_ORT
// <onnx without .onnx>_openarm.txt: "cameras 0 1", "text ..."
void read_info(const std::string& path, std::vector<int>& cams, std::vector<float>& text) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path + " (written by vla_openarm.py --onnx)");
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "cameras") {
            int c;
            while (ss >> c) cams.push_back(c);
        } else if (key == "text") {
            float v;
            while (ss >> v) text.push_back(v);
        }
    }
    if (cams.empty() || text.empty()) throw std::runtime_error(path + ": needs 'cameras' and 'text'");
}

class VlaPolicy : public ChunkPolicy {
public:
    VlaPolicy(const forge::RobotModel& m, int n, std::unique_ptr<vla::Model> model, std::vector<int> cams,
              std::vector<float> text, int exec, std::string path)
        : ChunkPolicy(m, n, model->size, model->chunk, exec), model_(std::move(model)), cams_(std::move(cams)),
          path_(std::move(path)) {
        if (model_->cams != int(cams_.size()) || model_->text_dim != int(text.size()))
            throw std::runtime_error(path_ + ": the model does not match its _openarm.txt");
        const size_t one = size_t(model_->size) * model_->size * 3;
        img_.resize(size_t(n) * cams_.size() * one);
        st_.resize(size_t(n) * kState);
        tx_.resize(size_t(n) * text.size());
        for (int e = 0; e < n; ++e) std::copy(text.begin(), text.end(), tx_.begin() + size_t(e) * text.size());
        noise_.resize(size_t(n) * chunk_ * kAct);
        out_.resize(noise_.size());
    }
    std::string describe() const override {
        return "VLA " + path_ + " (" + std::to_string(cams_.size()) + " cameras " + std::to_string(model_->size) + "x" +
               std::to_string(model_->size) + ", chunk " + std::to_string(chunk_) + ", runs " + std::to_string(exec_) +
               " of each, fresh noise)";
    }

protected:
    void plan(const forge::Env::HostCopy& h, const std::vector<char>&, const std::vector<char>& decide) override {
        const int C = int(cams_.size());
        const size_t one = size_t(model_->size) * model_->size * 3;
        for (int e = 0; e < n_; ++e)
            for (int j = 0; j < C; ++j) cams().copy_rgb(e, cams_[j], img_.data() + (size_t(e) * C + j) * one);
        for (int e = 0; e < n_; ++e)
            std::copy(h.obs.begin() + size_t(e) * Task::kObsDim, h.obs.begin() + size_t(e) * Task::kObsDim + kState,
                      st_.begin() + size_t(e) * kState);
        for (auto& z : noise_) z = gauss_(rng_);
        model_->run(n_, img_.data(), st_.data(), tx_.data(), noise_.data(), out_.data());
        const size_t row = size_t(chunk_) * kAct;
        for (int e = 0; e < n_; ++e)
            if (decide[e]) std::copy(out_.begin() + e * row, out_.begin() + (e + 1) * row, plan_.begin() + e * row);
    }

private:
    std::unique_ptr<vla::Model> model_;
    std::vector<int> cams_;
    std::string path_;
    std::vector<uint8_t> img_;
    std::vector<float> st_, tx_, noise_, out_;
    std::mt19937_64 rng_{1};
    std::normal_distribution<float> gauss_{0.0f, 1.0f};
};
#endif

}  // namespace

std::unique_ptr<VisionPolicy> make_student_policy(const forge::RobotModel& m, int num_envs, const std::string& path) {
#ifdef CRUCIBLE_HAS_STUDENT
    const temper::Checkpoint k = temper::Checkpoint::load(path);
    if (k.task != Task::kName) throw std::runtime_error(path + ": a checkpoint for task '" + k.task + "'");
    const temper::StudentConfig sc = temper::StudentPolicy::config_from(k);
    auto it = k.layer_sizes.find("student_run");
    if (it == k.layer_sizes.end() || it->second.size() < 2)
        throw std::runtime_error(path + ": no student_run (cameras, exec): made by an older distill_app");
    std::vector<int> cams(it->second.begin(), it->second.end() - 1);
    return std::make_unique<StudentPolicy>(m, num_envs, k, sc, cams, it->second.back(), path);
#else
    (void)m, (void)num_envs, (void)path;
    throw std::runtime_error("--student needs a build with CUDA and cuDNN (temper's Conv2d)");
#endif
}

std::unique_ptr<VisionPolicy> make_vla_policy(const forge::RobotModel& m, int num_envs, const std::string& path,
                                              int exec) {
#ifdef CRUCIBLE_HAS_ORT
    const size_t dot = path.rfind(".onnx");
    std::vector<int> cams;
    std::vector<float> text;
    read_info((dot == std::string::npos ? path : path.substr(0, dot)) + "_openarm.txt", cams, text);
    auto model = std::make_unique<vla::Model>(path, false);
    return std::make_unique<VlaPolicy>(m, num_envs, std::move(model), cams, text, exec, path);
#else
    (void)m, (void)num_envs, (void)path, (void)exec;
    throw std::runtime_error("--vla needs a build with ONNX Runtime (-DONNXRUNTIME_DIR=...)");
#endif
}

}  // namespace sandbox
