#pragma once
// CNN + GRU student policy (Phase 10: distilled from the OpenArm VLA, then RL
// fine-tuned). Small and fast: camera images at 64 x 64 (the VLA's 128 x 128 halved).
//
//   per camera:  Conv 3->32 5x5/2, ReLU, Conv 32->64 3x3/2, ReLU, Conv 64->64 3x3/2,
//                ReLU, Conv 64->64 3x3/1, ReLU, Linear -> 256, LayerNorm (BcPolicy's)
//   state:       Linear state_dim -> 128, ReLU
//   core:        [cam 0 | cam 1 | state] -> Linear hidden, ReLU -> GRU hidden
//   head:        Linear hidden, ReLU, Linear act_dim, Tanh (distill_app: act_dim = an
//                action chunk, chunk x 5 numbers: the next steps' actions)
//
// Sequences, time major: frame (t, b) is row t * B + b of every input. The encoders
// see all T * B frames at once; the GRU runs through the T steps (temper::Gru, BPTT
// in backward). reset [T * B] (1 = an episode starts at that frame) zeroes the
// memory there. Acting: T = 1, h0 = each env's memory, then last_hidden(). Images
// go in as uint8 [T * B][cams][size][size][3]; shifts as BcPolicy (DrQ, training).
#include <cstdint>
#include <memory>
#include <vector>

#include "temper/common/checkpoint.h"
#include "temper/nn/gru.h"
#include "temper/nn/sequential.h"
#include "temper/nn/tensor.h"

namespace temper {

struct StudentConfig {
    int state_dim = 24;
    int cams = 2;
    int size = 64;     // at least 36
    int act_dim = 5;
    int hidden = 256;  // GRU size
};

class StudentPolicy {
public:
    static constexpr int kFeature = 256, kStateHidden = 128;
    static constexpr int kShiftPad = 4;

    // Buffers for up to max_tb frames per forward (T * B). Weights as PyTorch's
    // defaults, from 'seed'.
    StudentPolicy(const StudentConfig& cfg, int max_tb, uint64_t seed);

    // state: device [T * B][state_dim] (normalized); images: device uint8; shifts:
    // device int [T * B][cams][2] or null; reset: device [T * B] or null; h0: device
    // [B][hidden] or null (zeros). Returns the action means, device [T * B][act_dim],
    // valid until the next forward.
    const float* forward(const float* state, const uint8_t* images, const int* shifts, const float* reset,
                         const float* h0, int T, int B, cudaStream_t stream);
    const float* last_hidden() const;  // device [B][hidden]: the GRU state after the last forward's last step
    // dy = dL/d(action means), device [T * B][act_dim]. Fills every parameter gradient.
    void backward(const float* dy, cudaStream_t stream);

    void params(std::vector<Tensor*>& p, std::vector<Tensor*>& g);
    size_t num_params();
    const StudentConfig& config() const { return cfg_; }
    int max_tb() const { return max_tb_; }

    // Weights to / from checkpoint arrays "student.0", ... (params() order) and
    // layer_sizes["student"] = {state_dim, cams, size, act_dim, hidden}.
    void save(Checkpoint& c);
    void load(const Checkpoint& c);
    static StudentConfig config_from(const Checkpoint& c);

private:
    StudentConfig cfg_;
    int max_tb_, cat_dim_, T_ = 0, B_ = 0;
    std::vector<std::unique_ptr<Sequential>> enc_;
    Sequential state_, pre_, head_;
    Gru gru_;
    Tensor img_;                  // [cams * max_tb][3 * size * size]
    Tensor cat_, dcat_;           // [max_tb][cat_dim]
    Tensor gy_, dgy_, dpre_;      // GRU output and gradients [max_tb][hidden]
    Tensor denc_, dstate_;
};

}  // namespace temper
