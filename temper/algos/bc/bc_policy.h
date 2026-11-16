#pragma once
// Behavior-cloning policy for panda_pick (Phase 9d): the same network as
// tools/bc_train.py, so the two can be compared on the same data.
//
//   per camera:  Conv 3->32 5x5/2, ReLU, Conv 32->64 3x3/2, ReLU, Conv 64->64 3x3/2,
//                ReLU, Conv 64->64 3x3/1, ReLU, Linear -> 256, LayerNorm
//   state:       Linear state_dim -> 128, ReLU
//   head:        [cam 0 | cam 1 | ... | state] -> Linear 512, ReLU, Linear 512, ReLU,
//                Linear act_dim, Tanh
// cams = 0: state only (no encoders). Images go in as uint8 straight from the demo
// file or the cameras; the policy turns them into floats (and shifts them when
// training, DrQ). Needs cuDNN (Conv2d).
#include <cstdint>
#include <memory>
#include <vector>

#include "temper/common/checkpoint.h"
#include "temper/nn/sequential.h"
#include "temper/nn/tensor.h"

namespace temper {

struct BcPolicyConfig {
    int state_dim = 24;
    int cams = 2;      // 0 = no images
    int size = 64;     // images are size x size (at least 36 for this encoder)
    int act_dim = 5;
};

class BcPolicy {
public:
    static constexpr int kFeature = 256, kStateHidden = 128, kHidden = 512;
    static constexpr int kShiftPad = 4;  // DrQ: random shifts of up to 4 pixels

    // Weights as PyTorch's defaults (uniform +-1/sqrt(fan_in)), from 'seed'.
    BcPolicy(const BcPolicyConfig& cfg, int max_batch, uint64_t seed);

    // state: device [batch][state_dim] (already normalized). images: device uint8
    // [batch][cams][size][size][3] (null when cams = 0). shifts: device int
    // [batch][cams][2] in [0, 2 kShiftPad], or null for no shift. Returns the actions,
    // device [batch][act_dim], valid until the next forward.
    const float* forward(const float* state, const uint8_t* images, const int* shifts, int batch, cudaStream_t stream);
    // dy = dL/d(actions), device [batch][act_dim]. Fills every parameter gradient.
    void backward(const float* dy, int batch, cudaStream_t stream);

    void params(std::vector<Tensor*>& p, std::vector<Tensor*>& g);
    size_t num_params();
    const BcPolicyConfig& config() const { return cfg_; }
    int max_batch() const { return max_batch_; }

    // Weights to / from checkpoint arrays "bc.0", "bc.1", ... (in params() order) and
    // layer_sizes["bc"] = {state_dim, cams, size, act_dim}. load() checks the sizes.
    void save(Checkpoint& c);
    void load(const Checkpoint& c);
    static BcPolicyConfig config_from(const Checkpoint& c);

private:
    BcPolicyConfig cfg_;
    int max_batch_, cat_dim_;
    std::vector<std::unique_ptr<Sequential>> enc_;
    Sequential state_, head_;
    Tensor img_;           // [cams * max_batch][3 * size * size]
    Tensor cat_, dcat_;    // [max_batch][cat_dim]
    Tensor denc_, dstate_; // gradient of one encoder's output, of the state branch's
};

}  // namespace temper
