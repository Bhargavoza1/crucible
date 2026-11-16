#include "temper/algos/bc/bc_policy.h"

#include <stdexcept>
#include <string>

#include "temper/algos/bc/bc_ops.h"
#include "temper/nn/activations.h"
#include "temper/nn/conv2d.h"
#include "temper/nn/layernorm.h"
#include "temper/nn/linear.h"

namespace temper {

BcPolicy::BcPolicy(const BcPolicyConfig& cfg, int max_batch, uint64_t seed) : cfg_(cfg), max_batch_(max_batch) {
    if (cfg.state_dim <= 0 || cfg.act_dim <= 0 || cfg.cams < 0 || max_batch <= 0)
        throw std::runtime_error("BcPolicy: bad sizes");
    if (cfg.cams > 0 && cfg.size < 36) throw std::runtime_error("BcPolicy: images must be at least 36 x 36");
    uint64_t s = seed * 1000;
    for (int c = 0; c < cfg.cams; ++c) {
        auto e = std::make_unique<Sequential>();
        auto c1 = std::make_unique<Conv2d>(3, cfg.size, cfg.size, 32, 5, 2, 0, ++s);
        auto c2 = std::make_unique<Conv2d>(32, c1->out_height(), c1->out_width(), 64, 3, 2, 0, ++s);
        auto c3 = std::make_unique<Conv2d>(64, c2->out_height(), c2->out_width(), 64, 3, 2, 0, ++s);
        auto c4 = std::make_unique<Conv2d>(64, c3->out_height(), c3->out_width(), 64, 3, 1, 0, ++s);
        const int n1 = c1->out_dim(), n2 = c2->out_dim(), n3 = c3->out_dim(), n4 = c4->out_dim();
        e->add(std::move(c1));
        e->add(std::make_unique<Relu>(n1));
        e->add(std::move(c2));
        e->add(std::make_unique<Relu>(n2));
        e->add(std::move(c3));
        e->add(std::make_unique<Relu>(n3));
        e->add(std::move(c4));
        e->add(std::make_unique<Relu>(n4));
        e->add(std::make_unique<Linear>(n4, kFeature, ++s));
        e->add(std::make_unique<LayerNorm>(kFeature));
        e->allocate(max_batch);
        enc_.push_back(std::move(e));
    }
    state_.add(std::make_unique<Linear>(cfg.state_dim, kStateHidden, ++s));
    state_.add(std::make_unique<Relu>(kStateHidden));
    state_.allocate(max_batch);
    cat_dim_ = kFeature * cfg.cams + kStateHidden;
    head_.add(std::make_unique<Linear>(cat_dim_, kHidden, ++s));
    head_.add(std::make_unique<Relu>(kHidden));
    head_.add(std::make_unique<Linear>(kHidden, kHidden, ++s));
    head_.add(std::make_unique<Relu>(kHidden));
    head_.add(std::make_unique<Linear>(kHidden, cfg.act_dim, ++s));
    head_.add(std::make_unique<Tanh>(cfg.act_dim));
    head_.allocate(max_batch);
    if (cfg.cams > 0) img_.resize(cfg.cams * max_batch, 3 * cfg.size * cfg.size);
    cat_.resize(max_batch, cat_dim_);
    dcat_.resize(max_batch, cat_dim_);
    denc_.resize(max_batch, kFeature);
    dstate_.resize(max_batch, kStateHidden);
}

const float* BcPolicy::forward(const float* state, const uint8_t* images, const int* shifts, int batch,
                               cudaStream_t stream) {
    if (batch > max_batch_) throw std::runtime_error("BcPolicy: batch > max_batch");
    if (cfg_.cams > 0) {
        if (!images) throw std::runtime_error("BcPolicy: this policy needs images");
        bc::images_to_input(images, batch, cfg_.cams, cfg_.size, cfg_.size, shifts, kShiftPad, img_.data(), stream);
        const size_t per_cam = size_t(batch) * 3 * cfg_.size * cfg_.size;
        for (int c = 0; c < cfg_.cams; ++c) {
            const float* f = enc_[c]->forward(img_.data() + c * per_cam, batch, stream);
            bc::copy_columns(f, kFeature, 0, cat_.data(), cat_dim_, c * kFeature, batch, kFeature, stream);
        }
    }
    const float* h = state_.forward(state, batch, stream);
    bc::copy_columns(h, kStateHidden, 0, cat_.data(), cat_dim_, cfg_.cams * kFeature, batch, kStateHidden, stream);
    return head_.forward(cat_.data(), batch, stream);
}

void BcPolicy::backward(const float* dy, int batch, cudaStream_t stream) {
    head_.backward(dy, batch, stream, dcat_.data());
    for (int c = 0; c < cfg_.cams; ++c) {
        bc::copy_columns(dcat_.data(), cat_dim_, c * kFeature, denc_.data(), kFeature, 0, batch, kFeature, stream);
        enc_[c]->backward(denc_.data(), batch, stream, nullptr);
    }
    bc::copy_columns(dcat_.data(), cat_dim_, cfg_.cams * kFeature, dstate_.data(), kStateHidden, 0, batch,
                     kStateHidden, stream);
    state_.backward(dstate_.data(), batch, stream, nullptr);
}

void BcPolicy::params(std::vector<Tensor*>& p, std::vector<Tensor*>& g) {
    for (auto& e : enc_) e->params(p, g);
    state_.params(p, g);
    head_.params(p, g);
}

size_t BcPolicy::num_params() {
    std::vector<Tensor*> p, g;
    params(p, g);
    size_t n = 0;
    for (auto* t : p) n += t->size();
    return n;
}

void BcPolicy::save(Checkpoint& c) {
    c.layer_sizes["bc"] = {cfg_.state_dim, cfg_.cams, cfg_.size, cfg_.act_dim};
    std::vector<Tensor*> p, g;
    params(p, g);
    for (size_t k = 0; k < p.size(); ++k) c.arrays["bc." + std::to_string(k)] = p[k]->download();
}

BcPolicyConfig BcPolicy::config_from(const Checkpoint& c) {
    auto it = c.layer_sizes.find("bc");
    if (it == c.layer_sizes.end() || it->second.size() != 4) throw std::runtime_error("checkpoint has no bc policy");
    return BcPolicyConfig{it->second[0], it->second[1], it->second[2], it->second[3]};
}

void BcPolicy::load(const Checkpoint& c) {
    const BcPolicyConfig k = config_from(c);
    if (k.state_dim != cfg_.state_dim || k.cams != cfg_.cams || k.size != cfg_.size || k.act_dim != cfg_.act_dim)
        throw std::runtime_error("BcPolicy::load: checkpoint sizes do not match this policy");
    std::vector<Tensor*> p, g;
    params(p, g);
    for (size_t i = 0; i < p.size(); ++i) p[i]->upload(c.array("bc." + std::to_string(i)));
}

}  // namespace temper
