#include "temper/algos/student/student_policy.h"

#include <stdexcept>
#include <string>

#include "temper/algos/bc/bc_ops.h"
#include "temper/nn/activations.h"
#include "temper/nn/conv2d.h"
#include "temper/nn/layernorm.h"
#include "temper/nn/linear.h"

namespace temper {

StudentPolicy::StudentPolicy(const StudentConfig& cfg, int max_tb, uint64_t seed)
    : cfg_(cfg), max_tb_(max_tb), gru_(cfg.hidden, cfg.hidden, seed * 1000 + 999) {
    if (cfg.state_dim <= 0 || cfg.act_dim <= 0 || cfg.cams < 0 || cfg.hidden <= 0 || max_tb <= 0)
        throw std::runtime_error("StudentPolicy: bad sizes");
    if (cfg.cams > 0 && cfg.size < 36) throw std::runtime_error("StudentPolicy: images must be at least 36 x 36");
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
        e->allocate(max_tb);
        enc_.push_back(std::move(e));
    }
    state_.add(std::make_unique<Linear>(cfg.state_dim, kStateHidden, ++s));
    state_.add(std::make_unique<Relu>(kStateHidden));
    state_.allocate(max_tb);
    cat_dim_ = kFeature * cfg.cams + kStateHidden;
    pre_.add(std::make_unique<Linear>(cat_dim_, cfg.hidden, ++s));
    pre_.add(std::make_unique<Relu>(cfg.hidden));
    pre_.allocate(max_tb);
    gru_.allocate(max_tb);
    head_.add(std::make_unique<Linear>(cfg.hidden, cfg.hidden, ++s));
    head_.add(std::make_unique<Relu>(cfg.hidden));
    head_.add(std::make_unique<Linear>(cfg.hidden, cfg.act_dim, ++s));
    head_.add(std::make_unique<Tanh>(cfg.act_dim));
    head_.allocate(max_tb);
    if (cfg.cams > 0) img_.resize(cfg.cams * max_tb, 3 * cfg.size * cfg.size);
    cat_.resize(max_tb, cat_dim_);
    dcat_.resize(max_tb, cat_dim_);
    gy_.resize(max_tb, cfg.hidden);
    dgy_.resize(max_tb, cfg.hidden);
    dpre_.resize(max_tb, cfg.hidden);
    denc_.resize(max_tb, kFeature);
    dstate_.resize(max_tb, kStateHidden);
}

const float* StudentPolicy::forward(const float* state, const uint8_t* images, const int* shifts, const float* reset,
                                    const float* h0, int T, int B, cudaStream_t stream) {
    const int TB = T * B;
    if (TB > max_tb_) throw std::runtime_error("StudentPolicy: T * B > max_tb");
    T_ = T, B_ = B;
    if (cfg_.cams > 0) {
        if (!images) throw std::runtime_error("StudentPolicy: this policy needs images");
        bc::images_to_input(images, TB, cfg_.cams, cfg_.size, cfg_.size, shifts, kShiftPad, img_.data(), stream);
        const size_t per_cam = size_t(TB) * 3 * cfg_.size * cfg_.size;
        for (int c = 0; c < cfg_.cams; ++c) {
            const float* f = enc_[c]->forward(img_.data() + c * per_cam, TB, stream);
            bc::copy_columns(f, kFeature, 0, cat_.data(), cat_dim_, c * kFeature, TB, kFeature, stream);
        }
    }
    const float* h = state_.forward(state, TB, stream);
    bc::copy_columns(h, kStateHidden, 0, cat_.data(), cat_dim_, cfg_.cams * kFeature, TB, kStateHidden, stream);
    const float* p = pre_.forward(cat_.data(), TB, stream);
    gru_.forward(p, h0, reset, T, B, gy_.data(), stream);
    return head_.forward(gy_.data(), TB, stream);
}

const float* StudentPolicy::last_hidden() const {
    return gy_.data() + size_t(T_ > 0 ? T_ - 1 : 0) * B_ * cfg_.hidden;
}

void StudentPolicy::backward(const float* dy, cudaStream_t stream) {
    const int TB = T_ * B_;
    head_.backward(dy, TB, stream, dgy_.data());
    gru_.backward(dgy_.data(), nullptr, dpre_.data(), nullptr, stream);
    pre_.backward(dpre_.data(), TB, stream, dcat_.data());
    for (int c = 0; c < cfg_.cams; ++c) {
        bc::copy_columns(dcat_.data(), cat_dim_, c * kFeature, denc_.data(), kFeature, 0, TB, kFeature, stream);
        enc_[c]->backward(denc_.data(), TB, stream, nullptr);
    }
    bc::copy_columns(dcat_.data(), cat_dim_, cfg_.cams * kFeature, dstate_.data(), kStateHidden, 0, TB, kStateHidden,
                     stream);
    state_.backward(dstate_.data(), TB, stream, nullptr);
}

void StudentPolicy::params(std::vector<Tensor*>& p, std::vector<Tensor*>& g) {
    for (auto& e : enc_) e->params(p, g);
    state_.params(p, g);
    pre_.params(p, g);
    gru_.params(p, g);
    head_.params(p, g);
}

size_t StudentPolicy::num_params() {
    std::vector<Tensor*> p, g;
    params(p, g);
    size_t n = 0;
    for (auto* t : p) n += t->size();
    return n;
}

void StudentPolicy::save(Checkpoint& c) {
    c.layer_sizes["student"] = {cfg_.state_dim, cfg_.cams, cfg_.size, cfg_.act_dim, cfg_.hidden};
    std::vector<Tensor*> p, g;
    params(p, g);
    for (size_t k = 0; k < p.size(); ++k) c.arrays["student." + std::to_string(k)] = p[k]->download();
}

StudentConfig StudentPolicy::config_from(const Checkpoint& c) {
    auto it = c.layer_sizes.find("student");
    if (it == c.layer_sizes.end() || it->second.size() != 5)
        throw std::runtime_error("checkpoint has no student policy");
    const auto& v = it->second;
    return StudentConfig{v[0], v[1], v[2], v[3], v[4]};
}

void StudentPolicy::load(const Checkpoint& c) {
    const StudentConfig k = config_from(c);
    if (k.state_dim != cfg_.state_dim || k.cams != cfg_.cams || k.size != cfg_.size || k.act_dim != cfg_.act_dim ||
        k.hidden != cfg_.hidden)
        throw std::runtime_error("StudentPolicy::load: checkpoint sizes do not match this policy");
    std::vector<Tensor*> p, g;
    params(p, g);
    for (size_t i = 0; i < p.size(); ++i) p[i]->upload(c.array("student." + std::to_string(i)));
}

}  // namespace temper
