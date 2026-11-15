#include "temper/nn/mlp.h"

#include <cmath>
#include <stdexcept>

namespace temper {

Mlp::Mlp(const std::vector<int>& dims, int max_batch, uint64_t seed) : dims_(dims), max_batch_(max_batch) {
    if (dims.size() < 2) throw std::runtime_error("Mlp: needs at least {in, out}");
    if (max_batch <= 0) throw std::runtime_error("Mlp: max_batch must be > 0");
    for (size_t i = 0; i + 1 < dims.size(); ++i) {
        layers_.push_back(std::make_unique<Linear>(dims[i], dims[i + 1], seed * 1000 + i));
        if (i + 2 < dims.size()) layers_.push_back(std::make_unique<Elu>(dims[i + 1]));
    }
    out_.resize(layers_.size());
    grad_.resize(layers_.size());
    for (size_t k = 0; k < layers_.size(); ++k) {
        out_[k].resize(max_batch, layers_[k]->out_dim());
        if (k > 0) grad_[k].resize(max_batch, layers_[k]->in_dim());
    }
}

void Mlp::init_orthogonal(float out_gain, uint64_t seed) {
    std::vector<Linear*> lin = linears();
    for (size_t i = 0; i < lin.size(); ++i)
        lin[i]->init_orthogonal(i + 1 == lin.size() ? out_gain : std::sqrt(2.0f), seed * 1000 + i);
}

const float* Mlp::forward(const float* x, int batch, cudaStream_t stream) {
    if (batch > max_batch_) throw std::runtime_error("Mlp::forward: batch larger than max_batch");
    const float* in = x;
    for (size_t k = 0; k < layers_.size(); ++k) {
        layers_[k]->forward(in, out_[k].data(), batch, stream);
        in = out_[k].data();
    }
    return in;
}

void Mlp::backward(const float* dy, int batch, cudaStream_t stream, float* dx) {
    if (batch > max_batch_) throw std::runtime_error("Mlp::backward: batch larger than max_batch");
    const float* g = dy;
    for (size_t k = layers_.size(); k-- > 0;) {
        float* gin = k > 0 ? grad_[k].data() : dx;
        layers_[k]->backward(g, gin, batch, stream);
        g = gin;
    }
}

void Mlp::params(std::vector<Tensor*>& p, std::vector<Tensor*>& g) {
    for (auto& l : layers_) l->params(p, g);
}

std::vector<Linear*> Mlp::linears() {
    std::vector<Linear*> v;
    for (auto& l : layers_)
        if (auto* lin = dynamic_cast<Linear*>(l.get())) v.push_back(lin);
    return v;
}

}  // namespace temper
