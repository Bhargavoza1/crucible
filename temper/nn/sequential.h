#pragma once
// A chain of layers (Phase 9d): out = L_n(... L_2(L_1(x))). Owns the layers and one
// output buffer and one gradient buffer per layer, sized for max_batch once.
//
//   Sequential s;
//   s.add(std::make_unique<Conv2d>(...));  s.add(std::make_unique<Relu>(n));  ...
//   s.allocate(max_batch);
//   const float* y = s.forward(x, batch, stream);   // x stays in use until backward
//   s.backward(dy, batch, stream, dx);              // dx may be null
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "temper/nn/layer.h"

namespace temper {

class Sequential {
public:
    void add(std::unique_ptr<Layer> layer) {
        if (!layers_.empty() && layer->in_dim() != layers_.back()->out_dim())
            throw std::runtime_error("Sequential: layer input " + std::to_string(layer->in_dim()) +
                                     " does not match the previous output " + std::to_string(layers_.back()->out_dim()));
        layers_.push_back(std::move(layer));
    }
    void allocate(int max_batch) {
        if (layers_.empty()) throw std::runtime_error("Sequential: no layers");
        max_batch_ = max_batch;
        out_.resize(layers_.size());
        grad_.resize(layers_.size());
        for (size_t k = 0; k < layers_.size(); ++k) {
            out_[k].resize(max_batch, layers_[k]->out_dim());
            grad_[k].resize(max_batch, layers_[k]->in_dim());  // dL/d(input of layer k); k = 0 unused
        }
    }
    const float* forward(const float* x, int batch, cudaStream_t stream) {
        check(batch);
        for (size_t k = 0; k < layers_.size(); ++k) {
            layers_[k]->forward(x, out_[k].data(), batch, stream);
            x = out_[k].data();
        }
        return x;
    }
    void backward(const float* dy, int batch, cudaStream_t stream, float* dx = nullptr) {
        check(batch);
        for (size_t k = layers_.size(); k-- > 0;) {
            float* d = k > 0 ? grad_[k].data() : dx;
            layers_[k]->backward(dy, d, batch, stream);
            dy = d;
        }
    }
    void params(std::vector<Tensor*>& p, std::vector<Tensor*>& g) {
        for (auto& l : layers_) l->params(p, g);
    }
    int in_dim() const { return layers_.front()->in_dim(); }
    int out_dim() const { return layers_.back()->out_dim(); }
    int max_batch() const { return max_batch_; }
    Layer& layer(size_t k) { return *layers_.at(k); }
    size_t size() const { return layers_.size(); }

private:
    void check(int batch) const {
        if (out_.empty()) throw std::runtime_error("Sequential: call allocate() first");
        if (batch > max_batch_) throw std::runtime_error("Sequential: batch " + std::to_string(batch) + " > max_batch");
    }
    std::vector<std::unique_ptr<Layer>> layers_;
    std::vector<Tensor> out_, grad_;
    int max_batch_ = 0;
};

}  // namespace temper
