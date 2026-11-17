#pragma once
// GRU layer (Phase 10: the CNN + GRU student policy), PyTorch's torch.nn.GRU (one
// layer, gate order r, z, n), over a whole sequence at once; backward is BPTT
// through all of it. Time major: x [T][B][in], y [T][B][H]. Per step, with
// hp = (1 - reset[t][b]) h_prev (an episode that starts at step t gets a zero state):
//   r = sigmoid(W_r x + b_r + U_r hp + c_r),  z = sigmoid(W_z x + b_z + U_z hp + c_z)
//   n = tanh(W_n x + b_n + r (U_n hp + c_n)),   h = (1 - z) n + z hp
// W_ih [3H][in], W_hh [3H][H], b_ih, b_hh [3H] (W, U, b, c above, rows r, z, n),
// initialized uniform in +-1/sqrt(H) (PyTorch's default). The answer key is
// temper::ref::gru_forward / gru_backward (reference_cpu.h, checked against PyTorch).
// One step at a time for acting: forward with T = 1 and the last h as h0.
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "temper/nn/tensor.h"

namespace temper {

class Gru {
public:
    Gru(int in, int hidden, uint64_t seed);
    // Buffers for sequences of up to max_tb = T * B steps.
    void allocate(int max_tb);

    // x: device [T][B][in], kept for backward (stay valid until then). h0: device
    // [B][H] or null (zeros). reset: device [T][B] (1 = episode starts at that step)
    // or null. y: device [T][B][H]. Asynchronous on 'stream'.
    void forward(const float* x, const float* h0, const float* reset, int T, int B, float* y, cudaStream_t stream);
    // dy: dL/dy [T][B][H]; dhT: dL/d(h after the last step) [B][H] or null. Fills
    // every parameter gradient (overwrites) and, when not null, dx [T][B][in] and
    // dh0 [B][H]. After a forward with the same T, B.
    void backward(const float* dy, const float* dhT, float* dx, float* dh0, cudaStream_t stream);

    void params(std::vector<Tensor*>& p, std::vector<Tensor*>& g) {
        p.insert(p.end(), {&W_ih, &W_hh, &b_ih, &b_hh});
        g.insert(g.end(), {&dW_ih, &dW_hh, &db_ih, &db_hh});
    }
    int in_dim() const { return in_; }
    int hidden() const { return H_; }

    Tensor W_ih, W_hh, b_ih, b_hh, dW_ih, dW_hh, db_ih, db_hh;

private:
    int in_, H_, max_tb_ = 0, T_ = 0, B_ = 0;
    const float* x_ = nullptr;
    const float* reset_ = nullptr;
    // per step: gx = x W_ih^T + b_ih, gh = hp W_hh^T + b_hh [TB][3H]; hp [TB][H];
    // backward: dgx, dgh [TB][3H], dh, dhp [B][H]
    Tensor gx_, gh_, hp_, dgx_, dgh_, dh_, dhp_;
};

}  // namespace temper
