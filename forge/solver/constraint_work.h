#pragma once
// Host storage for ConstraintWork (one env). For the CPU answer key and tests.
#include <vector>

#include "forge/solver/constraint_model.h"
#include "forge/solver/constraints.h"

namespace forge {

class ConstraintWorkHost {
public:
    ConstraintWorkHost(int nv, int max_rows)
        : type(max_rows), id(max_rows), J(size_t(max_rows) * nv), pos(max_rows), aref(max_rows), R(max_rows),
          D(max_rows), jar(max_rows), Jd(max_rows), force(max_rows), state(max_rows), qfrc_smooth(nv),
          qacc_smooth(nv), H(size_t(nv) * nv), LH(size_t(nv) * nv), grad(nv), dir(nv), Ma(nv), Mdir(nv) {}
    explicit ConstraintWorkHost(const ConstraintModel& c) : ConstraintWorkHost(c.nv, c.max_rows()) {}

    ConstraintWork view() {
        return {0,           0,          type.data(), id.data(),   J.data(),          pos.data(),
                aref.data(), R.data(),   D.data(),    jar.data(),  Jd.data(),         force.data(),
                state.data(), qfrc_smooth.data(), qacc_smooth.data(), H.data(), LH.data(), grad.data(),
                dir.data(),  Ma.data(),  Mdir.data()};
    }

    std::vector<int> type, id;
    std::vector<float> J, pos, aref, R, D, jar, Jd, force;
    std::vector<int> state;
    std::vector<float> qfrc_smooth, qacc_smooth, H, LH, grad, dir, Ma, Mdir;
};

}  // namespace forge
