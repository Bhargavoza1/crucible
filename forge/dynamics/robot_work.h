#pragma once
// Host storage for DynamicsWork (one robot). For the CPU answer key and tests.
#include <vector>

#include "forge/dynamics/articulated.h"
#include "forge/robots/model.h"

namespace forge {

class RobotWork {
public:
    explicit RobotWork(const RobotModel& m)
        : xpos(3 * m.nbody()), xquat(4 * m.nbody()), S(m.nv), I(m.nbody()), Ic(m.nbody()), V(m.nbody()),
          A(m.nbody()), F(m.nbody()), M(m.nv * m.nv), L(m.nv * m.nv), bias(m.nv) {}

    DynamicsWork view() {
        return {xpos.data(), xquat.data(), S.data(), I.data(), Ic.data(), V.data(),
                A.data(),    F.data(),     M.data(), L.data(), bias.data()};
    }

    std::vector<float> xpos, xquat;
    std::vector<SpatialVec> S;
    std::vector<SpatialInertia> I, Ic;
    std::vector<SpatialVec> V, A, F;
    std::vector<float> M, L, bias;
};

}  // namespace forge
