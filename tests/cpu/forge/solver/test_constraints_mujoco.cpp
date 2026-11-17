// Phase 6a done check: constraint rows, Newton solver and constrained steps vs
// real MuJoCo (tests/data/mujoco_scene_*.txt, written by tools/gen_mujoco_refs.py).
//
// Scenes: a ball on a floor (contact + friction), the PD quadruped on a floor
// (many contacts + actuators), arm3 (joint limits).
//
// Order of checks: the inputs of the solver first (invweight0, then every row:
// J, pos, R, D, aref), then its output (qacc, forces), then whole trajectories.
// If an input is wrong the later checks fail too; fix the first failure first.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/mujoco_setup.h"
#include "forge/dynamics/robot_work.h"
#include "forge/robots/actuators.h"
#include "forge/solver/constrained_dynamics.h"
#include "forge/solver/constraint_model.h"
#include "forge/solver/constraint_work.h"

using namespace forge;
using testutil::Setup;
using testutil::close;
using testutil::to_ours;

namespace {

constexpr double kRowTol = 1e-4;  // row inputs: same formulas, float vs double
// aref holds K imp pos with K = 1 / (dmax^2 tc^2) ~ 2770 (default solref/solimp):
// rounding qpos to float32 (~5e-8 m at the G1's 0.8 m height) moves it by ~1.4e-4.
constexpr double kArefTol = 3e-4;
// Row tolerances that depend on the contact distance: aref (via K imp pos) and R
// (via imp(pos): d ln R / d pos ~ 555 per m at the default solimp). Our float32
// forward kinematics puts bodies within ~2e-7 m of MuJoCo for the legged robots,
// but within 9.1e-7 m along the Panda's 9-body chain (ArticulatedVsMujoco.Panda:
// "worst xpos 9.1e-07"): K imp 9e-7 = 2.4e-3 for aref, 555 * 9e-7 = 5e-4 for R.
// The same scene's qacc and forces still match within 1e-4 (normwise).
struct RowTol {
    double aref, R;
};
RowTol row_tol(const std::string& scene) {
    if (scene == "panda_cube") return {2.5e-3, 5e-4};
    return {kArefTol, kRowTol};
}

// Our model of one scene + everything the solver needs.
struct Scene {
    Setup s;
    ConstraintModel cm;
    Actuators act;
    std::vector<int> act_mj;  // our actuator k -> MuJoCo actuator index (by joint name)

    explicit Scene(const std::string& name)
        : s(testutil::load_mujoco_setup("scene_" + name)), cm(s.model) {
        if (s.ref.floor) cm.add_ground();
        for (size_t k = 0; k < s.ref.act_joints.size(); ++k) {
            const float f = static_cast<float>(s.ref.act_force);
            act.add_position(s.model, s.ref.act_joints[k], static_cast<float>(s.ref.act_kp),
                             static_cast<float>(s.ref.act_kv), -f, f);
            act_mj.push_back(static_cast<int>(k));
        }
    }
    std::vector<float> ctrl(const std::vector<double>& mj) const {
        std::vector<float> c(act.nu());
        for (int k = 0; k < act.nu(); ++k) c[k] = static_cast<float>(mj[act_mj[k]]);
        return c;
    }
};

// Normwise relative error per vector: |a - b| / max(1, |b|).
double normwise(const std::vector<float>& ours, const std::vector<double>& mj_mapped) {
    double e = 0, n = 0;
    for (size_t i = 0; i < ours.size(); ++i) {
        e += (ours[i] - mj_mapped[i]) * (ours[i] - mj_mapped[i]);
        n += mj_mapped[i] * mj_mapped[i];
    }
    return std::sqrt(e) / std::max(1.0, std::sqrt(n));
}

std::vector<double> map_dofs(const std::vector<double>& mj, const std::vector<int>& d_mj) {
    std::vector<double> v(d_mj.size());
    for (size_t i = 0; i < d_mj.size(); ++i) v[i] = mj[d_mj[i]];
    return v;
}

class SceneTest : public ::testing::TestWithParam<std::string> {};

TEST_P(SceneTest, InvWeightMatchesMujoco) {
    Scene sc(GetParam());
    const auto& ref = sc.s.ref;
    for (int b = 0; b < sc.s.model.nbody(); ++b) {
        const int k = sc.s.b_mj[b];
        // A body with no mass (G1: IMU and camera mounts) has no center of mass;
        // MuJoCo then uses a point at twice the body's offset from its parent (an
        // import artifact), we use the body origin. Such bodies have no collision
        // shapes, so the value is never used: not compared.
        if (sc.s.model.mass[b] <= 0.0f) continue;
        for (int a = 0; a < 2; ++a)
            EXPECT_TRUE(close(sc.cm.body_invweight0[2 * b + a], ref.body_invweight0[2 * k + a], kRowTol))
                << "body " << sc.s.model.body_name[b] << (a ? " rot " : " tran ") << sc.cm.body_invweight0[2 * b + a]
                << " vs " << ref.body_invweight0[2 * k + a];
    }
    for (int i = 0; i < sc.s.model.nv; ++i)
        EXPECT_TRUE(close(sc.cm.dof_invweight0[i], ref.dof_invweight0[sc.s.d_mj[i]], kRowTol)) << "dof " << i;
}

TEST_P(SceneTest, RowsAndSolutionMatchMujoco) {
    Scene sc(GetParam());
    const ModelView m = sc.s.model.view();
    const ConstraintView cv = sc.cm.view();
    const ActuatorView av = sc.act.view();
    RobotWork work(sc.s.model);
    DynamicsWork w = work.view();
    ConstraintWorkHost cwork(sc.cm);
    ConstraintWork cw = cwork.view();
    const int nv = m.nv;
    double worst_qacc = 0, worst_force = 0, worst_row = 0;  // worst_row: J, pos, aref (relative to max(1, |x|))
    int max_iter = 0;
    auto rel = [](double a, double b) { return std::fabs(a - b) / std::max(1.0, std::fabs(b)); };

    for (size_t c = 0; c < sc.s.ref.cases.size(); ++c) {
        SCOPED_TRACE(GetParam() + " case " + std::to_string(c));
        const auto& rc = sc.s.ref.cases[c];
        const std::vector<float> qpos = to_ours(rc.qpos, sc.s.q_mj), qvel = to_ours(rc.qvel, sc.s.d_mj);
        const std::vector<float> ctrl = sc.ctrl(rc.ctrl);
        std::vector<float> qacc(nv);
        ASSERT_EQ(forward_dynamics_constrained(m, cv, av, qpos.data(), qvel.data(), nullptr,
                                               ctrl.empty() ? nullptr : ctrl.data(), w, cw, qacc.data()),
                  kDynamicsOk);
        ASSERT_EQ(cw.nefc, rc.nefc) << "number of constraint rows";
        for (int r = 0; r < cw.nefc; ++r) {
            EXPECT_EQ(cw.type[r], static_cast<int>(rc.efc_type[r])) << "row " << r;
            EXPECT_TRUE(close(cw.pos[r], rc.efc_pos[r], kRowTol)) << "pos row " << r << ": " << cw.pos[r] << " vs " << rc.efc_pos[r];
            for (int i = 0; i < nv; ++i) {
                const double mj = rc.efc_J[r * nv + sc.s.d_mj[i]];
                EXPECT_TRUE(close(cw.J[r * nv + i], mj, kRowTol)) << "J row " << r << " dof " << i << ": " << cw.J[r * nv + i] << " vs " << mj;
            }
            worst_row = std::max({worst_row, rel(cw.pos[r], rc.efc_pos[r]), rel(cw.aref[r], rc.efc_aref[r]),
                                  std::fabs(cw.R[r] / rc.efc_R[r] - 1.0)});
            for (int i = 0; i < nv; ++i) worst_row = std::max(worst_row, rel(cw.J[r * nv + i], rc.efc_J[r * nv + sc.s.d_mj[i]]));
            const RowTol tol = row_tol(GetParam());
            EXPECT_NEAR(cw.R[r] / rc.efc_R[r], 1.0, tol.R) << "R row " << r;
            EXPECT_NEAR(cw.D[r] / rc.efc_D[r], 1.0, tol.R) << "D row " << r;
            EXPECT_TRUE(close(cw.aref[r], rc.efc_aref[r], tol.aref)) << "aref row " << r << ": " << cw.aref[r] << " vs " << rc.efc_aref[r];
        }
        const double eq = normwise(qacc, map_dofs(rc.qacc_constrained, sc.s.d_mj));
        std::vector<float> f(cw.force, cw.force + cw.nefc);
        const double ef = normwise(f, rc.efc_force);
        worst_qacc = std::max(worst_qacc, eq);
        worst_force = std::max(worst_force, ef);
        max_iter = std::max(max_iter, cw.iterations);
        EXPECT_LE(eq, 1e-4) << "qacc";
        EXPECT_LE(ef, 1e-4) << "constraint forces";
    }
    std::printf("[ %s ] worst error: rows %.2e, qacc %.2e (normwise), forces %.2e (normwise); Newton iterations <= %d\n",
                GetParam().c_str(), worst_row, worst_qacc, worst_force, max_iter);
}

// The solver's answer is the true minimum (KKT conditions), checked without MuJoCo:
//   M qacc - qfrc_smooth = J^T f,  f >= 0,  f = -D (J qacc - aref) on rows with J qacc < aref, else 0.
TEST_P(SceneTest, NewtonSatisfiesOptimality) {
    Scene sc(GetParam());
    const ModelView m = sc.s.model.view();
    const ConstraintView cv = sc.cm.view();
    RobotWork work(sc.s.model);
    DynamicsWork w = work.view();
    ConstraintWorkHost cwork(sc.cm);
    ConstraintWork cw = cwork.view();
    const int nv = m.nv;
    for (size_t c = 0; c < sc.s.ref.cases.size(); ++c) {
        const auto& rc = sc.s.ref.cases[c];
        const std::vector<float> qpos = to_ours(rc.qpos, sc.s.q_mj), qvel = to_ours(rc.qvel, sc.s.d_mj);
        const std::vector<float> ctrl = sc.ctrl(rc.ctrl);
        std::vector<float> qacc(nv);
        ASSERT_EQ(forward_dynamics_constrained(m, cv, sc.act.view(), qpos.data(), qvel.data(), nullptr,
                                               ctrl.empty() ? nullptr : ctrl.data(), w, cw, qacc.data()),
                  kDynamicsOk);
        double res = 0, scale = 1;
        for (int i = 0; i < nv; ++i) {
            double s = -cw.qfrc_smooth[i];
            for (int j = 0; j < nv; ++j) s += double(work.M[i * nv + j]) * qacc[j];
            for (int r = 0; r < cw.nefc; ++r) s -= double(cw.J[r * nv + i]) * cw.force[r];
            res = std::max(res, std::fabs(s));
            scale = std::max(scale, double(std::fabs(cw.qfrc_smooth[i])));
        }
        for (int r = 0; r < cw.nefc; ++r) {
            EXPECT_GE(cw.force[r], 0.0f);
            scale = std::max(scale, double(cw.force[r]));
        }
        EXPECT_LE(res / scale, 1e-4) << GetParam() << " case " << c << ": M qacc - qfrc_smooth - J^T f = " << res;
    }
}

// Whole trajectory, 1 s at dt = 0.002 (500 steps), vs MuJoCo's mj_step.
TEST_P(SceneTest, TrajectoryMatchesMujoco) {
    Scene sc(GetParam());
    const auto& ref = sc.s.ref;
    const ModelView m = sc.s.model.view();
    const ConstraintView cv = sc.cm.view();
    const ActuatorView av = sc.act.view();
    RobotWork work(sc.s.model);
    DynamicsWork w = work.view();
    ConstraintWorkHost cwork(sc.cm);
    ConstraintWork cw = cwork.view();
    std::vector<float> qpos = to_ours(ref.traj_qpos0, sc.s.q_mj), qvel = to_ours(ref.traj_qvel0, sc.s.d_mj);
    std::vector<float> qacc(m.nv);
    const std::vector<float> ctrl = sc.ctrl(ref.traj_ctrl);
    double worst_q = 0, worst_v = 0;
    size_t next = 0;
    for (int step = 1; step <= ref.traj_steps; ++step) {
        ASSERT_EQ(step_constrained(m, cv, av, qpos.data(), qvel.data(), nullptr, ctrl.empty() ? nullptr : ctrl.data(),
                                   w, cw, qacc.data(), static_cast<float>(ref.traj_dt)),
                  kDynamicsOk);
        if (next < ref.traj.size() && ref.traj[next].step == step) {
            const auto& p = ref.traj[next++];
            // Positions: compare per qpos entry (absolute, m or rad); a quaternion q and -q agree.
            for (int i = 0; i < m.nq; ++i) worst_q = std::max(worst_q, std::fabs(qpos[i] - p.qpos[sc.s.q_mj[i]]));
            worst_v = std::max(worst_v, normwise(qvel, map_dofs(p.qvel, sc.s.d_mj)));
        }
    }
    std::printf("[ %s ] 1 s trajectory: worst |qpos - mujoco| %.2e, worst normwise qvel %.2e\n",
                GetParam().c_str(), worst_q, worst_v);
    EXPECT_LE(worst_q, 1e-3);
    EXPECT_LE(worst_v, 1e-2);
}

INSTANTIATE_TEST_SUITE_P(Mujoco, SceneTest, ::testing::Values("ball", "quadruped", "arm3", "g1_12dof", "go2", "ball_cube", "panda_cube",
                                                          "openarm", "openarm_cube", "g1_fallen", "go2_fallen"),
                         [](const ::testing::TestParamInfo<std::string>& p) { return p.param; });

}  // namespace
