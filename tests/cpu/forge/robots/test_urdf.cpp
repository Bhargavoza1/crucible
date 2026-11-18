#include <cmath>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "common/test_helpers.h"
#include "forge/dynamics/articulated.h"
#include "forge/dynamics/robot_work.h"
#include "forge/robots/skeleton.h"
#include "forge/robots/urdf.h"

using namespace forge;
using testutil::expect_vec_near;

namespace {

std::string asset(const std::string& name) { return std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/" + name; }

// Wraps links and joints in a <robot> element.
std::string robot(const std::string& body) { return "<robot name='t'>" + body + "</robot>"; }

const char* kLink = "<inertial><mass value='1'/><inertia ixx='1' iyy='1' izz='1'/></inertial>";

std::string link(const std::string& name) { return "<link name='" + name + "'>" + kLink + "</link>"; }

std::string joint(const std::string& name, const std::string& type, const std::string& parent, const std::string& child) {
    return "<joint name='" + name + "' type='" + type + "'><parent link='" + parent + "'/><child link='" + child +
           "'/></joint>";
}

}  // namespace

TEST(Urdf, RpyIsFixedAxisXThenYThenZ) {
    // Yaw 90 deg: x axis goes to y.
    expect_vec_near(rotate(quat_from_rpy(0, 0, 1.5707963f), {1, 0, 0}), {0, 1, 0});
    // General case: R = Rz(yaw) Ry(pitch) Rx(roll).
    const float r = 0.3f, p = -0.7f, y = 1.1f;
    const Mat3 expected = quat_to_mat3(quat_from_axis_angle({0, 0, 1}, y)) *
                          quat_to_mat3(quat_from_axis_angle({0, 1, 0}, p)) *
                          quat_to_mat3(quat_from_axis_angle({1, 0, 0}, r));
    testutil::expect_mat_near(quat_to_mat3(quat_from_rpy(r, p, y)), expected, 1e-6f);
}

TEST(Urdf, PendulumStructure) {
    const RobotModel m = load_urdf(asset("pendulum.urdf"));
    ASSERT_EQ(m.nbody(), 2);
    EXPECT_EQ(m.body_name[0], "base");
    EXPECT_EQ(m.parent[0], -1);
    EXPECT_EQ(m.jtype[0], kJointFixed);  // fixed base: root welded to the world
    EXPECT_EQ(m.jtype[1], kJointRevolute);
    EXPECT_EQ(m.joint_name[1], "hinge");
    EXPECT_EQ(m.nq, 1);
    EXPECT_EQ(m.nv, 1);
    EXPECT_FALSE(m.joint_limited[1]);  // continuous
    EXPECT_FLOAT_EQ(m.body_pos[3 * 1 + 2], 2.0f);
    ASSERT_EQ(m.spheres.size(), 1u);
    EXPECT_EQ(m.spheres[0].body, 1);
    EXPECT_FLOAT_EQ(m.spheres[0].radius, 0.1f);
}

TEST(Urdf, FloatingBaseAddsFreeJoint) {
    UrdfOptions opt;
    opt.floating_base = true;
    const RobotModel m = load_urdf(asset("quadruped.urdf"), opt);
    EXPECT_EQ(m.jtype[0], kJointFree);
    EXPECT_EQ(m.joint_name[0], "root");
    EXPECT_EQ(m.nq, 7 + 8);
    EXPECT_EQ(m.nv, 6 + 8);
    const auto q0 = m.qpos0();
    EXPECT_FLOAT_EQ(q0[3], 1.0f);  // qw = 1 (w first, like MuJoCo)
}

// Depth first, children in joint order: the same order MuJoCo uses.
TEST(Urdf, BodyOrderMatchesMujoco) {
    const RobotModel m = load_urdf(asset("quadruped.urdf"), {true});
    const char* expected[] = {"torso", "imu", "fl_thigh", "fl_shin", "fr_thigh", "fr_shin",
                              "bl_thigh", "bl_shin", "br_thigh", "br_shin"};
    ASSERT_EQ(m.nbody(), 10);
    for (int b = 0; b < 10; ++b) EXPECT_EQ(m.body_name[b], expected[b]);
    for (int b = 1; b < m.nbody(); ++b) EXPECT_LT(m.parent[b], b);  // topological order
}

TEST(Urdf, LimitsAndAxes) {
    const RobotModel m = load_urdf(asset("arm3.urdf"));
    const int j2 = m.joint_index("j2"), j3 = m.joint_index("j3");
    EXPECT_EQ(m.jtype[j2], kJointPrismatic);
    EXPECT_TRUE(m.joint_limited[j2]);
    EXPECT_FLOAT_EQ(m.joint_lower[j2], -0.5f);
    EXPECT_FLOAT_EQ(m.joint_upper[j2], 0.5f);
    const Vec3 a{m.jaxis[3 * j3], m.jaxis[3 * j3 + 1], m.jaxis[3 * j3 + 2]};
    EXPECT_NEAR(length(a), 1.0f, 1e-6f);  // normalized
    EXPECT_EQ(m.jtype[m.joint_index("tool_mount")], kJointFixed);
}

// Inertia given in a rotated inertial frame must be turned into the link frame:
// still symmetric, same trace (rotation keeps the trace).
TEST(Urdf, RotatedInertiaKeepsTrace) {
    const RobotModel m = load_urdf(asset("arm3.urdf"));
    const int b = m.body_index("link1");
    const float* I = &m.inertia[9 * b];
    EXPECT_NEAR(I[0] + I[4] + I[8], 0.012f + 0.011f + 0.004f, 1e-7f);
    EXPECT_FLOAT_EQ(I[1], I[3]);
    EXPECT_FLOAT_EQ(I[2], I[6]);
    EXPECT_NE(I[1], 0.001f);  // really rotated, not copied
}

TEST(Urdf, ErrorsAreClear) {
    auto expect_error = [](const std::string& xml, const std::string& part) {
        try {
            parse_urdf(xml, {}, "test.urdf");
            ADD_FAILURE() << "expected an error containing '" << part << "'";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find(part), std::string::npos) << e.what();
        }
    };
    expect_error("<notarobot/>", "no <robot>");
    expect_error(robot(link("a") + joint("j", "revolute", "a", "missing")), "unknown child link");
    expect_error(robot(link("a") + link("b")), "more than one root");
    expect_error(robot(link("a") + link("b") + joint("j", "floating", "a", "b")), "not supported");
    expect_error(robot(link("a") + link("b") + link("c") + joint("j1", "fixed", "a", "c") + joint("j2", "fixed", "b", "c")),
                 "two parent joints");
    expect_error(robot("<link name='a'><inertial><mass value='x'/><inertia ixx='1' iyy='1' izz='1'/></inertial></link>"),
                 "not a number");
    expect_error("<robot><link", "XML error");
}

TEST(Urdf, ValidateCatchesBrokenModel) {
    RobotModel m = load_urdf(asset("arm3.urdf"));
    m.parent[1] = 3;  // child before parent
    EXPECT_THROW(m.validate(), std::runtime_error);
}

// The skeleton of arm3 has a rail for the slider j2: points fixed to j2's parent
// that cover the slider's range, so the child body is always drawn on its rail.
TEST(Skeleton, PrismaticJointGetsRail) {
    const RobotModel m = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/arm3.urdf");
    const int j2 = m.joint_index("j2");
    ASSERT_GE(j2, 0);
    const int parent = m.parent[j2];
    const std::vector<SkeletonPoint> pts = make_skeleton(m);
    // Joint frame origin and slide axis in the parent frame.
    const Quat qb{m.body_quat[4 * j2], m.body_quat[4 * j2 + 1], m.body_quat[4 * j2 + 2], m.body_quat[4 * j2 + 3]};
    const Vec3 o{m.body_pos[3 * j2], m.body_pos[3 * j2 + 1], m.body_pos[3 * j2 + 2]};
    const Vec3 axis = rotate(qb, Vec3{m.jaxis[3 * j2], m.jaxis[3 * j2 + 1], m.jaxis[3 * j2 + 2]});
    float lo = 1e9f, hi = -1e9f;
    int on_rail = 0;
    for (const auto& p : pts) {
        if (p.body != parent || p.radius != 0.012f) continue;
        const Vec3 r = Vec3{p.local[0], p.local[1], p.local[2]} - o;
        const float s = dot(r, axis);
        EXPECT_NEAR(length(r - axis * s), 0.0f, 1e-5f) << "rail point off the axis";
        lo = std::min(lo, s);
        hi = std::max(hi, s);
        ++on_rail;
    }
    EXPECT_GE(on_rail, 10);
    EXPECT_NEAR(lo, m.joint_lower[j2], 1e-5f);
    EXPECT_NEAR(hi, m.joint_upper[j2], 1e-5f);
}

TEST(Urdf, DefaultSandboxArmHasOnlyLimitedHinges) {
    const RobotModel m = load_urdf(std::string(CRUCIBLE_SOURCE_DIR) + "/assets/robots/arm.urdf");
    EXPECT_EQ(m.nv, 3);
    for (int b = 0; b < m.nbody(); ++b) {
        if (m.parent[b] < 0) continue;
        EXPECT_EQ(m.jtype[b], kJointRevolute) << m.joint_name[b];
        EXPECT_TRUE(m.joint_limited[b]) << m.joint_name[b];
    }
}

// Phase 10: G1 and Go2 collide with spheres on the feet only plus capsules
// elsewhere; UrdfOptions::capsules = false (the training tasks) drops the capsules.
TEST(Urdf, UnitreeFeetAreTheOnlySpheresCapsulesOptional) {
    for (const auto& [file, feet] : {std::pair<std::string, size_t>{"g1/g1_12dof.urdf", 8}, {"go2/go2.urdf", 4}}) {
        UrdfOptions u;
        u.floating_base = true;
        const RobotModel with = load_urdf(asset(file), u);
        u.capsules = false;
        const RobotModel without = load_urdf(asset(file), u);
        EXPECT_EQ(with.spheres.size(), feet) << file;
        EXPECT_GT(with.capsules.size(), 10u) << file;
        EXPECT_EQ(without.spheres.size(), feet) << file;
        EXPECT_TRUE(without.capsules.empty()) << file;
    }
}

// Phase 11: the two-arm cell (tools/openarm_cell_to_urdf.py --bimanual): 18 joints (7 +
// 2 fingers per arm), 6 pad spheres per hand, the same links as openarm_cell.urdf
// (so openarm_cell.cmesh draws it). At the cell's home keyframe (joint 4 = 90 degrees,
// right fingers -45 degrees, left 0) the two hands mirror each other across y = 0.
TEST(Urdf, OpenArmBimanualCell) {
    const RobotModel m = load_urdf(asset("openarm/openarm_cell_bimanual.urdf"));
    const RobotModel one = load_urdf(asset("openarm/openarm_cell.urdf"));
    EXPECT_EQ(m.nq, 18);
    EXPECT_EQ(m.spheres.size(), 12u);
    EXPECT_EQ(m.nbody(), one.nbody());
    for (int b = 0; b < m.nbody(); ++b) EXPECT_EQ(m.body_name[size_t(b)], one.body_name[size_t(b)]);
    std::vector<float> q = m.qpos0();
    auto set = [&](const std::string& j, float v) { q[size_t(m.qadr[m.body_index(j)])] = v; };
    set("openarm_left_link4", 1.5708f);
    set("openarm_right_link4", 1.5708f);
    set("openarm_right_ee_inner_finger", -0.7854f);
    set("openarm_right_ee_outer_finger", -0.7854f);
    RobotWork work(m);
    DynamicsWork w = work.view();
    forward_kinematics(m.view(), q.data(), w);
    const Vec3 tcp{-0.00143f, 0.0f, -0.1635f};  // finger-pad center, hand frame
    auto tcp_of = [&](const char* hand) {
        const int b = m.body_index(hand);
        return load3(w.xpos + 3 * b) + rotate(load_quat(w.xquat + 4 * b), tcp);
    };
    const Vec3 l = tcp_of("openarm_left_ee_base_link"), r = tcp_of("openarm_right_ee_base_link");
    std::printf("bimanual cell home TCPs: left %.4f %.4f %.4f, right %.4f %.4f %.4f\n", l.x, l.y, l.z, r.x, r.y, r.z);
    EXPECT_NEAR(l.x, r.x, 1e-3f);
    EXPECT_NEAR(l.y, -r.y, 1e-3f);
    EXPECT_NEAR(l.z, r.z, 1e-3f);
    EXPECT_GT(l.y, 0.1f) << "the left hand is on the +y side";
}
