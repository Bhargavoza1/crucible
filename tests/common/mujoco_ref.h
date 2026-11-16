#pragma once
// Reader for tests/data/mujoco_*.txt (written by tools/gen_mujoco_refs.py).
// Each line: a keyword, then numbers or names.
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace testutil {

struct MujocoCase {
    std::vector<double> qpos, qvel, tau, xpos, xquat, M, bias, qacc;
    // Scene files (constraints on): ctrl, every constraint row, constrained qacc.
    std::vector<double> ctrl, efc_type, efc_J, efc_pos, efc_aref, efc_R, efc_D, efc_force, qacc_constrained;
    int nefc = 0;
};

struct MujocoTrajPoint {
    int step;
    std::vector<double> qpos, qvel;
};

struct MujocoJoint {
    std::string name;
    int qadr, dadr, type;  // MuJoCo type: 0 free, 1 ball, 2 slide, 3 hinge
};

struct MujocoRef {
    std::string robot, urdf;
    bool floating = false;
    double armature = 0.0, damping = -1.0;  // "joint_extra" line: applied to every hinge / slide joint
    int nq = 0, nv = 0;
    std::vector<std::string> bodies;
    std::vector<MujocoJoint> joints;
    std::vector<MujocoCase> cases;
    double traj_dt = 0;
    int traj_steps = 0, traj_every = 0;
    std::vector<double> traj_qpos0, traj_qvel0;
    std::vector<MujocoTrajPoint> traj;
    // Scene files only.
    bool floor = false;
    bool cube = false;  // a free solid box "cube" added after the robot (add_free_box)
    double cube_half[3] = {0, 0, 0}, cube_mass = 0;
    double act_kp = 0, act_kv = 0, act_force = 0;  // PD actuators on these joints, if any
    std::vector<std::string> act_joints;
    std::vector<double> body_invweight0, dof_invweight0, traj_ctrl;
};

inline std::vector<double> read_numbers(std::istringstream& in) {
    std::vector<double> v;
    double x;
    while (in >> x) v.push_back(x);
    return v;
}

inline MujocoRef load_mujoco_ref(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path + " (run tools/gen_mujoco_refs.py)");
    MujocoRef r;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        std::string key;
        in >> key;
        if (key == "robot") in >> r.robot;
        else if (key == "urdf") in >> r.urdf;
        else if (key == "floating") { int x; in >> x; r.floating = x != 0; }
        else if (key == "joint_extra") in >> r.armature >> r.damping;
        else if (key == "nq") in >> r.nq;
        else if (key == "nv") in >> r.nv;
        else if (key == "bodies") {
            int n; in >> n;
            r.bodies.resize(n);
            for (auto& b : r.bodies) in >> b;
        } else if (key == "joints") {
            int n; in >> n;
            r.joints.resize(n);
            for (auto& j : r.joints) in >> j.name >> j.qadr >> j.dadr >> j.type;
        } else if (key == "case") r.cases.emplace_back();
        else if (key == "qpos") r.cases.back().qpos = read_numbers(in);
        else if (key == "qvel") r.cases.back().qvel = read_numbers(in);
        else if (key == "tau") r.cases.back().tau = read_numbers(in);
        else if (key == "xpos") r.cases.back().xpos = read_numbers(in);
        else if (key == "xquat") r.cases.back().xquat = read_numbers(in);
        else if (key == "M") r.cases.back().M = read_numbers(in);
        else if (key == "bias") r.cases.back().bias = read_numbers(in);
        else if (key == "qacc") r.cases.back().qacc = read_numbers(in);
        else if (key == "floor") { int x; in >> x; r.floor = x != 0; }
        else if (key == "cube") { r.cube = true; in >> r.cube_half[0] >> r.cube_half[1] >> r.cube_half[2] >> r.cube_mass; }
        else if (key == "actuators") {
            int n; in >> r.act_kp >> r.act_kv >> r.act_force >> n;
            r.act_joints.resize(n);
            for (auto& j : r.act_joints) in >> j;
        }
        else if (key == "body_invweight0") r.body_invweight0 = read_numbers(in);
        else if (key == "dof_invweight0") r.dof_invweight0 = read_numbers(in);
        else if (key == "ctrl") r.cases.back().ctrl = read_numbers(in);
        else if (key == "nefc") in >> r.cases.back().nefc;
        else if (key == "efc_type") r.cases.back().efc_type = read_numbers(in);
        else if (key == "efc_J") r.cases.back().efc_J = read_numbers(in);
        else if (key == "efc_pos") r.cases.back().efc_pos = read_numbers(in);
        else if (key == "efc_aref") r.cases.back().efc_aref = read_numbers(in);
        else if (key == "efc_R") r.cases.back().efc_R = read_numbers(in);
        else if (key == "efc_D") r.cases.back().efc_D = read_numbers(in);
        else if (key == "efc_force") r.cases.back().efc_force = read_numbers(in);
        else if (key == "qacc_constrained") r.cases.back().qacc_constrained = read_numbers(in);
        else if (key == "traj_ctrl") r.traj_ctrl = read_numbers(in);
        else if (key == "traj") in >> r.traj_dt >> r.traj_steps >> r.traj_every;
        else if (key == "traj_qpos0") r.traj_qpos0 = read_numbers(in);
        else if (key == "traj_qvel0") r.traj_qvel0 = read_numbers(in);
        else if (key == "traj_qpos") {
            MujocoTrajPoint p;
            in >> p.step;
            p.qpos = read_numbers(in);
            r.traj.push_back(p);
        } else if (key == "traj_qvel") {
            int step; in >> step;
            r.traj.back().qvel = read_numbers(in);
        } else if (key != "gravity") throw std::runtime_error(path + ": unknown key '" + key + "'");
    }
    return r;
}

}  // namespace testutil
