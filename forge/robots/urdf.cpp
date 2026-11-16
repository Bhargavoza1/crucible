#include "forge/robots/urdf.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include <tinyxml2.h>

namespace forge {

namespace {

using tinyxml2::XMLElement;

struct Pose {
    Vec3 xyz{0, 0, 0};
    Quat rot = quat_identity();
};

struct Link {
    std::string name;
    float mass = 0.0f;
    Vec3 com{0, 0, 0};
    Mat3 inertia = mat3_zero();  // link frame, about com
    std::vector<RobotModel::Sphere> spheres;
    std::vector<RobotModel::Box> boxes;
    int other_shapes = 0;
};

struct Joint {
    std::string name, type, parent, child;
    Pose origin;
    Vec3 axis{1, 0, 0};
    float lower = 0, upper = 0, effort = 0, damping = 0;
    bool has_limit = false;
};

class Parser {
public:
    explicit Parser(std::string source) : src_(std::move(source)) {}

    [[noreturn]] void fail(const std::string& what) const { throw std::runtime_error("URDF " + src_ + ": " + what); }

    Vec3 parse_vec3(const char* text, Vec3 def, const std::string& where) const {
        if (!text) return def;
        std::istringstream in(text);
        Vec3 v;
        if (!(in >> v.x >> v.y >> v.z)) fail(where + ": expected 3 numbers, got '" + text + "'");
        return v;
    }

    float parse_float(const XMLElement* e, const char* attr, float def, bool required,
                      const std::string& where) const {
        const char* t = e ? e->Attribute(attr) : nullptr;
        if (!t) {
            if (required) fail(where + ": missing attribute '" + attr + "'");
            return def;
        }
        char* end = nullptr;
        const float v = std::strtof(t, &end);
        if (end == t || *end != '\0') fail(where + ": '" + attr + "' is not a number: '" + t + "'");
        return v;
    }

    Pose parse_origin(const XMLElement* parent, const std::string& where) const {
        Pose p;
        const XMLElement* o = parent->FirstChildElement("origin");
        if (!o) return p;
        p.xyz = parse_vec3(o->Attribute("xyz"), {0, 0, 0}, where + " origin xyz");
        const Vec3 rpy = parse_vec3(o->Attribute("rpy"), {0, 0, 0}, where + " origin rpy");
        p.rot = quat_from_rpy(rpy.x, rpy.y, rpy.z);
        return p;
    }

    Link parse_link(const XMLElement* e) const {
        Link l;
        const char* n = e->Attribute("name");
        if (!n) fail("<link> without a name");
        l.name = n;
        const std::string where = "link '" + l.name + "'";
        if (const XMLElement* in = e->FirstChildElement("inertial")) {
            const Pose o = parse_origin(in, where + " inertial");
            l.mass = parse_float(in->FirstChildElement("mass"), "value", 0, true, where + " mass");
            const XMLElement* ie = in->FirstChildElement("inertia");
            if (!ie) fail(where + ": <inertial> without <inertia>");
            const float ixx = parse_float(ie, "ixx", 0, true, where), ixy = parse_float(ie, "ixy", 0, false, where);
            const float ixz = parse_float(ie, "ixz", 0, false, where), iyy = parse_float(ie, "iyy", 0, true, where);
            const float iyz = parse_float(ie, "iyz", 0, false, where), izz = parse_float(ie, "izz", 0, true, where);
            const Mat3 I_frame{{ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz}};
            // The tensor is given in the (rotated) inertial frame: turn it into the link frame.
            l.inertia = rotate_inertia(quat_to_mat3(o.rot), I_frame);
            l.com = o.xyz;
        }
        for (const XMLElement* c = e->FirstChildElement("collision"); c; c = c->NextSiblingElement("collision")) {
            const XMLElement* g = c->FirstChildElement("geometry");
            const XMLElement* s = g ? g->FirstChildElement("sphere") : nullptr;
            const XMLElement* bx = g ? g->FirstChildElement("box") : nullptr;
            if (!s && !bx) {
                ++l.other_shapes;
                continue;
            }
            const Pose o = parse_origin(c, where + " collision");
            if (s) {
                l.spheres.push_back({-1, o.xyz, parse_float(s, "radius", 0, true, where + " sphere")});
            } else {
                const char* size = bx->Attribute("size");
                if (!size) fail(where + ": <box> without size");
                const Vec3 full = parse_vec3(size, {0, 0, 0}, where + " box size");
                if (!(full.x > 0.0f && full.y > 0.0f && full.z > 0.0f)) fail(where + ": box sizes must be > 0");
                l.boxes.push_back({-1, o.xyz, o.rot, full * 0.5f});
            }
        }
        return l;
    }

    Joint parse_joint(const XMLElement* e) const {
        Joint j;
        const char* n = e->Attribute("name");
        const char* t = e->Attribute("type");
        if (!n || !t) fail("<joint> needs a name and a type");
        j.name = n;
        j.type = t;
        const std::string where = "joint '" + j.name + "'";
        if (j.type != "revolute" && j.type != "continuous" && j.type != "prismatic" && j.type != "fixed")
            fail(where + ": joint type '" + j.type + "' is not supported");
        const XMLElement* p = e->FirstChildElement("parent");
        const XMLElement* c = e->FirstChildElement("child");
        if (!p || !p->Attribute("link") || !c || !c->Attribute("link")) fail(where + ": needs <parent link> and <child link>");
        j.parent = p->Attribute("link");
        j.child = c->Attribute("link");
        j.origin = parse_origin(e, where);
        if (const XMLElement* a = e->FirstChildElement("axis")) j.axis = parse_vec3(a->Attribute("xyz"), {1, 0, 0}, where + " axis");
        if (j.type != "fixed") {
            if (length(j.axis) < 1e-6f) fail(where + ": axis has zero length");
            j.axis = normalize(j.axis);
        }
        if (const XMLElement* lim = e->FirstChildElement("limit")) {
            j.lower = parse_float(lim, "lower", 0, false, where);
            j.upper = parse_float(lim, "upper", 0, false, where);
            j.effort = parse_float(lim, "effort", 0, false, where);
            j.has_limit = true;
        }
        if (const XMLElement* dyn = e->FirstChildElement("dynamics")) j.damping = parse_float(dyn, "damping", 0, false, where);
        return j;
    }

    RobotModel build(const tinyxml2::XMLDocument& doc, const UrdfOptions& opt) {
        const XMLElement* robot = doc.FirstChildElement("robot");
        if (!robot) fail("no <robot> element");
        RobotModel m;
        m.name = robot->Attribute("name") ? robot->Attribute("name") : "robot";

        std::vector<Link> links;
        std::map<std::string, int> link_index;
        for (const XMLElement* e = robot->FirstChildElement("link"); e; e = e->NextSiblingElement("link")) {
            Link l = parse_link(e);
            if (link_index.count(l.name)) fail("link '" + l.name + "' defined twice");
            link_index[l.name] = static_cast<int>(links.size());
            links.push_back(std::move(l));
        }
        if (links.empty()) fail("no links");

        std::vector<Joint> joints;
        std::map<std::string, int> joint_of_child;  // child link -> joint index
        for (const XMLElement* e = robot->FirstChildElement("joint"); e; e = e->NextSiblingElement("joint")) {
            Joint j = parse_joint(e);
            if (!link_index.count(j.parent)) fail("joint '" + j.name + "': unknown parent link '" + j.parent + "'");
            if (!link_index.count(j.child)) fail("joint '" + j.name + "': unknown child link '" + j.child + "'");
            if (joint_of_child.count(j.child)) fail("link '" + j.child + "' has two parent joints (loops are not supported)");
            joint_of_child[j.child] = static_cast<int>(joints.size());
            joints.push_back(std::move(j));
        }

        // Root: the one link that is nobody's child.
        int root = -1;
        for (size_t i = 0; i < links.size(); ++i)
            if (!joint_of_child.count(links[i].name)) {
                if (root >= 0) fail("more than one root link ('" + links[root].name + "', '" + links[i].name + "')");
                root = static_cast<int>(i);
            }
        if (root < 0) fail("no root link (the joints form a loop)");

        // Depth-first order; children in joint declaration order (like MuJoCo).
        std::vector<std::pair<int, int>> order;  // (link, joint index into the link, -1 for root)
        std::vector<int> stack_link{root}, stack_joint{-1};
        while (!stack_link.empty()) {
            const int l = stack_link.back(), j = stack_joint.back();
            stack_link.pop_back();
            stack_joint.pop_back();
            order.push_back({l, j});
            // Push children in reverse so the first declared is visited first.
            for (int k = static_cast<int>(joints.size()) - 1; k >= 0; --k)
                if (joints[k].parent == links[l].name) {
                    stack_link.push_back(link_index[joints[k].child]);
                    stack_joint.push_back(k);
                }
        }
        if (order.size() != links.size()) fail("some links are not connected to the root");

        std::map<int, int> body_of_link;
        int unsupported = 0;
        for (size_t b = 0; b < order.size(); ++b) {
            const auto [li, ji] = order[b];
            const Link& L = links[li];
            body_of_link[li] = static_cast<int>(b);
            m.body_name.push_back(L.name);
            m.mass.push_back(L.mass);
            for (float c : {L.com.x, L.com.y, L.com.z}) m.com.push_back(c);
            for (float x : L.inertia.m) m.inertia.push_back(x);
            for (auto s : L.spheres) {
                s.body = static_cast<int>(b);
                m.spheres.push_back(s);
            }
            for (auto bx : L.boxes) {
                bx.body = static_cast<int>(b);
                m.boxes.push_back(bx);
            }
            unsupported += L.other_shapes;

            int type = kJointFixed;
            Pose pose;
            Vec3 axis{1, 0, 0};
            float lo = 0, hi = 0, effort = 0, damping = 0;
            bool limited = false;
            std::string jname;
            if (ji < 0) {  // root link
                m.parent.push_back(-1);
                type = opt.floating_base ? kJointFree : kJointFixed;
                jname = opt.floating_base ? opt.free_joint_name : "";
            } else {
                const Joint& J = joints[ji];
                m.parent.push_back(body_of_link.at(link_index.at(J.parent)));
                type = J.type == "prismatic" ? kJointPrismatic : (J.type == "fixed" ? kJointFixed : kJointRevolute);
                pose = J.origin;
                axis = J.axis;
                lo = J.lower;
                hi = J.upper;
                effort = J.effort;
                damping = J.damping;
                limited = J.type != "continuous" && J.type != "fixed" && J.has_limit;
                jname = J.name;
            }
            m.jtype.push_back(type);
            m.joint_name.push_back(jname);
            for (float c : {pose.xyz.x, pose.xyz.y, pose.xyz.z}) m.body_pos.push_back(c);
            for (float c : {pose.rot.x, pose.rot.y, pose.rot.z, pose.rot.w}) m.body_quat.push_back(c);
            for (float c : {axis.x, axis.y, axis.z}) m.jaxis.push_back(c);
            m.joint_lower.push_back(lo);
            m.joint_upper.push_back(hi);
            m.joint_effort.push_back(effort);
            m.joint_damping.push_back(damping);
            m.joint_limited.push_back(limited);
            if (type == kJointFixed) {
                m.qadr.push_back(-1);
                m.dadr.push_back(-1);
            } else {
                m.qadr.push_back(m.nq);
                m.dadr.push_back(m.nv);
                for (int k = 0; k < joint_nv(type); ++k) m.dof_body.push_back(static_cast<int>(b));
                m.nq += joint_nq(type);
                m.nv += joint_nv(type);
            }
        }
        // Armature and damping per dof (free joint: none). Left empty when all zero.
        {
            std::vector<float> arm(m.nv, 0.0f), damp(m.nv, 0.0f);
            bool any = false;
            for (size_t b = 0; b < m.jtype.size(); ++b) {
                const int t = m.jtype[b];
                if (t != kJointRevolute && t != kJointPrismatic) continue;
                const float d = opt.damping >= 0.0f ? opt.damping : m.joint_damping[b];
                arm[m.dadr[b]] = opt.armature;
                damp[m.dadr[b]] = d;
                any = any || opt.armature != 0.0f || d != 0.0f;
            }
            if (any) {
                m.dof_armature = std::move(arm);
                m.dof_damping = std::move(damp);
            }
        }
        if (unsupported > 0)
            std::fprintf(stderr, "URDF %s: ignored %d collision shapes that are not spheres or boxes\n",
                         src_.c_str(), unsupported);
        m.validate();
        return m;
    }

private:
    std::string src_;
};

}  // namespace

Quat quat_from_rpy(float roll, float pitch, float yaw) {
    const Quat qx = quat_from_axis_angle({1, 0, 0}, roll);
    const Quat qy = quat_from_axis_angle({0, 1, 0}, pitch);
    const Quat qz = quat_from_axis_angle({0, 0, 1}, yaw);
    return normalize(qz * qy * qx);
}

RobotModel parse_urdf(const std::string& xml, const UrdfOptions& opt, const std::string& source) {
    tinyxml2::XMLDocument doc;
    if (doc.Parse(xml.c_str(), xml.size()) != tinyxml2::XML_SUCCESS)
        throw std::runtime_error("URDF " + source + ": XML error: " + (doc.ErrorStr() ? doc.ErrorStr() : "?"));
    return Parser(source).build(doc, opt);
}

RobotModel load_urdf(const std::string& path, const UrdfOptions& opt) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("URDF: cannot open '" + path + "'");
    std::stringstream ss;
    ss << f.rdbuf();
    return parse_urdf(ss.str(), opt, path);
}

}  // namespace forge
