"""enactic openarm_description (Apache-2) -> assets/robots/openarm/openarm.urdf (Phase 10).

One OpenArm v2.0 arm (the right one) with the pinch gripper, mounted on a post so
its shoulder is MOUNT_Z above the floor (the floor is the table). Input: the URDF
that openarm_description's xacro makes for the preset v2/right_arm_with_pinch_gripper
(see README there), and the openarm_description folder (for the collision meshes):

    git clone https://github.com/enactic/openarm_description
    # xacro needs ROS; without ROS see the note at the end of this docstring
    xacro openarm_description/urdf/openarm.urdf.xacro robot_preset:=v2/right_arm_with_pinch_gripper \\
        ros2_control:=false > openarm_right.urdf
    python tools/openarm_to_urdf.py openarm_right.urdf openarm_description

Kept as given: link masses, inertias, joint poses, axes, ranges, efforts.
Changes:
  * collision: every link's mesh becomes one CYLINDER fitted to the collision mesh
    (axis = the mesh's main direction, radius = 97th percentile of the vertex
    distance to the axis, length = the rest of the extent). crucible treats a
    <cylinder> as a capsule (a cylinder with round ends; see forge/robots/urdf.h).
  * fingers: the upper finger (finger_*_03 part) is a cylinder too; the pinch pad
    is 3 spheres (radius 4 mm) along the pad, their outer side on the pad's inner
    face (our engine has sphere-box contacts for gripping cubes).
  * finger_joint2 is a separate joint (the xacro couples it with <mimic>, which
    our engine does not have): give both fingers the same target. 0 = closed (pads
    touch), -0.7854 = open.
  * no visual meshes (Phase 10 adds mesh drawing separately).
Joint armature and damping from enactic openarm_mujoco v2 (motor classes): joints
1-2 (DM8009) 0.016 / 1.0, joints 3-4 (DM4340) 0.030 / 0.9, joints 5-7 (DM4310)
0.002 / 0.9, fingers 0.002 / 0.6; URDF has no field for them, so they are in
ARMATURE below and the task sets them (forge/env/tasks/openarm_pick.h).

xacro without ROS: pip install xacro, and give it a tiny ament_index_python module
whose get_package_share_directory(name) returns the folder that holds
openarm_description.
"""
import os
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

MOUNT_Z = 0.45  # shoulder (base_link origin) height above the floor
ARMATURE = {"joint1": (0.016, 1.0), "joint2": (0.016, 1.0), "joint3": (0.030, 0.9), "joint4": (0.030, 0.9),
            "joint5": (0.002, 0.9), "joint6": (0.002, 0.9), "joint7": (0.002, 0.9),
            "finger_joint1": (0.002, 0.6), "finger_joint2": (0.002, 0.6)}
PAD_R = 0.004
PAD_Z = -0.0955  # pad center height in the finger frame (pad plate spans z -0.102 .. -0.089)
PAD_X = (-0.02, 0.0, 0.02)


def stl_vertices(path):
    b = open(path, "rb").read()
    n = struct.unpack_from("<I", b, 80)[0]
    a = np.frombuffer(b, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=n, offset=84)
    return np.unique(a["v"].reshape(-1, 3), axis=0).astype(np.float64)


def fit_capsule(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    a = vt[0]
    if a[np.argmax(np.abs(a))] < 0:
        a = -a
    t = (P - c) @ a
    perp = np.linalg.norm((P - c) - np.outer(t, a), axis=1)
    r = float(np.percentile(perp, 97))
    mid = 0.5 * (t.max() + t.min())
    half = max(0.0, 0.5 * (t.max() - t.min()) - r)
    return c + mid * a, a, r, half


def rpy_from_z(a):
    """URDF rpy that turns +z into unit vector a (R = Rz(yaw) Ry(pitch) Rx(roll), roll = 0)."""
    pitch = np.arccos(np.clip(a[2], -1.0, 1.0))
    yaw = np.arctan2(a[1], a[0]) if np.hypot(a[0], a[1]) > 1e-12 else 0.0
    return 0.0, pitch, yaw


def g(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, desc = sys.argv[1], sys.argv[2]
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets/robots/openarm/openarm.urdf")
    root = ET.parse(src).getroot()
    pkg = "package://openarm_description/"

    def mesh_file(link, part=None):
        col = link.find("collision")
        if col is None:
            return None
        fn = col.find("geometry/mesh").get("filename").replace(pkg, desc.rstrip("/") + "/")
        if part is not None:
            d = os.path.join(os.path.dirname(fn), "normal_mesh_4parts")
            fn = os.path.join(d, os.path.basename(fn).replace(".stl", f"_{part:02d}.stl"))
        xyz = [float(x) for x in col.find("origin").get("xyz").split()] if col.find("origin") is not None else [0, 0, 0]
        return fn, np.array(xyz)

    lines = ['<?xml version="1.0"?>',
             "<!-- OpenArm v2.0, one arm (right) with the pinch gripper, on a post: the shoulder is",
             f"     {MOUNT_Z} m above the floor. From enactic/openarm_description (Apache License 2.0,",
             "     Copyright Enactic, Inc.; see LICENSE in this folder), made by tools/openarm_to_urdf.py.",
             "     Masses, inertias, joint poses, axes, ranges and efforts as given. Collision: one",
             "     cylinder per link fitted to its collision mesh (crucible treats cylinders as",
             "     capsules); 3 pad spheres (4 mm) per finger; no visual meshes; finger_joint2 is a",
             "     separate joint (no mimic): give both fingers the same target (0 closed, -0.7854 open).",
             "     Armature / damping (openarm_mujoco v2): joints 1-2 0.016 / 1.0, 3-4 0.030 / 0.9,",
             "     5-7 0.002 / 0.9, fingers 0.002 / 0.6 (set by the task). -->",
             '<robot name="openarm">',
             '  <mujoco><compiler fusestatic="false" discardvisual="true" balanceinertia="false"/></mujoco>',
             '  <link name="post">',
             '    <inertial><origin xyz="0 0 0.2"/><mass value="5"/>',
             '      <inertia ixx="0.07" ixy="0" ixz="0" iyy="0.07" iyz="0" izz="0.01"/></inertial>',
             "  </link>"]
    fits = []
    for link in root.findall("link"):
        name = link.get("name")
        inert = link.find("inertial")
        if inert is None:
            continue
        short = name.replace("openarm_right_", "")
        lines.append(f'  <link name="{short}">')
        o, m, i = inert.find("origin"), inert.find("mass"), inert.find("inertia")
        lines.append(f'    <inertial><origin xyz="{o.get("xyz")}" rpy="{o.get("rpy", "0 0 0")}"/>'
                     f'<mass value="{float(m.get("value")):.9g}"/>')
        lines.append("      <inertia " + " ".join(f'{k}="{float(i.get(k)):.9g}"' for k in
                                                   ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")) + "/></inertial>")
        finger = short in ("ee_link1", "ee_link2")
        mf = mesh_file(link, 3 if finger else None)
        if mf is not None:
            fn, off = mf
            c, a, r, half = fit_capsule(stl_vertices(fn))
            c = c + off
            rpy = rpy_from_z(a)
            lines.append(f'    <collision><origin xyz="{g(c)}" rpy="{g(rpy)}"/>'
                         f'<geometry><cylinder radius="{r:.6g}" length="{2 * half:.6g}"/></geometry></collision>')
            fits.append((short, r, 2 * half))
        if finger:  # pinch pad: inner finger's pad faces -y, outer finger's faces +y
            y = -0.018 + PAD_R if short == "ee_link1" else 0.018 - PAD_R
            for x in PAD_X:
                lines.append(f'    <collision><origin xyz="{x:.6g} {y:.6g} {PAD_Z:.6g}"/>'
                             f'<geometry><sphere radius="{PAD_R}"/></geometry></collision>')
        lines.append("  </link>")
    for j in root.findall("joint"):
        child = j.find("child").get("link").replace("openarm_right_", "")
        parent = j.find("parent").get("link").replace("openarm_right_", "")
        if child == "camera_view_frame":
            continue
        name = j.get("name").replace("openarm_right_", "")
        o = j.find("origin")
        xyz, rpy = o.get("xyz"), o.get("rpy", "0 0 0")
        if parent == "world":
            parent, xyz, name = "post", f"0 0 {MOUNT_Z}", "mount"
        lines.append(f'  <joint name="{name}" type="{j.get("type")}">')
        lines.append(f'    <origin xyz="{g([float(v) for v in xyz.split()])}" rpy="{g([float(v) for v in rpy.split()])}"/>'
                     f'<parent link="{parent}"/><child link="{child}"/>')
        if j.find("axis") is not None:
            lines.append(f'    <axis xyz="{j.find("axis").get("xyz")}"/>')
        lim = j.find("limit")
        if lim is not None:
            lines.append(f'    <limit lower="{lim.get("lower")}" upper="{lim.get("upper")}" effort="{lim.get("effort")}"'
                         f' velocity="{lim.get("velocity")}"/>')
        lines.append("  </joint>")
    lines.append("</robot>")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    for short, r, length in fits:
        print(f"  {short:14s} capsule radius {1000 * r:5.1f} mm, cylinder length {1000 * length:5.1f} mm")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
