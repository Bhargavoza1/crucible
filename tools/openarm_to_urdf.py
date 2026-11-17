"""enactic openarm_description (Apache-2) -> crucible OpenArm assets (Phase 10).

Two robots, from the URDFs that openarm_description's xacro makes:
  * one arm:  preset v2/right_arm_with_pinch_gripper -> assets/robots/openarm/openarm.urdf
    (on a post, shoulder MOUNT_Z above the floor; link names without "openarm_right_")
  * both arms: preset v2/default_bimanual without the body camera (it needs ZED's ROS
    package; copy the preset, set body_camera_mount and body_camera to enabled: false
    and drop them from robot.components) -> assets/robots/openarm/openarm_bimanual.urdf
    (the body's floor 0.697 - MOUNT_Z under our floor, so the shoulders are MOUNT_Z up:
    the floor is the table in front of the robot; link names without "openarm_")
and next to each URDF its visual meshes as <name>.cmesh (flux/robot_mesh.h): every
link's .dae meshes with their colors, in the link frame, simplified to two levels of
detail (LOD 0 for the viewer, LOD 1 for the small camera images).

    git clone https://github.com/enactic/openarm_description
    xacro openarm_description/urdf/openarm.urdf.xacro robot_preset:=v2/right_arm_with_pinch_gripper \\
        ros2_control:=false > openarm_right.urdf
    xacro openarm_description/urdf/openarm.urdf.xacro robot_preset:=v2/bimanual_nocam \\
        ros2_control:=false > openarm_bimanual.urdf
    pip install trimesh pycollada fast-simplification
    python tools/openarm_to_urdf.py openarm_right.urdf openarm_description openarm
    python tools/openarm_to_urdf.py openarm_bimanual.urdf openarm_description openarm_bimanual

Kept as given: link masses, inertias, joint poses, axes, ranges, efforts, and the
mirroring of the left arm (its meshes have scale 1 -1 1).
Changes:
  * collision: every collision mesh becomes one CYLINDER fitted to it (axis = the
    mesh's main direction, radius = 97th percentile of the vertex distance to the
    axis, length = the rest of the extent). crucible reads a <cylinder> as a capsule
    (forge/robots/urdf.h).
  * fingers: the upper finger (finger_*_03 part) is a cylinder; the pinch pad is 3
    spheres (radius 4 mm) along the pad, their outer side on the pad's inner face.
  * finger_joint2 is a separate joint (the xacro couples it with <mimic>, which our
    engine does not have): give both fingers the same target. Right arm: 0 = closed
    (pads touch), -0.7854 = open; left arm (mirrored): 0 closed, +0.7854 open.
  * links without mass that hold nothing (frames for cameras) are dropped.
Joint armature and damping from enactic openarm_mujoco v2 (motor classes): joints
1-2 (DM8009) 0.016 / 1.0, joints 3-4 (DM4340) 0.030 / 0.9, joints 5-7 (DM4310)
0.002 / 0.9, fingers 0.002 / 0.6 (in the URDF comment; tasks set them).

xacro without ROS: pip install xacro, and give it a tiny ament_index_python module
whose get_package_share_directory(name) returns the folder that holds
openarm_description.
"""
import os
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmesh import write_cmesh  # noqa: E402

MOUNT_Z = 0.45  # shoulder (arm base_link origin) height above the floor
BODY_ARM_Z = 0.697  # shoulder height above the bimanual body's origin (openarm v2 body)
PAD_R = 0.004
PAD_Z = -0.0955  # pad center in the finger frame (pad plate spans z -0.102 .. -0.089)
PAD_X = (-0.02, 0.0, 0.02)
LOD_KEEP = (0.35, 0.08)  # fraction of each mesh's triangles kept: viewer, camera images


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


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def g(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def origin_of(el):
    o = el.find("origin")
    xyz = np.array([float(x) for x in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
    rpy = np.array([float(x) for x in o.get("rpy", "0 0 0").split()]) if o is not None else np.zeros(3)
    return xyz, rpy


def mesh_of(el, desc):
    m = el.find("geometry/mesh")
    fn = m.get("filename").replace("package://openarm_description/", desc.rstrip("/") + "/")
    scale = np.array([float(x) for x in m.get("scale", "1 1 1").split()])
    return fn, scale


def visual_triangles(link, desc, keep):
    """All visual meshes of a link in its frame: (positions [n, 3, 3], colors [n, 3] uint8)."""
    import trimesh
    tris, cols = [], []
    for vis in link.findall("visual"):
        if vis.find("geometry/mesh") is None:
            continue
        fn, scale = mesh_of(vis, desc)
        xyz, rpy = origin_of(vis)
        R = rpy_matrix(rpy)
        scene = trimesh.load(fn, force="scene")
        for m in scene.dump(concatenate=False):
            color = np.array([160, 160, 160], np.uint8)
            mat = getattr(m.visual, "material", None)
            if mat is not None and getattr(mat, "main_color", None) is not None:
                color = np.asarray(mat.main_color[:3], np.uint8)
            if keep < 1.0 and len(m.faces) > 200:
                # weld duplicate vertices first (geometry only: uv seams and normals must
                # not keep them apart): decimating unconnected triangles tears holes
                m = trimesh.Trimesh(m.vertices, m.faces, process=False)
                m.merge_vertices(merge_tex=True, merge_norm=True)
                m = m.simplify_quadric_decimation(percent=1.0 - keep)
            v = (m.vertices * scale) @ R.T + xyz
            f = m.faces
            if np.prod(scale) < 0:  # mirrored: keep the triangles facing out
                f = f[:, ::-1]
            tris.append(v[f])
            cols.append(np.repeat(color[None], len(f), 0))
    if not tris:
        return np.zeros((0, 3, 3)), np.zeros((0, 3), np.uint8)
    return np.concatenate(tris), np.concatenate(cols)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, desc, name = sys.argv[1:]
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets/robots/openarm")
    root = ET.parse(src).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = root.findall("joint")
    bimanual = "openarm_left_base_link" in links
    strip = "openarm_" if bimanual else "openarm_right_"
    short = lambda s: s.replace(strip, "")
    # massless links that are no joint's parent: camera frames etc., dropped
    parents = {j.find("parent").get("link") for j in joints}
    keep = {n for n, l in links.items() if n != "world" and (l.find("inertial") is not None or n in parents)}
    keep -= {n for n in keep if links[n].find("inertial") is None and n not in parents}

    what = "both arms on the OpenArm v2.0 body" if bimanual else "one arm (right) on a post"
    floor = " (the floor is the table)" if bimanual else ""
    fingers = ("both fingers of an arm the same target (right arm 0 closed, -0.7854 open;",
               "     left arm 0 closed, +0.7854 open).") if bimanual else \
              ("both fingers the same target (0 closed, -0.7854 open).", "    ")
    lines = ['<?xml version="1.0"?>',
             f"<!-- OpenArm v2.0, {what}, with pinch gripper; the shoulder is {MOUNT_Z} m",
             f"     above the floor{floor}. From enactic/openarm_description (Apache",
             "     License 2.0, Copyright Enactic, Inc.; see LICENSE in this folder), made by",
             "     tools/openarm_to_urdf.py. Masses, inertias, joint poses, axes, ranges and efforts as",
             "     given. Collision: one cylinder per collision mesh, fitted to it (crucible reads",
             "     cylinders as capsules); 3 pad spheres (4 mm) per finger; finger_joint2 is a separate",
             "     joint (no mimic): give " + fingers[0],
             fingers[1] + " Visual meshes: " + name + ".cmesh.",
             "     Armature / damping (openarm_mujoco v2): joints 1-2 0.016 / 1.0, 3-4 0.030 / 0.9,",
             "     5-7 0.002 / 0.9, fingers 0.002 / 0.6 (set by the task). -->",
             f'<robot name="{name}">',
             '  <mujoco><compiler fusestatic="false" discardvisual="true" balanceinertia="false"/></mujoco>']
    if True:  # root: a post welded to the world (the robot stands on it)
        lines += ['  <link name="post">', '    <inertial><origin xyz="0 0 0.2"/><mass value="5"/>',
                  '      <inertia ixx="0.07" ixy="0" ixz="0" iyy="0.07" iyz="0" izz="0.01"/></inertial>', "  </link>"]
    order, fits, mesh_links = [], [], []
    for lname, link in links.items():
        if lname not in keep:
            continue
        s = short(lname)
        order.append(s)
        lines.append(f'  <link name="{s}">')
        inert = link.find("inertial")
        if inert is not None:
            xyz, rpy = origin_of(inert)
            m, i = inert.find("mass"), inert.find("inertia")
            o = inert.find("origin")  # as written in the source (full precision)
            lines.append(f'    <inertial><origin xyz="{o.get("xyz")}" rpy="{o.get("rpy", "0 0 0")}"/>'
                         f'<mass value="{float(m.get("value")):.9g}"/>')
            lines.append("      <inertia " + " ".join(f'{k}="{float(i.get(k)):.9g}"' for k in
                                                       ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")) + "/></inertial>")
        finger = s.endswith("ee_link1") or s.endswith("ee_link2")
        for col in link.findall("collision"):
            if col.find("geometry/mesh") is None:
                continue
            fn, scale = mesh_of(col, desc)
            if finger:  # the upper finger only; the pad is spheres
                fn = os.path.join(os.path.dirname(fn), "normal_mesh_4parts",
                                  os.path.basename(fn).replace(".stl", "_03.stl"))
            xyz, rpy = origin_of(col)
            P = stl_vertices(fn) * scale @ rpy_matrix(rpy).T + xyz
            c, a, r, half = fit_capsule(P)
            if half < 0.001:  # a ball: a sphere (MuJoCo needs capsule lengths > 0)
                lines.append(f'    <collision><origin xyz="{g(c)}"/><geometry><sphere radius="{r:.6g}"/></geometry></collision>')
            else:
                lines.append(f'    <collision><origin xyz="{g(c)}" rpy="{g(rpy_from_z(a))}"/>'
                             f'<geometry><cylinder radius="{r:.6g}" length="{2 * half:.6g}"/></geometry></collision>')
            fits.append((s, r, 2 * half))
            if finger:
                sy = scale[1]
                y = sy * (-0.018 + PAD_R) if s.endswith("ee_link1") else sy * (0.018 - PAD_R)
                for x in PAD_X:
                    lines.append(f'    <collision><origin xyz="{x:.6g} {y:.6g} {PAD_Z:.6g}"/>'
                                 f'<geometry><sphere radius="{PAD_R}"/></geometry></collision>')
        lines.append("  </link>")
        mesh_links.append((s, link))
    for j in joints:
        child, parent = j.find("child").get("link"), j.find("parent").get("link")
        if child not in keep:
            continue
        jname = short(j.get("name"))
        xyz, rpy = origin_of(j)
        if parent == "world":  # on the post; bimanual: the body's floor under ours, shoulders MOUNT_Z up
            parent_s, jname = "post", "mount"
            xyz = np.array([0, 0, MOUNT_Z - BODY_ARM_Z if bimanual else MOUNT_Z])
        else:
            parent_s = short(parent)
        lines.append(f'  <joint name="{jname}" type="{j.get("type")}">')
        lines.append(f'    <origin xyz="{g(xyz)}" rpy="{g(rpy)}"/><parent link="{parent_s}"/><child link="{short(child)}"/>')
        if j.find("axis") is not None:
            lines.append(f'    <axis xyz="{j.find("axis").get("xyz")}"/>')
        lim = j.find("limit")
        if lim is not None:
            lines.append(f'    <limit lower="{lim.get("lower")}" upper="{lim.get("upper")}" effort="{lim.get("effort")}"'
                         f' velocity="{lim.get("velocity")}"/>')
        lines.append("  </joint>")
    lines.append("</robot>")
    urdf = "\n".join(lines) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, name + ".urdf"), "w").write(urdf)
    for s, r, length in fits:
        print(f"  {s:20s} capsule radius {1000 * r:5.1f} mm, cylinder length {1000 * length:5.1f} mm")
    # visual meshes, two levels of detail
    names, lods = [], []
    for s, link in mesh_links:
        names.append(s)
        lods.append([visual_triangles(link, desc, k) for k in LOD_KEEP])
    write_cmesh(os.path.join(out_dir, name + ".cmesh"), names, lods)
    n0 = sum(len(l[0][0]) for l in lods)
    n1 = sum(len(l[1][0]) for l in lods)
    print(f"wrote {name}.urdf and {name}.cmesh ({n0} / {n1} triangles: viewer / camera)")


if __name__ == "__main__":
    main()
