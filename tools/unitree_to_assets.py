"""Unitree G1 / Go2 (unitree_rl_gym, BSD-3) -> crucible collisions and meshes (Phase 10).

    git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym
    pip install trimesh pycollada fast-simplification
    python tools/unitree_to_assets.py unitree_rl_gym/resources/robots/g1_description/g1_12dof.urdf g1
    python tools/unitree_to_assets.py unitree_rl_gym/resources/robots/go2/urdf/go2.urdf go2

Reads Unitree's URDF (the one our assets/robots/<name> URDF came from) and changes
our URDF in place, only its <collision> elements (masses, inertias, joints and limits
stay as they are), then writes <name>.cmesh next to it (flux/robot_mesh.h).

Collision (crucible reads a <cylinder> as a capsule; no self contacts, like the
MuJoCo references; spheres only on the feet, since the tasks take the robot's
spheres as its feet: other shapes that would be spheres are 2 mm long capsules):
  * G1: each collision mesh becomes one cylinder fitted to it (axis = main direction,
    radius = 97th percentile of the distance to the axis, as tools/openarm_to_urdf.py);
    Unitree's own cylinders (shoulders) stay; the 4 foot spheres per ankle_roll_link
    stay; logo_link (a thin plate on the torso) gets none.
  * Go2: Unitree's cylinders and spheres stay (hips, calves, head, feet; the head
    sphere as a capsule). Boxes become
    cylinders along their longest side (crucible has no box-box contacts and robot
    boxes would touch the robot's own capsules): base 0.376 x 0.094 x 0.114 m ->
    radius 0.05 m along x; thigh 0.11 x 0.0245 x 0.034 m -> radius 0.017 m along it.
Meshes: every <visual> mesh in its link's frame (STL color from the URDF, .dae colors
from its materials), welded and simplified to 2 levels of detail (about 60k and 12k triangles per robot),
with smooth normals (faces within 35 degrees share a vertex normal).
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmesh import write_cmesh  # noqa: E402
from openarm_to_urdf import fit_capsule, g, rpy_from_z, rpy_matrix  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOD_TRIANGLES = (60000, 12000)  # triangle budget per robot: viewer, camera images
SKIP_COLLISION = {"g1": {"logo_link"}, "go2": set()}


def origin(el):
    o = el.find("origin")
    xyz = np.array([float(x) for x in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
    rpy = np.array([float(x) for x in o.get("rpy", "0 0 0").split()]) if o is not None else np.zeros(3)
    return xyz, rpy


def cylinder_xml(pos, axis, r, half):
    half = max(half, 1e-3)  # MuJoCo rejects capsules of zero length
    rpy = rpy_from_z(axis / np.linalg.norm(axis))
    return (f'    <collision><origin xyz="{g(pos)}" rpy="{g(rpy)}"/>'
            f'<geometry><cylinder radius="{r:.4g}" length="{2 * half:.4g}"/></geometry></collision>')


def collisions(name, link, src_dir):
    """Our <collision> lines for one link of Unitree's URDF."""
    import trimesh
    out = []
    for c in link.findall("collision"):
        geo = c.find("geometry")[0]
        xyz, rpy = origin(c)
        R = rpy_matrix(rpy)
        if geo.tag == "sphere" and not link.get("name").endswith(("_foot", "ankle_roll_link")):
            # spheres stay the feet only (the tasks use the robot's spheres as its feet)
            out.append(cylinder_xml(xyz, np.array([0.0, 0.0, 1.0]), float(geo.get("radius")), 0.0))
        elif geo.tag == "sphere":
            out.append(f'    <collision><origin xyz="{g(xyz)}"/>'
                       f'<geometry><sphere radius="{geo.get("radius")}"/></geometry></collision>')
        elif geo.tag == "cylinder":
            r, length = float(geo.get("radius")), float(geo.get("length"))
            out.append(cylinder_xml(xyz, R[:, 2], r, 0.5 * length))
        elif geo.tag == "box":
            size = np.array([float(x) for x in geo.get("size").split()])
            k = int(np.argmax(size))
            r = 0.05 if link.get("name") == "base" else 0.5 * float(np.max(np.delete(size, k)))
            out.append(cylinder_xml(xyz, R[:, k], r, max(0.0, 0.5 * size[k] - r)))
        elif geo.tag == "mesh":
            m = trimesh.load(os.path.join(src_dir, geo.get("filename")), force="mesh")
            P = np.asarray(m.vertices) @ R.T + xyz
            c0, a, r, half = fit_capsule(P)
            out.append(cylinder_xml(c0, a, r, half))
    return out


def cluster(m, target):
    """Vertex clustering for meshes quadric decimation cannot bring down (G1's torso
    stops near 20k triangles: many small open parts): vertices snapped to a grid cell,
    the cell grown until at most 'target' triangles are left. Coarse, so only for the
    camera level of detail."""
    import trimesh
    cell = 0.002
    while True:
        q = np.floor(m.vertices / cell).astype(np.int64)
        _, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        v = np.zeros((len(cnt), 3))
        np.add.at(v, inv, m.vertices)
        v /= cnt[:, None]
        f = inv[m.faces]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        f = np.unique(f, axis=0)
        if len(f) <= target or cell > 0.05:
            return trimesh.Trimesh(v, f, process=False)
        cell *= 1.25


def visual_triangles(link, src_dir, keep, coarse=False):
    """(positions [n,3,3], colors [n,3] uint8) of a link's visual meshes."""
    import trimesh
    tris, norms, cols = [], [], []
    for vis in link.findall("visual"):
        geo = vis.find("geometry/mesh")
        if geo is None:
            continue
        xyz, rpy = origin(vis)
        R = rpy_matrix(rpy)
        urdf_rgba = vis.find("material/color")
        urdf_col = (np.array([float(x) for x in urdf_rgba.get("rgba").split()[:3]]) * 255).astype(np.uint8) \
            if urdf_rgba is not None else np.array([160, 160, 160], np.uint8)
        fn = os.path.join(src_dir, geo.get("filename"))
        parts = trimesh.load(fn, force="scene").dump(concatenate=False) if fn.lower().endswith(".dae") \
            else [trimesh.load(fn, force="mesh")]
        for m in parts:
            col = urdf_col
            mat = getattr(m.visual, "material", None)
            if fn.lower().endswith(".dae") and mat is not None and getattr(mat, "main_color", None) is not None:
                col = np.asarray(mat.main_color[:3], np.uint8)
            # weld first (geometry only: uv seams and normals must not keep vertices
            # apart): decimating unconnected triangles tears holes
            m = trimesh.Trimesh(m.vertices, m.faces, process=False)
            m.merge_vertices(merge_tex=True, merge_norm=True)
            if keep < 1.0 and len(m.faces) > 200:
                target = max(100, int(keep * len(m.faces)))
                m = m.simplify_quadric_decimation(face_count=target)
                if coarse and len(m.faces) > 2 * target:
                    m = cluster(m, target)
            t = (np.asarray(m.vertices)[m.faces]) @ R.T + xyz
            if np.linalg.det(R) < 0:
                t = t[:, ::-1]
            tris.append(t)
            cols.append(np.repeat(col[None], len(t), 0))
    if not tris:
        return None
    return np.concatenate(tris), np.concatenate(cols)


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in ("g1", "go2"):
        sys.exit(__doc__)
    src, name = sys.argv[1], sys.argv[2]
    src_dir = os.path.dirname(os.path.abspath(src))
    ours = os.path.join(ROOT, "assets/robots", name, {"g1": "g1_12dof", "go2": "go2"}[name] + ".urdf")
    text = open(ours).read()
    our_links = set(re.findall(r'<link name="([^"]+)"', text))
    unitree = {L.get("name"): L for L in ET.parse(src).getroot().findall("link")}

    # 1. collisions: drop ours, write the new ones before each </link>
    text = re.sub(r"\n\s*<collision>.*?</collision>", "", text, flags=re.S)
    for ln in sorted(our_links):
        if ln in SKIP_COLLISION[name] or ln not in unitree:
            continue
        lines = collisions(name, unitree[ln], src_dir)
        if not lines:
            continue
        pat = re.compile(r'(<link name="' + re.escape(ln) + r'"\s*>.*?)(\n\s*</link>)', re.S)
        text, n = pat.subn(lambda mm: mm.group(1) + "\n" + "\n".join(lines) + mm.group(2), text, count=1)
        if n != 1:
            sys.exit(f"link {ln}: no <link>...</link> block found in {ours}")
    open(ours, "w").write(text)

    # 2. meshes
    links = [x for x in re.findall(r'<link name="([^"]+)"', text) if x in unitree]
    full = sum(len(t[0]) for t in (visual_triangles(unitree[ln], src_dir, 1.0) for ln in links) if t is not None)
    keeps = [min(1.0, n / full) for n in LOD_TRIANGLES]
    names, lods = [], []
    for ln in links:
        levels = [visual_triangles(unitree[ln], src_dir, k, coarse=i > 0) for i, k in enumerate(keeps)]
        if levels[0] is None:
            continue
        names.append(ln)
        lods.append(levels)
    out = os.path.splitext(ours)[0] + ".cmesh"
    write_cmesh(out, names, lods)
    print(f"wrote {os.path.relpath(ours, ROOT)} collisions and {os.path.relpath(out, ROOT)} "
          f"({sum(len(l[0][0]) for l in lods)} / {sum(len(l[1][0]) for l in lods)} triangles)")


if __name__ == "__main__":
    main()
