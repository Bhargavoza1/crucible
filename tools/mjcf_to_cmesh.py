"""Visual meshes of a MuJoCo model (MJCF) -> <our urdf>.cmesh (flux/robot_mesh.h), Phase 10.

    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie
    pip install mujoco trimesh fast-simplification
    python tools/mjcf_to_cmesh.py mujoco_menagerie/franka_emika_panda/panda.xml assets/robots/panda/panda.urdf

For every link of our URDF, the visual geoms (contype 0 and conaffinity 0, or group 2)
of the MJCF body of the same name, in that body's frame (our URDF was converted from
this MJCF with the same body frames), with their material colors (sRGB, as MuJoCo
draws them). Simplified to about 60k / 12k triangles per robot (viewer / camera
images) with smooth normals, as tools/unitree_to_assets.py. Writes nothing but the
.cmesh next to the URDF.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmesh import write_cmesh  # noqa: E402
from unitree_to_assets import LOD_TRIANGLES, cluster  # noqa: E402


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def body_parts(m, body):
    """[(trimesh in the body frame, rgb uint8)] of a body's visual mesh geoms."""
    import mujoco
    import trimesh
    parts = []
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] != body or m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        visual = (m.geom_contype[gi] == 0 and m.geom_conaffinity[gi] == 0) or m.geom_group[gi] == 2
        if not visual:
            continue
        mid = m.geom_dataid[gi]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
        f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        v = v @ quat_mat(m.geom_quat[gi]).T + m.geom_pos[gi]  # mesh frame -> body frame
        rgba = m.mat_rgba[m.geom_matid[gi]] if m.geom_matid[gi] >= 0 else m.geom_rgba[gi]
        parts.append((trimesh.Trimesh(v, f, process=False), (np.clip(rgba[:3], 0, 1) * 255).astype(np.uint8)))
    return parts


def simplified(parts, keep, coarse):
    tris, cols = [], []
    for mesh, col in parts:
        mesh = mesh.copy()
        mesh.merge_vertices(merge_tex=True, merge_norm=True)  # weld: decimating loose triangles tears holes
        if keep < 1.0 and len(mesh.faces) > 200:
            target = max(100, int(keep * len(mesh.faces)))
            mesh = mesh.simplify_quadric_decimation(face_count=target)
            if coarse and len(mesh.faces) > 2 * target:
                mesh = cluster(mesh, target)
        t = np.asarray(mesh.vertices)[mesh.faces]
        tris.append(t)
        cols.append(np.repeat(col[None], len(t), 0))
    return (np.concatenate(tris), np.concatenate(cols)) if tris else None


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    import mujoco
    m = mujoco.MjModel.from_xml_path(sys.argv[1])
    urdf = sys.argv[2]
    links = re.findall(r'<link name="([^"]+)"', open(urdf).read())
    per_link = []
    for ln in links:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ln)
        if b < 0:
            print(f"  {ln}: no MJCF body of that name, no mesh")
            continue
        parts = body_parts(m, b)
        if parts:
            per_link.append((ln, parts))
    full = sum(len(p[0].faces) for _, parts in per_link for p in parts)
    keeps = [min(1.0, n / full) for n in LOD_TRIANGLES]
    names, lods = [], []
    for ln, parts in per_link:
        names.append(ln)
        lods.append([simplified(parts, k, coarse=i > 0) for i, k in enumerate(keeps)])
    out = os.path.splitext(urdf)[0] + ".cmesh"
    write_cmesh(out, names, lods)
    print(f"wrote {out}: {len(names)} links, {sum(len(l[0][0]) for l in lods)} / "
          f"{sum(len(l[1][0]) for l in lods)} triangles (from {full})")


if __name__ == "__main__":
    main()
