"""OpenArm Cell (openarm_mujoco v2, Apache-2.0, Enactic) -> crucible URDF + .cmesh (Phase 10).

    git clone --depth 1 https://github.com/enactic/openarm_mujoco
    pip install mujoco trimesh fast-simplification
    python tools/openarm_cell_to_urdf.py openarm_mujoco/v2/cell/cell.xml

The real OpenArm pick-and-place scene (as enactic/openarm_mjlab's pick_place task):
both arms hang from the lifter at the back of the cell's table. The left arm works,
the right arm and the lifter are frozen at the cell's "home" keyframe (joints
deleted, their home angles baked into the body frames, as openarm_mjlab does).

Writes assets/robots/openarm/openarm_cell.urdf and openarm_cell.cmesh:
  * links and joints from the compiled MuJoCo model (body frames, hinge axes, ranges,
    actuator force ranges, damping), inertials as full tensors in the link frame;
  * the world moved down by the table top height (1.005 m): our floor plane is the
    table top; the root link 'openarm_cell' carries the cell's visual meshes;
  * collision: each arm collision mesh -> one fitted cylinder (crucible reads it as a
    capsule); the left fingertips -> 3 pad spheres (4 mm) on each finger's gripping
    face (the only spheres), every finger base a capsule; nothing for the cell (table = floor plane,
    walls, roof and rail are out of the arm's way);
  * the meshes: the arms' visual geoms in full resolution, and the cell: its frame
    below the table top (walls and roof cut away so viewers and cameras see in), the
    table sheet, the rail and the lifter; 2 levels of detail as the other robots.
Armature is not in URDF: tasks set it per joint (openarm_pick_place reads the same
values from this cell: 0.016 / 0.016 / 0.030 / 0.030 / 0.002 x 3, fingers 0.002).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmesh import write_cmesh  # noqa: E402
from mjcf_to_cmesh import body_parts, quat_mat, simplified  # noqa: E402
from openarm_to_urdf import fit_capsule, g, rpy_from_z  # noqa: E402
from unitree_to_assets import LOD_TRIANGLES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_LINK = "openarm_cell"
FROZEN = ("openarm_lifter_joint", *(f"openarm_right_joint{i}" for i in range(1, 8)),
          "openarm_right_finger_joint1", "openarm_right_finger_joint2")
PAD_R, PAD_Y, PAD_Z, PAD_X = 0.004, 0.014, -0.0955, (-0.02, 0.0, 0.02)


def rpy_of(R):
    """Roll, pitch, yaw (URDF: R = Rz(y) Ry(p) Rx(r))."""
    p = np.arcsin(-np.clip(R[2, 0], -1, 1))
    if abs(np.cos(p)) > 1e-6:
        r = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(R[1, 0], R[0, 0])
    else:
        r, y = np.arctan2(-R[1, 2], R[1, 1]), 0.0
    return np.array([r, p, y])


def load_frozen(xml):
    import mujoco
    spec = mujoco.MjSpec.from_file(xml)
    m0 = spec.compile()
    key = m0.key("home")
    home = {mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j): float(key.qpos[m0.jnt_qposadr[j]])
            for j in range(m0.njnt)}
    table = m0.geom("cell_table_col")
    top = float(table.pos[2] + table.size[2])
    for name in FROZEN:  # bake the home angle into the body, then delete the joint
        j = spec.joint(name)
        if home[name] != 0.0:
            if j.type != mujoco.mjtJoint.mjJNT_HINGE or not np.allclose(j.pos, 0.0):
                sys.exit(f"{name}: only hinges at the body origin can be baked")
            q = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q, np.asarray(j.axis), home[name])
            b = j.parent
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, b.quat, q)
            b.quat = out
        spec.delete(j)
    for a in list(spec.actuators):
        spec.delete(a)
    for e in list(spec.equalities):
        spec.delete(e)
    for k in list(spec.keys):
        spec.delete(k)
    return spec.compile(), home, top


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    import mujoco
    m, home, top = load_frozen(sys.argv[1])
    name = lambda t, i: mujoco.mj_id2name(m, t, i)
    lines = ['<?xml version="1.0"?>',
             "<!-- OpenArm v2 Cell (both arms on the lifter at the back of the table): the left arm",
             "     moves (joints openarm_left_joint1-7, finger joints 1-2), the right arm and lifter are",
             "     frozen at the cell's home pose. From enactic/openarm_mujoco v2 cell/cell.xml (Apache",
             "     License 2.0, Copyright Enactic, Inc.; see LICENSE in this folder), made by",
             "     tools/openarm_cell_to_urdf.py. The world is moved down by the table top height",
             f"     ({top:.3f} m): z = 0 is the table top (the floor plane). Collision: one cylinder per",
             "     arm collision mesh (crucible reads cylinders as capsules), 3 pad spheres (4 mm) per",
             "     finger. Visual meshes: openarm_cell.cmesh. -->",
             '<robot name="openarm_cell">',
             f'  <link name="{ROOT_LINK}"/>']
    joints = []
    for b in range(1, m.nbody):
        bn = name(mujoco.mjtObj.mjOBJ_BODY, b)
        parent = m.body_parentid[b]
        pn = ROOT_LINK if parent == 0 else name(mujoco.mjtObj.mjOBJ_BODY, parent)
        pos = m.body_pos[b].copy()
        if parent == 0:
            pos[2] -= top
        R = quat_mat(m.body_quat[b])
        # inertial: full tensor in the link frame
        Ri = quat_mat(m.body_iquat[b])
        I = Ri @ np.diag(m.body_inertia[b]) @ Ri.T
        link = [f'  <link name="{bn}">',
                f'    <inertial><origin xyz="{g(m.body_ipos[b])}"/><mass value="{m.body_mass[b]:.6g}"/>'
                f'<inertia ixx="{I[0,0]:.6g}" ixy="{I[0,1]:.6g}" ixz="{I[0,2]:.6g}" iyy="{I[1,1]:.6g}" '
                f'iyz="{I[1,2]:.6g}" izz="{I[2,2]:.6g}"/></inertial>']
        finger = "finger" in bn
        for gi in range(m.ngeom):
            if m.geom_bodyid[gi] != b or (m.geom_contype[gi] == 0 and m.geom_conaffinity[gi] == 0):
                continue
            gname = name(mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
            if m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
                continue  # lifter box: out of the way
            if finger and not gname.endswith("_03"):
                continue  # fingertip meshes: pad spheres below (left hand)
            mid = m.geom_dataid[gi]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
            P = v @ quat_mat(m.geom_quat[gi]).T + m.geom_pos[gi]
            c0, a, r, half = fit_capsule(P)
            half = max(half, 1e-3)
            link.append(f'    <collision><origin xyz="{g(c0)}" rpy="{g(rpy_from_z(a))}"/>'
                        f'<geometry><cylinder radius="{r:.4g}" length="{2 * half:.4g}"/></geometry></collision>')
        if finger and "_left_" in bn:  # pads on the gripping face (toward the gripper center); the
            # frozen right hand needs none (the tasks take the robot's spheres as the working pads)
            sy = -np.sign(m.body_pos[b][1])
            for x in PAD_X:
                link.append(f'    <collision><origin xyz="{g([x, sy * PAD_Y, PAD_Z])}"/>'
                            f'<geometry><sphere radius="{PAD_R}"/></geometry></collision>')
        link.append("  </link>")
        lines += link
        org = f'<origin xyz="{g(pos)}" rpy="{g(rpy_of(R))}"/>'
        if m.body_jntnum[b] == 1:
            j = m.body_jntadr[b]
            jn = name(mujoco.mjtObj.mjOBJ_JOINT, j)
            lo, hi = m.jnt_range[j]
            eff = max(abs(m.jnt_actfrcrange[j][0]), abs(m.jnt_actfrcrange[j][1]))
            d = m.dof_damping[m.jnt_dofadr[j]]
            joints += [f'  <joint name="{jn}" type="revolute">',
                       f'    {org}<parent link="{pn}"/><child link="{bn}"/>',
                       f'    <axis xyz="{g(m.jnt_axis[j])}"/>',
                       f'    <limit lower="{lo:.6g}" upper="{hi:.6g}" effort="{eff:.6g}" velocity="10"/>',
                       f'    <dynamics damping="{d:.6g}"/>',
                       "  </joint>"]
        elif m.body_jntnum[b] == 0:
            joints += [f'  <joint name="{bn}_fixed" type="fixed">',
                       f'    {org}<parent link="{pn}"/><child link="{bn}"/>',
                       "  </joint>"]
        else:
            sys.exit(f"{bn}: more than one joint")
    lines += joints + ["</robot>", ""]
    out_dir = os.path.join(ROOT, "assets/robots/openarm")
    urdf = os.path.join(out_dir, "openarm_cell.urdf")
    open(urdf, "w").write("\n".join(lines))

    # meshes: every body's visual geoms; on the root link the cell (world frame, moved
    # down by the table top height): its frame below the table top (the walls and roof
    # are cut away, so the viewer and cameras see in), the white table sheet, the rail
    # and the lifter (boxes in the model, drawn from its collision boxes; the rail
    # drawn down to the table)
    import trimesh
    per = []
    world = []
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] != 0 or m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[gi]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
        f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        mesh = trimesh.Trimesh(v @ quat_mat(m.geom_quat[gi]).T + m.geom_pos[gi], f, process=False)
        if name(mujoco.mjtObj.mjOBJ_GEOM, gi) == "cell_vis":
            mesh = mesh.slice_plane([0, 0, top + 1e-3], [0, 0, -1])  # keep below the table top
        rgba = m.mat_rgba[m.geom_matid[gi]] if m.geom_matid[gi] >= 0 else m.geom_rgba[gi]
        world.append((mesh, (np.clip(rgba[:3], 0, 1) * 255).astype(np.uint8)))
    black = np.array([63, 63, 63], np.uint8)
    for lo, hi in (((0.0, -0.03, top), (0.06, 0.03, 1.825)), ((0.06, -0.039, 1.245), (0.123, 0.039, 1.825))):
        box = trimesh.creation.box(extents=np.subtract(hi, lo))
        box.apply_translation(np.add(lo, hi) / 2)
        world.append((box, black))
    lb = m.body("openarm_lifter_link")
    box = trimesh.creation.box(extents=(0.0915, 0.06, 0.122))
    box.apply_translation(lb.pos + np.array([-0.01575, 0.0, 0.0]))
    world.append((box, black))
    for mesh, _ in world:
        mesh.vertices[:, 2] -= top
    per.append((ROOT_LINK, world))
    for b in range(1, m.nbody):
        parts = body_parts(m, b)
        if parts:
            per.append((name(mujoco.mjtObj.mjOBJ_BODY, b), parts))
    full = sum(len(p[0].faces) for _, parts in per for p in parts)
    keeps = [1.0 if n is None else min(1.0, n / full) for n in LOD_TRIANGLES]
    names, lods = [], []
    for ln, parts in per:
        names.append(ln)
        lods.append([simplified(parts, k, coarse=i > 0) for i, k in enumerate(keeps)])
    write_cmesh(os.path.splitext(urdf)[0] + ".cmesh", names, lods)
    print(f"wrote {os.path.relpath(urdf, ROOT)} and openarm_cell.cmesh: {len(names)} links, "
          f"{sum(len(l[0][0]) for l in lods)} / {sum(len(l[1][0]) for l in lods)} triangles; table top {top} m; "
          f"left home {[round(home[f'openarm_left_joint{i}'], 4) for i in range(1, 8)]}")


if __name__ == "__main__":
    main()
