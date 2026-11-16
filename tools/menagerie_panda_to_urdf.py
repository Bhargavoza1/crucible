"""Menagerie panda.xml (Apache-2) -> assets/robots/panda/panda.urdf (Phase 9a).

Reads body poses, inertials, joints and limits from panda.xml's text (not the
compiled model, so values are written exactly as given there). Check: loading
both in MuJoCo gives the same body poses and mass matrix (minus armature) to ~1e-11.

    git clone https://github.com/google-deepmind/mujoco_menagerie
    python tools/menagerie_panda_to_urdf.py mujoco_menagerie/franka_emika_panda/panda.xml
"""
import math, os, sys, xml.etree.ElementTree as ET
import numpy as np
if len(sys.argv) != 2:
    sys.exit(__doc__)
SRC = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets/robots/panda/panda.urdf")
root = ET.parse(SRC).getroot()
DEF_RANGE = "-2.8973 2.8973"
EFFORT = {"joint1": 87, "joint2": 87, "joint3": 87, "joint4": 87, "joint5": 12, "joint6": 12, "joint7": 12}
FINGER_EFFORT = 50  # Menagerie: tendon actuator force 100, split 0.5 / 0.5 over the two fingers

def quat_to_rpy(q):
    w, x, y, z = np.array(q, float) / np.linalg.norm(q)
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    # URDF: R = Rz(yaw) Ry(pitch) Rx(roll)
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2]); yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1]); yaw = 0.0
    return roll, pitch, yaw

def g(v, n=9): return " ".join(f"{float(x):.{n}g}" for x in v)

links, joints = [], []
# Finger pad spheres (3 per finger, so the Panda + cube fits the GPU warp kernel): radius 4 mm, centered on Menagerie's main pad box (size 0.0085
# 0.004 0.0085 at 0 0.0055 0.0445), so each sphere's inner side is the pad's inner face.
PAD = [(-0.0045, 0.0055, 0.040), (0.0045, 0.0055, 0.040), (0.0, 0.0055, 0.049)]  # two low, one high
def walk(body, parent):
    name = body.get("name")
    inr = body.find("inertial")
    mass = float(inr.get("mass")); pos = [float(x) for x in inr.get("pos").split()]
    if inr.get("fullinertia"):
        ixx, iyy, izz, ixy, ixz, iyz = [float(x) for x in inr.get("fullinertia").split()]
    else:
        ixx, iyy, izz = [float(x) for x in inr.get("diaginertia").split()]; ixy = ixz = iyz = 0.0
    col = ""
    if name.endswith("_finger"):
        col = "".join(f'\n    <collision><origin xyz="{g(p)}"/><geometry><sphere radius="0.004"/></geometry></collision>' for p in PAD)
    links.append(f'''  <link name="{name}">
    <inertial><origin xyz="{g(pos)}"/><mass value="{mass:.9g}"/>
      <inertia ixx="{ixx:.9g}" ixy="{ixy:.9g}" ixz="{ixz:.9g}" iyy="{iyy:.9g}" iyz="{iyz:.9g}" izz="{izz:.9g}"/></inertial>{col}
  </link>''')
    if parent is not None:
        bpos = [float(x) for x in body.get("pos", "0 0 0").split()]
        rpy = quat_to_rpy([float(x) for x in body.get("quat", "1 0 0 0").split()])
        j = body.find("joint")
        if j is None:
            joints.append(f'''  <joint name="{name}_fixed" type="fixed"><parent link="{parent}"/><child link="{name}"/>
    <origin xyz="{g(bpos)}" rpy="{g(rpy, 12)}"/></joint>''')
        elif j.get("class") == "finger":
            joints.append(f'''  <joint name="{j.get("name")}" type="prismatic"><parent link="{parent}"/><child link="{name}"/>
    <origin xyz="{g(bpos)}" rpy="{g(rpy, 12)}"/><axis xyz="0 1 0"/>
    <limit lower="0" upper="0.04" effort="{FINGER_EFFORT}" velocity="0.2"/></joint>''')
        else:
            lo, hi = (j.get("range") or DEF_RANGE).split()
            joints.append(f'''  <joint name="{j.get("name")}" type="revolute"><parent link="{parent}"/><child link="{name}"/>
    <origin xyz="{g(bpos)}" rpy="{g(rpy, 12)}"/><axis xyz="0 0 1"/>
    <limit lower="{lo}" upper="{hi}" effort="{EFFORT[j.get("name")]}" velocity="2.6"/></joint>''')
    for c in body.findall("body"):
        walk(c, name)

walk(root.find("worldbody").find("body"), None)
head = '''<?xml version="1.0"?>
<!-- Franka Emika Panda arm + hand, 7 arm joints and 2 finger joints (fixed base).
     From google-deepmind/mujoco_menagerie, franka_emika_panda/panda.xml
     (Apache License 2.0, Copyright DeepMind Technologies Limited; see LICENSE in
     this folder). Converted to URDF for crucible by a script (masses, inertias,
     body poses, joint axes and ranges as in panda.xml). Changes:
       * no visual or mesh shapes (visual meshes come in Phase 10)
       * the arm links have no collision shapes; each finger has 3 collision
         spheres (radius 4 mm, two low and one high) on its pad, in place of
         Menagerie's pad boxes (our engine has sphere-box contacts but no box-box)
       * the two fingers are separate joints (Menagerie couples them with a tendon
         and an equality constraint, which our engine does not have); give both
         the same target
       * effort: joints 1-4 87 Nm, 5-7 12 Nm (Menagerie forcerange), fingers 50 N
     Menagerie's joints have armature 0.1 and damping 1 (UrdfOptions). -->
<robot name="panda">
  <mujoco><compiler fusestatic="false" discardvisual="true" balanceinertia="false"/></mujoco>
'''
open(OUT, "w").write(head + "\n".join(links + joints) + "\n</robot>\n")
print(f"wrote {os.path.relpath(OUT)}: {len(links)} links, {len(joints)} joints")
