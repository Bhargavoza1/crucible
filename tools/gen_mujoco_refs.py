#!/usr/bin/env python3
"""Generate MuJoCo reference numbers for the dynamics tests (doc 11.1).

Runs real MuJoCo on our test robots and writes plain text files into
tests/data/. The C++ tests only READ these files, so ctest never needs
Python or MuJoCo. Run this again only when a test robot changes:

    pip install mujoco
    python3 tools/gen_mujoco_refs.py

For each robot it writes:
  * 10 random states (fixed seed): body poses (xpos, xquat), mass matrix M,
    bias forces (qfrc_bias), and qacc_smooth = M^-1 (tau - bias) for a random tau
  * one 0.5 s trajectory with the Euler integrator, constraints off (no
    contacts, no joint limits): qpos and qvel every 25 steps

Phase 6 scenes (constraints ON), files tests/data/mujoco_scene_*.txt:
  * the robot + optional floor plane (robot geoms touch only the floor) +
    optional PD position actuators on every hinge / slide joint
  * body_invweight0, dof_invweight0 (constants the soft model uses)
  * 10 random states: every constraint row (type, J, pos, aref, R, D, force),
    contacts sorted by geom id so the order is the same as ours, and qacc
  * one 1 s trajectory (Euler, dt 0.002): qpos and qvel every 25 steps

Joint armature / damping (URDF has none): JOINT_EXTRA[robot] = (armature, damping)
is set on every hinge / slide joint and written as a "joint_extra" line; MuJoCo's
implicit Euler damping is turned off (eulerdamp), as our Euler treats it explicitly.

Conventions (same as our engine, see forge/robots/model.h):
  free joint qpos = x y z qw qx qy qz, qvel = world linear velocity of the body
  origin, then BODY-frame angular velocity. Quaternions are written w x y z.
"""
import os
import sys

import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("needs MuJoCo for Python: pip install mujoco")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOTS = [  # (name, urdf, floating base)
    ("pendulum", "assets/robots/pendulum.urdf", False),
    ("arm3", "assets/robots/arm3.urdf", False),
    ("quadruped", "assets/robots/quadruped.urdf", True),
    ("g1_12dof", "assets/robots/g1/g1_12dof.urdf", True),
    ("go2", "assets/robots/go2/go2.urdf", True),
    ("panda", "assets/robots/panda/panda.urdf", False),
    ("openarm", "assets/robots/openarm/openarm.urdf", False),
]
# Unitree's MuJoCo G1 model (g1_12dof.xml) gives every joint armature 0.01 and
# damping 0.001 (and frictionloss 0.1, which our solver does not have yet).
# Unitree's MuJoCo Go2 (unitree_mujoco go2.xml): armature 0.01, damping 0.1
# (frictionloss 0.2 left out, as for the G1).
# Menagerie's Panda (franka_emika_panda/panda.xml): armature 0.1, damping 1.
JOINT_EXTRA = {"g1_12dof": (0.01, 0.001), "go2": (0.01, 0.1), "panda": (0.1, 1.0), "panda_cube": (0.1, 1.0),
               "openarm": (0.016, 1.0), "openarm_cube": (0.016, 1.0), "g1_fallen": (0.01, 0.001),
               "go2_fallen": (0.01, 0.1)}
NUM_CASES = 10
TRAJ_DT = 0.002
TRAJ_STEPS = 250
TRAJ_EVERY = 25


def add_joint_extra(spec, name):
    if name not in JOINT_EXTRA:
        return
    arm, damp = JOINT_EXTRA[name]
    for j in spec.joints:
        if j.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            j.armature = arm
            j.damping = [damp, 0.0, 0.0]  # MuJoCo 4: damping is a 3-vector, [0] = linear damping


def cylinders_to_capsules(spec):
    """crucible reads a URDF <cylinder> as a capsule (forge/robots/urdf.h): the same here.
    MuJoCo's cylinder and capsule both have size = (radius, half length). Also: robot
    geoms get no mass (density 0), as in crucible."""
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        # collision shapes carry no mass in crucible: a link without <inertial> (Go2's
        # calflower links) stays massless, where MuJoCo would give it the geom's mass
        g.density = 0.0


def extra_line(name):
    return [f"joint_extra {JOINT_EXTRA[name][0]:.9g} {JOINT_EXTRA[name][1]:.9g}"] if name in JOINT_EXTRA else []


def load(urdf, floating, name=""):
    spec = mujoco.MjSpec.from_file(os.path.join(ROOT, urdf))
    cylinders_to_capsules(spec)
    if floating:
        spec.worldbody.first_body().add_freejoint(name="root")
    add_joint_extra(spec, name)
    m = spec.compile()
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    m.opt.timestep = TRAJ_DT
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)  # no contacts, no limits
    return m


def random_state(m, rng):
    qpos = np.zeros(m.nq)
    for j in range(m.njnt):
        a = m.jnt_qposadr[j]
        t = m.jnt_type[j]
        if t == mujoco.mjtJoint.mjJNT_FREE:
            qpos[a:a + 3] = rng.uniform(-0.5, 0.5, 3) + [0, 0, 0.8]
            q = rng.normal(size=4)
            qpos[a + 3:a + 7] = q / np.linalg.norm(q)
        elif t == mujoco.mjtJoint.mjJNT_SLIDE:
            qpos[a] = rng.uniform(-0.4, 0.4)
        else:
            qpos[a] = rng.uniform(-1.5, 1.5)
    return qpos, rng.uniform(-2.0, 2.0, m.nv)


def fmt(values):
    return " ".join(f"{v:.9g}" for v in np.asarray(values).ravel())


def write_robot(name, urdf, floating, rng):
    m = load(urdf, floating, name)
    d = mujoco.MjData(m)
    bodies = [m.body(i).name for i in range(1, m.nbody)]  # skip the world
    out = [f"# MuJoCo {mujoco.__version__} reference for {urdf}. Generated by tools/gen_mujoco_refs.py.",
           f"robot {name}", f"urdf {urdf}", f"floating {int(floating)}", f"nq {m.nq}", f"nv {m.nv}",
           f"gravity {fmt(m.opt.gravity)}",
           f"bodies {len(bodies)} " + " ".join(bodies)] + extra_line(name)
    joints = []
    for j in range(m.njnt):
        joints.append(f"{m.joint(j).name} {m.jnt_qposadr[j]} {m.jnt_dofadr[j]} {m.jnt_type[j]}")
    out.append(f"joints {m.njnt} " + " ".join(joints))

    M = np.zeros((m.nv, m.nv))
    for k in range(NUM_CASES):
        qpos, qvel = random_state(m, rng)
        tau = rng.uniform(-5.0, 5.0, m.nv)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        d.qfrc_applied[:] = tau
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, d, M)  # MuJoCo 4 API (3.x was mj_fullM(m, M, d.qM))
        out += [f"case {k}", "qpos " + fmt(d.qpos), "qvel " + fmt(d.qvel), "tau " + fmt(tau),
                "xpos " + fmt(d.xpos[1:]), "xquat " + fmt(d.xquat[1:]), "M " + fmt(M),
                "bias " + fmt(d.qfrc_bias), "qacc " + fmt(d.qacc_smooth)]

    # Trajectory: no applied force, gravity on, Euler, no constraints.
    qpos, qvel = random_state(m, rng)
    d.qpos[:] = qpos
    d.qvel[:] = qvel * 0.5
    d.qfrc_applied[:] = 0
    mujoco.mj_forward(m, d)
    out += [f"traj {TRAJ_DT} {TRAJ_STEPS} {TRAJ_EVERY}", "traj_qpos0 " + fmt(d.qpos), "traj_qvel0 " + fmt(d.qvel)]
    for step in range(1, TRAJ_STEPS + 1):
        mujoco.mj_step(m, d)
        if step % TRAJ_EVERY == 0:
            out += [f"traj_qpos {step} " + fmt(d.qpos), f"traj_qvel {step} " + fmt(d.qvel)]

    path = os.path.join(ROOT, "tests", "data", f"mujoco_{name}.txt")
    with open(path, "w") as f:
        f.write("\n".join(line.rstrip() for line in out) + "\n")
    print(f"wrote {os.path.relpath(path, ROOT)}  (nq {m.nq}, nv {m.nv}, {len(bodies)} bodies)")


# ---- Phase 6: scenes with constraints -------------------------------------------
SCENES = [  # (name, urdf, floating, floor, actuator (kp, kv, force limit) or None)
    ("ball", "assets/robots/ball.urdf", True, True, None),
    ("quadruped", "assets/robots/quadruped.urdf", True, True, (80.0, 2.0, 40.0)),
    ("arm3", "assets/robots/arm3.urdf", False, False, None),
    ("g1_12dof", "assets/robots/g1/g1_12dof.urdf", True, True, (100.0, 3.0, 100.0)),
    ("go2", "assets/robots/go2/go2.urdf", True, True, (20.0, 0.5, 100.0)),  # unitree_rl_gym Go2 PD
    ("ball_cube", "assets/robots/ball.urdf", True, True, None),  # + free cube (CUBES)
    ("panda_cube", "assets/robots/panda/panda.urdf", False, True, (1000.0, 30.0, 100.0)),  # + cube
    ("openarm", "assets/robots/openarm/openarm.urdf", False, True, (100.0, 3.0, 40.0)),  # capsules vs floor
    ("openarm_cube", "assets/robots/openarm/openarm.urdf", False, True, (100.0, 3.0, 40.0)),  # capsules vs cube
    ("g1_fallen", "assets/robots/g1/g1_12dof.urdf", True, True, (100.0, 3.0, 100.0)),  # body capsules on the floor
    ("go2_fallen", "assets/robots/go2/go2.urdf", True, True, (20.0, 0.5, 100.0)),
]
# Scenes with a free solid box named "cube" added after the robot (Phase 9a):
# half sizes (m) and mass (kg). It touches the floor and the robot's spheres.
CUBES = {"ball_cube": ((0.03, 0.04, 0.05), 0.2), "panda_cube": ((0.025, 0.025, 0.025), 0.1),
         "openarm_cube": ((0.03, 0.04, 0.05), 0.2)}
# OpenArm (Phase 10): elbow bent, forearm forward and level about 0.3 m up
OPENARM_READY = [0.6, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, -0.4, -0.4]
# Menagerie's Panda "home" keyframe (arm joints): hand pointing down
PANDA_HOME = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
PANDA_PAD_Z = 0.0584 + 0.0445  # finger pad center below the hand origin, along the hand's z
PANDA_PAD_FACE = 0.0015        # pad inner face, from the finger origin along its y
# Unitree's standing pose for the 12-joint G1 (unitree_rl_gym g1_config.py), per leg:
# hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll
G1_STAND = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2
# unitree_rl_gym's Go2 standing pose (go2_config.py), hip, thigh, calf for FL, FR, RL, RR
GO2_STAND = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]


def geom_low(m, d, g):
    """Lowest z of a sphere or capsule geom."""
    z = d.geom_xpos[g][2] - m.geom_size[g][0]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
        z -= abs(d.geom_xmat[g][8]) * m.geom_size[g][1]
    return z


def g1_set_height(m, qpos, gap):
    """Root height so the lowest collision shape is 'gap' above the floor (negative: inside)."""
    d = mujoco.MjData(m)
    qpos = qpos.copy()
    qpos[2] = 0.0
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    low = min(geom_low(m, d, g) for g in range(m.ngeom) if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE)
    qpos[2] = -low + gap
    return qpos
SCENE_TRAJ_STEPS = 500


def load_scene(urdf, floating, floor, act, name=""):
    spec = mujoco.MjSpec.from_file(os.path.join(ROOT, urdf))
    cylinders_to_capsules(spec)
    if floating:
        spec.worldbody.first_body().add_freejoint(name="root")
    add_joint_extra(spec, name)
    for g in spec.geoms:  # robot geoms touch the floor only: no self contacts
        g.contype = 1
        g.conaffinity = 0
    if name in CUBES:
        half, mass = CUBES[name]
        b = spec.worldbody.add_body()
        b.name = "cube"
        b.add_freejoint(name="cube")
        g = b.add_geom()
        g.name = "cube"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(half)
        g.mass = mass  # inertia from the geom: solid box
        g.contype = 1
        g.conaffinity = 1  # robot spheres (contype 1) touch it
    if floor:
        f = spec.worldbody.add_geom()
        f.name = "floor"
        f.type = mujoco.mjtGeom.mjGEOM_PLANE
        f.size = [0, 0, 1]
        f.contype = 0
        f.conaffinity = 1
    if act:
        kp, kv, flim = act
        for j in spec.joints:
            if j.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                a = spec.add_actuator()
                a.target = j.name
                a.trntype = mujoco.mjtTrn.mjTRN_JOINT
                a.set_to_position(kp=kp, kv=kv)
                a.forcelimited = 1
                a.forcerange = [-flim, flim]
    m = spec.compile()
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    m.opt.timestep = TRAJ_DT
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    return m


def scene_state(name, m, rng):
    """Random states that hit the constraints."""
    qpos = m.qpos0.copy()
    qvel = rng.uniform(-1.0, 1.0, m.nv)
    if name == "ball":
        qpos[:3] = [rng.uniform(-1, 1), rng.uniform(-1, 1), 0.1 + rng.uniform(-0.01, 0.003)]
        q = rng.normal(size=4)
        qpos[3:7] = q / np.linalg.norm(q)
        qvel *= 2.0
    elif name == "quadruped":
        qpos[:3] = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.5 + rng.uniform(-0.02, 0.01)]
        q = np.array([1.0, 0, 0, 0]) + 0.05 * rng.normal(size=4)
        qpos[3:7] = q / np.linalg.norm(q)
        qpos[7:] = rng.uniform(-0.3, 0.3, m.nq - 7)
    elif name == "g1_12dof":  # near the standing pose, feet touching or slightly inside the floor
        qpos[7:] = np.array(G1_STAND) + rng.uniform(-0.15, 0.15, 12)
        q = np.array([1.0, 0, 0, 0]) + 0.03 * rng.normal(size=4)
        qpos[3:7] = q / np.linalg.norm(q)
        qpos[:2] = rng.uniform(-0.5, 0.5, 2)
        qpos = g1_set_height(m, qpos, rng.uniform(-0.003, 0.001))
        qvel *= 0.5
    elif name == "go2":  # near the standing pose, feet touching or slightly inside the floor
        qpos[7:] = np.array(GO2_STAND) + rng.uniform(-0.15, 0.15, 12)
        q = np.array([1.0, 0, 0, 0]) + 0.03 * rng.normal(size=4)
        qpos[3:7] = q / np.linalg.norm(q)
        qpos[:2] = rng.uniform(-0.5, 0.5, 2)
        qpos = g1_set_height(m, qpos, rng.uniform(-0.003, 0.001))
        qvel *= 0.5
    elif name in ("g1_fallen", "go2_fallen"):  # any orientation, lowest shape touching the floor
        stand = G1_STAND if name == "g1_fallen" else GO2_STAND
        qpos[7:] = np.array(stand) + rng.uniform(-0.3, 0.3, 12)
        q = rng.normal(size=4)
        qpos[3:7] = q / np.linalg.norm(q)
        qpos[:2] = rng.uniform(-0.5, 0.5, 2)
        qpos = g1_set_height(m, qpos, rng.uniform(-0.003, 0.001))
        qvel *= 0.5
    elif name == "ball_cube":  # cube on the floor (a corner or face slightly in), ball touching the cube
        half = np.array(CUBES[name][0])
        q = np.array([1.0, 0, 0, 0]) + 0.25 * rng.normal(size=4)
        qc = q / np.linalg.norm(q)
        R = quat_mat(qc)
        corners = np.array([[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]) * half
        low = min((R @ c)[2] for c in corners)
        cube = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), -low + rng.uniform(-0.003, 0.001)])
        face = rng.integers(0, 5)  # +x -x +y -y +z: a point on that face, maybe past its edge
        axis, sign = face // 2, (1.0 if face % 2 == 0 else -1.0)
        if face == 4:
            axis, sign = 2, 1.0
        local = rng.uniform(-1.2, 1.2, 3) * half
        local[axis] = sign * half[axis]
        r = 0.1
        outward = np.zeros(3)
        outward[axis] = sign
        ball_local = local + outward * (r + rng.uniform(-0.004, 0.001))
        qpos[0:3] = cube + R @ ball_local
        qpos[3:7] = [1, 0, 0, 0]
        qpos[7:10] = cube
        qpos[10:14] = qc
        qvel *= 0.5
    elif name == "panda_cube":  # arm near home, cube between the finger pads (pads slightly in or just off)
        half = CUBES[name][0][0]
        qpos[:7] = np.array(PANDA_HOME) + rng.uniform(-0.3, 0.3, 7)
        qpos[7:9] = half - PANDA_PAD_FACE - rng.uniform(-0.001, 0.003, 2)
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        R = d.xmat[hand].reshape(3, 3)
        offset = np.array([rng.uniform(-0.008, 0.008), 0.0, PANDA_PAD_Z + rng.uniform(-0.01, 0.01)])
        qc = np.zeros(4)
        mujoco.mju_mat2Quat(qc, R.flatten())
        yaw = rng.uniform(-0.1, 0.1)  # turn the cube a little about the hand's z
        qy = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        qcube = np.zeros(4)
        mujoco.mju_mulQuat(qcube, qc, qy)
        qpos[9:12] = d.xpos[hand] + R @ offset
        qpos[12:16] = qcube
        qvel *= 0.3
    elif name == "openarm":  # random joints in range: some capsules in the floor (shoulder 0.45 m up)
        for j in range(m.njnt):
            lo, hi = m.jnt_range[j]
            qpos[m.jnt_qposadr[j]] = rng.uniform(lo, hi)
        qvel *= 0.5
    elif name == "openarm_cube":  # cube touching a random capsule: end, side, any angle
        for j in range(len(OPENARM_READY)):  # the arm's hinges (the cube's free joint is last)
            lo, hi = m.jnt_range[j]
            qpos[m.jnt_qposadr[j]] = np.clip(OPENARM_READY[j] + rng.uniform(-0.4, 0.4), lo, hi)
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        caps = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]
        g = caps[rng.integers(0, len(caps))]
        r, hl = m.geom_size[g][0], m.geom_size[g][1]
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
        on_axis = d.geom_xpos[g] + axis * rng.uniform(-1.2, 1.2) * hl
        q = rng.normal(size=4)
        qc = q / np.linalg.norm(q)
        Rc = quat_mat(qc)
        half = np.array(CUBES[name][0])
        dirn = rng.normal(size=3)
        dirn /= np.linalg.norm(dirn)
        # move the cube out along dirn until its surface is about r from the segment point
        local = Rc.T @ dirn
        reach = np.min(half / np.maximum(np.abs(local), 1e-9))  # center to the box surface along dirn
        qpos[9:12] = on_axis + dirn * (reach + r + rng.uniform(-0.006, 0.001))
        qpos[12:16] = qc
        qvel *= 0.3
    else:  # arm3: some joints past their limits
        for j in range(m.njnt):
            a = m.jnt_qposadr[j]
            if m.jnt_limited[j]:
                lo, hi = m.jnt_range[j]
                qpos[a] = rng.choice([lo, hi]) + rng.uniform(-0.1, 0.1)
            else:
                qpos[a] = rng.uniform(-3, 3)
    return qpos, qvel


def scene_traj_start(name, m):
    qpos = m.qpos0.copy()
    qvel = np.zeros(m.nv)
    ctrl = np.zeros(m.nu)
    if name == "ball":  # dropped while sliding and spinning: bounce, slip, then roll
        qpos[:3] = [0, 0, 0.3]
        qvel[:3] = [1.5, 0.0, 0.0]
        qvel[3:6] = [0.0, -5.0, 3.0]
    elif name == "quadruped":  # dropped from 6 cm above standing height, PD holds a pose
        qpos[:3] = [0, 0, 0.56]
        ctrl[:] = [0.0, 0.3] * 4  # hip, knee for each leg
    elif name == "g1_12dof":  # dropped 2 cm onto the floor, PD holds Unitree's standing pose
        qpos[7:] = G1_STAND
        qpos = g1_set_height(m, qpos, 0.02)
        ctrl[:] = G1_STAND
    elif name == "go2":  # dropped 2 cm onto the floor, PD holds Unitree's standing pose
        qpos[7:] = GO2_STAND
        qpos = g1_set_height(m, qpos, 0.02)
        ctrl[:] = GO2_STAND
    elif name in ("g1_fallen", "go2_fallen"):  # dropped 5 cm, tipped over, PD holds the standing pose
        # Starts picked so no contact starts at a near-tie (|dist| < 1e-5 m when it
        # appears): float32 vs double would put such a contact one step apart. G1:
        # pitched forward about 80 degrees (rolled 70 had a foot touch at 5e-9 m);
        # Go2: rolled about 70 degrees onto its side.
        stand = G1_STAND if name == "g1_fallen" else GO2_STAND
        qpos[7:] = stand
        qpos[3:7] = [np.cos(0.7), 0.0, np.sin(0.7), 0.1] if name == "g1_fallen" else [np.cos(0.6), np.sin(0.6), 0.1, 0.0]
        qpos[3:7] /= np.linalg.norm(qpos[3:7])
        qpos = g1_set_height(m, qpos, 0.05)
        ctrl[:] = stand
    elif name == "ball_cube":  # tilted cube dropped onto the floor, ball dropped onto the cube
        q = np.array([1.0, 0.15, -0.1, 0.05])
        qpos[0:3] = [0.02, 0.0, 0.35]
        qpos[3:7] = [1, 0, 0, 0]
        qvel[0:3] = [0.2, 0.0, 0.0]
        qpos[7:10] = [0.0, 0.0, 0.1]
        qpos[10:14] = q / np.linalg.norm(q)
    elif name == "panda_cube":  # hand down around a cube on the floor, fingers open; they close and squeeze
        half = CUBES[name][0][0]
        arm = panda_ik(m, np.array([0.5, 0.0, half + PANDA_PAD_Z]))
        qpos[:7] = arm
        qpos[7:9] = 0.04
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        R = d.xmat[hand].reshape(3, 3)
        pad = d.xpos[hand] + R @ np.array([0, 0, PANDA_PAD_Z])
        yaw = np.arctan2(R[1, 0], R[0, 0])  # cube faces line up with the fingers
        qpos[9:12] = [pad[0], pad[1], half]
        qpos[12:16] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        ctrl[:7] = arm
        ctrl[7:9] = 0.0
    elif name == "openarm":  # from the ready pose, PD drives the arm down: it lands on the floor
        qpos[:9] = OPENARM_READY
        ctrl[:] = OPENARM_READY
        ctrl[0] = 1.6  # shoulder pitch: hand down past the floor
        ctrl[3] = 0.3
    elif name == "openarm_cube":  # PD holds the ready pose; a tilted cube falls onto the forearm
        qpos[:9] = OPENARM_READY
        ctrl[:] = OPENARM_READY
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        fore = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link5")]
        q = np.array([1.0, 0.2, -0.1, 0.05])
        qpos[9:12] = fore + [0.0, 0.0, 0.16]
        qpos[12:16] = q / np.linalg.norm(q)
    else:  # arm3: slider starts inside its range and falls onto the limit
        qpos[:] = [0.3, 0.2, 2.5]
    return qpos, qvel, ctrl


def panda_ik(m, target, iters=300):
    """Arm joints (from home) that put the hand origin at 'target' with home's
    orientation (pointing down): damped least squares on position and rotation,
    small steps, clipped to the joint ranges."""
    d = mujoco.MjData(m)
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    q = np.zeros(m.nq)
    q[:7] = PANDA_HOME
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    R0 = d.xmat[hand].reshape(3, 3).copy()
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacBody(m, d, jp, jr, hand)
        R = d.xmat[hand].reshape(3, 3)
        er = 0.5 * (np.cross(R[:, 0], R0[:, 0]) + np.cross(R[:, 1], R0[:, 1]) + np.cross(R[:, 2], R0[:, 2]))
        e = np.concatenate([target - d.xpos[hand], er])
        J = np.vstack([jp, jr])[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        q[:7] += dq * min(1.0, 0.1 / max(1e-12, np.abs(dq).max()))  # steps of at most 0.1 rad
        q[:7] = np.clip(q[:7], m.jnt_range[:7, 0], m.jnt_range[:7, 1])  # stay inside the joint limits
    assert np.linalg.norm(e) < 1e-6, f"panda_ik did not converge: {e}"
    return q[:7]


def quat_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def contact_order(m):
    """Sort key for MuJoCo contacts in our row order (constraints.h): per sphere
    (planes, then boxes), then per box (planes; corners keep MuJoCo's order), then per
    capsule (planes, then boxes)."""
    P, S, B = mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_BOX
    C = mujoco.mjtGeom.mjGEOM_CAPSULE
    rank = {}
    for t in (P, S, B, C):
        ids = [g for g in range(m.ngeom) if m.geom_type[g] == t]
        rank.update({g: i for i, g in enumerate(ids)})
    def key(con):
        g1, g2 = con.geom[0], con.geom[1]
        t1, t2 = m.geom_type[g1], m.geom_type[g2]
        if t1 == P and t2 == S:
            return (0, rank[g2], 0, rank[g1])
        if t1 == S and t2 == B:
            return (0, rank[g1], 1, rank[g2])
        if t1 == P and t2 == B:
            return (1, rank[g2], 0, rank[g1])
        if t1 == P and t2 == C:  # capsule ends: MuJoCo's order kept (stable sort)
            return (2, rank[g2], 0, rank[g1])
        if t1 == C and t2 == B:
            return (2, rank[g1], 1, rank[g2])
        raise ValueError(f"contact type {t1}-{t2} not supported")
    return key


def write_scene(name, urdf, floating, floor, act, rng):
    m = load_scene(urdf, floating, floor, act, name)
    d = mujoco.MjData(m)
    bodies = [m.body(i).name for i in range(1, m.nbody)]
    out = [f"# MuJoCo {mujoco.__version__} scene '{name}' with constraints. Generated by tools/gen_mujoco_refs.py.",
           f"robot {name}", f"urdf {urdf}", f"floating {int(floating)}", f"nq {m.nq}", f"nv {m.nv}",
           f"gravity {fmt(m.opt.gravity)}", f"bodies {len(bodies)} " + " ".join(bodies)] + extra_line(name)
    out.append(f"joints {m.njnt} " + " ".join(
        f"{m.joint(j).name} {m.jnt_qposadr[j]} {m.jnt_dofadr[j]} {m.jnt_type[j]}" for j in range(m.njnt)))
    out.append(f"floor {int(floor)}")
    if name in CUBES:
        out.append(f"cube {fmt(CUBES[name][0])} {CUBES[name][1]:.9g}")
    if act:
        names = [m.joint(m.actuator_trnid[k][0]).name for k in range(m.nu)]
        out.append(f"actuators {act[0]:.9g} {act[1]:.9g} {act[2]:.9g} {m.nu} " + " ".join(names))
    out += ["body_invweight0 " + fmt(m.body_invweight0[1:]), "dof_invweight0 " + fmt(m.dof_invweight0)]

    for k in range(NUM_CASES):
        qpos, qvel = scene_state(name, m, rng)
        ctrl = rng.uniform(-0.5, 0.5, m.nu)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        d.ctrl[:] = ctrl
        mujoco.mj_forward(m, d)
        nl = d.nl
        rows = list(range(nl))  # limits first (MuJoCo order = ours)
        key = contact_order(m)
        cons = sorted(range(d.ncon), key=lambda c: key(d.contact[c]))  # stable: box corners keep their order
        for c in cons:
            a = d.contact[c].efc_address
            rows += list(range(a, a + (2 * (d.contact[c].dim - 1) if d.contact[c].dim > 1 else 1)))
        assert len(rows) == d.nefc
        J = d.efc_J.reshape(d.nefc, m.nv)[rows] if d.nefc else np.zeros((0, m.nv))
        typ = [0 if d.efc_type[r] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT else 1 for r in rows]
        out += [f"case {k}", "qpos " + fmt(d.qpos), "qvel " + fmt(d.qvel), "tau " + fmt(np.zeros(m.nv)),
                "ctrl " + fmt(ctrl), f"nefc {d.nefc}", "efc_type " + " ".join(map(str, typ)),
                "efc_J " + fmt(J), "efc_pos " + fmt(d.efc_pos[rows]), "efc_aref " + fmt(d.efc_aref[rows]),
                "efc_R " + fmt(d.efc_R[rows]), "efc_D " + fmt(d.efc_D[rows]),
                "efc_force " + fmt(d.efc_force[rows]), "qacc_constrained " + fmt(d.qacc)]

    qpos, qvel, ctrl = scene_traj_start(name, m)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    d.ctrl[:] = ctrl
    mujoco.mj_forward(m, d)
    out += [f"traj {TRAJ_DT} {SCENE_TRAJ_STEPS} {TRAJ_EVERY}", "traj_ctrl " + fmt(ctrl),
            "traj_qpos0 " + fmt(d.qpos), "traj_qvel0 " + fmt(d.qvel)]
    for step in range(1, SCENE_TRAJ_STEPS + 1):
        mujoco.mj_step(m, d)
        if step % TRAJ_EVERY == 0:
            out += [f"traj_qpos {step} " + fmt(d.qpos), f"traj_qvel {step} " + fmt(d.qvel)]

    path = os.path.join(ROOT, "tests", "data", f"mujoco_scene_{name}.txt")
    with open(path, "w") as f:
        f.write("\n".join(line.rstrip() for line in out) + "\n")
    print(f"wrote {os.path.relpath(path, ROOT)}  (nv {m.nv}, nu {m.nu})")


CAPSULE_BOX_CASES = 400


def write_capsule_box():
    """Phase 10: MuJoCo's capsule-box contacts (mjc_CapsuleBox) for many random
    capsules and boxes that touch or nearly touch: every corner, edge and face case.
    tests/data/mujoco_capsule_box.txt; checked by tests/cpu/forge/solver/test_capsule_box.cpp."""
    rng = np.random.default_rng(20261117)
    spec = mujoco.MjSpec()
    cb = spec.worldbody.add_body()
    cb.name, cb.mocap = "capsule", True
    gc = cb.add_geom()
    gc.type, gc.size = mujoco.mjtGeom.mjGEOM_CAPSULE, [0.05, 0.1, 0]
    bb = spec.worldbody.add_body()
    bb.name = "box"
    bb.add_freejoint()
    gb = bb.add_geom()
    gb.type, gb.size = mujoco.mjtGeom.mjGEOM_BOX, [0.1, 0.1, 0.1]
    m = spec.compile()
    d = mujoco.MjData(m)
    out = [f"# MuJoCo {mujoco.__version__} capsule-box contacts (mjc_CapsuleBox), {CAPSULE_BOX_CASES} random pairs.",
           "# Generated by tools/gen_mujoco_refs.py. Per case: 'case' line with capsule pos (3) quat (4, w x y z)",
           "# radius half_length, box pos (3) quat (4) half sizes (3), ncon; then per contact: dist pos (3) normal (3)",
           "# (normal from the capsule into the box)."]
    for k in range(CAPSULE_BOX_CASES):
        r, hl = rng.uniform(0.01, 0.06), rng.uniform(0.0, 0.15)
        half = rng.uniform(0.02, 0.12, 3)
        qc = rng.normal(size=4)
        qc /= np.linalg.norm(qc)
        qb = rng.normal(size=4)
        qb /= np.linalg.norm(qb)
        if k % 4 == 0:  # capsules NEARLY lined up with the box (faces / edges almost parallel): 2-contact
            # cases. Not exactly: with an exact tie, float and double may pick different (equally
            # good) contact points, as any two implementations could.
            base = np.array([1.0, 0, 0, 0]) if k % 8 == 0 else np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
            tilt = rng.normal(size=3)
            tilt *= rng.uniform(0.002, 0.01) / np.linalg.norm(tilt)
            qt = np.zeros(4)
            mujoco.mju_axisAngle2Quat(qt, tilt / np.linalg.norm(tilt), np.linalg.norm(tilt))
            mujoco.mju_mulQuat(qc, base, qt)
            qb = np.array([1.0, 0, 0, 0])
        Rb = quat_mat(qb)
        dirn = rng.normal(size=3)
        dirn /= np.linalg.norm(dirn)
        local = Rb.T @ dirn
        reach = np.min(half / np.maximum(np.abs(local), 1e-9))
        Rc = quat_mat(qc)
        pc = np.zeros(3)
        pb = pc - dirn * (reach + r + rng.uniform(-0.02, 0.005)) - Rc[:, 2] * rng.uniform(-1.0, 1.0) * hl
        m.geom_size[0] = [r, hl, 0]
        m.geom_rbound[0] = r + hl
        m.geom_size[1] = half
        m.geom_rbound[1] = np.linalg.norm(half)
        d.mocap_pos[0], d.mocap_quat[0] = pc, qc
        d.qpos[:3], d.qpos[3:7] = pb, qb
        mujoco.mj_forward(m, d)
        cons = [c for c in d.contact[:d.ncon]]
        out.append(f"case {fmt(pc)} {fmt(qc)} {r:.9g} {hl:.9g} {fmt(pb)} {fmt(qb)} {fmt(half)} {len(cons)}")
        for c in cons:
            assert c.geom[0] == 0 and c.geom[1] == 1
            out.append(f"{c.dist:.9g} {fmt(c.pos)} {fmt(c.frame[:3])}")
    path = os.path.join(ROOT, "tests/data/mujoco_capsule_box.txt")
    open(path, "w").write("\n".join(out) + "\n")
    print(f"wrote tests/data/mujoco_capsule_box.txt  ({CAPSULE_BOX_CASES} pairs)")


BOX_BOX_CASES = 600


def write_box_box():
    """Phase 10b: MuJoCo's box-box contacts (mjraw_BoxBox) for random pairs: any
    turn and place (edge-edge, face, corner cases), a box resting on another (face
    contacts, up to 8 points) and boxes tilted a little. Pairs apart give 0 contacts.
    tests/data/mujoco_box_box.txt; checked by tests/cpu/forge/solver/test_box_box.cpp."""
    rng = np.random.default_rng(20261118)
    spec = mujoco.MjSpec()
    for name in ("box1", "box2"):
        b = spec.worldbody.add_body()
        b.name = name
        b.add_freejoint()
        g = b.add_geom()
        g.type, g.size = mujoco.mjtGeom.mjGEOM_BOX, [0.1, 0.1, 0.1]
    m = spec.compile()
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY
    d = mujoco.MjData(m)

    def unit_quat():
        q = rng.normal(size=4)
        return q / np.linalg.norm(q)

    def small_quat(a):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        t = rng.uniform(-a, a)
        return np.r_[np.cos(t / 2), np.sin(t / 2) * ax]

    out = [f"# MuJoCo {mujoco.__version__} box-box contacts (mjraw_BoxBox), {BOX_BOX_CASES} random pairs.",
           "# Generated by tools/gen_mujoco_refs.py. Per case: 'case' line with box 1 pos (3) quat (4, w x y z)",
           "# half sizes (3), box 2 pos (3) quat (4) half sizes (3), ncon; then per contact: dist pos (3) normal (3)",
           "# (normal from box 1 to box 2)"]
    for t in range(BOX_BOX_CASES):
        h1, h2 = rng.uniform(0.01, 0.2, 3), rng.uniform(0.01, 0.2, 3)
        kind = t % 3
        if kind == 0:  # any turn, any place
            q1, q2, p2 = unit_quat(), unit_quat(), rng.normal(size=3) * 0.15
        elif kind == 1:  # box 2 resting on box 1's top face, a little in
            q1, q2 = np.array([1.0, 0, 0, 0]), small_quat(0.05)
            p2 = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), h1[2] + h2[2] - rng.uniform(0, 0.01)])
        else:  # both tilted a little
            q1, q2, p2 = small_quat(0.3), small_quat(0.3), rng.normal(size=3) * 0.2
        p1 = np.zeros(3)
        m.geom_size[0], m.geom_size[1] = h1, h2
        m.geom_rbound[0], m.geom_rbound[1] = np.linalg.norm(h1), np.linalg.norm(h2)
        m.geom_aabb[0], m.geom_aabb[1] = np.r_[0, 0, 0, h1], np.r_[0, 0, 0, h2]  # the broad phase uses these too
        d.qpos[:] = np.r_[p1, q1, p2, q2]
        mujoco.mj_forward(m, d)
        cons = [c for c in d.contact[:d.ncon]]
        out.append(f"case {fmt(p1)} {fmt(q1)} {fmt(h1)} {fmt(p2)} {fmt(q2)} {fmt(h2)} {len(cons)}")
        for c in cons:
            assert c.geom[0] == 0 and c.geom[1] == 1
            out.append(f"{c.dist:.9g} {fmt(c.pos)} {fmt(c.frame[:3])}")
    path = os.path.join(ROOT, "tests/data/mujoco_box_box.txt")
    open(path, "w").write("\n".join(out) + "\n")
    print(f"wrote tests/data/mujoco_box_box.txt  ({BOX_BOX_CASES} pairs)")


PRIM_CASES = 300  # per kind


def write_round_pairs():
    """Phase 10d: MuJoCo's sphere-sphere, sphere-capsule and capsule-capsule contacts
    (mjraw_SphereSphere, mjraw_SphereCapsule, mjraw_CapsuleCapsule) for random pairs
    that touch or nearly touch; a third of the capsule pairs parallel (2 contacts).
    tests/data/mujoco_round_pairs.txt; checked by tests/cpu/forge/solver/test_round_pairs.cpp."""
    rng = np.random.default_rng(20261119)
    kinds = [("ss", mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_SPHERE),
             ("sc", mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE),
             ("cc", mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CAPSULE)]

    def unit_quat():
        q = rng.normal(size=4)
        return q / np.linalg.norm(q)

    out = [f"# MuJoCo {mujoco.__version__} sphere / capsule pair contacts, {PRIM_CASES} random pairs per kind.",
           "# Generated by tools/gen_mujoco_refs.py. Per case: 'case' kind (ss, sc, cc), shape 1 pos (3) quat (4,",
           "# w x y z) radius half_length, shape 2 the same, ncon; then per contact: dist pos (3) normal (3)",
           "# (normal from shape 1 to shape 2; a sphere's half_length is 0)"]
    for name, t1, t2 in kinds:
        spec = mujoco.MjSpec()
        for k, t in enumerate((t1, t2)):
            b = spec.worldbody.add_body()
            b.name = f"g{k}"
            b.add_freejoint()
            g = b.add_geom()
            g.type, g.size = t, [0.1, 0.1, 0]
        m = spec.compile()
        d = mujoco.MjData(m)
        for t in range(PRIM_CASES):
            r1, r2 = rng.uniform(0.01, 0.08, 2)
            h1 = 0.0 if t1 == mujoco.mjtGeom.mjGEOM_SPHERE else rng.uniform(0.0, 0.15)
            h2 = rng.uniform(0.0, 0.15) if t2 == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
            q1 = unit_quat()
            q2 = q1.copy() if (name == "cc" and t % 3 == 0) else unit_quat()  # parallel
            if name == "cc" and t % 3 == 0 and t % 2 == 0:
                q2 = np.array([-q1[0], -q1[1], -q1[2], -q1[3]])  # the same axis, other sign
            p1 = np.zeros(3)
            dirn = rng.normal(size=3)
            dirn /= np.linalg.norm(dirn)
            p2 = dirn * (r1 + r2 + rng.uniform(-0.03, 0.01)) + rng.normal(size=3) * 0.5 * (h1 + h2)
            for k, (r, h) in enumerate(((r1, h1), (r2, h2))):
                m.geom_size[k] = [r, h, 0]
                m.geom_rbound[k] = r + h
                m.geom_aabb[k] = [0, 0, 0, r, r, r + h]
            d.qpos[:] = np.r_[p1, q1, p2, q2]
            mujoco.mj_forward(m, d)
            cons = [c for c in d.contact[:d.ncon]]
            out.append(f"case {name} {fmt(p1)} {fmt(q1)} {r1:.9g} {h1:.9g} {fmt(p2)} {fmt(q2)} {r2:.9g} {h2:.9g} "
                       f"{len(cons)}")
            for c in cons:
                assert c.geom[0] == 0 and c.geom[1] == 1
                out.append(f"{c.dist:.9g} {fmt(c.pos)} {fmt(c.frame[:3])}")
    path = os.path.join(ROOT, "tests/data/mujoco_round_pairs.txt")
    open(path, "w").write("\n".join(out) + "\n")
    print(f"wrote tests/data/mujoco_round_pairs.txt  ({PRIM_CASES} pairs per kind)")


def main():
    rng = np.random.default_rng(20261115)  # fixed seed: same files every run
    for name, urdf, floating in ROBOTS:
        write_robot(name, urdf, floating, rng)
    rng = np.random.default_rng(20261116)  # own seed: adding scenes does not change the files above
    for scene in SCENES:
        write_scene(*scene, rng)
    write_capsule_box()
    write_box_box()
    write_round_pairs()


if __name__ == "__main__":
    main()
