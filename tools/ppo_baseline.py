#!/usr/bin/env python3
"""PPO baseline in PyTorch (doc 5.7, Phase 7 done check), on the same tasks
(cartpole, ant) as forge/env/tasks/.

The point: the same algorithm, written in a well-known way (CleanRL's
ppo_continuous_action.py structure), with the SAME settings as temper's PPO, on
the same physics (MuJoCo, our cartpole URDF). If our C++/CUDA PPO learns about as
fast as this script, the implementation is right. If it is much slower or
worse, there is a bug.

Same as temper (and therefore NOT CleanRL's defaults, on purpose):
  * separate actor and critic MLPs, 3 x 256, ELU (CleanRL: 2 x 64, tanh)
  * orthogonal init: gain sqrt(2) hidden, 0.01 actor output, 1.0 critic output
  * state-independent log_std, init 0
  * obs normalizer: running mean/var, updated with every step's obs, clip 5
    (CleanRL: gym NormalizeObservation, clip 10, plus NormalizeReward: not used here)
  * time-limit bootstrap: r + gamma V(final_obs) where truncated (CleanRL: none)
  * GAE gamma 0.99, lambda 0.95; horizon 24 steps per env per iteration
  * 5 epochs x 4 minibatches, shuffled each epoch, advantages normalized per minibatch
  * clipped loss eps 0.2, value loss 0.5 * mean (V - R)^2 with vf_coef 0.5 and no
    value clipping (CleanRL clips by default), entropy coef 0
  * one Adam, lr 3e-4, eps 1e-8 (CleanRL: 1e-5), global grad-norm clip 1.0 (CleanRL: 0.5)
Cartpole (forge/env/tasks/cartpole.h): force = 10 N x action (clipped to [-1, 1]),
physics dt 0.005 x 4 per action (Euler), reward 1 - 0.01 a^2, terminated at
|angle| > 0.5 or |x| > 2.4, truncated at 500 steps, reset uniform +-0.05.
Ant (forge/env/tasks/ant.h): same obs (36, with the world +x direction in the base frame), actions (8 PD targets over each
joint's range, kp 20, kv 0.5, 5 Nm), reward, termination and reset; floor
contacts only between the robot and the floor (like our engine).

Writes OUT/log.csv with the same columns as train_app. Plot both with
tools/plot_log.py.

  conda activate crucible-tools
  python3 tools/ppo_baseline.py --seed 1 --out runs_baseline/cartpole_seed1
  python3 tools/ppo_baseline.py --task ant --iterations 300 --out runs_baseline/ant_seed1
"""
import argparse
import csv
import math
import os
import time

import mujoco
import numpy as np
import torch
import torch.nn as nn
from mujoco import rollout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- cartpole, many envs, MuJoCo -------------------------------------------------
class CartpoleBatch:
    obs_dim, act_dim = 4, 1
    FORCE, X_LIMIT, ANGLE_LIMIT, MAX_STEPS, NOISE, DT, DECIMATION = 10.0, 2.4, 0.5, 500, 0.05, 0.005, 4

    def __init__(self, n, seed):
        spec = mujoco.MjSpec.from_file(os.path.join(ROOT, "assets/robots/cartpole.urdf"))
        act = spec.add_actuator()  # a motor on the slider: force = gear * ctrl, like our tau
        act.target = "slider"
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gear[0] = self.FORCE
        self.m = spec.compile()
        self.m.opt.timestep = self.DT
        self.m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self.datas = [mujoco.MjData(self.m) for _ in range(max(1, os.cpu_count() or 1))]
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.kind = mujoco.mjtState.mjSTATE_FULLPHYSICS
        d = mujoco.MjData(self.m)
        self.state0 = np.zeros(mujoco.mj_stateSize(self.m, self.kind))
        mujoco.mj_getState(self.m, d, self.state0, self.kind)
        self.qpos = slice(1, 1 + self.m.nq)  # FULLPHYSICS state: time, qpos, qvel, ...
        self.qvel = slice(1 + self.m.nq, 1 + self.m.nq + self.m.nv)
        self.state = np.tile(self.state0, (n, 1))
        self.steps = np.zeros(n, dtype=np.int64)
        self.ep_return = np.zeros(n)
        self.last_return = np.zeros(n)
        self.last_length = np.zeros(n, dtype=np.int64)
        self.reset(np.arange(n))

    def reset(self, ids):
        k = len(ids)
        self.state[ids] = self.state0
        self.state[ids, self.qpos] = self.rng.uniform(-self.NOISE, self.NOISE, (k, self.m.nq))
        self.state[ids, self.qvel] = self.rng.uniform(-self.NOISE, self.NOISE, (k, self.m.nv))
        self.steps[ids] = 0
        self.ep_return[ids] = 0.0

    def obs(self):
        s = self.state
        return np.stack([s[:, 1], s[:, 3], s[:, 2], s[:, 4]], axis=1)  # x, xdot, angle, angle_dot

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        ctrl = np.repeat(a[:, None, :], self.DECIMATION, axis=1)
        out, _ = rollout.rollout(self.m, self.datas, self.state, ctrl, nstep=self.DECIMATION)
        self.state = out[:, -1, :].copy()
        o = self.obs()
        r = 1.0 - 0.01 * a[:, 0] ** 2
        self.steps += 1
        self.ep_return += r
        term = (np.abs(o[:, 2]) > self.ANGLE_LIMIT) | (np.abs(o[:, 0]) > self.X_LIMIT)
        trunc = ~term & (self.steps >= self.MAX_STEPS)
        done = term | trunc
        final = o.copy()
        ids = np.nonzero(done)[0]
        if len(ids):
            self.last_return[ids] = self.ep_return[ids]
            self.last_length[ids] = self.steps[ids]
            self.reset(ids)
            o = self.obs()
        return o, r, term, trunc, final


def _quat_rotate_inv(q, v):  # q: (n,4) w x y z, v: (n,3) -> R^T v
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    u = np.concatenate([-x, -y, -z], axis=1)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


class AntBatch:
    KP, KV, FLIM, STAND, MIN_H, DT, DECIMATION, MAX_STEPS = 20.0, 0.5, 5.0, 0.50, 0.28, 0.005, 4, 1000
    FWD, ALIVE, TORQUE, CHANGE, FALL, JNOISE, VNOISE = 1.0, 0.5, 0.005, 0.01, 2.0, 0.1, 0.1
    obs_dim, act_dim = 36, 8

    def __init__(self, n, seed):
        spec = mujoco.MjSpec.from_file(os.path.join(ROOT, "assets/robots/ant.urdf"))
        spec.worldbody.first_body().add_freejoint(name="root")
        for g in spec.geoms:  # robot geoms touch the floor only
            g.contype, g.conaffinity = 1, 0
        fl = spec.worldbody.add_geom()
        fl.type, fl.size, fl.contype, fl.conaffinity = mujoco.mjtGeom.mjGEOM_PLANE, [0, 0, 1], 0, 1
        names = []
        for j in spec.joints:
            if j.type == mujoco.mjtJoint.mjJNT_HINGE:
                a = spec.add_actuator()
                a.target, a.trntype = j.name, mujoco.mjtTrn.mjTRN_JOINT
                a.set_to_position(kp=self.KP, kv=self.KV)
                a.forcelimited, a.forcerange = 1, [-self.FLIM, self.FLIM]
                names.append(j.name)
        self.m = spec.compile()
        self.m.opt.timestep = self.DT
        self.m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        jid = [self.m.joint(nm).id for nm in names]
        lo, hi = self.m.jnt_range[jid, 0], self.m.jnt_range[jid, 1]
        self.mid, self.half = (lo + hi) / 2, (hi - lo) / 2
        self.datas = [mujoco.MjData(self.m) for _ in range(max(1, os.cpu_count() or 1))]
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.kind = mujoco.mjtState.mjSTATE_FULLPHYSICS
        d = mujoco.MjData(self.m)
        self.state0 = np.zeros(mujoco.mj_stateSize(self.m, self.kind))
        mujoco.mj_getState(self.m, d, self.state0, self.kind)
        self.qp, self.qv = 1, 1 + self.m.nq  # FULLPHYSICS: time, qpos, qvel, act, ...
        self.state = np.tile(self.state0, (n, 1))
        self.ctrl = np.tile(self.mid, (n, 1))
        self.last_a = np.zeros((n, 8))
        self.steps = np.zeros(n, dtype=np.int64)
        self.ep_return = np.zeros(n)
        self.last_return = np.zeros(n)
        self.last_length = np.zeros(n, dtype=np.int64)
        self.reset(np.arange(n))

    def reset(self, ids):
        k = len(ids)
        s = self.state0.copy()
        st = np.tile(s, (k, 1))
        st[:, self.qp:self.qp + 7] = [0, 0, self.STAND + 0.05, 1, 0, 0, 0]
        j = self.mid + self.rng.uniform(-self.JNOISE, self.JNOISE, (k, 8))
        st[:, self.qp + 7:self.qp + 15] = j
        st[:, self.qv:self.qv + 14] = self.rng.uniform(-self.VNOISE, self.VNOISE, (k, 14))
        self.state[ids] = st
        self.ctrl[ids] = j
        self.last_a[ids] = 0.0
        self.steps[ids] = 0
        self.ep_return[ids] = 0.0

    def obs(self):
        q = self.state[:, self.qp:self.qp + 15]
        v = self.state[:, self.qv:self.qv + 14]
        quat = q[:, 3:7] / np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
        g = _quat_rotate_inv(quat, np.tile([0.0, 0.0, -1.0], (self.n, 1)))
        fwd = _quat_rotate_inv(quat, np.tile([1.0, 0.0, 0.0], (self.n, 1)))
        vlin = _quat_rotate_inv(quat, v[:, 0:3])
        return np.concatenate([g, fwd, vlin, v[:, 3:6], q[:, 7:15] - self.mid, v[:, 6:14], self.last_a], axis=1)

    def _failed(self):
        q = self.state[:, self.qp:self.qp + 7]
        quat = q[:, 3:7] / np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
        up_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)  # z of the rotated up axis
        return (q[:, 2] < self.MIN_H) | (up_z < 0.0)

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        prev = self.last_a.copy()
        self.last_a = a
        target = self.mid + self.half * a
        ctrl = np.repeat(target[:, None, :], self.DECIMATION, axis=1)
        out, _ = rollout.rollout(self.m, self.datas, self.state, ctrl, nstep=self.DECIMATION)
        self.state = out[:, -1, :].copy()
        q = self.state[:, self.qp:self.qp + 15]
        v = self.state[:, self.qv:self.qv + 14]
        torque = np.clip(self.KP * (target - q[:, 7:15]) - self.KV * v[:, 6:14], -self.FLIM, self.FLIM)
        term = self._failed()
        r = (self.FWD * v[:, 0] + self.ALIVE - self.TORQUE * (torque ** 2).sum(1)
             - self.CHANGE * ((a - prev) ** 2).sum(1) - self.FALL * term)
        o = self.obs()
        self.steps += 1
        self.ep_return += r
        trunc = ~term & (self.steps >= self.MAX_STEPS)
        done = term | trunc
        final = o.copy()
        ids = np.nonzero(done)[0]
        if len(ids):
            self.last_return[ids] = self.ep_return[ids]
            self.last_length[ids] = self.steps[ids]
            self.reset(ids)
            o = self.obs()
        return o, r, term, trunc, final


# ---- PPO -----------------------------------------------------------------------
def mlp(dims, out_gain):
    layers = []
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1])
        last = i == len(dims) - 2
        nn.init.orthogonal_(lin.weight, out_gain if last else math.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if not last:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class RunningMeanStd:
    def __init__(self, dim, device):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        tot = self.count + bc
        d = bm - self.mean
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    def norm(self, x, clip):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -clip, clip)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="cartpole", choices=["cartpole", "ant"])
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="runs_baseline/cartpole_seed1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    T, EPOCHS, MB, LR, CLIP, GAMMA, LAM, VF, ENT, MAXG, OCLIP = 24, 5, 4, 3e-4, 0.2, 0.99, 0.95, 0.5, 0.0, 1.0, 5.0

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device)
    N = a.num_envs
    env = (AntBatch if a.task == "ant" else CartpoleBatch)(N, a.seed)
    D, A = env.obs_dim, env.act_dim
    actor = mlp([D, 256, 256, 256, A], 0.01).to(dev)
    critic = mlp([D, 256, 256, 256, 1], 1.0).to(dev)
    log_std = nn.Parameter(torch.zeros(A, device=dev))
    params = list(actor.parameters()) + list(critic.parameters()) + [log_std]
    opt = torch.optim.Adam(params, lr=LR, eps=1e-8)
    rms = RunningMeanStd(D, dev)

    os.makedirs(a.out, exist_ok=True)
    cols = ["iteration", "total_steps", "time_s", "steps_per_s", "mean_reward", "mean_episode_len", "value_loss",
            "policy_loss", "entropy", "lr", "kl", "clip_fraction", "grad_norm", "action_std"]
    f = open(os.path.join(a.out, "log.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(cols)
    print(f"ppo_baseline: {a.task}, {N} envs, {a.iterations} iterations, device {dev}, out {a.out}")

    obs = torch.tensor(env.obs(), dtype=torch.float32, device=dev)
    t0 = time.time()
    total = 0
    for it in range(1, a.iterations + 1):
        ti = time.time()
        b_obs = torch.zeros(T, N, D, device=dev)
        b_act = torch.zeros(T, N, A, device=dev)
        b_logp = torch.zeros(T, N, device=dev)
        b_val = torch.zeros(T, N, device=dev)
        b_rew = torch.zeros(T, N, device=dev)
        b_done = torch.zeros(T, N, device=dev)
        with torch.no_grad():
            for t in range(T):
                rms.update(obs)
                on = rms.norm(obs, OCLIP)
                mu = actor(on)
                std = torch.exp(log_std)
                act = mu + std * torch.randn_like(mu)
                logp = (-0.5 * ((act - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(1)
                v = critic(on)[:, 0]
                o2, r, term, trunc, final = env.step(act.cpu().numpy())
                r = torch.tensor(r, dtype=torch.float32, device=dev)
                trunc_t = torch.tensor(trunc, device=dev)
                if trunc.any():  # time limit: bootstrap from the real last obs
                    vf = critic(rms.norm(torch.tensor(final, dtype=torch.float32, device=dev), OCLIP))[:, 0]
                    r = r + GAMMA * vf * trunc_t.float()
                b_obs[t], b_act[t], b_logp[t], b_val[t], b_rew[t] = on, act, logp, v, r
                b_done[t] = torch.tensor(term | trunc, dtype=torch.float32, device=dev)
                obs = torch.tensor(o2, dtype=torch.float32, device=dev)
            last_v = critic(rms.norm(obs, OCLIP))[:, 0]
            adv = torch.zeros(T, N, device=dev)
            nadv = torch.zeros(N, device=dev)
            nval = last_v
            for t in reversed(range(T)):
                nd = 1.0 - b_done[t]
                delta = b_rew[t] + GAMMA * nd * nval - b_val[t]
                nadv = delta + GAMMA * LAM * nd * nadv
                adv[t] = nadv
                nval = b_val[t]
            ret = adv + b_val
        total += T * N

        f_obs, f_act, f_logp = b_obs.reshape(-1, D), b_act.reshape(-1, A), b_logp.reshape(-1)
        f_adv, f_ret = adv.reshape(-1), ret.reshape(-1)
        M = T * N
        mb = M // MB
        sums = np.zeros(5)
        n_upd = 0
        for ep in range(EPOCHS):
            perm = torch.randperm(M, device=dev)
            for k in range(MB):
                idx = perm[k * mb:(k + 1) * mb]
                ad = f_adv[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                mu = actor(f_obs[idx])
                std = torch.exp(log_std)
                lp = (-0.5 * ((f_act[idx] - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(1)
                logr = lp - f_logp[idx]
                ratio = logr.exp()
                pg = torch.max(-ad * ratio, -ad * torch.clamp(ratio, 1 - CLIP, 1 + CLIP)).mean()
                v = critic(f_obs[idx])[:, 0]
                vloss = 0.5 * ((v - f_ret[idx]) ** 2).mean()
                ent = (log_std + 0.5 * math.log(2 * math.pi) + 0.5).sum()
                loss = pg + VF * vloss - ENT * ent
                opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(params, MAXG)
                opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - logr).mean()
                    cf = ((ratio - 1).abs() > CLIP).float().mean()
                sums += [pg.item(), vloss.item(), kl.item(), cf.item(), gn.item()]
                n_upd += 1
        s = sums / n_upd
        done_envs = env.last_length > 0
        mr = env.last_return[done_envs].mean() if done_envs.any() else float("nan")
        ml = env.last_length[done_envs].mean() if done_envs.any() else float("nan")
        now = time.time() - t0
        sps = T * N / (time.time() - ti)
        ent_v = float((log_std.detach() + 0.5 * math.log(2 * math.pi) + 0.5).sum())
        w.writerow([it, total, now, sps, mr, ml, s[1], s[0], ent_v, LR, s[2], s[3], s[4], float(torch.exp(log_std.detach()).mean())])
        f.flush()
        print(f"it {it:5d}  steps {total:10d}  {sps:8.3g} steps/s  return {mr:8.2f}  ep_len {ml:7.1f}  "
              f"vloss {s[1]:.4f}  kl {s[2]:.4f}  std {float(torch.exp(log_std.detach()).mean()):.3f}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
