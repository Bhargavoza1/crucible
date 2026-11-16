#!/usr/bin/env python3
"""Redo one temper SAC or TQC gradient step in PyTorch and compare (Phase 7e check).

temper writes the files from a GPU test:
  CRUCIBLE_SAC_STEP_DIR=/tmp/sac_step ./tests/tests_gpu --gtest_filter=Sac.DumpOneStepForPytorchCheck
  (CRUCIBLE_SAC_STEP_ALGO=tqc for TQC)
    before.bin  all weights, log_alpha and Adam state before the step
    step.bin    the batch, the noise (eps) of both samples, y, all gradients
    after.bin   all weights after the step (Adam, alpha, soft target update)
Then:
  python3 tools/sac_check_step.py /tmp/sac_step

This uses temper's batch and noise, so the two must agree up to float rounding
(different summation order): about 1e-6 relative. A bigger number names the part
that differs.
"""
import math
import struct
import sys

import numpy as np
import torch


def read_ckpt(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 0

    def get(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return v[0]

    def get_str():
        nonlocal pos
        n = get("I")
        s = data[pos:pos + n].decode()
        pos += n
        return s

    if data[:8] != b"CRUCKPT1":
        raise SystemExit(f"{path}: not a crucible checkpoint")
    pos = 8
    c = {"version": get("I"), "algo": get_str(), "task": get_str(), "obs_dim": get("i"), "act_dim": get("i")}
    sizes = {}
    for _ in range(get("I")):
        name = get_str()
        sizes[name] = [get("i") for _ in range(get("I"))]
    c["layer_sizes"] = sizes
    c["iteration"], c["seed"], c["optimizer_steps"] = get("q"), get("Q"), get("q")
    arrays = {}
    for _ in range(get("I")):
        name = get_str()
        n = get("Q")
        arrays[name] = np.frombuffer(data, dtype="<f4", count=n, offset=pos).copy()
        pos += 4 * n
    c["arrays"] = arrays
    return c


class Net:
    """MLP with ReLU, weights as torch tensors (W is out x in, as nn.Linear)."""

    def __init__(self, arrays, prefix, dims):
        self.p = []
        for i in range(len(dims) - 1):
            W = torch.tensor(arrays[f"{prefix}.{i}.W"].reshape(dims[i + 1], dims[i]), requires_grad=True)
            b = torch.tensor(arrays[f"{prefix}.{i}.b"], requires_grad=True)
            self.p += [W, b]

    def __call__(self, x):
        n = len(self.p) // 2
        for i in range(n):
            x = x @ self.p[2 * i].T + self.p[2 * i + 1]
            if i < n - 1:
                x = torch.relu(x)
        return x


def squash(out, eps, A):
    mu, raw = out[:, :A], out[:, A:]
    log_std = -5.0 + 3.5 * (torch.tanh(raw) + 1)
    u = mu + log_std.exp() * eps
    a = torch.tanh(u)
    log1m_tanh2 = 2.0 * (math.log(2.0) - u - torch.nn.functional.softplus(-2.0 * u))
    logp = (-0.5 * eps ** 2 - log_std - 0.5 * math.log(2 * math.pi) - log1m_tanh2).sum(1)
    return a, logp


def adam_step(params, grads, m_list, v_list, steps, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = steps + 1
    out = []
    for p, g, m, v in zip(params, grads, m_list, v_list):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        out.append(p - lr * (m / (1 - b1 ** t)) / (torch.sqrt(v / (1 - b2 ** t)) + eps))
    return out


worst = 0.0


def compare(name, ours, ref, tol):
    global worst
    ours = np.asarray(ours, dtype=np.float64).ravel()
    ref = np.asarray(ref.detach().numpy() if torch.is_tensor(ref) else ref, dtype=np.float64).ravel()
    err = np.linalg.norm(ours - ref) / max(np.linalg.norm(ref), 1e-12)
    worst = max(worst, err / tol)
    print(f"  {name:28s} rel err {err:9.2e}  {'ok' if err < tol else 'DIFFERENT'}")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sac_step"
    B0, S, A1 = read_ckpt(f"{d}/before.bin"), read_ckpt(f"{d}/step.bin"), read_ckpt(f"{d}/after.bin")
    w, s, w1 = B0["arrays"], S["arrays"], A1["arrays"]
    D, A = B0["obs_dim"], B0["act_dim"]
    da, dc = B0["layer_sizes"]["actor"], B0["layer_sizes"]["critic"]
    gamma, tau = float(s["gamma"][0]), float(s["tau"][0])
    q_lr, pi_lr, alpha_lr = (float(x) for x in s["lr"])
    t = lambda name, *shape: torch.tensor(s[name].reshape(*shape) if shape else s[name])
    Bn = len(s["b_rew"])
    obs, act, rew, nxt, done = t("b_obs", Bn, D), t("b_act", Bn, A), t("b_rew"), t("b_next", Bn, D), t("b_done")
    actor = Net(w, "actor", da)
    tqc_on, nc, NQ, drop = (int(x) for x in s.get("config", [0, 2, 1, 0]))  # older files: SAC
    qs = [Net(w, f"q{c + 1}", dc) for c in range(nc)]
    qts = [Net(w, f"q{c + 1}t", dc) for c in range(nc)]
    M = nc * (NQ - drop) if tqc_on else 1
    taus = (torch.arange(NQ, dtype=torch.float32) + 0.5) / NQ
    log_alpha = float(w["log_alpha"][0])
    alpha = math.exp(log_alpha)
    print(f"{'TQC' if tqc_on else 'SAC'} step check: batch {Bn}, obs {D}, act {A}, {nc} critics x {NQ} outputs, "
          f"alpha {alpha:.5f}")

    # 1. critic target
    with torch.no_grad():
        na, nlogp = squash(actor(nxt), t("eps_next", Bn, A), A)
        nxa = torch.cat([nxt, na], 1)
        zt = torch.stack([q(nxa) for q in qts], 1)  # [B, nc, NQ]
        if tqc_on:
            atoms = torch.sort(zt.reshape(Bn, -1), dim=1)[0][:, :M]
            y = rew[:, None] + gamma * (1 - done[:, None]) * (atoms - alpha * nlogp[:, None])
        else:
            y = rew + gamma * (1 - done) * (zt[:, :, 0].min(1)[0] - alpha * nlogp)
    compare("logp(a'|s')", s["logp_next"], nlogp, 1e-4)
    compare("critic target y", s["y"], y, 1e-4)

    # 2. critic gradients
    oa = torch.cat([obs, act], 1)
    z = torch.stack([q(oa) for q in qs], 1)
    if tqc_on:
        delta = y[:, None, None, :] - z[:, :, :, None]
        ad = delta.abs()
        huber = torch.where(ad > 1, ad - 0.5, 0.5 * delta ** 2)
        q_loss = (torch.abs(taus[None, None, :, None] - (delta.detach() < 0).float()) * huber).mean()
    else:
        q_loss = ((z[:, :, 0] - y[:, None]) ** 2).mean(0).sum()
    qp = [p for q in qs for p in q.p]
    gq = torch.autograd.grad(q_loss, qp)
    per = len(qs[0].p)
    for i, g in enumerate(gq):
        compare(f"critic grad {i} (q{i // per + 1})", s[f"grad_q.{i}"], g, 1e-3)
    m = [torch.tensor(w[f"adam_q.m.{i}"]).reshape(p.shape) for i, p in enumerate(qp)]
    v = [torch.tensor(w[f"adam_q.v.{i}"]).reshape(p.shape) for i, p in enumerate(qp)]
    new_q = adam_step([p.detach() for p in qp], gq, m, v, int(w["adam_q.steps"][0]), q_lr)
    for i, p in enumerate(new_q):
        qs[i // per].p[i % per] = p.clone().requires_grad_(True)

    # 3. actor gradients through the updated critics
    a_pi, logp = squash(actor(obs), t("eps_pi", Bn, A), A)
    compare("logp(a|s)", s["logp_pi"], logp, 1e-4)
    opa = torch.cat([obs, a_pi], 1)
    zpi = torch.stack([q(opa) for q in qs], 1)
    qagg = zpi.mean((1, 2)) if tqc_on else zpi[:, :, 0].min(1)[0]
    pi_loss = (alpha * logp - qagg).mean()
    gpi = torch.autograd.grad(pi_loss, actor.p)
    for i, g in enumerate(gpi):
        compare(f"actor grad {i}", s[f"grad_pi.{i}"], g, 1e-3)
    dla = -alpha * (logp.detach().mean() + (-A))
    compare("d loss / d log_alpha", s["dlog_alpha"], dla, 1e-4)

    # 4. weights after: Adam for actor and alpha, soft update of the targets
    m = [torch.tensor(w[f"adam_pi.m.{i}"]).reshape(p.shape) for i, p in enumerate(actor.p)]
    v = [torch.tensor(w[f"adam_pi.v.{i}"]).reshape(p.shape) for i, p in enumerate(actor.p)]
    new_pi = adam_step([p.detach() for p in actor.p], gpi, m, v, int(w["adam_pi.steps"][0]), pi_lr)
    new_la = adam_step([torch.tensor([log_alpha])], [torch.tensor([float(dla)])], [torch.tensor(w["adam_alpha.m.0"])],
                       [torch.tensor(w["adam_alpha.v.0"])], int(w["adam_alpha.steps"][0]), alpha_lr)[0]
    nl = len(dc) - 1
    for i in range(nl):
        for k, part in enumerate("Wb"):
            j = 2 * i + k
            compare(f"after: actor.{i}.{part}", w1[f"actor.{i}.{part}"], new_pi[j], 1e-4)
            for c in (0, nc - 1):
                qn = f"q{c + 1}"
                compare(f"after: {qn}.{i}.{part}", w1[f"{qn}.{i}.{part}"], qs[c].p[j], 1e-4)
                tgt = (1 - tau) * torch.tensor(w[f"{qn}t.{i}.{part}"]).reshape(qs[c].p[j].shape) + tau * qs[c].p[j].detach()
                compare(f"after: {qn}t.{i}.{part}", w1[f"{qn}t.{i}.{part}"], tgt, 1e-4)
    compare("after: log_alpha", w1["log_alpha"], new_la, 1e-4)
    print("RESULT:", "MATCH" if worst < 1 else "DIFFERENT (see the lines marked DIFFERENT)")


if __name__ == "__main__":
    main()
