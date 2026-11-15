#!/usr/bin/env python3
"""SAC baseline in PyTorch (doc 5.1, 5.7), on the same tasks as forge/env/tasks/
(the env classes come from ppo_baseline.py). Written like CleanRL's
sac_continuous_action.py, with temper's settings, so temper's CUDA SAC can be
checked against it curve by curve.

Algorithm (same as temper's SAC):
  * actor: MLP (256, 256, ReLU) -> mean and raw log_std per action;
    log_std = -5 + 0.5 * (2 - (-5)) * (tanh(raw) + 1); a = tanh(mean + std * eps);
    log pi uses log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u)) (stable; CleanRL: log(1 - a^2 + 1e-6))
  * two critics Q1, Q2: MLP (256, 256, ReLU) on [obs, action]; target critics,
    soft update tau each gradient step
  * critic target y = r + gamma (1 - terminated) (min Q'(s', a') - alpha log pi(a'|s')),
    a' ~ pi(s'); s' is final_obs where the episode ended (doc 3.6)
  * critic loss = mean (Q1 - y)^2 + mean (Q2 - y)^2
  * actor loss = mean (alpha log pi(a|s) - min(Q1, Q2)(s, a)), a ~ pi(s) (reparameterized)
  * alpha: learned, log_alpha loss = mean(-alpha (log pi + target_entropy)), target = -act_dim
  * many envs: every env step adds num_envs transitions; then updates_per_step
    gradient steps on random batches; uniform random actions until learning_starts
  * no obs normalization (like CleanRL)

  python3 tools/sac_baseline.py --task cartpole --out runs_baseline/sac_cartpole_seed1
The defaults are the same as configs/sac_cartpole.txt (keep them in step).
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppo_baseline import AntBatch, CartpoleBatch  # noqa: E402

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def sample(actor, obs, A):
    out = actor(obs)
    mu, raw = out[:, :A], out[:, A:]
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(raw) + 1)
    eps = torch.randn_like(mu)
    u = mu + log_std.exp() * eps
    a = torch.tanh(u)
    # log(1 - tanh(u)^2) in its stable form (same as temper/algos/sac/sac_math.h)
    log1m_tanh2 = 2.0 * (math.log(2.0) - u - torch.nn.functional.softplus(-2.0 * u))
    logp = (-0.5 * eps ** 2 - log_std - 0.5 * math.log(2 * math.pi) - log1m_tanh2).sum(1)
    return a, logp, torch.tanh(mu)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="cartpole", choices=["cartpole", "ant"])
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--steps_per_iteration", type=int, default=24)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--updates_per_step", type=int, default=4)
    p.add_argument("--learning_starts", type=int, default=2000, help="transitions")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--q_lr", type=float, default=1e-3)
    p.add_argument("--policy_lr", type=float, default=3e-4)
    p.add_argument("--out", default="runs_baseline/sac_cartpole_seed1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device)
    N = a.num_envs
    env = (AntBatch if a.task == "ant" else CartpoleBatch)(N, a.seed)
    D, A = env.obs_dim, env.act_dim
    actor = mlp([D, 256, 256, 2 * A]).to(dev)
    q1, q2 = mlp([D + A, 256, 256, 1]).to(dev), mlp([D + A, 256, 256, 1]).to(dev)
    q1t, q2t = mlp([D + A, 256, 256, 1]).to(dev), mlp([D + A, 256, 256, 1]).to(dev)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    log_alpha = torch.zeros(1, device=dev, requires_grad=True)
    target_entropy = -float(A)
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=a.q_lr, eps=1e-8)
    pi_opt = torch.optim.Adam(actor.parameters(), lr=a.policy_lr, eps=1e-8)
    a_opt = torch.optim.Adam([log_alpha], lr=a.q_lr, eps=1e-8)

    C = a.buffer_size
    rb_o = torch.zeros(C, D, device=dev)
    rb_a = torch.zeros(C, A, device=dev)
    rb_r = torch.zeros(C, device=dev)
    rb_n = torch.zeros(C, D, device=dev)
    rb_d = torch.zeros(C, device=dev)
    pos, size = 0, 0

    os.makedirs(a.out, exist_ok=True)
    f = open(os.path.join(a.out, "log.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["iteration", "total_steps", "time_s", "steps_per_s", "mean_reward", "mean_episode_len", "q_loss",
                "actor_loss", "alpha", "entropy", "q_mean", "updates"])
    print(f"sac_baseline: {a.task}, {N} envs, batch {a.batch_size}, {a.updates_per_step} updates/step, device {dev}")
    obs = torch.tensor(env.obs(), dtype=torch.float32, device=dev)
    t0 = time.time()
    total, n_updates = 0, 0
    for it in range(1, a.iterations + 1):
        ti = time.time()
        sums = np.zeros(5)
        k_upd = 0
        for _ in range(a.steps_per_iteration):
            with torch.no_grad():
                if size < a.learning_starts:
                    act = torch.rand(N, A, device=dev) * 2 - 1
                else:
                    act, _, _ = sample(actor, obs, A)
            o2, r, term, trunc, final = env.step(act.cpu().numpy())
            done = term | trunc
            nxt = np.where(done[:, None], final, o2)
            idx = (torch.arange(N, device=dev) + pos) % C
            rb_o[idx], rb_a[idx] = obs, act
            rb_r[idx] = torch.tensor(r, dtype=torch.float32, device=dev)
            rb_n[idx] = torch.tensor(nxt, dtype=torch.float32, device=dev)
            rb_d[idx] = torch.tensor(term, dtype=torch.float32, device=dev)
            pos, size = (pos + N) % C, min(size + N, C)
            obs = torch.tensor(o2, dtype=torch.float32, device=dev)
            total += N
            if size < a.learning_starts:
                continue
            for _ in range(a.updates_per_step):
                b = torch.randint(0, size, (a.batch_size,), device=dev)
                o, ac, rr, nx, dn = rb_o[b], rb_a[b], rb_r[b], rb_n[b], rb_d[b]
                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    na, nlogp, _ = sample(actor, nx, A)
                    qn = torch.min(q1t(torch.cat([nx, na], 1)), q2t(torch.cat([nx, na], 1)))[:, 0] - alpha * nlogp
                    y = rr + a.gamma * (1 - dn) * qn
                oa = torch.cat([o, ac], 1)
                v1, v2 = q1(oa)[:, 0], q2(oa)[:, 0]
                q_loss = ((v1 - y) ** 2).mean() + ((v2 - y) ** 2).mean()
                q_opt.zero_grad()
                q_loss.backward()
                q_opt.step()
                pa, logp, _ = sample(actor, o, A)
                opa = torch.cat([o, pa], 1)
                qmin = torch.min(q1(opa), q2(opa))[:, 0]
                pi_loss = (alpha * logp - qmin).mean()
                pi_opt.zero_grad()
                pi_loss.backward()
                pi_opt.step()
                a_loss = (-log_alpha.exp() * (logp.detach() + target_entropy)).mean()
                a_opt.zero_grad()
                a_loss.backward()
                a_opt.step()
                with torch.no_grad():
                    for net, tgt in ((q1, q1t), (q2, q2t)):
                        for pp, tp in zip(net.parameters(), tgt.parameters()):
                            tp.mul_(1 - a.tau).add_(a.tau * pp)
                sums += [q_loss.item(), pi_loss.item(), float(alpha), float(-logp.mean()), float(v1.mean())]
                k_upd += 1
                n_updates += 1
        s = sums / max(k_upd, 1)
        done_envs = env.last_length > 0
        mr = env.last_return[done_envs].mean() if done_envs.any() else float("nan")
        ml = env.last_length[done_envs].mean() if done_envs.any() else float("nan")
        sps = a.steps_per_iteration * N / (time.time() - ti)
        w.writerow([it, total, time.time() - t0, sps, mr, ml, *s, n_updates])
        f.flush()
        print(f"it {it:5d}  steps {total:9d}  {sps:8.3g} steps/s  return {mr:8.2f}  ep_len {ml:7.1f}  qloss {s[0]:.3f}  "
              f"alpha {s[2]:.4f}  entropy {s[3]:.2f}  q {s[4]:.1f}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
