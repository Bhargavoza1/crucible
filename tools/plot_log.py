#!/usr/bin/env python3
"""Plot learning curves from log.csv files (train_app or ppo_baseline.py).

Each argument is a run folder or a log.csv, optionally with a label:
  python3 tools/plot_log.py build/runs/cartpole_2026-11-15_seed1=crucible \\
                            runs_baseline/cartpole_seed1=pytorch -o cartpole.png
Runs with the same label are drawn as their mean with a band from min to max
(use it for several seeds). Default: mean_reward vs total_steps; --y picks
another column (value_loss, kl, action_std, ...).

--terms: one small plot per reward term (the r_* columns, doc 7.4), all runs in
each, to see which term is weak:
  python3 tools/plot_log.py build/runs/g1_walk_2026-11-16_seed1 --terms -o terms.png
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path, xcol, ycol):
    if os.path.isdir(path):
        path = os.path.join(path, "log.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: empty")
    for c in (xcol, ycol):
        if c not in rows[0]:
            raise SystemExit(f"{path}: no column '{c}' (has: {', '.join(rows[0])})")
    x = np.array([float(r[xcol]) for r in rows])
    y = np.array([float(r[ycol]) for r in rows])
    return x, y


def plot_terms(a):
    specs = [s.partition("=") for s in a.runs]
    first = specs[0][0]
    first = os.path.join(first, "log.csv") if os.path.isdir(first) else first
    with open(first) as f:
        cols = [c for c in next(csv.reader(f)) if c.startswith("r_")]
    if not cols:
        raise SystemExit(f"{first}: no r_* columns (this task does not name its reward terms)")
    nc = 4
    nr = (len(cols) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.6 * nr), dpi=110, squeeze=False)
    for i, col in enumerate(cols):
        ax = axes[i // nc][i % nc]
        for path, _, label in specs:
            x, y = read(path, a.x, col)
            ax.plot(x, y, lw=1.5, label=label or os.path.basename(os.path.normpath(path)))
        ax.set_title(col[2:], fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for j in range(len(cols), nr * nc):
        axes[j // nc][j % nc].axis("off")
    if len(specs) > 1:
        axes[0][0].legend(fontsize=7)
    fig.suptitle(a.title or "reward terms (mean per env-step) vs " + a.x.replace("_", " "))
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"wrote {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run folder or log.csv, optional =label")
    p.add_argument("--x", default="total_steps")
    p.add_argument("--y", default="mean_reward")
    p.add_argument("-o", "--out", default="curves.png")
    p.add_argument("--title", default="")
    p.add_argument("--terms", action="store_true", help="one plot per r_* column (reward terms)")
    a = p.parse_args()
    if a.terms:
        plot_terms(a)
        return

    groups = defaultdict(list)
    for spec in a.runs:
        path, _, label = spec.partition("=")
        groups[label or os.path.basename(os.path.normpath(path))].append(read(path, a.x, a.y))

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
    for label, runs in groups.items():
        n = min(len(x) for x, _ in runs)  # common length
        x = runs[0][0][:n]
        ys = np.stack([y[:n] for _, y in runs])
        ax.plot(x, ys.mean(0), label=f"{label} ({len(runs)} run{'s' if len(runs) > 1 else ''})", lw=2)
        if len(runs) > 1:
            ax.fill_between(x, ys.min(0), ys.max(0), alpha=0.2)
    ax.set_xlabel(a.x.replace("_", " "))
    ax.set_ylabel(a.y.replace("_", " "))
    if a.title:
        ax.set_title(a.title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
