#!/usr/bin/env python3
"""Plot learning curves from log.csv files (train_app or ppo_baseline.py).

Each argument is a run folder or a log.csv, optionally with a label:
  python3 tools/plot_log.py build/runs/cartpole_2026-11-15_seed1=crucible \\
                            runs_baseline/cartpole_seed1=pytorch -o cartpole.png
Runs with the same label are drawn as their mean with a band from min to max
(use it for several seeds). Default: mean_reward vs total_steps; --y picks
another column (value_loss, kl, action_std, ...).
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run folder or log.csv, optional =label")
    p.add_argument("--x", default="total_steps")
    p.add_argument("--y", default="mean_reward")
    p.add_argument("-o", "--out", default="curves.png")
    p.add_argument("--title", default="")
    a = p.parse_args()

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
