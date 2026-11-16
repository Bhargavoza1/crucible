"""Reads a demonstration file from collect_app (format: forge/env/demo_file.h).

    python tools/read_demos.py demos/pick.bin                 summary
    python tools/read_demos.py demos/pick.bin --npz pick.npz  also save flat arrays for training

In Python:  from read_demos import load_demos;  task, episodes = load_demos("demos/pick.bin")
each episode is a dict: obs [T, obs_dim], expert [T, act_dim], executed [T, act_dim],
reward [T], final_obs [obs_dim], success (bool), noise (float).
"""
import argparse
import struct

import numpy as np


def load_demos(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"CRUCDEMO":
        raise ValueError(f"{path}: not a CRUCDEMO file")
    version, obs_dim, act_dim = struct.unpack_from("<3I", data, 8)
    if version != 1:
        raise ValueError(f"{path}: unknown version {version}")
    task = data[20:52].split(b"\0", 1)[0].decode()
    at, episodes = 52, []
    while at < len(data):
        steps, success, noise = struct.unpack_from("<IB3xf", data, at)
        at += 12

        def take(n):
            nonlocal at
            a = np.frombuffer(data, dtype="<f4", count=n, offset=at).copy()
            at += 4 * n
            return a

        ep = {"obs": take(steps * obs_dim).reshape(steps, obs_dim),
              "expert": take(steps * act_dim).reshape(steps, act_dim),
              "executed": take(steps * act_dim).reshape(steps, act_dim),
              "reward": take(steps), "final_obs": take(obs_dim),
              "success": bool(success), "noise": float(noise)}
        episodes.append(ep)
    return task, episodes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--npz", help="save obs, expert, executed, episode index, success to this .npz")
    a = ap.parse_args()
    task, eps = load_demos(a.path)
    steps = sum(len(e["reward"]) for e in eps)
    ok = sum(e["success"] for e in eps)
    print(f"{a.path}: task {task}, {len(eps)} episodes, {steps} steps, "
          f"success {ok} ({100.0 * ok / max(1, len(eps)):.1f}%), "
          f"noise {min(e['noise'] for e in eps):.3f}-{max(e['noise'] for e in eps):.3f}")
    if a.npz:
        np.savez_compressed(a.npz,
                            obs=np.concatenate([e["obs"] for e in eps]),
                            expert=np.concatenate([e["expert"] for e in eps]),
                            executed=np.concatenate([e["executed"] for e in eps]),
                            episode=np.concatenate([np.full(len(e["reward"]), i) for i, e in enumerate(eps)]),
                            success=np.array([e["success"] for e in eps]))
        print(f"wrote {a.npz}")


if __name__ == "__main__":
    main()
