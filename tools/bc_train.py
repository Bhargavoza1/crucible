"""Behavior cloning baseline in PyTorch (Phase 9d): learn the panda_pick expert's
actions from a collect_app demo file. The reference for the CNN in our own engine
(same data, same split, same metrics).

    python tools/bc_train.py demos/pick_img.bin                      images + robot state (default)
    python tools/bc_train.py demos/pick_img.bin --input state        full state incl. the true cube pose (upper bound)

Inputs (--input image): the two camera images (front, wrist; 64 x 64 RGB) and the
robot's own state: obs 0-23 (joint positions 9, joint velocities 9, hand position
3, sin / cos of the hand turn 2, finger opening 1). Not the cube's position (the
cameras replace it), not the last action (a known behavior-cloning shortcut: copy
the last action). --input state: all of obs 0-33 (with the cube pose), no images.
Label: the expert's clean action (the file also has the executed, noisy one).
Only steps with a camera frame are used (every image_every-th step).

Model: per camera Conv 32 5x5/2, Conv 64 3x3/2, Conv 64 3x3/2, Conv 64 3x3/1 (ReLU),
Linear 256 + LayerNorm; state Linear 128; MLP 512-512; tanh. Random shifts of up
to 4 pixels on training images (DrQ). Adam, MSE loss.
Split: --val_frac of the episodes (default 10%), chosen by --seed, held out.
Metrics on the held-out episodes: MSE per action, gripper right (sign of a[4]
matches the expert), and the move error in cm per step (a[0..2] x 2 cm).
Writes the model and the metrics to --out (a .pt file).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_demos import load_demos  # noqa: E402

STATE_IMAGE = list(range(0, 24))   # robot only
STATE_FULL = list(range(0, 34))    # + cube position, quaternion, cube - pads


def load(path, max_episodes, need_images):
    task, eps = load_demos(path)
    if task != "panda_pick":
        raise SystemExit(f"expected panda_pick demos, got {task}")
    if need_images and "images" not in eps[0]:
        raise SystemExit("no camera images in this file: record with collect_app --images true")
    eps = eps[:max_episodes] if max_episodes else eps
    every = eps[0].get("image_every", 1) if need_images else 2
    obs, act, img, ep_id = [], [], [], []
    for i, e in enumerate(eps):
        steps = np.arange(0, len(e["reward"]), every)
        obs.append(e["obs"][steps])
        act.append(e["expert"][steps])
        ep_id.append(np.full(len(steps), i))
        if need_images:
            img.append(e["images"][: len(steps)])  # [frames, cams, H, W, 3]
    out = {"obs": np.concatenate(obs), "act": np.concatenate(act), "ep": np.concatenate(ep_id), "n_eps": len(eps)}
    if need_images:
        out["img"] = np.concatenate(img)
    return out


class Encoder(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 5, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 1), nn.ReLU())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 3, size, size)).numel()
        self.fc = nn.Sequential(nn.Linear(n, 256), nn.LayerNorm(256))

    def forward(self, x):  # x: [B, 3, H, W] in [0, 1]
        return self.fc(self.conv(x).flatten(1))


class Policy(nn.Module):
    def __init__(self, state_dim, cams, size):
        super().__init__()
        self.cams = cams
        self.enc = nn.ModuleList([Encoder(size) for _ in range(cams)])
        self.state = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256 * cams + 128, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                  nn.Linear(512, 5), nn.Tanh())

    def forward(self, state, img=None):  # img: [B, cams, 3, H, W] float in [0, 1]
        parts = [enc(img[:, c]) for c, enc in enumerate(self.enc)] if self.cams else []
        return self.head(torch.cat(parts + [self.state(state)], dim=1))


def random_shift(x, pad=4):
    """DrQ: pad by replication, crop back at a random offset (per image)."""
    b, c, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    rows = (oy[:, None] + torch.arange(h, device=x.device)[None]).clamp(max=h + 2 * pad - 1)
    cols = (ox[:, None] + torch.arange(w, device=x.device)[None]).clamp(max=w + 2 * pad - 1)
    x = x[torch.arange(b, device=x.device)[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None],
          rows[:, None, :, None], cols[:, None, None, :]]
    return x


def to_images(u8, device, shift):
    x = torch.from_numpy(u8).to(device).permute(0, 1, 4, 2, 3).float() / 255.0  # [B, cams, 3, H, W]
    if shift:
        b, cams = x.shape[:2]
        x = random_shift(x.flatten(0, 1)).view(b, cams, *x.shape[2:])
    return x


@torch.no_grad()
def evaluate(model, d, idx, cols, mean, std, device, use_images, batch=1024):
    model.eval()
    se, n, grip_ok, move_err = np.zeros(5), 0, 0, 0.0
    for k in range(0, len(idx), batch):
        b = idx[k:k + batch]
        s = torch.from_numpy((d["obs"][b][:, cols] - mean) / std).float().to(device)
        im = to_images(d["img"][b], device, False) if use_images else None
        pred = model(s, im).cpu().numpy()
        y = d["act"][b]
        se += ((pred - y) ** 2).sum(0)
        grip_ok += int((np.sign(pred[:, 4]) == np.sign(y[:, 4])).sum())
        move_err += float(np.linalg.norm((pred[:, :3] - y[:, :3]) * 2.0, axis=1).sum())  # cm (2 cm per unit)
        n += len(b)
    model.train()
    return {"mse": (se / n).tolist(), "mse_mean": float((se / n).mean()), "gripper_right": grip_ok / n,
            "move_error_cm": move_err / n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos")
    ap.add_argument("--input", choices=["image", "state"], default="image")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--max_episodes", type=int, default=0, help="0 = all (images: about 3 MB of RAM per episode)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="bc_pick.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    use_images = a.input == "image"
    t0 = time.time()
    d = load(a.demos, a.max_episodes, use_images)
    cols = STATE_IMAGE if use_images else STATE_FULL
    perm = rng.permutation(d["n_eps"])
    n_val = max(1, int(round(a.val_frac * d["n_eps"])))
    val_eps = set(perm[:n_val].tolist())
    is_val = np.array([e in val_eps for e in d["ep"]])
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    mean = d["obs"][tr][:, cols].mean(0)
    std = d["obs"][tr][:, cols].std(0) + 1e-6
    cams = d["img"].shape[1] if use_images else 0
    size = d["img"].shape[2] if use_images else 0
    print(f"bc: {d['n_eps']} episodes ({d['n_eps'] - n_val} train, {n_val} held out), {len(tr)} / {len(va)} frames, "
          f"input {a.input} ({len(cols)} state numbers{f', {cams} cameras {size}x{size}' if cams else ''}), "
          f"device {a.device}, loaded in {time.time() - t0:.0f} s")

    model = Policy(len(cols), cams, size).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(f"bc: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    history = []
    for epoch in range(1, a.epochs + 1):
        t = time.time()
        order = rng.permutation(tr)
        total, nb = 0.0, 0
        for k in range(0, len(order) - a.batch + 1, a.batch):
            b = np.sort(order[k:k + a.batch])
            s = torch.from_numpy((d["obs"][b][:, cols] - mean) / std).float().to(a.device)
            im = to_images(d["img"][b], a.device, True) if use_images else None
            y = torch.from_numpy(d["act"][b]).float().to(a.device)
            loss = F.mse_loss(model(s, im), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        m = evaluate(model, d, va, cols, mean, std, a.device, use_images)
        m.update(epoch=epoch, train_loss=total / max(1, nb))
        history.append(m)
        print(f"epoch {epoch:3d}  train loss {m['train_loss']:.4f}  held-out: mse {m['mse_mean']:.4f} "
              f"(x {m['mse'][0]:.3f} y {m['mse'][1]:.3f} z {m['mse'][2]:.3f} turn {m['mse'][3]:.3f} "
              f"grip {m['mse'][4]:.3f})  gripper right {100 * m['gripper_right']:.1f}%  "
              f"move error {m['move_error_cm']:.2f} cm/step  ({time.time() - t:.0f} s)")
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "cols": cols, "cams": cams, "size": size,
                "args": vars(a), "history": history}, a.out)
    with open(os.path.splitext(a.out)[0] + "_metrics.json", "w") as f:
        json.dump(history, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
