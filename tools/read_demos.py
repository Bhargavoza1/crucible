"""Reads a demonstration file from collect_app (format: forge/env/demo_file.h).

    python tools/read_demos.py demos/pick.bin                 summary
    python tools/read_demos.py demos/pick.bin --npz pick.npz  also save flat arrays for training

    python tools/read_demos.py demos/pick.bin --png ep0.png  camera frames of episode 0 as a picture

In Python:  from read_demos import load_demos;  task, episodes = load_demos("demos/pick.bin")
Big image files (10 000 OpenArm episodes: 17 GB on disk, 180 GB decompressed): pass
images=False, then load_images(path, ep) gives one episode's images when needed.
each episode is a dict: obs [T, obs_dim], expert [T, act_dim], executed [T, act_dim],
reward [T], final_obs [obs_dim], success (bool), noise (float), and with camera
images (file version 2): images uint8 [frames, cameras, height, width, 3] and
image_every (frame f is before step f * image_every). panda_pick_color (Phase 9e)
episodes also have target_color ("red", "green" or "yellow") and instruction
("pick the red cube").
"""
import argparse
import mmap
import struct
import zlib

import numpy as np


COLORS = ["red", "green", "yellow"]  # panda_pick_color's color ids 0, 1, 2


def load_demos(path, images=True):
    """images=False: the camera images stay in the file (ep["image_at"], ep["image_bytes"],
    ep["image_shape"]); load_images(path, ep) reads one episode's. The file is memory
    mapped, not read whole."""
    with open(path, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if data[:8] != b"CRUCDEMO":
        raise ValueError(f"{path}: not a CRUCDEMO file")
    version, obs_dim, act_dim = struct.unpack_from("<3I", data, 8)
    if version not in (1, 2):
        raise ValueError(f"{path}: unknown version {version}")
    task = data[20:52].split(b"\0", 1)[0].decode()
    at, episodes = 52, []
    cams = width = height = every = 0
    if version == 2:
        cams, width, height, every = struct.unpack_from("<4I", data, 52)
        at = 68
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
        if cams:
            frames, nbytes = struct.unpack_from("<2I", data, at)
            at += 8
            ep["image_shape"] = (frames, cams, height, width, 3)
            ep["image_at"], ep["image_bytes"] = at, nbytes
            if images:
                raw = zlib.decompress(data[at:at + nbytes])
                ep["images"] = np.frombuffer(raw, dtype=np.uint8).reshape(ep["image_shape"])
            at += nbytes
            ep["image_every"] = every
        if task == "panda_pick_color":  # Phase 9e: which cube to pick (obs 39-41, one-hot color)
            ep["target_color"] = COLORS[int(np.argmax(ep["obs"][0, 39:42]))]
            ep["instruction"] = f"pick the {ep['target_color']} cube"
        episodes.append(ep)
    data.close()
    return task, episodes


def load_images(path, ep):
    """One episode's camera images, uint8 [frames, cameras, height, width, 3], from a
    file loaded with images=False."""
    with open(path, "rb") as f:
        f.seek(ep["image_at"])
        raw = zlib.decompress(f.read(ep["image_bytes"]))
    return np.frombuffer(raw, dtype=np.uint8).reshape(ep["image_shape"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--npz", help="save obs, expert, executed, episode index, success to this .npz")
    ap.add_argument("--png", help="save the camera frames of one episode (--episode) to this picture")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=int, default=10, help="frames in the picture (evenly spaced)")
    a = ap.parse_args()
    task, eps = load_demos(a.path, images=False)  # images only for --png, one episode
    steps = sum(len(e["reward"]) for e in eps)
    ok = sum(e["success"] for e in eps)
    print(f"{a.path}: task {task}, {len(eps)} episodes, {steps} steps, "
          f"success {ok} ({100.0 * ok / max(1, len(eps)):.1f}%), "
          f"noise {min(e['noise'] for e in eps):.3f}-{max(e['noise'] for e in eps):.3f}")
    if "image_shape" in eps[0]:
        sh = eps[0]["image_shape"]
        frames = [e["image_shape"][0] for e in eps]
        print(f"camera images: {sh[1]} cameras, {sh[3]} x {sh[2]}, every {eps[0]['image_every']} steps "
              f"({min(frames)}-{max(frames)} frames per episode, mean {sum(frames) / len(frames):.0f}), "
              f"{sum(e['image_bytes'] for e in eps) / 1e9:.1f} GB compressed")
    if a.png:
        ep = eps[a.episode]
        if "image_shape" in ep:
            ep = dict(ep, images=load_images(a.path, ep))
        save_png(ep, a.png, a.frames)
        print(f"wrote {a.png} (episode {a.episode}: one row per camera, time to the right)")
    if a.npz:
        np.savez_compressed(a.npz,
                            obs=np.concatenate([e["obs"] for e in eps]),
                            expert=np.concatenate([e["expert"] for e in eps]),
                            executed=np.concatenate([e["executed"] for e in eps]),
                            episode=np.concatenate([np.full(len(e["reward"]), i) for i, e in enumerate(eps)]),
                            success=np.array([e["success"] for e in eps]))
        print(f"wrote {a.npz}")


def save_png(ep, path, n):
    """One row per camera, n frames evenly spaced in time, 2x size."""
    if "images" not in ep:
        raise SystemExit("this file has no camera images (record with collect_app --images true)")
    im = ep["images"]
    picks = np.linspace(0, im.shape[0] - 1, min(n, im.shape[0])).round().astype(int)
    rows = [np.concatenate([im[f, c] for f in picks], axis=1) for c in range(im.shape[1])]
    grid = np.concatenate(rows, axis=0).repeat(2, axis=0).repeat(2, axis=1)
    write_png(path, grid)


def write_png(path, rgb):
    """Minimal PNG writer (no extra packages): rgb uint8 [H, W, 3]."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\0" + rgb[y].tobytes() for y in range(h))
    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


if __name__ == "__main__":
    main()
