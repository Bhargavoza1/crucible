"""Small VLA for openarm_pick_place (Phase 10): supervised training (SFT) on the scripted
expert's demos, the first step of the pipeline (then DAgger, then distillation to a
CNN + GRU, then RL fine-tuning).

    python tools/vla_openarm.py demos/openarm_img.bin --out runs/oa_vla.pt --onnx runs/oa_vla.onnx
    python tools/vla_openarm.py demos/openarm_img.bin --max_episodes 300 --epochs 2 --text words   # quick test
    python tools/vla_openarm.py --load runs/oa_vla.pt --onnx runs/oa_vla.onnx                      # export only
    then: ./apps/vla_app --task openarm_pick_place --onnx runs/oa_vla.onnx --eval_envs 500
    DAgger: ./apps/vla_app --task openarm_pick_place --onnx runs/oa_vla.onnx --eval_envs 2000 --eval_seed 2000 \
                --record demos/oa_dagger1.bin
            python tools/vla_openarm.py demos/openarm_img.bin demos/oa_dagger1.bin --init runs/oa_vla.pt \
                --epochs 3 --out runs/oa_vla_d1.pt --onnx runs/oa_vla_d1.onnx

Inputs per frame: two of the cell's cameras, head (0) and left wrist (1) (--cameras),
the robot's own state (obs 0-23: joint positions and velocities, TCP, turn, finger
opening; not the cube or the tray: those it must see; --no_velocity: without the
joint velocities, so a stopped arm does not tell the model "stay stopped": the
copycat problem of behavior cloning, a suspect for VLAs that freeze above the cube) and the instruction ("put the
cube in the tray"). Output: an action chunk, the next --chunk (8) actions, by flow
matching (as pi0 / SmolVLA): from noise, --flow_steps (10) Euler steps along a learned
velocity field. The network is vla_train.py's (per camera a CNN with FiLM from the
text, state and text MLPs, a 512 x 3 velocity MLP).

Data: collect_app's file with images (17 GB for 10 000 episodes; 180 GB unpacked) is
read a few episodes at a time: --workers processes each unpack --pool episodes, mix
their frames and hand out batches (a frame every image_every steps; its label is the
expert's next --chunk actions, the last one repeated at the episode's end). The
failed demos are left out (DAgger files, after the first: every episode; of their
chunks only the first action, --dagger_chunk first: the rest are the expert's answers
to where the VLA went, not one plan; with all of it, DAgger dropped the closed-loop
success from 91% to 63%, while only the first action of the same DAgger model
reached 91% with --exec 1). Held out: the first
file's episodes i % --val_every == 0 (the first --val_episodes of them are scored
each epoch: first action of a sampled chunk with fixed noise: MSE per action, gripper
right, move error cm / step). Random shifts of 4 pixels. State normalized by the
training frames' mean / std.

Writes --out after every epoch (weights, sizes, cameras, state mean / std, the
instruction's text vector) and <out>_metrics.json. --onnx: the whole sampler as one
ONNX graph (vla_train.py's: images uint8 [B, 2, H, W, 3], state [B, 24], text
[B, E], noise [B, chunk * 5] -> chunk [B, chunk, 5]) and <onnx without .onnx>_openarm.txt
(the camera ids and the text vector) for vla_app.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_demos import load_demos, load_images  # noqa: E402
from vla_train import ACT, STATE_COLS, Exported, Vla, WordText, encode_texts, to_images  # noqa: E402

TASK = "openarm_pick_place"
TRAIN_TEMPLATES = ["put the cube in the tray", "place the cube into the tray", "pick up the cube and put it in the tray",
                   "move the orange cube to the tray", "put the block in the tray"]
EVAL_TEMPLATE = "put the orange cube into the tray"


# ---- data ---------------------------------------------------------------------------

def index_episodes(paths, max_episodes, val_every):
    """Episodes without their images, split into training and held out."""
    train, val = [], []
    for n, path in enumerate(paths):
        task, eps = load_demos(path, images=False)
        if task != TASK:
            raise SystemExit(f"{path}: expected {TASK} demos, got {task}")
        if "image_shape" not in eps[0]:
            raise SystemExit(f"{path}: no camera images: record with collect_app --images true")
        if n == 0 and max_episodes:
            eps = eps[:max_episodes]
        for i, e in enumerate(eps):
            e["path"] = path
            if n == 0 and i % val_every == 0:
                if e["success"]:
                    val.append(e)
            elif n > 0 or e["success"]:  # first file: the expert's successes; DAgger files: all
                e["dagger"] = n > 0
                train.append(e)
    return train, val


def frames_of(e, chunk):
    """State [F, 24] and chunk labels [F, chunk, 5] of an episode's image frames."""
    T, every = len(e["reward"]), e["image_every"]
    steps = np.arange(0, T, every)[: e["image_shape"][0]]
    idx = np.minimum(steps[:, None] + np.arange(chunk)[None, :], T - 1)
    return e["obs"][steps, :STATE_COLS].astype(np.float32), e["expert"][idx].astype(np.float32)


def episode_images(e, cams):
    return load_images(e["path"], e)[:, cams]  # [F, len(cams), H, W, 3]


class Batches(torch.utils.data.IterableDataset):
    """One pass over the training episodes in batches: each worker unpacks 'pool'
    episodes at a time and hands out their frames in random order."""

    def __init__(self, eps, cams, chunk, batch, pool, seed, dagger_first):
        super().__init__()
        self.eps, self.cams, self.chunk, self.batch, self.pool, self.seed = eps, cams, chunk, batch, pool, seed
        self.dagger_first = dagger_first
        self.epoch = 0

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng((self.seed, self.epoch, wid))
        order = np.random.default_rng((self.seed, self.epoch)).permutation(len(self.eps))[wid::nw]
        img = st = lab = w = None
        for g in range(0, len(order), self.pool):
            parts = [(episode_images(self.eps[i], self.cams), *frames_of(self.eps[i], self.chunk),
                      self.label_weights(self.eps[i])) for i in order[g:g + self.pool]]
            im2 = np.concatenate([p[0] for p in parts])
            st2 = np.concatenate([p[1] for p in parts])
            lab2 = np.concatenate([p[2] for p in parts])
            w2 = np.concatenate([p[3] for p in parts])
            if img is not None:  # frames left over from the last pool
                im2, st2 = np.concatenate([img, im2]), np.concatenate([st, st2])
                lab2, w2 = np.concatenate([lab, lab2]), np.concatenate([w, w2])
            p = rng.permutation(len(st2))
            n = len(p) // self.batch * self.batch
            for b in range(0, n, self.batch):
                r = p[b:b + self.batch]
                yield (torch.from_numpy(im2[r]), torch.from_numpy(st2[r]), torch.from_numpy(lab2[r]),
                       torch.from_numpy(w2[r]))
            keep = p[n:]
            img, st, lab, w = im2[keep], st2[keep], lab2[keep], w2[keep]

    def label_weights(self, e):
        """[F, chunk]: which chunk actions are labels. A DAgger frame's chunk is the
        expert's answer to the state it shows, then its answers to where the VLA went
        next (not one plan): with dagger_first only the first action counts."""
        w = np.ones((e["image_shape"][0], self.chunk), np.float32)
        if e.get("dagger") and self.dagger_first:
            w[:, 1:] = 0.0
        return w


# ---- language -----------------------------------------------------------------------

class Text:
    """The instruction's vector: CLIP's (frozen, computed once) or learned words."""

    def __init__(self, kind):
        self.sentences = TRAIN_TEMPLATES + [EVAL_TEMPLATE]
        vec = encode_texts(kind, self.sentences)
        self.table = None if vec is None else {s: v for s, v in zip(self.sentences, vec)}
        self.words = WordText(self.sentences) if vec is None else None
        self.dim = vec.shape[1] if vec is not None else self.words.emb.embedding_dim

    def __call__(self, sentences, device):
        if self.words is not None:
            return self.words.to(device)(sentences)
        return torch.from_numpy(np.stack([self.table[s] for s in sentences])).to(device)


# ---- metrics ------------------------------------------------------------------------

def evaluate(model, tx, val, mean, std, device, steps, batch=512):
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(0)
    preds, chunks = [], []
    with torch.no_grad():
        for i in range(0, len(val["state"]), batch):
            img = to_images(val["images"][i:i + batch], device, False)
            st = torch.from_numpy((val["state"][i:i + batch] - mean) / std).to(device)
            n = len(st)
            noise = torch.randn(n, model.chunk * ACT, generator=g).to(device)
            out = model.sample(model.condition(img, st, tx.expand(n, -1)), noise, steps).cpu().numpy()
            chunks.append(out)
            preds.append(out[:, 0])
    model.train()
    pred, chunk, lab = np.concatenate(preds), np.concatenate(chunks), val["labels"]
    err = pred - lab[:, 0]
    mse = (err ** 2).mean(0)
    return {"mse": mse.tolist(), "mse_mean": float(mse.mean()), "chunk_mse": float(((chunk - lab) ** 2).mean()),
            "gripper_right": float((np.sign(pred[:, 4]) == np.sign(lab[:, 0, 4])).mean()),
            "move_error_cm": float(np.linalg.norm(err[:, :3], axis=1).mean() * 2.0)}


# ---- ONNX ---------------------------------------------------------------------------

def export_onnx(ck, path):
    model = Vla(len(ck["cameras"]), ck["size"], ck["text_dim"], ck["chunk"])
    model.load_state_dict(ck["model"])
    ex = Exported(model.eval(), ck["mean"], ck["std"], ck["flow_steps"]).eval()
    B, C, S = 2, len(ck["cameras"]), ck["size"]
    args = (torch.zeros(B, C, S, S, 3, dtype=torch.uint8), torch.zeros(B, STATE_COLS), torch.zeros(B, ck["text_dim"]),
            torch.zeros(B, ck["chunk"] * ACT))
    names = ["images", "state", "text", "noise"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.onnx.export(ex, args, path, input_names=names, output_names=["chunk"], opset_version=17,
                      dynamic_axes={n: {0: "batch"} for n in names + ["chunk"]}, dynamo=False)
    side = os.path.splitext(path)[0] + "_openarm.txt"
    with open(side, "w") as f:
        f.write(f"# {TASK} VLA: the cell's camera ids it sees (0 head, 1 left wrist, 2 ceiling, 3 right wrist)\n")
        f.write("cameras " + " ".join(str(c) for c in ck["cameras"]) + "\n")
        f.write(f"# text vector ({ck['text_dim']}) of: {EVAL_TEMPLATE}\n")
        f.write("text " + " ".join(f"{v:.7g}" for v in ck["text_vector"]) + "\n")
    try:
        import onnxruntime as ort
        rng = np.random.default_rng(0)
        feed = {"images": rng.integers(0, 256, size=args[0].shape, dtype=np.uint8),
                "state": rng.normal(size=args[1].shape).astype(np.float32),
                "text": np.stack([ck["text_vector"]] * B).astype(np.float32),
                "noise": rng.normal(size=args[3].shape).astype(np.float32)}
        out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, feed)[0]
        with torch.no_grad():
            ref = ex(*[torch.from_numpy(feed[n]) for n in names]).numpy()
        print(f"onnx check: max difference to PyTorch {np.abs(out - ref).max():.2e}")
    except ImportError:
        print("onnx check skipped (pip install onnxruntime to compare with PyTorch)")
    print(f"wrote {path} and {side}")


# ---- main ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos", nargs="*", help="demo files: collect_app demos first, then DAgger files")
    ap.add_argument("--cameras", default="0,1", help="the cell's camera ids: 0 head, 1 left wrist, 2 ceiling, 3 right")
    ap.add_argument("--text", choices=["clip", "words"], default="clip")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--flow_steps", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_every", type=int, default=10)
    ap.add_argument("--val_episodes", type=int, default=50)
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pool", type=int, default=16, help="episodes each worker unpacks at a time")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="oa_vla.pt")
    ap.add_argument("--load", default="")
    ap.add_argument("--init", default="", help="continue training this checkpoint (DAgger)")
    ap.add_argument("--onnx", default="")
    ap.add_argument("--no_velocity", action="store_true",
                    help="hide the joint velocities (obs 9-17) from the model (their std is set to 1e30)")
    ap.add_argument("--dagger_chunk", choices=["first", "all"], default="first",
                    help="DAgger frames: learn only the first action of the chunk (first) or all of it")
    a = ap.parse_args()

    if a.load:
        ck = torch.load(a.load, map_location="cpu", weights_only=False)
        if a.onnx:
            export_onnx(ck, a.onnx)
        return
    if not a.demos:
        raise SystemExit("give a demo file (or --load CHECKPOINT --onnx FILE)")
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    cams = [int(c) for c in a.cameras.split(",")]
    t0 = time.time()
    init = torch.load(a.init, map_location="cpu", weights_only=False) if a.init else None
    if init is not None and (init["chunk"] != a.chunk or init["text"] != a.text or init["cameras"] != cams):
        raise SystemExit("--init: that model has another chunk, text or cameras: use the same")
    train, val_eps = index_episodes(a.demos, a.max_episodes, a.val_every)
    if not train or not val_eps:
        raise SystemExit("no training or held-out episodes (too few episodes?)")
    val_eps = val_eps[:a.val_episodes]
    n_frames = sum(e["image_shape"][0] for e in train)
    all_state = np.concatenate([frames_of(e, a.chunk)[0] for e in train])
    mean = all_state.mean(0) if init is None else init["mean"]
    std = all_state.std(0) + 1e-6 if init is None else init["std"]
    if init is None and a.no_velocity:  # joint velocities (obs 9-17) normalized to 0: the model never sees them
        mean, std = mean.copy(), std.copy()
        mean[9:18], std[9:18] = 0.0, 1e30
    if init is not None and a.no_velocity != bool(init.get("no_velocity", False)):
        raise SystemExit("--init: that model was trained with another --no_velocity: use the same")
    vs, vl = zip(*[frames_of(e, a.chunk) for e in val_eps])
    val = {"images": np.concatenate([episode_images(e, cams) for e in val_eps]), "state": np.concatenate(vs),
           "labels": np.concatenate(vl)}
    size = val["images"].shape[2]
    text = Text(a.text)
    n_dagger = sum(e["dagger"] for e in train)
    print(f"vla_openarm: {len(train)} training episodes ({n_dagger} from DAgger files, chunk: {a.dagger_chunk}; "
          f"{n_frames} frames), "
          f"{len(val_eps)} held out "
          f"({len(val['state'])} frames), cameras {cams} {size}x{size}, text {a.text} ({text.dim}), chunk {a.chunk}, "
          f"{a.flow_steps} flow steps, device {a.device}, indexed in {time.time() - t0:.0f} s", flush=True)
    model = Vla(len(cams), size, text.dim, a.chunk)
    if init is not None:
        model.load_state_dict(init["model"])
        if text.words is not None and "words" in init:
            text.words.load_state_dict(init["words"])
        print(f"vla_openarm: starting from {a.init}")
    model = model.to(a.device)
    params = list(model.parameters()) + (list(text.words.parameters()) if text.words is not None else [])
    print(f"vla_openarm: {sum(p.numel() for p in params) / 1e6:.2f} M parameters")
    opt = torch.optim.Adam(params, lr=a.lr)
    data = Batches(train, cams, a.chunk, a.batch, a.pool, a.seed, a.dagger_chunk == "first")
    rng = np.random.default_rng(a.seed)
    mean_t, std_t = torch.from_numpy(mean).to(a.device), torch.from_numpy(std).to(a.device)
    history = []

    def save():
        vec = (text([EVAL_TEMPLATE], "cpu") if text.words is None else text.words.cpu()([EVAL_TEMPLATE]))
        ck = {"model": model.state_dict(), "cameras": cams, "size": size, "text_dim": text.dim, "text": a.text,
              "chunk": a.chunk, "flow_steps": a.flow_steps, "mean": mean, "std": std, "task": TASK,
              "no_velocity": a.no_velocity,
              "text_vector": vec.detach().numpy()[0], "eval_template": EVAL_TEMPLATE, "train_templates": TRAIN_TEMPLATES}
        if text.words is not None:
            ck["words"] = text.words.state_dict()
            text.words.to(a.device)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        torch.save(ck, a.out)
        with open(os.path.splitext(a.out)[0] + "_metrics.json", "w") as f:
            json.dump(history, f, indent=1)
        return ck

    ck = None
    for epoch in range(1, a.epochs + 1):
        te = time.time()
        data.epoch = epoch
        loader = torch.utils.data.DataLoader(data, batch_size=None, num_workers=a.workers,
                                             pin_memory=a.device.startswith("cuda"),
                                             persistent_workers=False, prefetch_factor=4 if a.workers else None)
        nb, total = 0, 0.0
        for img_u8, st, lab, w in loader:
            img = to_images(img_u8.numpy(), a.device, True)
            st = (st.to(a.device, non_blocking=True) - mean_t) / std_t
            words = [TRAIN_TEMPLATES[j] for j in rng.integers(0, len(TRAIN_TEMPLATES), len(st))]
            tx = text(words, a.device)
            x1 = lab.to(a.device, non_blocking=True).flatten(1)
            x0 = torch.randn_like(x1)
            t = torch.rand(len(st), device=a.device)
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1
            w = w.to(a.device, non_blocking=True).repeat_interleave(ACT, dim=1)  # [B, chunk * 5]
            v = model.velocity(xt, t, model.condition(img, st, tx))
            loss = (w * (v - (x1 - x0)) ** 2).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
            if nb % 500 == 0:
                print(f"  epoch {epoch} batch {nb} / {n_frames // a.batch}: flow loss {total / nb:.4f} "
                      f"({time.time() - te:.0f} s)", flush=True)
        m = evaluate(model, text([EVAL_TEMPLATE], a.device), val, mean, std, a.device, a.flow_steps)
        m.update(epoch=epoch, train_loss=total / max(1, nb), seconds=time.time() - te)
        history.append(m)
        x = m["mse"]
        print(f"epoch {epoch:3d}  flow loss {m['train_loss']:.4f}  held-out first action: mse {m['mse_mean']:.4f} "
              f"(x {x[0]:.3f} y {x[1]:.3f} z {x[2]:.3f} turn {x[3]:.3f} grip {x[4]:.3f})  gripper right "
              f"{100 * m['gripper_right']:.1f}%  move error {m['move_error_cm']:.2f} cm/step  chunk mse "
              f"{m['chunk_mse']:.4f}  ({m['seconds']:.0f} s)", flush=True)
        ck = save()
    print(f"wrote {a.out}")
    if a.onnx:
        export_onnx(ck, a.onnx)


if __name__ == "__main__":
    main()
