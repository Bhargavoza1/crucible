"""Small vision-language-action policy for panda_pick_color (Phase 9e, plan B).

    python tools/vla_train.py demos/color_img.bin --out runs/vla.pt
    python tools/vla_train.py demos/color_img.bin --text words     # no download (tests)
    python tools/vla_train.py --load runs/vla.pt --onnx runs/vla.onnx   # export for TensorRT
    # DAgger: vla_app --record dagger1.bin, then continue training on demos + DAgger data
    python tools/vla_train.py demos/color_img.bin dagger1.bin --init runs/vla.pt --epochs 5 --out runs/vla_d1.pt

Inputs per frame: the two camera images (front, wrist), the robot state (obs 0-23)
and the instruction ("pick the red cube"). Output: an action chunk, the next
--chunk (8) actions, made by flow matching (as pi0 / SmolVLA): start from noise and
follow a learned velocity field for --flow_steps (10) Euler steps. Flow matching can
put its answer on one of two good actions (turn left OR right) where MSE behavior
cloning learns their average (no turn: the hover failures of Phase 9d).

Language: --text clip (default) encodes each instruction once with CLIP's frozen
text encoder (openai/clip-vit-base-patch32, downloaded by transformers the first
time) and keeps the vectors; the policy only sees those vectors, so the engine needs
no tokenizer (design doc 8.1: language computed once per task). --text words: a
small learned word embedding instead (no download; for tests).
Instructions: TRAIN_TEMPLATES, one picked at random per training frame; the held-out
metrics use EVAL_TEMPLATE, a phrasing never seen in training (with CLIP it should
still work; with --text words it shares the color word only).

Network: per camera the bc_train.py CNN with FiLM (text-made scale and shift of the
conv channels after conv 2 and 4, so the image features can depend on which color
is asked), then Linear 256 + LayerNorm; state Linear 128; text Linear 128; the
velocity MLP (512 x 3) gets [noisy chunk, time features, all of that].

Data, split, augmentation as bc_train.py: frames every image_every steps, episode i
held out when i % val_every == 0, random shifts of 4 pixels. Several files (DAgger):
the held-out episodes come from the first file only; every episode of the other
files (vla_app --record: the policy's states, labeled with the expert's actions)
is for training. --init CHECKPOINT: start from that model (its state mean / std and
text vectors) instead of a new one. State normalized by
the training frames' mean / std. Held-out metrics on the first action of a sampled
chunk (fixed noise seed): MSE, gripper right, move error cm / step; and how often
the first move points toward the named cube vs another cube ("toward named").

Writes --out (weights, sizes, state mean / std, text vectors per color) and
<out>_metrics.json. --onnx: the whole sampler (preprocess, encoders, 10 Euler steps)
as one ONNX graph: inputs images uint8 [B, cams, H, W, 3], state [B, 24] (raw:
normalized inside), text [B, E], noise [B, chunk * 5]; output chunk [B, chunk, 5].
The text vectors for the C++ side go to <onnx without .onnx>_text.txt (one line
per color: name then E numbers; instruction EVAL_TEMPLATE).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_train import random_shift  # noqa: E402
from read_demos import COLORS, load_demos  # noqa: E402

STATE_COLS = 24
ACT = 5
TRAIN_TEMPLATES = ["pick the {} cube", "pick up the {} cube", "lift the {} block", "grab the {} one",
                   "take the {} cube"]
EVAL_TEMPLATE = "get the {} cube"


# ---- data ---------------------------------------------------------------------------

def load(paths, max_episodes, chunk):
    eps, first = [], 0
    for n, path in enumerate(paths):
        task, e = load_demos(path)
        if task != "panda_pick_color":
            raise SystemExit(f"{path}: expected panda_pick_color demos, got {task}")
        if "images" not in e[0]:
            raise SystemExit(f"{path}: no camera images: record with collect_app --images true")
        if n == 0:
            e = e[:max_episodes] if max_episodes else e
            first = len(e)
        eps += e
    every = eps[0]["image_every"]
    state, labels, color, frame_ep, frame_k, cubes = [], [], [], [], [], []
    for i, e in enumerate(eps):
        T = len(e["reward"])
        steps = np.arange(0, T, every)[: len(e["images"])]
        state.append(e["obs"][steps, :STATE_COLS])
        idx = np.minimum(steps[:, None] + np.arange(chunk)[None, :], T - 1)  # chunk labels, last action repeated
        labels.append(e["expert"][idx])
        color.append(np.full(len(steps), COLORS.index(e["target_color"])))
        frame_ep.append(np.full(len(steps), i))
        frame_k.append(np.arange(len(steps)))
        o = e["obs"][steps]  # positions of all cubes and the finger pads, for "toward named"
        cubes.append(np.stack([o[:, 24:27], o[:, 42:45], o[:, 45:48], o[:, 24:27] - o[:, 31:34]], axis=1))
    return {"eps": eps, "first_file_eps": first, "state": np.concatenate(state).astype(np.float32),
            "labels": np.concatenate(labels).astype(np.float32), "color": np.concatenate(color),
            "ep": np.concatenate(frame_ep), "k": np.concatenate(frame_k), "cubes": np.concatenate(cubes),
            "n_eps": len(eps)}


def gather_images(d, rows):
    return np.stack([d["eps"][d["ep"][r]]["images"][d["k"][r]] for r in rows])  # [B, cams, H, W, 3]


# ---- language -----------------------------------------------------------------------

def encode_texts(kind, sentences):
    """Vectors for each sentence: CLIP (frozen, once) or None for the learned words."""
    if kind != "clip":
        return None
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection
    name = "openai/clip-vit-base-patch32"
    tok = CLIPTokenizer.from_pretrained(name)
    model = CLIPTextModelWithProjection.from_pretrained(name).eval()
    with torch.no_grad():
        out = model(**tok(sentences, padding=True, return_tensors="pt")).text_embeds
    return F.normalize(out, dim=-1).numpy().astype(np.float32)


class WordText(nn.Module):
    """--text words: mean of learned word vectors (vocabulary of the templates + colors)."""

    def __init__(self, sentences, dim=64):
        super().__init__()
        self.vocab = sorted({w for s in sentences for w in s.split()})
        self.emb = nn.Embedding(len(self.vocab), dim)

    def ids(self, s):
        return [self.vocab.index(w) for w in s.split() if w in self.vocab]

    def forward(self, sentences):
        return torch.stack([self.emb(torch.tensor(self.ids(s), device=self.emb.weight.device)).mean(0)
                            for s in sentences])


# ---- network ------------------------------------------------------------------------

class FilmEncoder(nn.Module):
    def __init__(self, size, text_dim):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(3, 32, 5, 2), nn.Conv2d(32, 64, 3, 2)
        self.c3, self.c4 = nn.Conv2d(64, 64, 3, 2), nn.Conv2d(64, 64, 3, 1)
        self.film = nn.Linear(text_dim, 4 * 64)  # scale and shift after conv 2 and conv 4
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        with torch.no_grad():
            n = self.c4(self.c3(self.c2(self.c1(torch.zeros(1, 3, size, size))))).numel()
        self.fc = nn.Sequential(nn.Linear(n, 256), nn.LayerNorm(256))

    def forward(self, x, t):  # x [B, 3, H, W] in [0, 1], t [B, text_dim]
        g = self.film(t).view(-1, 4, 64, 1, 1)
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x) * (1 + g[:, 0]) + g[:, 1])
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x) * (1 + g[:, 2]) + g[:, 3])
        return self.fc(x.flatten(1))


def time_features(t, n=32):  # t [B] in [0, 1] -> [B, 2n]
    f = torch.exp(torch.linspace(0, math.log(1000.0), n, device=t.device))
    a = t[:, None] * f[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class Vla(nn.Module):
    def __init__(self, cams, size, text_dim, chunk, state_dim=STATE_COLS, act=ACT):
        super().__init__()
        self.cams, self.chunk, self.act = cams, chunk, act
        self.enc = nn.ModuleList([FilmEncoder(size, text_dim) for _ in range(cams)])
        self.state = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.text = nn.Sequential(nn.Linear(text_dim, 128), nn.ReLU())
        cond = 256 * cams + 128 + 128
        self.vel = nn.Sequential(nn.Linear(chunk * act + 64 + cond, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                                 nn.Linear(512, 512), nn.GELU(), nn.Linear(512, chunk * act))

    def condition(self, img, state, text):  # img [B, cams, 3, H, W]
        parts = [enc(img[:, c], text) for c, enc in enumerate(self.enc)]
        return torch.cat(parts + [self.state(state), self.text(text)], dim=1)

    def velocity(self, x, t, cond):  # x [B, chunk * act], t [B]
        return self.vel(torch.cat([x, time_features(t), cond], dim=1))

    def sample(self, cond, noise, steps):  # Euler from noise (t = 0) to actions (t = 1)
        x = noise
        for k in range(steps):
            t = torch.full((x.shape[0],), k / steps, device=x.device)
            x = x + self.velocity(x, t, cond) / steps
        return x.clamp(-1, 1).view(-1, self.chunk, self.act)


def to_images(u8, device, shift):  # [B, cams, H, W, 3] uint8 -> [B, cams, 3, H, W] float
    x = torch.from_numpy(u8).to(device).permute(0, 1, 4, 2, 3).float() / 255.0
    if shift:
        B, C = x.shape[:2]
        x = random_shift(x.flatten(0, 1)).view(B, C, *x.shape[2:])
    return x


class Text:
    """Instruction vectors: fixed (CLIP, looked up) or computed by WordText (trained)."""

    def __init__(self, kind):
        self.kind = kind
        self.sentences = [t.format(c) for c in COLORS for t in TRAIN_TEMPLATES + [EVAL_TEMPLATE]]
        vec = encode_texts(kind, self.sentences)
        self.table = None if vec is None else {s: v for s, v in zip(self.sentences, vec)}
        self.words = WordText(self.sentences) if vec is None else None
        self.dim = vec.shape[1] if vec is not None else self.words.emb.embedding_dim

    def __call__(self, sentences, device):
        if self.words is not None:
            return self.words.to(device)(sentences)
        return torch.from_numpy(np.stack([self.table[s] for s in sentences])).to(device)


# ---- metrics ------------------------------------------------------------------------

def evaluate(model, text, d, rows, mean, std, device, steps, batch=1024):
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(0)
    first, chunks = [], []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            r = rows[i:i + batch]
            img = to_images(gather_images(d, r), device, False)
            st = torch.from_numpy((d["state"][r] - mean) / std).to(device)
            tx = text([EVAL_TEMPLATE.format(COLORS[c]) for c in d["color"][r]], device)
            noise = torch.randn(len(r), model.chunk * ACT, generator=g).to(device)
            out = model.sample(model.condition(img, st, tx), noise, steps).cpu().numpy()
            chunks.append(out)
            first.append(out[:, 0])
    model.train()
    pred, chunk = np.concatenate(first), np.concatenate(chunks)
    lab = d["labels"][rows]
    err = pred - lab[:, 0]
    mse = (err ** 2).mean(0)
    # toward named: when the expert moves sideways (> 0.5 of full speed), does the
    # policy's sideways move point closer to the named cube than to the others?
    cubes = d["cubes"][rows]  # named, other 1, other 2, pad center
    pad = cubes[:, 3, :2]
    mv = pred[:, :2]
    moving = np.linalg.norm(lab[:, 0, :2], axis=1) > 0.5
    def cos_to(k):
        v = cubes[:, k, :2] - pad
        return (mv * v).sum(1) / (np.linalg.norm(mv, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    named_best = (cos_to(0) > np.maximum(cos_to(1), cos_to(2)))[moving]
    return {"mse": mse.tolist(), "mse_mean": float(mse.mean()), "chunk_mse": float(((chunk - lab) ** 2).mean()),
            "gripper_right": float((np.sign(pred[:, 4]) == np.sign(lab[:, 0, 4])).mean()),
            "move_error_cm": float(np.linalg.norm(err[:, :3], axis=1).mean() * 2.0),
            "toward_named": float(named_best.mean()) if len(named_best) else float("nan")}


# ---- ONNX ---------------------------------------------------------------------------

class Exported(nn.Module):
    """The whole sampler for TensorRT: raw inputs in, action chunk out."""

    def __init__(self, model, mean, std, steps):
        super().__init__()
        self.model, self.steps = model, steps
        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("std", torch.from_numpy(std))

    def forward(self, images, state, text, noise):  # images uint8 [B, cams, H, W, 3]
        img = images.permute(0, 1, 4, 2, 3).float() / 255.0
        cond = self.model.condition(img, (state - self.mean) / self.std, text)
        return self.model.sample(cond, noise, self.steps)


def export_onnx(ck, path):
    model = Vla(ck["cams"], ck["size"], ck["text_dim"], ck["chunk"])
    model.load_state_dict(ck["model"])
    model.eval()
    ex = Exported(model, ck["mean"], ck["std"], ck["flow_steps"]).eval()
    B = 2
    args = (torch.zeros(B, ck["cams"], ck["size"], ck["size"], 3, dtype=torch.uint8),
            torch.zeros(B, STATE_COLS), torch.zeros(B, ck["text_dim"]), torch.zeros(B, ck["chunk"] * ACT))
    names = ["images", "state", "text", "noise"]
    torch.onnx.export(ex, args, path, input_names=names, output_names=["chunk"], opset_version=17,
                      dynamic_axes={n: {0: "batch"} for n in names + ["chunk"]}, dynamo=False)
    tpath = os.path.splitext(path)[0] + "_text.txt"
    with open(tpath, "w") as f:
        f.write(f"# {EVAL_TEMPLATE} -> text vector ({ck['text_dim']}), one line per color\n")
        for c in COLORS:
            f.write(c + " " + " ".join(f"{v:.7g}" for v in ck["text_vectors"][c]) + "\n")
    # check: the ONNX graph gives the same chunk as PyTorch (onnxruntime, if installed)
    try:
        import onnxruntime as ort
        rng = np.random.default_rng(0)
        feed = {"images": rng.integers(0, 256, size=args[0].shape, dtype=np.uint8),
                "state": rng.normal(size=args[1].shape).astype(np.float32),
                "text": np.stack([ck["text_vectors"][COLORS[i % 3]] for i in range(B)]).astype(np.float32),
                "noise": rng.normal(size=args[3].shape).astype(np.float32)}
        out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, feed)[0]
        with torch.no_grad():
            ref = ex(*[torch.from_numpy(feed[n]) for n in names]).numpy()
        print(f"onnx check: max difference to PyTorch {np.abs(out - ref).max():.2e}")
    except ImportError:
        print("onnx check skipped (pip install onnxruntime to compare with PyTorch)")
    print(f"wrote {path} and {tpath}")


# ---- main ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos", nargs="*", help="demo files: collect_app demos first, then DAgger files")
    ap.add_argument("--text", choices=["clip", "words"], default="clip")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--flow_steps", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_every", type=int, default=10)
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="vla.pt")
    ap.add_argument("--load", default="")
    ap.add_argument("--init", default="", help="continue training this checkpoint (DAgger)")
    ap.add_argument("--onnx", default="")
    a = ap.parse_args()

    if a.load:
        ck = torch.load(a.load, map_location="cpu", weights_only=False)
        if a.onnx:
            export_onnx(ck, a.onnx)
        return
    if not a.demos:
        raise SystemExit("give a demo file (or --load CHECKPOINT --onnx FILE)")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    init = torch.load(a.init, map_location="cpu", weights_only=False) if a.init else None
    if init is not None and (init["chunk"] != a.chunk or init["text"] != a.text):
        raise SystemExit(f"--init: that model has chunk {init['chunk']} and text {init['text']}: use the same")
    d = load(a.demos, a.max_episodes, a.chunk)
    is_val = (d["ep"] < d["first_file_eps"]) & (d["ep"] % a.val_every == 0)
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    cams, size = d["eps"][0]["images"].shape[1], d["eps"][0]["images"].shape[2]
    mean = d["state"][tr].mean(0) if init is None else init["mean"]
    std = d["state"][tr].std(0) + 1e-6 if init is None else init["std"]
    text = Text(a.text)
    print(f"vla: {d['n_eps']} episodes ({d['n_eps'] - d['first_file_eps']} from DAgger files), "
          f"{len(tr)} / {len(va)} frames (train / held out), {cams} cameras {size}x{size}, "
          f"text {a.text} ({text.dim}), chunk {a.chunk}, {a.flow_steps} flow steps, device {a.device}, "
          f"loaded in {time.time() - t0:.0f} s")
    model = Vla(cams, size, text.dim, a.chunk)
    if init is not None:
        model.load_state_dict(init["model"])
        if text.words is not None and "words" in init:
            text.words.load_state_dict(init["words"])
        print(f"vla: starting from {a.init}")
    model = model.to(a.device)
    params = list(model.parameters()) + (list(text.words.parameters()) if text.words is not None else [])
    print(f"vla: {sum(p.numel() for p in params) / 1e6:.2f} M parameters")
    opt = torch.optim.Adam(params, lr=a.lr)
    if text.words is not None:
        text.words.to(a.device)
    history = []
    for epoch in range(1, a.epochs + 1):
        te = time.time()
        perm = rng.permutation(tr)
        nb, total = len(perm) // a.batch, 0.0
        for b in range(nb):
            r = perm[b * a.batch:(b + 1) * a.batch]
            img = to_images(gather_images(d, r), a.device, True)
            st = torch.from_numpy((d["state"][r] - mean) / std).to(a.device)
            words = [TRAIN_TEMPLATES[j].format(COLORS[c])
                     for j, c in zip(rng.integers(0, len(TRAIN_TEMPLATES), len(r)), d["color"][r])]
            tx = text(words, a.device)
            x1 = torch.from_numpy(d["labels"][r]).to(a.device).flatten(1)
            x0 = torch.randn_like(x1)
            t = torch.rand(len(r), device=a.device)
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1
            loss = F.mse_loss(model.velocity(xt, t, model.condition(img, st, tx)), x1 - x0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        m = evaluate(model, text, d, va, mean, std, a.device, a.flow_steps)
        m.update(epoch=epoch, train_loss=total / max(1, nb), seconds=time.time() - te)
        history.append(m)
        x = m["mse"]
        print(f"epoch {epoch:3d}  flow loss {m['train_loss']:.4f}  held-out first action: mse {m['mse_mean']:.4f} "
              f"(x {x[0]:.3f} y {x[1]:.3f} z {x[2]:.3f} turn {x[3]:.3f} grip {x[4]:.3f})  gripper right "
              f"{100 * m['gripper_right']:.1f}%  move error {m['move_error_cm']:.2f} cm/step  chunk mse "
              f"{m['chunk_mse']:.4f}  toward named {100 * m['toward_named']:.1f}%  ({m['seconds']:.0f} s)", flush=True)
    vectors = {c: text([EVAL_TEMPLATE.format(c)], "cpu").detach().numpy()[0] if text.words is None else
               text.words.cpu()([EVAL_TEMPLATE.format(c)]).detach().numpy()[0] for c in COLORS}
    ck = {"model": model.state_dict(), "cams": cams, "size": size, "text_dim": text.dim, "text": a.text,
          "chunk": a.chunk, "flow_steps": a.flow_steps, "mean": mean, "std": std, "text_vectors": vectors,
          "eval_template": EVAL_TEMPLATE, "train_templates": TRAIN_TEMPLATES}
    if text.words is not None:
        ck["words"] = text.words.state_dict()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(ck, a.out)
    with open(os.path.splitext(a.out)[0] + "_metrics.json", "w") as f:
        json.dump(history, f, indent=1)
    print(f"wrote {a.out}")
    if a.onnx:
        export_onnx(ck, a.onnx)


if __name__ == "__main__":
    main()
