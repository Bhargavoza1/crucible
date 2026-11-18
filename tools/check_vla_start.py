"""Does an exported OpenArm VLA (vla_openarm.py --onnx) move the right way on DEMO frames?

    python tools/check_vla_start.py runs/st_nv_e3.onnx demos/openarm_stack.bin
    python tools/check_vla_start.py runs/st_nv_e3.onnx demos/openarm_stack.bin --frame 10 --episodes 100

Runs the ONNX file (the same graph vla_app runs, with ONNX Runtime on the CPU) on image
frame --frame (0: the episode's start) of the first --episodes episodes of a demo file:
the demo's own camera images (the model's cameras, from <onnx without .onnx>_openarm.txt),
the demo's own state (obs 0 .. state-1), the instruction's text vector from that file and
flow noise 0 (vla_app --noise zero). It compares the model's TCP move over the first
--steps (4: vla_app's --exec) actions of its chunk with the expert's labels for the same
steps, per arm: the direction (cosine), the x move (is the hand pulled back while the
expert goes forward?) and the size.
Reading it: the model agrees with the expert here but goes the wrong way in the sim -> the
live inputs differ from the demo inputs (a bug in the closed loop); it is wrong here too ->
the model itself is the problem.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_demos import load_demos, load_images  # noqa: E402


def read_info(path):
    cams, text, task = [], [], ""
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p or p[0].startswith("#"):
                continue
            if p[0] == "cameras":
                cams = [int(x) for x in p[1:]]
            elif p[0] == "text":
                text = [float(x) for x in p[1:]]
            elif p[0] == "task":
                task = p[1]
    if not cams or not text:
        raise SystemExit(f"{path}: needs 'cameras' and 'text' lines")
    return cams, np.array(text, np.float32), task or "openarm_pick_place"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("demos")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--frame", type=int, default=0, help="image frame of each episode (0: the start)")
    ap.add_argument("--steps", type=int, default=4, help="chunk actions to add up (vla_app --exec)")
    ap.add_argument("--show", type=int, default=10, help="episodes printed one by one")
    a = ap.parse_args()

    import onnxruntime as ort
    cams, text, task = read_info(os.path.splitext(a.onnx)[0] + "_openarm.txt")
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    shapes = {i.name: i.shape for i in sess.get_inputs()}
    S, A = shapes["state"][1], sess.get_outputs()[0].shape[2]
    C = shapes["noise"][1] // A
    if a.steps < 1 or a.steps > C:
        raise SystemExit(f"--steps must be 1 .. {C} (the chunk)")
    dtask, eps = load_demos(a.demos, images=False)
    if dtask != task:
        raise SystemExit(f"{a.onnx} is a {task} model, {a.demos} has {dtask} demos")
    arms = [("left", [0, 1, 2]), ("right", [5, 6, 7])] if A == 10 else [("arm", [0, 1, 2])]
    print(f"check_vla_start: {a.onnx} ({task}, cameras {cams}, state {S}, act {A}, chunk {C}); "
          f"{a.demos}: frame {a.frame} of {min(a.episodes, len(eps))} episodes, the first {a.steps} actions, noise 0")

    rows = []  # per (episode, arm): cosine, model x, expert x, model size, expert size
    for i, e in enumerate(eps[:a.episodes]):
        F = e["image_shape"][0]
        if a.frame >= F:
            continue
        step = a.frame * e["image_every"]
        img = load_images(a.demos, e)[a.frame:a.frame + 1][:, cams]
        st = e["obs"][step:step + 1, :S].astype(np.float32)
        out = sess.run(None, {"images": np.ascontiguousarray(img), "state": st, "text": text[None, :],
                              "noise": np.zeros((1, C * A), np.float32)})[0][0]  # [chunk, act]
        idx = np.minimum(step + np.arange(a.steps), len(e["reward"]) - 1)
        lab = e["expert"][idx]
        line = []
        for k, (name, m) in enumerate(arms):
            pm, em = out[:a.steps, m].sum(0), lab[:, m].sum(0)  # summed moves (x 2 cm = metres)
            npm, nem = np.linalg.norm(pm), np.linalg.norm(em)
            cos = float(pm @ em / (npm * nem)) if npm > 1e-6 and nem > 1e-6 else float("nan")
            rows.append((k, cos, pm[0], em[0], npm, nem))
            line.append(f"{name}: model ({pm[0]:+.2f} {pm[1]:+.2f} {pm[2]:+.2f}) expert ({em[0]:+.2f} {em[1]:+.2f} "
                        f"{em[2]:+.2f}) cos {cos:+.2f}")
        if i < a.show:
            print(f"  ep {i:4d}  " + "  |  ".join(line))

    r = np.array(rows, dtype=np.float64)
    print("summary (summed move actions; x 0.02 m = metres):")
    for k, (name, _) in enumerate(arms):
        q = r[r[:, 0] == k]
        moving = q[q[:, 5] > 0.2]  # the expert really moves this arm
        if len(moving) == 0:
            print(f"  {name}: the expert does not move this arm at this frame")
            continue
        cos = moving[:, 1]
        back = np.mean((moving[:, 3] > 0.1) & (moving[:, 2] < -0.1))
        print(f"  {name}: {len(moving)} frames where the expert moves | direction agrees (cos > 0.7) "
              f"{100 * np.mean(cos > 0.7):.0f}%, opposite (cos < 0) {100 * np.mean(cos < 0):.0f}%, "
              f"mean cos {np.nanmean(cos):+.2f} | model pulls back while the expert goes forward in x "
              f"{100 * back:.0f}% | size model {moving[:, 4].mean():.2f} vs expert {moving[:, 5].mean():.2f}")


if __name__ == "__main__":
    main()
