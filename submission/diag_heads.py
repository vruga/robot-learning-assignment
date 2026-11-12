"""Diagnostics from the paper (arXiv:2610.20429): does a head over-smooth, and is it reactive?

Held-out episodes (same split as train.py: seed 0, 6 episodes). For chunks predicted every 10 steps:
  * L1 error to the expert chunk
  * jerk ratio = mean |third difference of predicted arm-joint chunk| / same for the target chunk
    (ratio < 1: the head over-smooths in time, the paper's explanation for why continuous heads lose)
  * reactivity = median L2 change of the predicted chunk between frames 4 steps apart / the
    expert's change over the same pair (1.0 matches the expert)

    .venv-train/bin/python -m submission.diag_heads --packed /var/tmp/sra_data/packed --ckpt A.pt B.pt
"""

import argparse
import json

import numpy as np
import torch

from .bc_policy import load_checkpoint


def jerk(a):  # a: (H, D) -> mean |third difference| over time and dims
    d3 = a[3:] - 3 * a[2:-1] + 3 * a[1:-2] - a[:-3]
    return np.abs(d3).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=0.0)
    args = ap.parse_args()
    from pathlib import Path

    pk = Path(args.packed)
    rgb = np.load(pk / "rgb.npy", mmap_mode="r")
    state, action, episode = (np.load(pk / f"{k}.npy") for k in ("state", "action", "episode"))
    eps = np.unique(episode)
    rng = np.random.RandomState(0)
    val = rng.choice(eps, 6, replace=False)  # identical to train.py's held-out split
    dev = "cuda"
    out = {}
    for path in args.ckpt:
        model, mean, std, ck = load_checkpoint(path, dev)
        H = ck["horizon"]
        l1, jr, reac_p, reac_t = [], [], [], []
        for e in val:
            ii = np.where(episode == e)[0]
            for k in range(0, len(ii) - H - 4, args.stride):
                i0 = ii[k]
                tgt = action[i0 : i0 + H]
                pred = {}
                for off in (0, 4):
                    i = i0 + off
                    r = torch.from_numpy(np.ascontiguousarray(rgb[i])).to(dev)[None].float() / 255
                    s = (torch.from_numpy(state[i]).to(dev)[None] - mean) / std
                    with torch.no_grad():
                        out_t = model.act(r, s, args.cfg)
                        if ck.get("relative"):
                            from .model import action_limits, arm_ref

                            lo, hi = (torch.tensor(v, device=dev) for v in action_limits())
                            raw = torch.from_numpy(state[i]).to(dev)[None]
                            out_t = out_t * ck["rel_range"].to(dev) + arm_ref(raw, lo, hi)[:, None]
                        pred[off] = out_t[0].cpu().numpy()
                l1.append(np.abs(pred[0] - tgt).mean())
                jt = jerk(tgt[:, 4:18])
                if jt > 1e-6:
                    jr.append(jerk(pred[0][:, 4:18]) / jt)
                tgt4 = action[i0 + 4 : i0 + 4 + H]
                reac_p.append(np.linalg.norm(pred[4].mean(0) - pred[0].mean(0)))
                reac_t.append(np.linalg.norm(tgt4.mean(0) - tgt.mean(0)))
        out[path] = {
            "head": ck.get("head", "continuous"),
            "loss": ck["args"].get("loss", "l1"),
            "chunks": len(l1),
            "l1": float(np.mean(l1)),
            "jerk_ratio_median": float(np.median(jr)),
            "jerk_ratio_mean": float(np.mean(jr)),
            "reactivity": float(np.median(reac_p) / np.median(reac_t)),
        }
        print(json.dumps({path: out[path]}), flush=True)


if __name__ == "__main__":
    main()
