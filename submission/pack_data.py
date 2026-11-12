"""Pack human demos + expert episodes into uncompressed .npy files that trainings can memory-map.

Several concurrent trainings then share ONE copy of the images in the OS page cache
(instead of each holding ~20 GB in RAM).

    uv run python -m submission.pack_data --demos datasets/move_plate.npz \
        --expert-dir datasets/expert_train --max-expert 298 --out /var/tmp/sra_data/packed
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="datasets/move_plate.npz")
    ap.add_argument("--expert-dir", default="")
    ap.add_argument("--max-expert", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Keep only paths (never ~1400 open zip handles): lengths first, then load each file on demand.
    srcs = []  # (kind, path, length)
    if args.demos:
        with np.load(args.demos) as d:
            srcs.append(("demo", args.demos, len(d["action"])))
    files = sorted(Path(args.expert_dir).glob("*.npz")) if args.expert_dir else []
    if args.max_expert:
        files = files[: args.max_expert]
    for f in files:
        with np.load(f) as e:
            srcs.append(("expert", f, len(e["action"])))
    total = sum(s[2] for s in srcs)
    rgb = np.lib.format.open_memmap(
        out / "rgb.npy", mode="w+", dtype=np.uint8, shape=(total, 3, 3, 84, 84)
    )
    state = np.empty((total, 50), np.float32)
    action = np.empty((total, 20), np.float32)
    episode = np.empty(total, np.int64)
    slot = np.full(total, -1, np.int8)  # source slot (0-based); -1 = unknown (human demos)
    at, ep = 0, 0
    for kind, path, n in srcs:
        with np.load(path) as h:
            rgb[at : at + n] = h["rgb"]
            state[at : at + n] = h["state"]
            action[at : at + n] = h["action"]
            episode[at : at + n] = h["episode"] if kind == "demo" else 100_000 + ep
            if kind == "expert":
                slot[at : at + n] = int(h["slot"])
        if kind == "expert":
            ep += 1
        at += n
    rgb.flush()
    for k, v in (("state", state), ("action", action), ("episode", episode), ("slot", slot)):
        np.save(out / f"{k}.npy", v)
    print("packed", total, "steps ->", out)


if __name__ == "__main__":
    main()
