"""Export the BiGym2 move_plate demos to one npz (needs the main `uv sync --locked` env).

MUJOCO_GL=egl uv run python -m submission.export_demos --out datasets/move_plate.npz
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/move_plate.npz")
    ap.add_argument("--num", type=int, default=60)
    args = ap.parse_args()

    from bigym.loco import make_gym

    env = make_gym("move_plate")
    demos = env.get_demos(args.num)
    env.close()
    rgb, state, action, ep, step = [], [], [], [], []
    for i, d in enumerate(demos):
        if not d["success"]:
            continue
        n = len(d["action"])
        rgb.append(d["obs"]["rgb"][:n])  # obs has T+1 entries, action T
        state.append(d["obs"]["state"][:n])
        action.append(d["action"])
        ep.append(np.full(n, i))
        step.append(np.arange(n))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        rgb=np.concatenate(rgb),
        state=np.concatenate(state).astype(np.float32),
        action=np.concatenate(action).astype(np.float32),
        episode=np.concatenate(ep),
        step=np.concatenate(step),
    )
    print(f"saved {len(rgb)} episodes, {sum(map(len, ep))} steps -> {args.out}")


if __name__ == "__main__":
    main()
