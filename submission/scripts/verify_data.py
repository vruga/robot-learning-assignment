"""Verify the privileged-expert dataset: distribution vs human demos, and exact simulator replay.

    MUJOCO_GL=egl .venv/bin/python submission/scripts/verify_data.py \
        --demos datasets/move_plate.npz --expert-dir datasets/expert_train

Replays recorded actions of a few expert episodes through the real simulator and checks that the
recorded states are reproduced exactly and that the episode ends in success. (Replay is a data
integrity check only; it is never used as a policy.)
"""

import argparse
import glob

import numpy as np


def describe(demos, expert_dir):
    h = np.load(demos)
    ha, hs, he = h["action"], h["state"], h["episode"]
    files = sorted(glob.glob(f"{expert_dir}/*.npz"))
    ea = np.concatenate([np.load(f)["action"] for f in files])
    elen = [len(np.load(f)["action"]) for f in files]
    hlen = [int((he == e).sum()) for e in np.unique(he)]
    share = sum(hlen) / (sum(hlen) + sum(elen))
    print(f"episodes human {len(hlen)} expert {len(elen)}; human share of steps {share:.3f}")
    print(f"episode length mean human {np.mean(hlen):.0f} expert {np.mean(elen):.0f}")
    groups = (
        ("base vx,vy,h,yaw", slice(0, 4)),
        ("left arm", slice(4, 11)),
        ("right arm", slice(11, 18)),
        ("grippers", slice(18, 20)),
    )
    for name, sl in groups:
        print(name, "mean human", ha[:, sl].mean(0).round(2), "expert", ea[:, sl].mean(0).round(2))
    print("right-arm state std human", hs[:, 9:16].std(0).round(3))
    return files


def replay(files, count=4, step=75):
    from advaitic_eval.environment import make_environment

    env = make_environment()
    for f in files[::step][:count]:
        z = np.load(f)
        actions, states = z["action"], z["state"]
        ts = env.reset(seed=int(z["seed"]))
        worst = 0.0
        for t in range(len(actions)):
            worst = max(worst, float(np.abs(ts.low_dim_obs - states[t]).max()))
            ts = env.step(actions[t])
        print(
            f"replay {f.split('/')[-1]} steps {len(actions)} max|state diff| {worst:.5f} "
            f"success {bool(env.episode_succeeded())} fell {bool(env.episode_fell())}"
        )
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="datasets/move_plate.npz")
    ap.add_argument("--expert-dir", default="datasets/expert_train")
    ap.add_argument("--replays", type=int, default=4)
    args = ap.parse_args()
    files = describe(args.demos, args.expert_dir)
    seeds = [int(np.load(f)["seed"]) for f in files]
    print("expert seed range", min(seeds), max(seeds))
    replay(files, args.replays)


if __name__ == "__main__":
    main()
