"""Convert human demos + expert episodes to a LeRobotDataset for VLA fine-tuning (pi0.5).

Run in the lerobot env (not the sim env):
    ~/envs/pi05/bin/python -m submission.make_lerobot_dataset --out datasets/lerobot_plate \
        --demos datasets/move_plate.npz --expert-dir datasets/expert_train --max-expert 100

Cameras are renamed to the pi0.5 key names (head=base_0_rgb, right/left wrist). The state is a
24-d subset of the 50-d proprioception (pi0.5 pads to 32). The 20-d action is already in [-1, 1].
"""

import argparse
from pathlib import Path

import numpy as np

TASK = "pick up the plate from the left rack and place it in the right rack"
STATE_IDX = np.r_[0:18, 20:22, 44:46, 48:50]  # arms+fingers, pelvis z/yaw, grippers, ctrl z/yaw
CAMS = ["base_0_rgb", "right_wrist_0_rgb", "left_wrist_0_rgb"]


def episodes(demos, expert_dir, max_expert):
    if demos:
        d = np.load(demos)
        for e in np.unique(d["episode"]):
            m = d["episode"] == e
            yield f"human_{e}", d["rgb"][m], d["state"][m], d["action"][m]
    files = sorted(Path(expert_dir).glob("*.npz")) if expert_dir else []
    for f in files[:max_expert]:
        z = np.load(f)
        yield f"expert_{f.stem}", z["rgb"], z["state"], z["action"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--demos", default="")
    ap.add_argument("--expert-dir", default="")
    ap.add_argument("--max-expert", type=int, default=100)
    ap.add_argument("--max-episodes", type=int, default=0, help="debug: stop after N episodes")
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        f"observation.images.{c}": {
            "dtype": "video",
            "shape": (84, 84, 3),
            "names": ["height", "width", "channels"],
        }
        for c in CAMS
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(STATE_IDX),),
        "names": [f"s{i}" for i in STATE_IDX],
    }
    features["action"] = {"dtype": "float32", "shape": (20,), "names": [f"a{i}" for i in range(20)]}
    ds = LeRobotDataset.create(
        repo_id="local/g1-plate",
        fps=50,
        features=features,
        root=args.out,
        robot_type="unitree_g1",
        use_videos=True,
    )
    n = 0
    for name, rgb, state, action in episodes(args.demos, args.expert_dir, args.max_expert):
        for t in range(len(action)):
            frame = {
                f"observation.images.{c}": np.ascontiguousarray(rgb[t, i].transpose(1, 2, 0))
                for i, c in enumerate(CAMS)
            }
            frame["observation.state"] = state[t, STATE_IDX].astype(np.float32)
            frame["action"] = action[t].astype(np.float32)
            frame["task"] = TASK
            ds.add_frame(frame)
        ds.save_episode()
        n += 1
        print(name, len(action), flush=True)
        if args.max_episodes and n >= args.max_episodes:
            break
    ds.finalize()
    print("episodes", n)


if __name__ == "__main__":
    main()
