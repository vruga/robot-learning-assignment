"""Zero-shot pi0.5 (lerobot/pi05_base) adapter. A *reference* baseline, not a trained solution.

The base model has never seen this robot's 20-d action layout, so its raw outputs have no
defined meaning here. This only measures what an un-adapted VLA does on the task. Fine-tuning
on expert data is the real VLA comparison.

    PI05_PATH=/usr/local/models/hf-hub/models--lerobot--pi05_base/snapshots/<rev> \
    --policy submission.pi05_zeroshot:make_policy --policy-python ~/envs/pi05/bin/python
"""

import json
import os
from importlib.resources import files

import numpy as np

PROMPT = "pick up the plate from the left rack and place it in the right rack"


class Pi05ZeroShot:
    def __init__(self, path):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_CACHE", "/usr/local/models/hf-hub")
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        self.torch = torch
        self.dev = "cuda"
        self.policy = PI05Policy.from_pretrained(path).to(self.dev).eval()
        # Saved processor json comes from a newer lerobot than the pinned 0.4.4, and there are no
        # normalisation stats for this robot anyway: build the default pipeline from the config.
        self.pre, self.post = make_pre_post_processors(self.policy.config, dataset_stats=None)
        layout = json.loads(files("advaitic_eval").joinpath("action_layout.json").read_text())
        self.low = np.asarray(layout["action_low"], np.float32)
        self.high = np.asarray(layout["action_high"], np.float32)

    def reset(self):
        self.policy.reset()

    def _state32(self, state):
        raw = np.zeros(20, np.float32)
        raw[2] = state[48]
        raw[4:11] = state[:7]
        raw[11:18] = state[9:16]
        raw[18:20] = state[44:46]
        norm = np.clip(2 * (raw - self.low) / (self.high - self.low) - 1, -1, 1)
        out = np.zeros(32, np.float32)
        out[:20] = norm
        return out

    def __call__(self, obs):
        t = self.torch
        rgb = np.asarray(obs["rgb"])  # (3 cams, 3, 84, 84): head, right wrist, left wrist

        def img(i):
            x = t.from_numpy(rgb[i]).float().div(255)[None]
            return t.nn.functional.interpolate(x, size=(224, 224), mode="bilinear").to(self.dev)

        batch = {
            "observation.images.base_0_rgb": img(0),
            "observation.images.right_wrist_0_rgb": img(1),
            "observation.images.left_wrist_0_rgb": img(2),
            "observation.state": t.from_numpy(self._state32(obs["state"]))[None].to(self.dev),
            "task": PROMPT,
        }
        with t.no_grad():
            action = self.post(self.policy.select_action(self.pre(batch)))
        a = action.detach().float().cpu().numpy().reshape(-1)[:20]
        return np.clip(np.nan_to_num(a), -1, 1).astype(np.float32)


def make_policy():
    path = os.environ.get("PI05_PATH")
    if not path:
        raise RuntimeError("Set PI05_PATH to a lerobot pi05 checkpoint directory")
    return Pi05ZeroShot(path)
