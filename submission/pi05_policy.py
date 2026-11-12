"""Fine-tuned pi0.5 policy (LeRobot checkpoint trained on our dataset by `lerobot-train`).

    PI05_CKPT=<.../checkpoints/005000/pretrained_model> PI05_EXECUTE=50 \
    --policy submission.pi05_policy:make_policy --policy-python ~/envs/pi05/bin/python

Uses the processors saved with the checkpoint (they hold the dataset normalisation statistics), the
same 24-d state subset and camera naming as `make_lerobot_dataset.py`, and returns a chunk of actions.
"""

import os

import numpy as np

STATE_IDX = np.r_[0:18, 20:22, 44:46, 48:50]
PROMPT = "pick up the plate from the left rack and place it in the right rack"


class Pi05Finetuned:
    def __init__(self, path, execute):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_CACHE", "/usr/local/models/hf-hub")
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        self.torch, self.dev, self.execute = torch, "cuda", execute
        # trained with fp32 weights (~14.5 GB); bf16 for inference (~7 GB) so it fits next to other jobs
        self.policy = PI05Policy.from_pretrained(path).to(dtype=torch.bfloat16).to(self.dev).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": self.dev}},
        )

    def reset(self):
        self.policy.reset()

    def __call__(self, obs):
        t = self.torch
        rgb = np.asarray(obs["rgb"])  # (3, 3, 84, 84): head, right wrist, left wrist

        def img(i):
            return t.from_numpy(rgb[i]).float().div(255)[None].to(self.dev)

        batch = {
            "observation.images.base_0_rgb": img(0),
            "observation.images.right_wrist_0_rgb": img(1),
            "observation.images.left_wrist_0_rgb": img(2),
            "observation.state": t.from_numpy(np.asarray(obs["state"], np.float32)[STATE_IDX])[
                None
            ].to(self.dev),
            "task": PROMPT,
        }
        with t.no_grad(), t.autocast("cuda", dtype=t.bfloat16):
            chunk = self.policy.predict_action_chunk(self.pre(batch))  # (1, T, D)
            acts = [self.post(chunk[:, i])[0] for i in range(min(self.execute, chunk.shape[1]))]
        a = np.stack([x.detach().float().cpu().numpy()[:20] for x in acts])
        return np.clip(np.nan_to_num(a), -1, 1).astype(np.float32)


def make_policy():
    path = os.environ.get("PI05_CKPT")
    if not path:
        raise RuntimeError("Set PI05_CKPT to a lerobot pretrained_model directory")
    return Pi05Finetuned(path, int(os.environ.get("PI05_EXECUTE", "50")))
