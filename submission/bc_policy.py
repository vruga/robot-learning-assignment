"""Inference wrapper for the trained chunked BC policy.

--policy submission.bc_policy:make_policy   with BC_CHECKPOINT=checkpoints/bc/last.pt
"""

import os
from pathlib import Path

import numpy as np


def load_checkpoint(path, dev):
    """Build the right head from a checkpoint; returns (model, state_mean, state_std, ckpt)."""
    import torch

    from .model import ChunkPolicy, DiscreteChunkPolicy, DiscreteDiTPolicy

    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = ck.get("head", "continuous")
    if head in ("dit-dr", "dit-df"):
        model = DiscreteDiTPolicy(
            ck["horizon"], bins=ck["bins"], width=ck["width"], blocks=ck["blocks"], mode=head[-2:]
        )
        model.flow_steps = int(os.environ.get("BC_FLOW_STEPS", "20"))
    elif head == "discrete":
        model = DiscreteChunkPolicy(ck["horizon"], bins=ck["bins"])
    else:
        model = ChunkPolicy(ck["horizon"])
    model = model.to(dev).eval()
    model.load_state_dict(ck["model"])
    return model, ck["state_mean"].to(dev), ck["state_std"].to(dev), ck


class BCPolicy:
    def __init__(self, path, execute=20):
        import torch

        self.torch = torch
        # BC_DEVICE=cpu keeps the small ResNet policy off a busy GPU (inference is ~0.3 s/episode).
        self.dev = os.environ.get("BC_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.dev == "cpu":
            torch.set_num_threads(int(os.environ.get("BC_THREADS", "2")))
        self.model, self.mean, self.std, ck = load_checkpoint(path, self.dev)
        self.cfg = float(os.environ.get("BC_CFG", "0"))  # classifier-free guidance (discrete heads)
        self.execute = min(execute, ck["horizon"])  # replan every `execute` steps
        self.relative = bool(ck.get("relative", False))
        # Stall breaker (observations only): a deterministic single-frame policy can sit in a fixed
        # point (e.g. gripper closed, arm frozen). After `stall_calls` replans with an unchanged robot
        # state, perturb the proprioceptive input; sigma grows with each consecutive kick.
        self.stall_calls = int(os.environ.get("BC_STALL_CALLS", "3"))
        self.stall_eps = float(os.environ.get("BC_STALL_EPS", "0.02"))
        self.kick_sigma = float(os.environ.get("BC_KICK_SIGMA", "1.0"))
        # BC_HOLD_GRIP (default 1): while a kick is active and the gripper is closed, keep the previous gripper
        # command so a perturbation can move the arm but never open the hand mid-hold.
        self.hold_grip = os.environ.get("BC_HOLD_GRIP", "1") == "1"
        self.reset()
        if self.relative:  # relative arm-joint actions: add the current pose back at decode time
            from .model import action_limits

            lo, hi = action_limits()
            self.lo = torch.tensor(lo, dtype=torch.float32, device=self.dev)
            self.hi = torch.tensor(hi, dtype=torch.float32, device=self.dev)
            self.rel_range = ck["rel_range"].to(self.dev)

    def reset(self):
        self.prev = None
        self.still = 0
        self.kicks = 0
        self.total_kicks = 0
        self.last_grip = None
        self.rng = np.random.RandomState(0)

    def __call__(self, obs):
        t = self.torch
        with t.no_grad():
            rgb = t.from_numpy(np.asarray(obs["rgb"])).to(self.dev)[None].float() / 255
            raw = t.from_numpy(np.asarray(obs["state"])).to(self.dev)[None]
            sn = (raw - self.mean) / self.std
            kicked = False
            if self.stall_calls:
                cur = sn.cpu().numpy()
                joints = np.r_[0:16, 44:46]  # arm + gripper joints; base velocities always jitter
                moved = None if self.prev is None else np.abs(cur - self.prev)[..., joints].max()
                if moved is not None and moved < self.stall_eps:
                    self.still += 1
                else:
                    self.still, self.kicks = 0, 0
                self.prev = cur
                if self.still >= self.stall_calls:
                    self.kicks += 1
                    self.total_kicks += 1
                    kicked = True
                    # re-draw the strength each kick (0.3x-3x): a fixed or ramped sigma often stays inside
                    # the same fixed point; a random one escapes it within 1-2 kicks
                    sigma = self.kick_sigma * float(self.rng.uniform(0.3, 3.0))
                    noise = t.from_numpy(self.rng.randn(*cur.shape).astype(np.float32)).to(self.dev)
                    mode = os.environ.get("BC_KICK_MODE", "all")
                    if mode == "joints":
                        mask = t.zeros_like(noise)
                        mask[..., :16] = 1
                        noise = noise * mask
                    if mode == "rgb":
                        rgb = (rgb + 0.1 * sigma * t.randn_like(rgb)).clamp(0, 1)
                    else:
                        sn = sn + sigma * noise
            chunk = self.model.act(rgb, sn, self.cfg)
            if self.relative:
                from .model import arm_ref

                chunk = chunk * self.rel_range + arm_ref(raw, self.lo, self.hi)[:, None, :]
            chunk = chunk[0, : self.execute].clamp(-1, 1)
            if self.hold_grip:
                closed = bool((sn[0, 44:46] * self.std[44:46] + self.mean[44:46]).max() > 0.5)
                if kicked and closed and self.last_grip is not None:
                    chunk[:, 18:20] = self.last_grip
                self.last_grip = chunk[-1, 18:20].clone()
        return chunk.cpu().numpy().astype(np.float32)


def make_policy():
    # Default: the final checkpoint shipped next to this file (BC_CHECKPOINT overrides for experiments).
    path = os.environ.get("BC_CHECKPOINT") or str(Path(__file__).parent / "weights" / "final.pt")
    return BCPolicy(path, int(os.environ.get("BC_EXECUTE", "20")))
