"""Interface example only: holds the observed pose, does not solve the task."""

import numpy as np


class HoldPose:
    def reset(self):
        self.action = None

    def __call__(self, obs):
        if self.action is None:
            # Fixed robot limits are public metadata, not live simulator state.
            import json
            from importlib.resources import files

            layout = json.loads(files("advaitic_eval").joinpath("action_layout.json").read_text())
            low = np.asarray(layout["action_low"])
            high = np.asarray(layout["action_high"])
            state = obs["state"]
            raw = np.zeros(20, dtype=np.float32)
            raw[2] = state[48]  # observed pelvis height
            raw[4:11] = state[:7]
            raw[11:18] = state[9:16]
            raw[18:20] = state[44:46]
            self.action = np.clip(2 * (raw - low) / (high - low) - 1, -1, 1).astype(np.float32)
        return self.action


def make_policy():
    return HoldPose()
