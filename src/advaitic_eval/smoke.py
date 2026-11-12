"""Build/reset the real environment and check all six reset slots."""

import argparse
import json

import numpy as np

from .environment import make_environment, observed_source_slot
from .integrity import check_integrity
from .protocol import make_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resets", type=int, default=6)
    args = parser.parse_args()
    integrity = check_integrity()
    if not integrity["ok"]:
        raise RuntimeError(integrity)
    env = make_environment()
    results = []
    try:
        chosen = {}
        for ep in make_plan():
            chosen.setdefault(ep.source_slot, ep)
        selected = list(chosen.values()) if args.resets == 6 else make_plan()[: args.resets]
        for ep in selected:
            ts = env.reset(seed=ep.seed)
            slot = observed_source_slot(env)
            assert slot == ep.source_slot
            assert ts.rgb_obs.shape == (3, 3, 84, 84)
            assert ts.low_dim_obs.shape == (50,)
            action = env.normalize_action(env.raw_hold_action())
            for _ in range(5):
                ts = env.step(action)
            assert np.isfinite(ts.low_dim_obs).all()
            results.append(
                {
                    "seed": ep.seed,
                    "source_slot": slot + 1,
                    "fell": bool(env.episode_fell()),
                    "success": bool(env.episode_succeeded()),
                }
            )
        print(
            json.dumps(
                {
                    "integrity": integrity,
                    "resets": results,
                    "action_layout": env.wholebody_action_layout(),
                    "substrate": env.substrate_fingerprint(),
                },
                indent=2,
                default=lambda value: value.tolist(),
            )
        )
    finally:
        env.close()
