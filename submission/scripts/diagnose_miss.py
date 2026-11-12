"""Replay a recorded evaluator episode (its saved actions) and log privileged probes every N steps.

Diagnostics only: reads simulator truth, never feeds it to the policy. The evaluator is deterministic, so
replaying `episode_XXX.actions.npz` under the same seed reproduces the original episode exactly.

    MUJOCO_GL=egl .venv-train/bin/python submission/scripts/diagnose_miss.py results/final300_B7360000 7360022
"""

import json
import sys
from pathlib import Path

import numpy as np


def main(result_dir, seed, every=40):
    from advaitic_eval.environment import make_environment, observed_source_slot
    from submission.expert import Expert

    rep = json.loads(Path(result_dir, "report.json").read_text())
    ep = next(e for e in rep["episodes"] if e["seed"] == seed)
    acts = np.load(Path(result_dir, f"episode_{ep['episode']:03}.actions.npz"))["actions"]
    env = make_environment()
    ex = Expert(env)
    ts = env.reset(seed=seed)
    slots = ex.slots()
    inner = env.inner_env
    p0 = ex.plate().copy()
    rows = []
    for t, a in enumerate(acts):
        if t % every == 0:
            plate, hand = ex.plate(), ex.hand()
            st = np.asarray(ts.low_dim_obs)
            d = np.linalg.norm(slots - plate, axis=1)
            rows.append(
                dict(
                    t=t,
                    plate=plate.round(3).tolist(),
                    hand_to_plate=float(np.linalg.norm(hand - plate)),
                    grip=st[44:46].round(2).tolist(),
                    pelvis=st[18:20].round(3).tolist(),
                    dz=float(plate[2] - p0[2]),
                    dest_d=float(d.min()),
                    dest_slot=int(d.argmin()),
                    up_z=float(ex.plate_up()[2]),
                )
            )
        ts = env.step(a)
        if ts.last():
            break
    print(
        f"seed {seed} slot {observed_source_slot(env) + 1} steps {len(acts)} success={ep['success']}"
    )
    print(f"start rack off: {np.round(inner.rack_start.get_position()[:2], 3)}") if hasattr(
        inner.rack_start, "get_position"
    ) else None
    print("   t   plate_xyz              h-p   grip   dz    dest_d  up_z")
    for r in rows:
        print(
            f"{r['t']:5d} {r['plate']} {r['hand_to_plate']:.3f} {r['grip']} "
            f"{r['dz']:+.3f} {r['dest_d']:.3f} {r['up_z']:.2f}"
        )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 40)
