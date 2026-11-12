"""Run the privileged expert on many seeds in parallel; keep successful episodes.

    uv run python -m submission.generate_data --start-seed 5000000 --count 600 --workers 16 \
        --out datasets/expert_v1

Seeds must stay out of the evaluation blocks (public 7_310_000+, and any reviewer block).
Writes one npz per successful episode plus summary.json with per-source-slot success rates.
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

EVAL_BLOCK = range(7_310_000, 7_320_000)


class DartNoise:
    """Brief random perturbation bursts on base + left-arm commands; the recorded label stays clean.

    Constant noise breaks the expert's tolerance-based phase switches, so noise is applied in 1-3 bursts
    of 10-30 steps at random times. The expert then has to recover from the perturbed state, and its
    labels (clean, own integrator) show how.
    """

    def __init__(self, seed, lo, hi, t0=60, t1=700):
        rng = np.random.RandomState(seed + 17)
        self.rng = rng
        self.bursts = []
        for _ in range(rng.randint(1, 4)):
            start = int(rng.uniform(t0, t1))
            self.bursts.append((start, start + int(rng.randint(10, 31)), rng.uniform(lo, hi)))
        self.i = 0
        keep = np.zeros(20, dtype=np.float32)
        keep[0:2] = 1  # base vx, vy
        keep[4:11] = 1  # left-arm joint targets
        self.keep = keep
        self.n = np.zeros(20, dtype=np.float32)

    def __call__(self, ts, a):
        t = self.i
        self.i += 1
        active = [sg for (s0, s1, sg) in self.bursts if s0 <= t < s1]
        if not active:
            return a  # same object: the expert treats it as its own command
        if t % 5 == 0 or not self.n.any():
            self.n = self.rng.randn(20).astype(np.float32) * max(active) * self.keep
        return np.clip(a + self.n, -1, 1).astype(np.float32)


def hard_seeds(start, count, min_off=0.035, slots=(2, 3, 4, 5)):
    slots = tuple(int(x) for x in os.environ.get("HARD_SLOTS", "").split(",") if x) or slots
    """Seeds whose slot is 3-6 and whose start- or target-rack offset is large in x (miss configs)."""
    from advaitic_eval.protocol import source_slot_for_seed

    out, seed = [], start
    while len(out) < count:
        rng = np.random.RandomState(seed)
        b = np.array([0.05, 0.05, 0.0])
        a1, a2 = rng.uniform(-b, b), rng.uniform(-b, b)
        if source_slot_for_seed(seed) in slots and max(abs(a1[0]), abs(a2[0])) >= min_off:
            out.append(seed)
        seed += 1
    return out


def worker(args):
    seeds, out, params = args
    dart = params.pop("dart", None)
    from advaitic_eval.environment import make_environment, observed_source_slot

    from .expert import Expert, Params

    env = make_environment()
    rows = []
    for seed in seeds:
        try:
            ex = Expert(env, Params(**params))
            r = ex.run(seed, executor=DartNoise(seed, *dart) if dart else None)
            slot = observed_source_slot(env)
        except Exception as exc:  # keep the batch alive; report the failure
            rows.append({"seed": seed, "slot": -1, "success": False, "error": repr(exc)})
            continue
        if r["success"] and not r["fell"]:
            t = r["trace"]
            np.savez_compressed(
                Path(out) / f"ep_{seed}.npz",
                rgb=t["obs_rgb"],
                state=t["obs_state"],
                action=t["act"].astype(np.float32),
                seed=seed,
                slot=slot,
            )
        rows.append(
            {
                "seed": seed,
                "slot": slot,
                "success": bool(r["success"]),
                "fell": r["fell"],
                "steps": r["steps"],
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-seed", type=int, default=5_000_000)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", default="", help="comma-separated seeds (overrides start/count)")
    ap.add_argument("--out", default="datasets/expert_v1")
    ap.add_argument("--dart", default="", help="sigma_lo,sigma_hi: noisy execution, clean labels")
    ap.add_argument(
        "--hard", type=int, default=0, help="pick this many hard-config seeds from start"
    )
    ap.add_argument("--params", default="{}", help="JSON overrides for expert Params")
    args = ap.parse_args()
    seeds = (
        [int(x) for x in args.seeds.split(",")]
        if args.seeds
        else list(range(args.start_seed, args.start_seed + args.count))
    )
    if args.hard:
        seeds = hard_seeds(args.start_seed, args.hard)
    if any(s in EVAL_BLOCK for s in seeds):
        raise SystemExit("refusing to generate on evaluation seeds")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    params = json.loads(args.params)
    if args.dart:
        params["dart"] = [float(x) for x in args.dart.split(",")]
        params["adopt_executed"] = False
    chunks = [(seeds[i :: args.workers], args.out, params) for i in range(args.workers)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [r for part in pool.map(worker, chunks) for r in part]
    summary = {"total": len(rows), "success": sum(r["success"] for r in rows), "per_slot": {}}
    for slot in range(6):
        sel = [r for r in rows if r["slot"] == slot]
        summary["per_slot"][slot + 1] = {"n": len(sel), "ok": sum(r["success"] for r in sel)}
    summary["rows"] = rows
    Path(args.out, "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
