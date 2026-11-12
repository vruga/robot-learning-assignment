"""DAgger rollouts: the student drives, the privileged expert labels every visited state.

    uv run python -m submission.dagger --ckpt checkpoints/bc_expert/last.pt --beta 0.3 \
        --start-seed 6000000 --count 100 --workers 12 --out datasets/dagger_1

Each `execute`-step window is executed by the expert with probability `beta`, otherwise by the
student. The label recorded at every step is always the expert's action for that state. Seeds must
not be in the evaluation block. Training-time only; the evaluated policy never sees privileged state.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

EVAL_BLOCK = range(7_310_000, 7_320_000)


class Mixer:
    def __init__(self, ckpt, beta, execute, seed):
        import torch

        from .model import ChunkPolicy

        torch.set_num_threads(1)
        self.torch = torch
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.dev = "cuda"
        self.model = ChunkPolicy(ck["horizon"]).to(self.dev).eval()
        self.model.load_state_dict(ck["model"])
        self.mean, self.std = ck["state_mean"].to(self.dev), ck["state_std"].to(self.dev)
        self.beta, self.execute = beta, min(execute, ck["horizon"])
        self.rng = np.random.RandomState(seed)
        self.i = 0
        self.use_expert = False
        self.chunk = None
        self.expert_steps = 0

    def __call__(self, ts, a_expert):
        t = self.torch
        if self.i % self.execute == 0:
            self.use_expert = self.rng.rand() < self.beta
            with t.no_grad():
                rgb = t.from_numpy(np.asarray(ts.rgb_obs)).to(self.dev)[None].float() / 255
                st = t.from_numpy(np.asarray(ts.low_dim_obs)).to(self.dev)[None]
                self.chunk = self.model(rgb, (st - self.mean) / self.std)[0].cpu().numpy()
        k = self.i % self.execute
        self.i += 1
        if self.use_expert:
            self.expert_steps += 1
            return a_expert
        return self.chunk[k].astype(np.float32)


def worker(args):
    seeds, ckpt, out, beta, execute = args
    from advaitic_eval.environment import make_environment, observed_source_slot

    from .expert import Expert, Params

    env = make_environment()
    rows = []
    for seed in seeds:
        try:
            mixer = Mixer(ckpt, beta, execute, seed)
            r = Expert(env, Params()).run(seed, max_steps=1500, executor=mixer)
            slot = observed_source_slot(env)
        except Exception as exc:
            rows.append({"seed": seed, "slot": -1, "success": False, "error": repr(exc)})
            continue
        t = r["trace"]
        np.savez_compressed(
            Path(out) / f"ep_{seed}.npz",
            rgb=t["obs_rgb"],
            state=t["obs_state"],
            action=t["act"].astype(np.float32),
            seed=seed,
            slot=slot,
            success=r["success"],
        )
        rows.append(
            {
                "seed": seed,
                "slot": slot,
                "success": bool(r["success"]),
                "steps": r["steps"],
                "expert_frac": round(mixer.expert_steps / max(1, mixer.i), 3),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--execute", type=int, default=10)
    ap.add_argument("--start-seed", type=int, default=6_000_000)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = list(range(args.start_seed, args.start_seed + args.count))
    if any(s in EVAL_BLOCK for s in seeds):
        raise SystemExit("refusing to roll out on evaluation seeds")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    chunks = [
        (seeds[i :: args.workers], args.ckpt, args.out, args.beta, args.execute)
        for i in range(args.workers)
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [r for part in pool.map(worker, chunks) for r in part]
    ok = sum(r["success"] for r in rows)
    summary = {"total": len(rows), "student_mix_success": ok, "beta": args.beta, "rows": rows}
    Path(args.out, "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))


if __name__ == "__main__":
    main()
