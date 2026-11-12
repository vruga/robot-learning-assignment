"""Run fixed plate-transfer evaluation in a process separate from candidate code."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .environment import make_environment, observation, observed_source_slot, validate_actions
from .integrity import check_integrity, submission_digest
from .policy_process import PolicyProcess
from .protocol import MIN_EPISODES, PUBLIC_SEED_START, make_plan, plan_digest, summarize


def run(args) -> dict:
    scored = not args.smoke
    if scored and args.max_steps is not None:
        raise ValueError("--max-steps is only available with --smoke")
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    plan = make_plan(args.episodes if scored else MIN_EPISODES, args.seed_start)
    if not scored:
        if args.episodes > MIN_EPISODES:
            raise ValueError("Smoke runs support 1–30 episodes")
        plan = plan[: args.episodes]
    integrity = check_integrity()
    if not integrity["ok"]:
        raise RuntimeError(f"Frozen environment integrity check failed: {integrity['failures']}")
    # New directory only: never blend evidence from different runs.
    args.out.mkdir(parents=True, exist_ok=False)
    root = args.policy_root.resolve()
    before = submission_digest(root)
    records = []
    env = make_environment()
    process = None
    fingerprint = None
    infrastructure_error = None
    try:
        process = PolicyProcess(
            args.policy,
            root,
            args.out / "policy.stderr.log",
            timeout=args.policy_timeout,
            python=args.policy_python,
        )
        for index, episode in enumerate(plan):
            ts = env.reset(seed=episode.seed)
            actual_slot = observed_source_slot(env)
            if actual_slot != episode.source_slot:
                raise RuntimeError("Source-slot prediction does not match actual reset")
            if fingerprint is None:
                fingerprint = env.substrate_fingerprint()
            actions_log = []
            length = 0
            policy_seconds = 0.0
            policy_error = None
            forced_stop = False
            writer = None
            if index < args.video_episodes:
                import imageio.v2 as imageio

                writer = imageio.get_writer(
                    str(args.out / f"episode_{index:03}.mp4"), fps=10, macro_block_size=1
                )
            try:
                process.reset()
                while not ts.last():
                    started = time.monotonic()
                    actions = validate_actions(process.act(observation(ts)))
                    policy_seconds += time.monotonic() - started
                    for action in actions:
                        actions_log.append(action.copy())
                        ts = env.step(action)
                        length += 1
                        if writer is not None and length % 5 == 0:
                            cameras = np.asarray(ts.rgb_obs)
                            frame = np.concatenate([c.transpose(1, 2, 0) for c in cameras], axis=1)
                            writer.append_data(frame)
                        if args.max_steps is not None and length >= args.max_steps:
                            forced_stop = True
                        if ts.last() or forced_stop:
                            break
                    if forced_stop:
                        break
            except (ValueError, RuntimeError, TimeoutError, BrokenPipeError, KeyError) as exc:
                policy_error = f"{type(exc).__name__}: {exc}"
            finally:
                if writer is not None:
                    writer.close()
            # A policy error is a scored failure, not an omitted trial.
            success = bool(
                env.episode_succeeded()
                and not env.episode_fell()
                and not forced_stop
                and policy_error is None
            )
            termination = (
                "policy_error"
                if policy_error
                else "smoke_cutoff"
                if forced_stop
                else env.episode_termination()
            )
            np.savez_compressed(
                args.out / f"episode_{index:03}.actions.npz",
                actions=np.asarray(actions_log, dtype=np.float32).reshape(-1, 20),
            )
            records.append(
                {
                    "episode": index,
                    **asdict(episode),
                    "success": success,
                    "completed": True,
                    "steps": length,
                    "simulated_seconds": length * 0.02,
                    "policy_seconds": policy_seconds,
                    "fell": bool(env.episode_fell()),
                    "termination": termination,
                    "policy_error": policy_error,
                }
            )
            print(
                f"{index + 1}/{len(plan)} slot={actual_slot + 1} success={success} "
                f"end={termination}",
                flush=True,
            )
            _write_records(args.out, records)
    except Exception as exc:
        infrastructure_error = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            process.close()
        env.close()
    after_integrity = check_integrity()
    unchanged = before == submission_digest(root)
    integrity_ok = after_integrity["ok"] and unchanged and infrastructure_error is None
    result = summarize(records, len(plan), scored=scored, integrity_ok=integrity_ok)
    result.update(
        {
            "task": "move_plate",
            "comparison": "Advaitic balanced-slot protocol; not BiGym leaderboard",
            "method": args.method,
            "policy": args.policy,
            "checkpoint": args.checkpoint,
            "seed_start": args.seed_start,
            "plan_sha256": plan_digest(plan),
            "plan": [asdict(ep) for ep in plan],
            "episodes": records,
            "upstream_integrity": after_integrity,
            "submission_sha256": before,
            "submission_unchanged": unchanged,
            "infrastructure_error": infrastructure_error,
            "substrate": fingerprint,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "smoke": args.smoke,
        }
    )
    (args.out / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "successes",
                    "completed_episodes",
                    "success_rate",
                    "meets_success_criterion",
                ]
            },
            indent=2,
        )
    )
    return result


def _write_records(out: Path, records: list[dict]):
    with (out / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", required=True, help="module:factory callable; see submission/policy.py"
    )
    parser.add_argument("--policy-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--policy-python",
        default=sys.executable,
        help="Python from a separate model environment, if needed",
    )
    parser.add_argument("--method", default="candidate-method")
    parser.add_argument("--checkpoint", default="none", help="Checkpoint identifier/hash, if used")
    parser.add_argument("--episodes", type=int, default=MIN_EPISODES)
    parser.add_argument("--seed-start", type=int, default=PUBLIC_SEED_START)
    parser.add_argument("--out", type=Path, required=True, help="New result directory")
    parser.add_argument("--video-episodes", type=int, default=0)
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument("--smoke", action="store_true", help="Never counts as a scored pass")
    parser.add_argument("--max-steps", type=int, help="Smoke-only early cutoff")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    if args.video_episodes < 0 or args.policy_timeout <= 0:
        parser.error("video-episodes must be nonnegative; policy-timeout must be positive")
    try:
        result = run(args)
    except (ValueError, RuntimeError, FileExistsError) as exc:
        parser.exit(2, f"{exc}\n")
    raise SystemExit(
        0
        if result["meets_success_criterion"]
        or (args.smoke and result["infrastructure_error"] is None)
        else 1
    )


if __name__ == "__main__":
    main()
