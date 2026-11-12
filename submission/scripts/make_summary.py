"""Summarise evaluator reports into a markdown table (and optional per-model cards).

python submission/scripts/make_summary.py --results results --out results/SUMMARY.md
"""

import argparse
import json
from pathlib import Path


def load(path):
    r = json.loads(Path(path).read_text())
    slots = ",".join(str(v["successes"]) for _, v in sorted(r["per_source_slot"].items()))
    return {
        "name": Path(path).parent.name,
        "method": r.get("method", ""),
        "trials": r["completed_episodes"],
        "ok": r["successes"],
        "rate": r["success_rate"],
        "slots": slots,
        "seed_start": r["seed_start"],
        "valid": bool(
            r["integrity_ok"] and r["submission_unchanged"] and not r["infrastructure_error"]
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/SUMMARY.md")
    args = ap.parse_args()
    rows = []
    for rep in sorted(Path(args.results).glob("*/report.json")):
        if rep.parent.name.startswith("INVALID"):
            continue
        rows.append(load(rep))
    lines = [
        "| run | method | seeds | trials | success | rate | per slot (1..6) | valid |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['method']} | {r['seed_start']}+ | {r['trials']} | {r['ok']} | "
            f"{r['rate']:.1%} | {r['slots']} | {'yes' if r['valid'] else 'NO'} |"
        )
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
