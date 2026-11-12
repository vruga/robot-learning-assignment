#!/bin/bash
# Reproduce the 3,000-trial certification run: 10 blocks x 300 trials, seeds 7,900,000 + k*10,000.
# usage: bash submission/scripts/repro_cert.sh [PARALLEL_BLOCKS]   (default 5; each block needs ~1 GPU/CPU slot)
# Output: results/cert_B<seed>/report.json, then a pooled summary. Run from the repo root after the setup in README.md.
set -e
export PATH=$HOME/.local/bin:$PATH MUJOCO_GL=egl ORT_DISABLE_TELEMETRY=1
P=${1:-5}
mkdir -p logs results
run_block() {
  s=$1
  uv run advaitic-eval --policy submission.bc_policy:make_policy --policy-python .venv-train/bin/python \
    --method cr_v5_hold --checkpoint submission/weights/final.pt --episodes 300 --seed-start "$s" \
    --out "results/cert_B$s" > "logs/cert_B$s.log" 2>&1
}
export -f run_block
seq 7900000 10000 7990000 | xargs -P "$P" -I{} bash -c 'run_block {}'
python3 - <<'PY'
import glob, json, math
s = n = 0
slots = {}
miss = []
for f in sorted(glob.glob("results/cert_B*/report.json")):
    r = json.load(open(f))
    assert r["integrity_ok"] and r["submission_unchanged"] and not r["infrastructure_error"], f
    s += r["successes"]; n += r["completed_episodes"]
    for k, v in r["per_source_slot"].items():
        slots[k] = slots.get(k, 0) + v["successes"]
    miss += [(e["seed"], e["termination"]) for e in r["episodes"] if not e["success"]]
p = s / n; z = 1.96
c = (p + z * z / 2 / n) / (1 + z * z / n)
h = z * math.sqrt(p * (1 - p) / n + z * z / 4 / n / n) / (1 + z * z / n)
print(f"{s}/{n} = {p:.4f}  Wilson 95% [{c-h:.4f}, {c+h:.4f}]  per slot {slots}")
print("misses:", miss)
PY
