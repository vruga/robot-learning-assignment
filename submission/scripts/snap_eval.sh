#!/bin/bash
# usage: snap_eval.sh NAME CKPT [EXECUTE] [EPISODES] [SEED_START]
# Evaluates from an isolated snapshot dir so training/other writes cannot change the submission digest.
set -e
NAME=$1; CKPT=$2; EXEC=${3:-10}; EPS=${4:-30}; SEED=${5:-7310000}
export PATH=$HOME/.local/bin:$PATH MUJOCO_GL=egl
R=$HOME/robot-learning-assignment; SNAP=$HOME/eval_snap/$NAME
rm -rf $SNAP; mkdir -p $SNAP
rsync -a --exclude __pycache__ $R/submission $SNAP/
cp $CKPT $SNAP/ckpt.pt
cd $R
BC_CHECKPOINT=$SNAP/ckpt.pt BC_EXECUTE=$EXEC uv run advaitic-eval --policy-root $SNAP --policy submission.bc_policy:make_policy --policy-python $R/.venv-train/bin/python --method $NAME --checkpoint $CKPT --episodes $EPS --seed-start $SEED --out results/$NAME > logs/eval_$NAME.log 2>&1
python3 - <<PY
import json
r=json.load(open("results/$NAME/report.json"))
print("$NAME", r["successes"],"/",r["completed_episodes"],"unchanged",r["submission_unchanged"],{k:v["successes"] for k,v in r["per_source_slot"].items()})
PY
