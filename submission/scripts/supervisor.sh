#!/bin/bash
# Runs inside tmux (session ra_pipeline) on the training machine. No credentials needed except an
# optional ~/.ra_hf_token for the final upload. Waits for training to finish, fires evaluations on
# fresh seed blocks, summarises, stages artifacts, and uploads to Hugging Face if a token is present.
export PATH=$HOME/.local/bin:$PATH MUJOCO_GL=egl
R=$HOME/robot-learning-assignment
CK=/var/tmp/sra_ckpt
STAGE=/var/tmp/sra_publish
BLOCKS="7340000 7350000"
cd "$R" || exit 1

wait_for() {  # wait_for FILE [max_seconds]
  local f=$1 max=${2:-28800} t=0
  until [ -f "$f" ]; do sleep 60; t=$((t + 60)); [ "$t" -ge "$max" ] && { echo "TIMEOUT waiting for $f"; return 1; }; done
}
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "stage 1: wait for the large-data continuous training"
if wait_for $CK/cr_all/train_log.json; then
  for b in $BLOCKS; do
    BC_DEVICE=cpu nohup logs/snap_eval.sh cr_all_B$b $CK/cr_all/last.pt 20 60 $b \
      > logs/cr_all_B$b.out 2>&1 &
    log "launched cr_all eval on block $b"
  done
fi

log "stage 2: wait for every evaluation report"
for n in best_B7340000 best_B7350000 df_B7340000 df_B7350000 cr_all_B7340000 cr_all_B7350000 pi05_ft5k_30; do
  wait_for results/$n/report.json 21600 && log "have $n" || log "missing $n"
done

log "stage 3: summary and staging"
.venv/bin/python submission/scripts/make_summary.py --results results --out results/SUMMARY.md
rm -rf $STAGE; mkdir -p $STAGE/vruga__g1-plate-bc-ablations/discrete_flow_dit \
                        $STAGE/vruga__g1-plate-bc-expert-xl
cp $CK/dit_df/last.pt $STAGE/vruga__g1-plate-bc-ablations/discrete_flow_dit/model.pt 2>/dev/null
cp $CK/dit_df/train_log.json $STAGE/vruga__g1-plate-bc-ablations/discrete_flow_dit/ 2>/dev/null
cp $CK/cr_all/last.pt $STAGE/vruga__g1-plate-bc-expert-xl/model.pt 2>/dev/null
cp $CK/cr_all/train_log.json $STAGE/vruga__g1-plate-bc-expert-xl/ 2>/dev/null
for d in $STAGE/*/; do
  mkdir -p $d/reports
  cp results/SUMMARY.md $d/ ; cp submission/model.py submission/bc_policy.py $d/
  for n in best_B7340000 best_B7350000 df_B7340000 df_B7350000 cr_all_B7340000 cr_all_B7350000 pi05_ft5k_30; do
    [ -f results/$n/report.json ] && cp results/$n/report.json $d/reports/$n.json
  done
done
log "staged under $STAGE"

log "stage 4: publish"
if [ -f "$HOME/.ra_hf_token" ] || [ -n "$HF_TOKEN" ]; then
  .venv-train/bin/python submission/scripts/publish_hf.py --public --stage $STAGE && log "published"
else
  log "no ~/.ra_hf_token: artifacts left staged in $STAGE (run publish_hf.py when a token is available)"
fi
log "pipeline done"
