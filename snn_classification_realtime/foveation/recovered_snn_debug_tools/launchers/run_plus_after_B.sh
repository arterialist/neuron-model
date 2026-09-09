#!/bin/zsh
# Phase 2 (Part A+): waits until Part B (exp_settle) is done so feature extraction runs
# UNCONTENDED, then pushes the encoder-ceiling accuracy with full 50k + hflip aug + big MLP.
# Resumable: skips configs whose JSON already exists. Detached, single-instance locked.
set -u
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=. OMP_NUM_THREADS=9
SC=foveation_results/minibrain/scale
ST=foveation_results/minibrain/settle
LOG=$SC/plus_phase2.log
LOCK=$SC/plus_phase2.lock
mkdir -p $SC
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK 2>/dev/null)" 2>/dev/null; then
  echo "[$(date +%H:%M)] phase2 already running; exit" >> $LOG; exit 0; fi
echo $$ > $LOCK; trap 'rm -f $LOCK' EXIT
if [ -f "$SC/PUSH_COMPLETE" ]; then echo "[$(date +%H:%M)] push already complete; exit" >> $LOG; exit 0; fi

echo "[$(date +%H:%M)] phase2 waiting for Part B to finish ..." >> $LOG
# wait until Part B aggregate exists (all 3 settle arms done) or no exp_settle running
while [ ! -f "$ST/AGG_settle_summary.json" ]; do
  if ! pgrep -f exp_settle >/dev/null 2>&1 && [ ! -f "$ST/AGG_settle_summary.json" ]; then
    # exp_settle not running and no aggregate: give the overnight script a moment; if still
    # absent after a short grace it likely finished arms without aggregate -> proceed anyway
    sleep 30
    [ -f "$ST/AGG_settle_summary.json" ] && break
    ls "$ST"/settle_*.json >/dev/null 2>&1 && break   # arms exist, proceed
  fi
  sleep 20
done
echo "[$(date +%H:%M)] Part B done -> starting push (uncontended)" >> $LOG

run_plus () {  # tag extra-args...
  local tag=$1; shift
  if [ -f "$SC/$tag.json" ]; then echo "[$(date +%H:%M)] [skip] $tag" >> $LOG; return; fi
  echo "[$(date +%H:%M)] push start $tag $*" >> $LOG
  .venv/bin/python -m snn_classification_realtime.foveation.minibrain.exp_scale_plus \
    --grid 16 --test 10000 --hidden 1024 --dropout 0.3 --epochs 250 --batch 512 \
    --device mps --tag $tag --out $SC "$@" >> $LOG 2>&1
  echo "[$(date +%H:%M)] push done $tag -> $(cat $SC/$tag.json 2>/dev/null | tr ',' '\n' | grep '\"acc\"')" >> $LOG
}

run_plus plusA_full_noaug --samples 0
run_plus plusA_full_aug   --samples 0 --aug

echo "[$(date +%H:%M)] phase2 complete" >> $LOG
touch $SC/PUSH_COMPLETE
