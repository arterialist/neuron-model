#!/bin/zsh
# PAULA attribution controls -- waits until Part B (exp_settle) is done so the substrate
# passes run UNCONTENDED, then runs exp_controls (pixel/retina/substrate/randproj x
# linear/mlp) for all 3 substrates. Resumable (skips complete arm JSONs). Detached, locked.
set -u
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=. OMP_NUM_THREADS=1
CT=foveation_results/minibrain/controls
ST=foveation_results/minibrain/settle
LOG=$CT/controls.log
LOCK=$CT/controls.lock
mkdir -p $CT
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK 2>/dev/null)" 2>/dev/null; then
  echo "[$(date +%H:%M)] controls already running; exit" >> $LOG; exit 0; fi
echo $$ > $LOCK; trap 'rm -f $LOCK' EXIT
if [ -f "$CT/CONTROLS_COMPLETE" ]; then echo "[$(date +%H:%M)] controls already complete; exit" >> $LOG; exit 0; fi

echo "[$(date +%H:%M)] controls waiting for Part B ..." >> $LOG
while [ ! -f "$ST/AGG_settle_summary.json" ]; do
  if ! pgrep -f exp_settle >/dev/null 2>&1; then
    sleep 30
    { [ -f "$ST/AGG_settle_summary.json" ] || ls "$ST"/settle_*.json >/dev/null 2>&1; } && break
  fi
  sleep 20
done
echo "[$(date +%H:%M)] Part B done -> running controls (uncontended, 3 arms parallel)" >> $LOG
.venv/bin/python -m snn_classification_realtime.foveation.minibrain.exp_controls \
  --dataset cifar10 --archs conv,reservoir_random,reservoir_retino \
  --samples 600 --test 300 --dwell 200 --warmup 500 --shards 9 --device mps \
  --out $CT >> $LOG 2>&1
echo "[$(date +%H:%M)] controls complete" >> $LOG
touch $CT/CONTROLS_COMPLETE
