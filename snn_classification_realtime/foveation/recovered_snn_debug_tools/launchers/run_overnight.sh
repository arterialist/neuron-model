#!/bin/zsh
# Overnight A+B launcher. Resumable: each step is guarded so a re-launch skips finished
# work. Logs to $LOG. Designed to be re-invoked by a periodic wakeup if the tree is killed.
set -u
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=. OMP_NUM_THREADS=1
SC=foveation_results/minibrain/scale
ST=foveation_results/minibrain/settle
LOG=$SC/overnight.log
mkdir -p $SC $ST
# single-instance lock: a stale relaunch exits if one is already running
LOCK=$SC/overnight.lock
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK 2>/dev/null)" 2>/dev/null; then
  echo "[$(date +%H:%M)] already running (pid $(cat $LOCK)); exit" >> $LOG; exit 0
fi
echo $$ > $LOCK
trap 'rm -f $LOCK' EXIT
if [ -f "$SC/OVERNIGHT_COMPLETE" ]; then echo "[$(date +%H:%M)] already complete; exit" >> $LOG; exit 0; fi
echo "==== overnight run start $(date) ====" >> $LOG

run_ceil () {  # tag samples test
  local tag=$1 samp=$2 test=$3
  if [ -f "$SC/$tag.json" ]; then echo "[skip] $tag done" >> $LOG; return; fi
  echo "[$(date +%H:%M)] PART A start $tag (samples=$samp)" >> $LOG
  .venv/bin/python -m snn_classification_realtime.foveation.minibrain.exp_scale \
    --mode convfeat --head mlp --dataset cifar10 --grid 16 \
    --samples $samp --test $test --batch 256 --epochs 150 --mlp-hidden 512 \
    --device mps --tag $tag --out $SC >> $LOG 2>&1
  echo "[$(date +%H:%M)] PART A done $tag -> $(grep -o 'TEST ACC [0-9.]*' $SC/$tag.json 2>/dev/null || cat $SC/$tag.json 2>/dev/null | tr ',' '\n' | grep acc)" >> $LOG
}

# ---------- PART A: encoder ceiling pushed for >=0.50 ----------
run_ceil ceilA_color_15k 15000 5000
run_ceil ceilA_color_40k 40000 8000

# ---------- PART B: settling sweep (substrate preservation + mean vs dynamics) ----------
echo "[$(date +%H:%M)] PART B start (exp_settle 3 arms)" >> $LOG
.venv/bin/python -m snn_classification_realtime.foveation.minibrain.exp_settle \
  --dataset cifar10 --images 1000 --max-dwell 800 --warmup 500 --par 9 \
  --archs conv,reservoir_random,reservoir_retino --out $ST >> $LOG 2>&1
echo "[$(date +%H:%M)] PART B done" >> $LOG

echo "==== overnight run end $(date) ====" >> $LOG
touch $SC/OVERNIGHT_COMPLETE
