#!/bin/zsh
# Attractor-as-representation study: 4 arms = {retino,random} x {rgb,grayscale}, full image,
# 500 imgs/label, settle 500 + observe 500, sharded 9-way. Each arm self-contained (writes
# figures+animations+JSON as it finishes). Resumable (skips complete arm JSON), detached, locked.
# Order: retino-rgb first (most promising), then retino-gray, random-rgb, random-gray.
set -u
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=. OMP_NUM_THREADS=1
AC=foveation_results/minibrain/attractor_class
LOG=$AC/run.log
LOCK=$AC/run.lock
mkdir -p $AC
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK 2>/dev/null)" 2>/dev/null; then
  echo "[$(date +%H:%M)] already running; exit" >> $LOG; exit 0; fi
echo $$ > $LOCK; trap 'rm -f $LOCK' EXIT
if [ -f "$AC/ATTRACTOR_CLASS_COMPLETE" ]; then echo "[$(date +%H:%M)] already complete; exit" >> $LOG; exit 0; fi
echo "==== attractor-class run start $(date) ====" >> $LOG

arm () {  # arch dataset
  local arch=$1 ds=$2 tag="attr_${1}_${2/cifar10/c10}"
  if [ -f "$AC/$tag.json" ]; then echo "[$(date +%H:%M)] [skip] $tag" >> $LOG; return; fi
  echo "[$(date +%H:%M)] START $tag" >> $LOG
  .venv/bin/python -m snn_classification_realtime.foveation.minibrain.exp_attractor_class \
    --arch $arch --dataset $ds --per-class 500 --settle 500 --observe 500 --shards 9 \
    --out $AC >> $LOG 2>&1
  echo "[$(date +%H:%M)] DONE $tag -> $(cat $AC/$tag.json 2>/dev/null | tr ',' '\n' | grep -E 'full|centroid|dominant_type|fisher' | head -4 | tr '\n' ' ')" >> $LOG
}

arm reservoir_retino cifar10
arm reservoir_retino cifar10_grayscale
arm reservoir_random cifar10
arm reservoir_random cifar10_grayscale

echo "==== attractor-class run end $(date) ====" >> $LOG
touch $AC/ATTRACTOR_CLASS_COMPLETE
