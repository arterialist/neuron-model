#!/bin/bash
# Cluttered-canvas active vision: 4 modes x 3 seeds = 12 parallel single-threaded runs.
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY="$ROOT/.venv/bin/python"
OUT=foveation_results/minibrain/gaze_clutter
DS=${1:-cifar10_grayscale}
mkdir -p $OUT
for seed in 0 1 2; do
  for mode in static staticlong random learned; do
    $PY snn_classification_realtime/foveation/minibrain/exp_gaze.py \
      --dataset-name $DS --images 1600 --saccades 6 --ticks-per-sacc 10 \
      --canvas 72 --distractors 4 --distractor-size 16 --fovea 10 --grid 14 \
      --max-step 18 --readout all --probe-every 400 --mode $mode --seed $seed --out $OUT \
      > $OUT/run_${DS}_${mode}_seed${seed}.stdout 2>&1 &
  done
done
wait
echo "ALL 12 DONE — aggregating"
$PY snn_classification_realtime/foveation/minibrain/exp_gaze.py --aggregate \
   --dataset-name $DS --readout all --out $OUT
