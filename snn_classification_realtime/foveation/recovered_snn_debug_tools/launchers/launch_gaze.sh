#!/bin/bash
# 9 parallel single-threaded runs (3 modes x 3 seeds) on 12 cores, then aggregate.
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY="$ROOT/.venv/bin/python"
OUT=foveation_results/minibrain/gaze9
DS=${1:-cifar10_grayscale}
mkdir -p $OUT
for seed in 0 1 2; do
  for mode in static staticlong random learned; do
    $PY snn_classification_realtime/foveation/minibrain/exp_gaze.py \
      --dataset-name $DS --images 1500 --saccades 5 --ticks-per-sacc 12 \
      --readout all --probe-every 300 --mode $mode --seed $seed --out $OUT \
      > $OUT/run_${DS}_${mode}_seed${seed}.stdout 2>&1 &
  done
done
wait
echo "ALL 9 DONE — aggregating"
$PY snn_classification_realtime/foveation/minibrain/exp_gaze.py --aggregate \
   --dataset-name $DS --readout all --out $OUT
