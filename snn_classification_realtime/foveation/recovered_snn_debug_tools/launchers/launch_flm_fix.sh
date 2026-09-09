#!/bin/bash
# Cheap gaze fix: bigger fovea (44) captures object at imperfect aim + fovea-only
# memorize (suppress clutter). oracle=ceiling, random=baseline, flm/flmlock=learned.
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY="$ROOT/.venv/bin/python"
OUT=foveation_results/minibrain/flm_fix
mkdir -p $OUT
for seed in 0 1 2; do
  for mode in oracle random flm flmlock; do
    $PY snn_classification_realtime/foveation/minibrain/exp_flm.py \
      --images 2400 --saccades 8 --ticks-per-sacc 8 --memorize-ticks 30 \
      --canvas 72 --distractors 4 --fovea 44 --grid 16 --max-step 18 --lock-std 0.02 \
      --where periph --memorize-fovea-only 1 --readout all --probe-every 400 \
      --mode $mode --seed $seed --out $OUT \
      > $OUT/run_${mode}_seed${seed}.stdout 2>&1 &
  done
done
wait
echo "ALL FLM-FIX DONE — aggregating"
$PY snn_classification_realtime/foveation/minibrain/exp_flm.py --aggregate --out $OUT
