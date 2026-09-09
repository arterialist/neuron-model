#!/bin/bash
# Does the retinotopic 'where' periphery let the gaze precisely foveate?
# flm + flmlock x 3 seeds, where=periph, longer run. Compare to readout baseline.
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY="$ROOT/.venv/bin/python"
OUT=foveation_results/minibrain/flm_periph
mkdir -p $OUT
for seed in 0 1 2; do
  for mode in random flm flmlock; do
    $PY snn_classification_realtime/foveation/minibrain/exp_flm.py \
      --images 2400 --saccades 8 --ticks-per-sacc 8 --memorize-ticks 30 \
      --canvas 72 --distractors 4 --fovea 28 --grid 14 --max-step 18 --lock-std 0.02 \
      --where periph --readout all --probe-every 400 --mode $mode --seed $seed --out $OUT \
      > $OUT/run_${mode}_seed${seed}.stdout 2>&1 &
  done
done
wait
echo "ALL FLM-PERIPH DONE — aggregating"
$PY snn_classification_realtime/foveation/minibrain/exp_flm.py --aggregate --out $OUT
