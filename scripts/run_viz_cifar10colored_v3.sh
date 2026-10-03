#!/usr/bin/env bash
# Run all visualizations on cifar10colored_v3 outputs.
# Run from project root.

set -euo pipefail
cd "$(dirname "$0")/.."

# --- cifar10colored_v3 paths (from script 03) ---
ACTIVITY_PATH="activity_datasets/cifar10colored_v3_cifar10_color_run3"
NETWORK_PATH="networks/cifar10colored_v3.json"
EVAL_SUMMARY="evals/cifar10colored_v3_cifar10_color_run3_firings_avg_S_e100_lr0.001_b32_eval_run3_summary.json"
EVAL_JSONL="evals/cifar10colored_v3_cifar10_color_run3_firings_avg_S_e100_lr0.001_b32_eval_run3.jsonl"
VIZ_OUT="viz/cifar10colored_v3_run3"

echo "=== Running all visualizations on cifar10colored_v3 ==="
echo "Activity:  ${ACTIVITY_PATH}"
echo "Network:   ${NETWORK_PATH}"
echo "Output:    ${VIZ_OUT}/"
echo ""

# 1. Activity dataset plots (all plot types)
# echo "=== 1. Activity dataset visualizations ==="
# for plot in firing_rate_per_layer avg_S_per_layer_per_label firings_time_series avg_S_time_series total_fired_cumulative; do
#   echo "  Plot: ${plot}"
#   uv run python visualize_activity_dataset.py "${ACTIVITY_PATH}" --plot "${plot}" --out-dir "${VIZ_OUT}/activity_dataset" || true
# done
# echo "  -> ${VIZ_OUT}/activity_dataset/"
# echo ""

# 2. Network activity visualizations
# echo "=== 2. Network activity visualizations ==="
# uv run python visualize_network_activity.py \
#   --input-file "${ACTIVITY_PATH}" \
#   --output-dir "${VIZ_OUT}/network_activity" \
#   --num-classes 10 \
#   --plots all
# echo "  -> ${VIZ_OUT}/network_activity/"
# echo ""

# 3. 3D brain visualization

echo "=== 3. Activity 3D brain visualization ==="
uv run python -m snn_classification_realtime.viz.activity_3d.run \
  --network "${NETWORK_PATH}" \
  --output-dir "${VIZ_OUT}/activity_3d" \
  --dataset cifar10 \
  --ticks 200 \
  --samples-per-class 5 \
  --clustering firings
echo "  -> ${VIZ_OUT}/activity_3d/"
echo ""

# 4. Neuron clustering
echo "=== 4. Neuron clustering ==="
uv run python -m snn_classification_realtime.viz.cluster_neurons.run \
  --input-file "${ACTIVITY_PATH}" \
  --output-dir "${VIZ_OUT}/cluster_neurons" \
  --clustering-mode fixed \
  --num-clusters 10 \
  --num-classes 10
echo "  -> ${VIZ_OUT}/cluster_neurons/"
echo ""

# 5. Activity clustering
echo "=== 5. Activity clustering ==="
uv run python -m snn_classification_realtime.viz.cluster_activity.run \
  --input-file "${ACTIVITY_PATH}" \
  --output-dir "${VIZ_OUT}/cluster_activity" \
  --feature-types "firings,avg_S" \
  --clustering-mode fixed \
  --num-clusters 10 \
  --num-classes 10
echo "  -> ${VIZ_OUT}/cluster_activity/"
echo ""

# 6. Concept hierarchy (from eval summary)
echo "=== 6. Concept hierarchy ==="
if [[ -f "${EVAL_SUMMARY}" ]]; then
  uv run python plot_concept_hierarchy.py \
    --json-file "${EVAL_SUMMARY}" \
    --results-file "${EVAL_JSONL}" \
    --output-dir "${VIZ_OUT}/concept_hierarchy"
  echo "  -> ${VIZ_OUT}/concept_hierarchy/"
else
  echo "  Skipped (eval summary not found: ${EVAL_SUMMARY})"
fi
echo ""

# 7. Synaptic analysis (requires network_states from activity build with --export-network-states)
echo "=== 7. Synaptic analysis ==="
NETWORK_STATES="network_state/cifar10colored_v3_cifar10_color_run3"
if [[ -d "${NETWORK_STATES}" ]]; then
  uv run python -m snn_classification_realtime.viz.synaptic_analysis.run \
    --network-states-dir "${NETWORK_STATES}" \
    --output-dir "${VIZ_OUT}/synaptic_analysis" \
    --method louvain \
    --n-clusters 5
  echo "  -> ${VIZ_OUT}/synaptic_analysis/"
else
  echo "  Skipped (network_states not found; rebuild activity with --export-network-states)"
fi
echo ""

echo "=== Visualizations complete ==="
echo "Output: ${VIZ_OUT}/"
