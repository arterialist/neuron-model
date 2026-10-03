#!/usr/bin/env bash
# CIFAR-10 Color pipeline — deterministic artifact naming for agent feedback loop.
# Run from project root. No --silent; output visible for manual runs.
#
# Artifact chain (all derived from config vars below):
#   network → activity_dataset → prepared_data → model → eval → analysis

set -euo pipefail
TRAIN_DEVICE="${TRAIN_DEVICE:-mps}"
cd "$(dirname "$0")/.."

# --- Config (edit to change run) ---
CONFIG_PATH="configs/cifar10colored_v5.yaml"
NETWORK_NAME="cifar10colored_v5"
DATASET_NAME="cifar10_color"
RUN_ID="run1"
FEATURE_TYPES="firings avg_S"
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.001
BETA=0.9
IMAGES_PER_LABEL=1000
EVAL_SAMPLES=1000

# --- Derived paths (do not edit) ---
NETWORKS_DIR="networks"
ACTIVITY_DIR="activity_datasets"
PREPARED_DIR="prepared_data"
MODELS_DIR="models"
EVALS_DIR="evals"
ANALYSIS_DIR="analysis"

NETWORK_PATH="${NETWORKS_DIR}/${NETWORK_NAME}.json"
ACTIVITY_PATH="${ACTIVITY_DIR}/${NETWORK_NAME}_${DATASET_NAME}_${RUN_ID}"
PREPARED_PATH="${PREPARED_DIR}/${NETWORK_NAME}_${DATASET_NAME}_${RUN_ID}_${FEATURE_TYPES// /_}"
MODEL_RUN_DIR="${MODELS_DIR}/${NETWORK_NAME}_${DATASET_NAME}_${RUN_ID}_${FEATURE_TYPES// /_}_e${EPOCHS}_lr${LEARNING_RATE}_b${BATCH_SIZE}"
MODEL_PATH="${MODEL_RUN_DIR}/model.pth"
EVAL_JSONL="${EVALS_DIR}/$(basename "${MODEL_RUN_DIR}")_eval_${RUN_ID}.jsonl"
ANALYSIS_OUT="${ANALYSIS_DIR}/${NETWORK_NAME}_${RUN_ID}"

echo "=== Step 1: Build network ==="
# uv run python build_network.py --config "${CONFIG_PATH}" --output-dir "${NETWORKS_DIR}" --name "${NETWORK_NAME}"
echo "  -> ${NETWORK_PATH}"

echo ""
echo "=== Step 2: Generate activity recording ==="
# uv run python -m snn_classification_realtime.build_activity_dataset \
#   --network-path "${NETWORK_PATH}" \
#   --dataset-name "${DATASET_NAME}" \
#   --output-dir "${ACTIVITY_DIR}" \
#   --dataset-base "${NETWORK_NAME}" \
#   --output-suffix "${RUN_ID}" \
#   --ticks-per-image 150 \
#   --images-per-label "${IMAGES_PER_LABEL}" \
#   --tick-ms 0 \
#   --ablation none
echo "  -> ${ACTIVITY_PATH}/activity_dataset.h5"

echo ""
echo "=== Step 3: Prepare data ==="
# uv run python -m snn_classification_realtime.prepare_activity_data \
#   --input-file "${ACTIVITY_PATH}" \
#   --output-dir "${PREPARED_DIR}" \
#   --feature-types ${FEATURE_TYPES} \
#   --train-split 0.8 \
#   --scaler none
echo "  -> ${PREPARED_PATH}/"

echo ""
echo "=== Step 4: Train SNN classifier ==="
# Note: For best accuracy, use the checkpoint with lowest test loss from *_config.json
# (e.g. model_epoch15.pth). The script uses the final model by default.
uv run python -m snn_classification_realtime.train_snn_classifier \
  --dataset-dir "${PREPARED_PATH}" \
  --output-dir "${MODELS_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --beta "${BETA}" \
  --test-every 1 \
  --device "${TRAIN_DEVICE}"
echo "  -> ${MODEL_PATH}"

echo ""
echo "=== Step 5: Run evaluation ==="
uv run python -m snn_classification_realtime.realtime_classification \
  --snn-model-path "${MODEL_PATH}" \
  --neuron-model-path "${NETWORK_PATH}" \
  --dataset-name "${DATASET_NAME}" \
  --evaluation-mode \
  --eval-samples "${EVAL_SAMPLES}" \
  --output-dir "${EVALS_DIR}" \
  --eval-output-suffix "${RUN_ID}" \
  --window-size 200 \
  --ticks-per-image 50 \
  --device cpu \
  --think-longer \
  --max-thinking-multiplier 4 \
  --bistability-rescue
echo "  -> ${EVAL_JSONL}"

echo ""
echo "=== Step 6: Analyze results ==="
uv run python -m snn_classification_realtime.analyze_eval \
  --jsonl "${EVAL_JSONL}" \
  --output-dir "${ANALYSIS_OUT}" \
  --num-classes 10 \
  --class-labels "airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck"
echo "  -> ${ANALYSIS_OUT}/metrics.json"
echo "  -> ${ANALYSIS_OUT}/metrics.md"

echo ""
echo "=== Pipeline complete ==="
echo "Artifacts for agent feedback:"
echo "  - Network:     ${NETWORK_PATH}"
echo "  - Activity:    ${ACTIVITY_PATH}/"
echo "  - Prepared:    ${PREPARED_PATH}/"
echo "  - Model:       ${MODEL_PATH}"
echo "  - Eval JSONL:  ${EVAL_JSONL}"
echo "  - Metrics:     ${ANALYSIS_OUT}/metrics.json"
