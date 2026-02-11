"""
Evaluator step for the pipeline.

Evaluates trained classifiers on neural network activity.
Wraps functionality from snn_classification_realtime/realtime_classification.py.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    import snntorch as snn
except ImportError:
    snn = None

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuron.nn_core import NNCore
from neuron.network import NeuronNetwork
from neuron.network_config import NetworkConfig as NeuronNetworkConfig
from neuron import setup_neuron_logger

from pipeline.config import EvaluationConfig, DatasetType
from pipeline.steps.base import (
    PipelineStep,
    StepContext,
    StepResult,
    StepStatus,
    Artifact,
    StepRegistry,
    StepCancelledException,
)
from pipeline.steps.activity_recorder import (
    load_dataset_for_recording,
    infer_layers_from_metadata,
    determine_input_mapping,
    image_to_signals,
)
from pipeline.steps.classifier_trainer import SNNClassifier


def apply_scaling(snapshot: List[float], scaler_state: Dict[str, Any]) -> List[float]:
    """Apply saved scaler state to a snapshot."""
    method = scaler_state.get("method", "none")
    eps = scaler_state.get("eps", 1e-8)

    arr = np.array(snapshot)

    if method == "zscore":
        mean = np.array(scaler_state.get("mean", []))
        std = np.array(scaler_state.get("std", []))
        if len(mean) == len(arr):
            arr = (arr - mean) / (std + eps)
    elif method == "minmax":
        min_val = np.array(scaler_state.get("min", []))
        max_val = np.array(scaler_state.get("max", []))
        if len(min_val) == len(arr):
            arr = (arr - min_val) / (max_val - min_val + eps)
    elif method == "maxabs":
        max_abs = np.array(scaler_state.get("max_abs", []))
        if len(max_abs) == len(arr):
            arr = arr / (max_abs + eps)

    return arr.tolist()


def collect_features_snapshot(
    network_sim: NeuronNetwork,
    layers: List[List[int]],
    feature_types: List[str],
) -> List[float]:
    """Collect features from current network state."""
    features = []

    for ft in feature_types:
        for layer_ids in layers:
            for nid in layer_ids:
                neuron = network_sim.network.neurons[nid]
                if ft == "firings":
                    features.append(float(neuron.O > 0))
                elif ft == "avg_S":
                    features.append(float(neuron.S))
                elif ft == "avg_t_ref":
                    features.append(float(neuron.t_ref))

    return features


@StepRegistry.register
class EvaluatorStep(PipelineStep):
    """Pipeline step for evaluating trained classifiers."""

    @property
    def name(self) -> str:
        return "evaluation"

    @property
    def display_name(self) -> str:
        return "Evaluation"

    def run(self, context: StepContext) -> StepResult:
        start_time = datetime.now()
        logs: list[str] = []

        try:
            config: EvaluationConfig = context.config
            log = context.logger or logging.getLogger(__name__)

            if snn is None:
                raise ImportError("snntorch is required for evaluation")

            # Get network from first step
            network_artifact = context.get_artifact("network", "network.json")
            if not network_artifact:
                raise ValueError("Network artifact not found")

            # Get trained model from training step
            training_artifacts = context.previous_artifacts.get("training", [])
            if not training_artifacts:
                raise ValueError("Training artifacts not found")

            training_dir = training_artifacts[0].path.parent

            # Get activity recording config for dataset info
            activity_artifacts = context.previous_artifacts.get(
                "activity_recording", []
            )

            # Create output directory
            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Evaluation output directory: {step_dir}")

            # Load model config
            with open(training_dir / "model_config.json", "r") as f:
                model_config = json.load(f)

            input_size = model_config["input_size"]
            hidden_size = model_config["hidden_size"]
            num_classes = model_config["num_classes"]
            feature_types = model_config.get("feature_types", ["avg_S", "firings"])
            scaler_state = model_config.get("scaler_state", {})

            log.info(
                f"Loading model: input={input_size}, hidden={hidden_size}, output={num_classes}"
            )
            logs.append(
                f"Model: input={input_size}, hidden={hidden_size}, output={num_classes}"
            )

            # Load model
            device = torch.device(config.device)
            model = SNNClassifier(input_size, hidden_size, num_classes).to(device)
            model.load_state_dict(
                torch.load(training_dir / "best_model.pth", map_location=device)
            )
            model.eval()

            # Load network
            network_path = str(network_artifact.path)
            log.info(f"Loading network from {network_path}")
            logs.append(f"Loading network from {network_path}")

            network_sim = NeuronNetworkConfig.load_network_config(network_path)
            layers = infer_layers_from_metadata(network_sim)
            input_layer_ids, input_synapses_per_neuron = determine_input_mapping(
                network_sim, layers
            )

            # Initialize NNCore
            nn_core = NNCore()
            nn_core.neural_net = network_sim
            try:
                setup_neuron_logger("CRITICAL")
                nn_core.set_log_level("CRITICAL")
            except:
                pass

            # Load dataset
            # Try to infer dataset from activity recording metadata
            dataset_type = DatasetType.MNIST
            if activity_artifacts:
                activity_meta = activity_artifacts[0].metadata or {}
                dataset_name = activity_meta.get("dataset", "mnist")
                for dt in DatasetType:
                    if dt.value == dataset_name:
                        dataset_type = dt
                        break

            log.info(f"Loading dataset: {dataset_type.value}")
            logs.append(f"Dataset: {dataset_type.value}")

            ds_test, _, _, _, is_colored = load_dataset_for_recording(
                dataset_type, train=False
            )

            # Determine ticks based on window size
            ticks_per_image = config.window_size

            # Limit samples
            n_samples = min(config.samples, len(ds_test))
            indices = list(range(len(ds_test)))
            np.random.shuffle(indices)
            eval_indices = indices[:n_samples]

            log.info(
                f"Evaluating on {n_samples} samples with {ticks_per_image} ticks each"
            )
            logs.append(f"Evaluating on {n_samples} samples")

            # Evaluation loop
            results = []
            correct = 0
            total = 0

            for idx in tqdm(eval_indices, desc="Evaluating", disable=True):
                # Check for cancellation/pause
                context.check_control_signals()

                image_tensor, true_label = ds_test[idx]

                # Reset network
                network_sim.reset_simulation()
                nn_core.state.current_tick = 0
                network_sim.current_tick = 0

                if (total + 1) % 10 == 0 or (total + 1) == n_samples:
                    log.info(f"Evaluating sample {total + 1}/{n_samples}")
                    logs.append(f"Processed {total + 1}/{n_samples} samples")

                # Generate signals
                signals = image_to_signals(
                    image_tensor,
                    input_layer_ids,
                    input_synapses_per_neuron,
                    network_sim,
                    is_colored,
                )

                # Collect features over ticks
                feature_history = []
                mem1, mem2, mem3 = None, None, None
                spk_sum = torch.zeros(num_classes, device=device)

                for tick in range(ticks_per_image):
                    nn_core.send_batch_signals(signals)
                    nn_core.do_tick()

                    # Collect features
                    snapshot = collect_features_snapshot(
                        network_sim, layers, feature_types
                    )
                    scaled = apply_scaling(snapshot, scaler_state)
                    feature_history.append(scaled)

                    # Feed to classifier
                    x = torch.tensor([scaled], dtype=torch.float32, device=device)
                    spk, mem1, mem2, mem3 = model(x, mem1, mem2, mem3)
                    spk_sum += spk[0]

                # Prediction
                predicted = spk_sum.argmax().item()
                is_correct = predicted == true_label

                if is_correct:
                    correct += 1
                total += 1

                results.append(
                    {
                        "index": int(idx),
                        "true_label": int(true_label),
                        "predicted": int(predicted),
                        "correct": is_correct,
                        "confidence": float(spk_sum[predicted].item()),
                    }
                )

            accuracy = 100.0 * correct / total if total > 0 else 0.0

            log.info(f"Evaluation accuracy: {accuracy:.2f}% ({correct}/{total})")
            logs.append(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

            # Compute per-class accuracy
            per_class_correct = {}
            per_class_total = {}
            for r in results:
                label = r["true_label"]
                per_class_total[label] = per_class_total.get(label, 0) + 1
                if r["correct"]:
                    per_class_correct[label] = per_class_correct.get(label, 0) + 1

            per_class_accuracy = {
                label: 100.0 * per_class_correct.get(label, 0) / per_class_total[label]
                for label in per_class_total
            }

            # Save results
            eval_results = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "per_class_accuracy": per_class_accuracy,
                "config": {
                    "samples": config.samples,
                    "window_size": config.window_size,
                    "think_longer": config.think_longer,
                    "max_thinking_multiplier": config.max_thinking_multiplier,
                },
                "model_config": model_config,
            }

            with open(step_dir / "evaluation_results.json", "w") as f:
                json.dump(eval_results, f, indent=2)

            # Save detailed results
            with open(step_dir / "detailed_results.json", "w") as f:
                json.dump({"results": results}, f, indent=2)

            # Create artifacts
            artifacts = [
                Artifact(
                    name="evaluation_results.json",
                    path=step_dir / "evaluation_results.json",
                    artifact_type="metadata",
                    size_bytes=(step_dir / "evaluation_results.json").stat().st_size,
                    metadata={"accuracy": accuracy},
                ),
                Artifact(
                    name="detailed_results.json",
                    path=step_dir / "detailed_results.json",
                    artifact_type="metadata",
                    size_bytes=(step_dir / "detailed_results.json").stat().st_size,
                ),
            ]

            return StepResult(
                status=StepStatus.COMPLETED,
                artifacts=artifacts,
                metrics={
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                },
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs,
            )

        except StepCancelledException:
            raise
        except Exception as e:
            import traceback

            return StepResult(
                status=StepStatus.FAILED,
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs + [f"ERROR: {e}", traceback.format_exc()],
            )
