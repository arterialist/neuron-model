"""
Evaluator step for the pipeline.

Delegates to snn_classification_realtime.realtime_classifier for evaluation.
"""

import json
import logging
import os
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.config import EvaluationConfig
from pipeline.steps.base import (
    PipelineStep,
    StepContext,
    StepResult,
    StepStatus,
    Artifact,
    StepRegistry,
    StepCancelledException,
)


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
        logs: List[str] = []

        try:
            config: EvaluationConfig = context.config
            log = context.logger or logging.getLogger(__name__)

            # Get network artifact
            network_artifact = context.get_artifact("network", "network.json")
            if not network_artifact:
                raise ValueError("Network artifact not found")

            # Get trained model from training step
            training_artifacts = context.previous_artifacts.get("training", [])
            if not training_artifacts:
                raise ValueError("Training artifacts not found")

            # Find model.pth artifact (snn_trainer saves model.pth)
            model_artifact = None
            for a in training_artifacts:
                if a.name.endswith(".pth"):
                    model_artifact = a
                    break
            if not model_artifact or not model_artifact.path.exists():
                raise ValueError("Trained model (.pth) not found in training artifacts")

            snn_model_path = str(model_artifact.path)
            neuron_model_path = str(network_artifact.path)

            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)
            evals_dir = step_dir / "evals"
            evals_dir.mkdir(parents=True, exist_ok=True)

            log.info(f"Evaluating with SNN model: {snn_model_path}")
            log.info(f"Neuron network: {neuron_model_path}")
            logs.append(f"Evaluating with SNN model: {snn_model_path}")

            # Build args for realtime_classifier
            args = Namespace(
                snn_model_path=snn_model_path,
                neuron_model_path=neuron_model_path,
                dataset_name=config.dataset_name,
                ticks_per_image=config.window_size,
                window_size=config.window_size,
                evaluation_mode=True,
                eval_samples=config.samples,
                device=config.device if config.device != "cpu" else None,
                think_longer=config.think_longer,
                max_thinking_multiplier=float(config.max_thinking_multiplier),
                enable_web_server=False,
                ablation=None,
                bistability_rescue=False,
                cifar10_color_upper_bound=1.0,
            )

            # Run evaluation via realtime_classifier (writes to cwd/evals/)
            from snn_classification_realtime.realtime_classifier.run import run

            orig_cwd = os.getcwd()
            try:
                os.chdir(step_dir)
                run(args)
            finally:
                os.chdir(orig_cwd)

            # Find the generated JSONL and summary
            jsonl_files = list(evals_dir.glob("*_eval_*.jsonl"))
            summary_files = list(evals_dir.glob("*_eval_*_summary.json"))

            if not jsonl_files:
                raise RuntimeError("Evaluation did not produce JSONL results")

            # Parse JSONL for results
            results_list: List[Dict[str, Any]] = []
            with open(jsonl_files[0]) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    results_list.append(json.loads(line))

            # Compute metrics
            total = len(results_list)
            correct = sum(1 for r in results_list if r.get("correct", False))
            accuracy = 100.0 * correct / total if total > 0 else 0.0

            # Per-class accuracy
            per_class_correct: Dict[int, int] = {}
            per_class_total: Dict[int, int] = {}
            for r in results_list:
                label = int(r.get("actual_label", 0))
                per_class_total[label] = per_class_total.get(label, 0) + 1
                if r.get("correct", False):
                    per_class_correct[label] = per_class_correct.get(label, 0) + 1
            per_class_accuracy = {
                str(k): 100.0 * per_class_correct.get(k, 0) / per_class_total[k]
                for k in per_class_total
            }

            # Load model config for artifact metadata
            model_config_path = model_artifact.path.parent / "model_config.json"
            model_config: Dict[str, Any] = {}
            if model_config_path.exists():
                with open(model_config_path) as f:
                    model_config = json.load(f)

            # Build evaluation_results.json (pipeline + concept_hierarchy compatible)
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
                    "dataset_name": config.dataset_name,
                },
                "model_config": model_config,
                "results": results_list,
                "evaluation_metadata": {
                    "dataset_name": config.dataset_name,
                    "snn_model_path": snn_model_path,
                    "neuron_model_path": neuron_model_path,
                },
            }

            eval_results_path = step_dir / "evaluation_results.json"
            with open(eval_results_path, "w") as f:
                json.dump(eval_results, f, indent=2, default=str)

            # Build detailed_results.json (pipeline format)
            detailed = [
                {
                    "index": r.get("image_idx", i),
                    "true_label": int(r.get("actual_label", 0)),
                    "predicted": r.get("predicted_label"),
                    "correct": r.get("correct", False),
                    "confidence": float(r.get("confidence", 0.0) or 0.0),
                }
                for i, r in enumerate(results_list)
            ]
            detailed_path = step_dir / "detailed_results.json"
            with open(detailed_path, "w") as f:
                json.dump({"results": detailed}, f, indent=2)

            log.info(f"Evaluation accuracy: {accuracy:.2f}% ({correct}/{total})")
            logs.append(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

            artifacts = [
                Artifact(
                    name="evaluation_results.json",
                    path=eval_results_path,
                    artifact_type="metadata",
                    size_bytes=eval_results_path.stat().st_size,
                    metadata={"accuracy": accuracy},
                ),
                Artifact(
                    name="detailed_results.json",
                    path=detailed_path,
                    artifact_type="metadata",
                    size_bytes=detailed_path.stat().st_size,
                ),
            ]
            for jf in jsonl_files:
                artifacts.append(
                    Artifact(
                        name=jf.name,
                        path=jf,
                        artifact_type="metadata",
                        size_bytes=jf.stat().st_size,
                    )
                )
            for sf in summary_files:
                artifacts.append(
                    Artifact(
                        name=sf.name,
                        path=sf,
                        artifact_type="metadata",
                        size_bytes=sf.stat().st_size,
                    )
                )

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
