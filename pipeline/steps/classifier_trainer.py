"""
Classifier trainer step for the pipeline.

Delegates to snn_classification_realtime.snn_trainer for SNN classifier training.
"""

import io
import json
import logging
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import List

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.config import TrainingConfig
from pipeline.steps.base import (
    PipelineStep,
    StepContext,
    StepResult,
    StepStatus,
    Artifact,
    StepRegistry,
    StepCancelledException,
)

from snn_classification_realtime.snn_trainer.config import TrainConfig
from snn_classification_realtime.snn_trainer.run import run_train


@StepRegistry.register
class ClassifierTrainerStep(PipelineStep):
    """Pipeline step for training SNN classifiers."""

    @property
    def name(self) -> str:
        return "training"

    @property
    def display_name(self) -> str:
        return "Classifier Training"

    def run(self, context: StepContext) -> StepResult:
        start_time = datetime.now()
        logs: List[str] = []

        try:
            config: TrainingConfig = context.config
            log = context.logger or logging.getLogger(__name__)

            prep_artifacts = context.previous_artifacts.get("data_preparation", [])
            if not prep_artifacts:
                raise ValueError("Data preparation artifacts not found")

            # Dataset dir is parent of first artifact (e.g. train_data.pt)
            dataset_dir = str(prep_artifacts[0].path.parent)

            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)

            log.info(f"Training classifier on dataset from {dataset_dir}")
            logs.append(f"Training classifier on dataset from {dataset_dir}")

            train_config = TrainConfig(
                dataset_dir=dataset_dir,
                model_save_path="model.pth",
                load_model_path=None,
                output_dir=str(step_dir),
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                test_every=config.test_every,
                device=config.device if config.device != "cpu" else None,
                beta=config.beta,
            )

            # Capture trainer stdout (print) and stderr (tqdm) as logs
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                run_train(train_config)
            trainer_output = (
                stdout_capture.getvalue() + "\n" + stderr_capture.getvalue()
            ).strip()
            if trainer_output:
                ansi_escape = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\r")
                for line in trainer_output.split("\n"):
                    line = ansi_escape.sub("", line).strip()
                    if not line:
                        continue
                    # Skip tqdm progress bar lines (noise)
                    if re.search(r"\d+%\|[^\|]*\|", line) or "it/s]" in line:
                        continue
                    logs.append(line)

            # snn_trainer creates run_dir = output_dir/{dataset_basename}_e{epochs}_lr{lr}_b{batch}
            dataset_basename = os.path.basename(os.path.normpath(dataset_dir))
            run_dir_name = (
                f"{dataset_basename}_e{config.epochs}_lr{config.learning_rate}_b{config.batch_size}"
            )
            run_dir = step_dir / run_dir_name

            if not run_dir.exists():
                # Fallback: use first subdir or step_dir
                subdirs = [d for d in step_dir.iterdir() if d.is_dir()]
                run_dir = subdirs[0] if subdirs else step_dir

            model_path = run_dir / "model.pth"
            if not model_path.exists():
                model_path = run_dir / "best_model.pth"
            if not model_path.exists():
                model_path = next(run_dir.glob("*.pth"), None)

            if model_path is None or not model_path.exists():
                raise RuntimeError(f"No model file found in {run_dir}")

            # Load metrics from model_config.json (snn_trainer produces this)
            config_path = run_dir / "model_config.json"
            metrics: dict = {}
            if config_path.exists():
                with open(config_path) as f:
                    train_cfg = json.load(f)
                metrics = {
                    "final_train_loss": train_cfg.get("final_train_loss"),
                    "final_train_accuracy": train_cfg.get("final_train_accuracy"),
                    "final_test_accuracy": train_cfg.get("final_test_accuracy"),
                    "final_test_loss": train_cfg.get("final_test_loss"),
                    "epochs_trained": config.epochs,
                }
                # Add summary to logs
                if train_cfg.get("final_test_accuracy") is not None:
                    logs.append(
                        f"Final test accuracy: {train_cfg['final_test_accuracy']:.2f}%"
                    )
                if train_cfg.get("final_test_loss") is not None:
                    logs.append(f"Final test loss: {train_cfg['final_test_loss']:.4f}")
                if train_cfg.get("final_train_accuracy") is not None:
                    logs.append(
                        f"Final train accuracy: {train_cfg['final_train_accuracy']:.2f}%"
                    )
                if train_cfg.get("final_train_loss") is not None:
                    logs.append(f"Final train loss: {train_cfg['final_train_loss']:.4f}")

            logs.append(f"Model saved to {model_path}")

            artifacts = [
                Artifact(
                    name=model_path.name,
                    path=model_path,
                    artifact_type="model",
                    size_bytes=model_path.stat().st_size,
                ),
            ]

            for fname in ["training_history.json", "model_config.json"]:
                fp = run_dir / fname
                if fp.exists():
                    artifacts.append(
                        Artifact(
                            name=fname,
                            path=fp,
                            artifact_type="metadata",
                            size_bytes=fp.stat().st_size,
                        )
                    )

            return StepResult(
                status=StepStatus.COMPLETED,
                artifacts=artifacts,
                metrics=metrics,
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
