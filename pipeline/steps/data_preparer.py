"""
Data preparer step for the pipeline.

Delegates to snn_classification_realtime.activity_preparer for feature extraction.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.config import DataPreparationConfig
from pipeline.steps.base import (
    PipelineStep,
    StepContext,
    StepResult,
    StepStatus,
    Artifact,
    StepRegistry,
    StepCancelledException,
)
from pipeline.utils.activity_data import is_binary_dataset

from snn_classification_realtime.activity_preparer.config import PrepareConfig
from snn_classification_realtime.activity_preparer.run import run_prepare


def _scaler_to_activity_preparer(scaling_method: str) -> str:
    """Map pipeline scaling_method to activity_preparer scaler."""
    mapping = {"zscore": "standard", "minmax": "minmax", "maxabs": "maxabs", "none": "none"}
    return mapping.get(scaling_method, "none")


@StepRegistry.register
class DataPreparerStep(PipelineStep):
    """Pipeline step for preparing activity data for training."""

    @property
    def name(self) -> str:
        return "data_preparation"

    @property
    def display_name(self) -> str:
        return "Data Preparation"

    def run(self, context: StepContext) -> StepResult:
        start_time = datetime.now()
        logs: List[str] = []

        try:
            config: DataPreparationConfig = context.config
            log = context.logger or logging.getLogger(__name__)

            activity_artifacts = context.previous_artifacts.get("activity_recording", [])
            if not activity_artifacts:
                raise ValueError("Activity recording artifact not found")

            activity_artifact = activity_artifacts[0]
            activity_path = str(activity_artifact.path)

            # Binary format: use directory; JSON: use file path
            if activity_path.endswith(".h5"):
                input_file = str(Path(activity_path).parent)
            else:
                input_file = activity_path

            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)

            log.info(f"Preparing activity data from {input_file}")
            logs.append(f"Preparing activity data from {input_file}")

            prepare_config = PrepareConfig(
                input_file=input_file,
                output_dir=str(step_dir),
                feature_types=config.feature_types,
                train_split=config.train_split,
                use_streaming=False,
                scaler=_scaler_to_activity_preparer(config.scaling_method),
                scale_eps=1e-8,
                max_ticks=config.max_ticks,
                max_samples=None,
                legacy_json=not is_binary_dataset(activity_path),
            )

            run_prepare(prepare_config)

            # Resolve structured output dir (activity_preparer adds input_basename_feature_suffix)
            input_basename = Path(input_file).name
            if input_basename == ".":
                input_basename = Path(input_file).stem or "dataset"
            feature_suffix = "_".join(config.feature_types)
            structured_dir = step_dir / f"{input_basename}_{feature_suffix}"

            if not structured_dir.exists():
                raise RuntimeError(
                    f"Activity preparer did not create expected output dir: {structured_dir}"
                )

            # Load metadata for metrics
            metadata_path = structured_dir / "dataset_metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

            num_train = metadata.get("train_samples", 0)
            num_test = metadata.get("test_samples", 0)
            total_dim = metadata.get("total_feature_dim", 0)

            logs.append(f"Saved prepared data to {structured_dir}")
            logs.append(f"Train samples: {num_train}, Test samples: {num_test}")

            artifacts = [
                Artifact(
                    name="train_data.pt",
                    path=structured_dir / "train_data.pt",
                    artifact_type="dataset",
                    size_bytes=(structured_dir / "train_data.pt").stat().st_size,
                ),
                Artifact(
                    name="train_labels.pt",
                    path=structured_dir / "train_labels.pt",
                    artifact_type="dataset",
                    size_bytes=(structured_dir / "train_labels.pt").stat().st_size,
                ),
                Artifact(
                    name="test_data.pt",
                    path=structured_dir / "test_data.pt",
                    artifact_type="dataset",
                    size_bytes=(structured_dir / "test_data.pt").stat().st_size,
                ),
                Artifact(
                    name="test_labels.pt",
                    path=structured_dir / "test_labels.pt",
                    artifact_type="dataset",
                    size_bytes=(structured_dir / "test_labels.pt").stat().st_size,
                ),
                Artifact(
                    name="dataset_metadata.json",
                    path=metadata_path,
                    artifact_type="metadata",
                    size_bytes=metadata_path.stat().st_size,
                    metadata=metadata,
                ),
            ]

            if (structured_dir / "scaler.pt").exists():
                artifacts.append(
                    Artifact(
                        name="scaler.pt",
                        path=structured_dir / "scaler.pt",
                        artifact_type="metadata",
                        size_bytes=(structured_dir / "scaler.pt").stat().st_size,
                    )
                )

            return StepResult(
                status=StepStatus.COMPLETED,
                artifacts=artifacts,
                metrics={
                    "num_train": num_train,
                    "num_test": num_test,
                    "num_features": total_dim,
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
