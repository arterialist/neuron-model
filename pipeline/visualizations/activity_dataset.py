"""
Visualization step for activity dataset visualizations.

Wraps visualize_activity_dataset.py to generate plots from pipeline.
Uses subprocess to call the original script with proper arguments.
"""

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.steps.base import (
    PipelineStep,
    StepContext,
    StepResult,
    StepStatus,
    Artifact,
)
from pipeline.utils.activity_data import is_binary_dataset

# Available plot types from visualize_activity_dataset.py
AVAILABLE_PLOTS = [
    "firing_rate_per_layer",
    "firing_rate_per_layer_3d",
    "avg_S_per_layer_per_label",
    "avg_S_per_layer_per_label_3d",
    "firings_time_series",
    "firings_time_series_3d",
    "avg_S_time_series",
    "avg_S_time_series_3d",
    "total_fired_cumulative",
    "total_fired_cumulative_3d",
    "network_state_progression",
]


def run_activity_dataset_visualization(
    data_path: str,
    output_dir: str,
    plot_types: List[str],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run activity dataset visualization using subprocess.

    Args:
        data_path: Path to activity dataset (JSON or binary dir)
        output_dir: Output directory for plots
        plot_types: List of plot types to generate
        plot_types: List of plot types to generate
        logger: Optional logger

    Returns:
        Dict with status and generated file paths
    """
    log = logger or logging.getLogger(__name__)

    try:
        # Resolve script path
        script_path = (
            Path(__file__).parent.parent.parent / "visualize_activity_dataset.py"
        )
        if not script_path.exists():
            return {
                "success": False,
                "error": f"Visualization script not found: {script_path}",
                "files": [],
            }

        os.makedirs(output_dir, exist_ok=True)
        all_generated_files = []

        if not is_binary_dataset(data_path):
            return {
                "success": False,
                "error": f"Path is not a valid binary dataset: {data_path}",
                "files": [],
            }

        # Run each plot type separately
        for plot_type in plot_types:
            if plot_type not in AVAILABLE_PLOTS:
                log.warning(f"Unknown plot type: {plot_type}, skipping")
                continue

            log.info(f"Generating plot: {plot_type}")

            cmd = [
                sys.executable,
                str(script_path),
                data_path,
                "--plot",
                plot_type,
                "--out-dir",
                output_dir,
            ]

            log.debug(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per plot
                cwd=str(script_path.parent),
            )

            if result.returncode != 0:
                log.warning(f"Plot {plot_type} failed: {result.stderr}")
                # Continue with other plots
            else:
                log.info(f"Generated plot: {plot_type}")

        # Collect generated files recursively
        out_path_obj = Path(output_dir)
        for ext in ["*.png", "*.html", "*.svg", "*.pdf"]:
            for f in out_path_obj.rglob(ext):
                all_generated_files.append(str(f.relative_to(out_path_obj)))

        return {"success": True, "files": all_generated_files}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Visualization timed out", "files": []}
    except Exception as e:
        import traceback

        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "files": [],
        }


class ActivityDatasetVisualizationStep(PipelineStep):
    """Visualization step for activity dataset plots."""

    @property
    def name(self) -> str:
        return "visualize_activity_dataset"

    @property
    def display_name(self) -> str:
        return "Activity Dataset Visualization"

    def run(self, context: StepContext) -> StepResult:
        start_time = datetime.now()
        logs = []

        try:
            config = context.config
            log = context.logger or logging.getLogger(__name__)

            # Get activity recording artifact
            activity_artifacts = context.previous_artifacts.get(
                "activity_recording", []
            )
            if not activity_artifacts:
                raise ValueError("Activity recording artifact not found")

            # Determine data path - could be .h5 file or .json file
            artifact = activity_artifacts[0]
            data_path = str(artifact.path)

            # If artifact is a directory (binary format), use directory path
            # If artifact is a file, use directly
            if artifact.path.is_dir():
                # Binary format - look for activity_dataset.h5 or use directory
                data_path = str(artifact.path)
            elif artifact.path.suffix == ".h5":
                # Binary format - use parent directory
                data_path = str(artifact.path.parent)
            else:
                raise ValueError(
                    f"Artifact path {artifact.path} is not a valid binary dataset (dir or .h5)"
                )

            # Create output directory
            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)

            # Get plot types from config
            plot_types = config.get(
                "plots", ["avg_S_per_layer_per_label", "firing_rate_per_layer"]
            )

            log.info(f"Generating visualizations: {plot_types}")
            logs.append(f"Generating visualizations: {plot_types}")

            result = run_activity_dataset_visualization(
                data_path=data_path,
                output_dir=str(step_dir),
                plot_types=plot_types,
                logger=log,
            )

            if not result["success"]:
                raise RuntimeError(result.get("error", "Unknown visualization error"))

            # Collect artifacts
            artifacts = []
            for fname in result.get("files", []):
                fpath = step_dir / fname
                if fpath.exists():
                    # Use absolute path for registration to ensure orchestrator finds it
                    artifacts.append(
                        Artifact(
                            name=fname,
                            path=fpath.absolute(),
                            artifact_type="visualization",
                            size_bytes=fpath.stat().st_size,
                        )
                    )

            logs.append(f"Generated {len(artifacts)} visualization files")

            return StepResult(
                status=StepStatus.COMPLETED,
                artifacts=artifacts,
                metrics={"num_visualizations": len(artifacts)},
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs,
            )

        except Exception as e:
            import traceback

            return StepResult(
                status=StepStatus.FAILED,
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs + [f"ERROR: {e}", traceback.format_exc()],
            )
