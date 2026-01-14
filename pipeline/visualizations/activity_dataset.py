"""
Visualization step for activity dataset visualizations.

Wraps visualize_activity_dataset.py to generate plots from pipeline.
"""

import logging
import os
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
    StepRegistry,
)

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
    """Run activity dataset visualization.

    Args:
        data_path: Path to activity dataset (JSON or binary dir)
        output_dir: Output directory for plots
        plot_types: List of plot types to generate
        logger: Optional logger

    Returns:
        Dict with status and generated file paths
    """
    log = logger or logging.getLogger(__name__)

    try:
        # Import the visualization module
        import visualize_activity_dataset as viz_module

        # Load dataset
        log.info(f"Loading activity data from {data_path}")
        records = viz_module.load_dataset(data_path)

        if not records:
            return {"success": False, "error": "No records loaded", "files": []}

        log.info(f"Loaded {len(records)} records")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate all requested plots
        for plot_type in plot_types:
            if plot_type in AVAILABLE_PLOTS:
                log.info(f"Generating plot: {plot_type}")
                viz_module.generate_plots(
                    records=records,
                    plot_type=plot_type,
                    output_dir=output_dir,
                )

        # Collect generated files recursively
        generated_files = []
        out_path_obj = Path(output_dir)
        for ext in ["*.png", "*.html", "*.svg", "*.pdf"]:
            for f in out_path_obj.rglob(ext):
                generated_files.append(str(f.relative_to(out_path_obj)))

        return {"success": True, "files": generated_files}

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

            data_path = str(activity_artifacts[0].path)

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
