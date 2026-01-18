"""
Visualization step for network activity visualizations.

Wraps visualize_network_activity.py to generate plots from pipeline.
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

# Available plot types from visualize_network_activity.py
# This matches the original script's available choices
AVAILABLE_PLOTS = [
    "s_heatmap_by_class",
    "favg_tref_scatter",
    "firing_rate_hist_by_layer",
    "tref_timeline",
    "layerwise_s_average",
    "spike_raster",
    "phase_portrait",
    "attractor_landscape",
    "attractor_landscape_3d",
    "attractor_landscape_animated",
    "tref_bounds_box",
    "favg_stability",
    "homeostatic_response",
    "affinity_heatmap",
    "tref_by_preferred_digit",
    "temporal_corr_graph",
    "s_variance_decay",
]


def run_network_activity_visualization(
    data_path: str,
    output_dir: str,
    plot_types: List[str],
    num_classes: int = 10,
    network_config: Optional[str] = None,
    max_representative_neurons: int = 10,
    skip_static_images: bool = False,
    raster_max_images: int = 50,
    animation_frames: int = 50,
    legacy_json: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run network activity visualization using subprocess.

    Args:
        data_path: Path to activity dataset
        output_dir: Output directory for plots
        plot_types: List of plot types to generate
        num_classes: Number of classes in the dataset
        network_config: Optional path to network config for theoretical bounds
        max_representative_neurons: Max neurons for t_ref timeline
        skip_static_images: Skip PNG/SVG exports (HTML only)
        raster_max_images: Max images in spike raster
        animation_frames: Number of frames for animations
        legacy_json: Force loading as legacy JSON format
        logger: Optional logger

    Returns:
        Dict with status and generated file paths
    """
    log = logger or logging.getLogger(__name__)

    try:
        # Resolve script path
        script_path = (
            Path(__file__).parent.parent.parent / "visualize_network_activity.py"
        )
        if not script_path.exists():
            return {
                "success": False,
                "error": f"Visualization script not found: {script_path}",
                "files": [],
            }

        os.makedirs(output_dir, exist_ok=True)

        # Auto-detect if legacy JSON is needed
        if not legacy_json and not is_binary_dataset(data_path):
            if data_path.endswith(".json"):
                legacy_json = True

        # Build command
        cmd = [
            sys.executable,
            str(script_path),
            "--input-file",
            data_path,
            "--output-dir",
            output_dir,
            "--num-classes",
            str(num_classes),
            "--plots",
        ]

        # Filter to only valid plot types
        valid_plots = [p for p in plot_types if p in AVAILABLE_PLOTS]
        if not valid_plots:
            # Default to some sensible plots
            valid_plots = ["s_heatmap_by_class", "spike_raster", "layerwise_s_average"]

        cmd.extend(valid_plots)

        # Add optional parameters
        cmd.extend(["--max-representative-neurons", str(max_representative_neurons)])
        cmd.extend(["--raster-max-images", str(raster_max_images)])
        cmd.extend(["--animation-frames", str(animation_frames)])

        if network_config:
            cmd.extend(["--network-config", network_config])

        if skip_static_images:
            cmd.append("--skip-static-images")

        if legacy_json:
            cmd.append("--legacy-json")

        log.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout for all plots
            cwd=str(script_path.parent),
        )

        if result.returncode != 0:
            log.warning(f"Visualization script returned non-zero: {result.stderr}")
            # Try to continue if some plots were generated

        # Collect generated files recursively
        generated_files = []
        out_path_obj = Path(output_dir)
        for ext in ["*.png", "*.html", "*.svg", "*.pdf", "*.json"]:
            for f in out_path_obj.rglob(ext):
                generated_files.append(str(f.relative_to(out_path_obj)))

        return {"success": True, "files": generated_files}

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


class NetworkActivityVisualizationStep(PipelineStep):
    """Visualization step for network activity plots."""

    @property
    def name(self) -> str:
        return "visualize_network_activity"

    @property
    def display_name(self) -> str:
        return "Network Activity Visualization"

    def run(self, context: StepContext) -> StepResult:
        start_time = datetime.now()
        logs = []

        try:
            config = context.config or {}
            log = context.logger or logging.getLogger(__name__)

            # Get activity recording artifact
            activity_artifacts = context.previous_artifacts.get(
                "activity_recording", []
            )
            if not activity_artifacts:
                raise ValueError("Activity recording artifact not found")

            # Determine data path
            artifact = activity_artifacts[0]
            if artifact.path.is_dir():
                data_path = str(artifact.path)
            elif artifact.path.suffix == ".h5":
                data_path = str(artifact.path.parent)
            else:
                data_path = str(artifact.path)

            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)

            # Get config parameters
            plot_types = config.get(
                "plots", ["s_heatmap_by_class", "spike_raster", "layerwise_s_average"]
            )
            num_classes = config.get("num_classes", 10)
            max_representative_neurons = config.get("max_representative_neurons", 10)
            skip_static_images = config.get("skip_static_images", False)
            raster_max_images = config.get("raster_max_images", 50)
            animation_frames = config.get("animation_frames", 50)
            legacy_json = config.get("legacy_json", False)

            # Get network config path if available
            network_config = None
            network_artifacts = context.previous_artifacts.get("network", [])
            if network_artifacts:
                network_config = str(network_artifacts[0].path)

            log.info(f"Generating visualizations: {plot_types}")
            logs.append(f"Generating visualizations: {plot_types}")

            result = run_network_activity_visualization(
                data_path=data_path,
                output_dir=str(step_dir),
                plot_types=plot_types,
                num_classes=num_classes,
                network_config=network_config,
                max_representative_neurons=max_representative_neurons,
                skip_static_images=skip_static_images,
                raster_max_images=raster_max_images,
                animation_frames=animation_frames,
                legacy_json=legacy_json,
                logger=log,
            )

            if not result["success"]:
                raise RuntimeError(result.get("error", "Unknown visualization error"))

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
