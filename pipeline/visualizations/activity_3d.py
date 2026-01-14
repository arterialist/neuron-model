"""
3D brain visualization step.

Wraps visualize_activity_3d.py to create interactive 3D visualizations.
"""

import logging
import os
import sys
import subprocess
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


def run_3d_visualization(
    network_path: str,
    output_dir: str,
    dataset: str = "mnist",
    samples_per_class: int = 10,
    ticks: int = 20,
    clustering_method: str = "firings",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run 3D brain visualization.

    Args:
        network_path: Path to network JSON file
        output_dir: Output directory for visualization files
        dataset: Dataset name (mnist, cifar10)
        samples_per_class: Number of samples per class to process
        ticks: Number of ticks per image
        clustering_method: Method for clustering (firings or avg_S)
        logger: Optional logger

    Returns:
        Dict with status and generated file paths
    """
    log = logger or logging.getLogger(__name__)

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Build command
        script_path = Path(__file__).parent.parent.parent / "visualize_activity_3d.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--network",
            network_path,
            "--output",
            output_dir,
            "--dataset",
            dataset,
            "--samples-per-class",
            str(samples_per_class),
            "--ticks",
            str(ticks),
            "--clustering",
            clustering_method,
        ]

        log.info(f"Running 3D visualization: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Script failed: {result.stderr}",
                "files": [],
            }

        # Collect generated files recursively
        generated_files = []
        out_path_obj = Path(output_dir)
        for ext in ["*.png", "*.html", "*.svg", "*.pdf"]:
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


class Activity3DVisualizationStep(PipelineStep):
    """Visualization step for 3D brain visualization."""

    @property
    def name(self) -> str:
        return "visualize_activity_3d"

    @property
    def display_name(self) -> str:
        return "3D Brain Visualization"

    def run(self, context: StepContext) -> StepResult:
        start_time = datetime.now()
        logs = []

        try:
            config = context.config
            log = context.logger or logging.getLogger(__name__)

            # Get network from first step
            network_artifact = context.get_artifact("network", "network.json")
            if not network_artifact:
                raise ValueError("Network artifact not found")

            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)

            # Get config params
            dataset = config.get("dataset", "mnist")
            samples_per_class = config.get("samples_per_class", 5)
            ticks = config.get("ticks", 20)
            clustering_method = config.get("clustering_method", "firings")

            log.info(f"Generating 3D visualization with {clustering_method} clustering")
            logs.append(
                f"3D visualization: {dataset}, {samples_per_class} samples/class"
            )

            result = run_3d_visualization(
                network_path=str(network_artifact.path),
                output_dir=str(step_dir),
                dataset=dataset,
                samples_per_class=samples_per_class,
                ticks=ticks,
                clustering_method=clustering_method,
                logger=log,
            )

            if not result["success"]:
                raise RuntimeError(result.get("error", "Unknown error"))

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

            logs.append(f"Generated {len(artifacts)} files")

            return StepResult(
                status=StepStatus.COMPLETED,
                artifacts=artifacts,
                metrics={"num_files": len(artifacts)},
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
