"""
Neuron clustering visualization step.

Wraps cluster_neurons.py to generate neuron clustering visualizations.
"""

import logging
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.steps.base import (
    PipelineStep,
    StepContext,
    StepResult,
    StepStatus,
    Artifact,
)


def run_neuron_clustering(
    data_path: str,
    output_dir: str,
    method: str = "hierarchical",
    num_clusters: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run neuron clustering visualization.

    Args:
        data_path: Path to activity dataset
        output_dir: Output directory for clustering results
        method: Clustering method (hierarchical, dbscan, kmeans)
        num_clusters: Number of clusters (optional)
        logger: Optional logger

    Returns:
        Dict with status and generated file paths
    """
    log = logger or logging.getLogger(__name__)

    try:
        os.makedirs(output_dir, exist_ok=True)

        script_path = (
            Path(__file__).parent.parent.parent
            / "snn_classification_realtime"
            / "cluster_neurons.py"
        )

        cmd = [
            sys.executable,
            str(script_path),
            "--input-file",
            data_path,
            "--output-dir",
            output_dir,
            "--clustering-mode",
            method,
        ]

        if num_clusters:
            cmd.extend(["--num-clusters", str(num_clusters)])

        log.info(f"Running neuron clustering: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
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
        return {"success": False, "error": "Clustering timed out", "files": []}
    except Exception as e:
        import traceback

        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "files": [],
        }


class NeuronClusteringStep(PipelineStep):
    """Visualization step for neuron clustering."""

    @property
    def name(self) -> str:
        return "cluster_neurons"

    @property
    def display_name(self) -> str:
        return "Neuron Clustering"

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

            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)

            method = config.get("method", "hierarchical")
            num_clusters = config.get("num_clusters")

            log.info(f"Running neuron clustering with {method} method")
            logs.append(f"Clustering neurons with {method}")

            result = run_neuron_clustering(
                data_path=data_path,
                output_dir=str(step_dir),
                method=method,
                num_clusters=num_clusters,
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

            logs.append(f"Generated {len(artifacts)} clustering files")

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
