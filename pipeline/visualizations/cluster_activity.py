"""
Activity data clustering visualization step.

Wraps cluster_activity_data.py to generate activity clustering visualizations.
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
)


def run_activity_clustering(
    data_path: str,
    output_dir: str,
    feature_types: List[str] = None,
    method: str = "dbscan",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run activity clustering visualization.

    Args:
        data_path: Path to activity dataset
        output_dir: Output directory for clustering results
        feature_types: Features to use for clustering
        method: Clustering method
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
            / "cluster_activity_data.py"
        )

        if feature_types is None:
            feature_types = ["firings", "avg_S"]

        cmd = [
            sys.executable,
            str(script_path),
            "--input-file",
            data_path,
            "--output-dir",
            output_dir,
            "--feature-types",
            ",".join(feature_types),
            "--clustering-mode",
            method,
        ]

        log.info(f"Running activity clustering: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Script failed: {result.stderr}",
                "files": [],
            }

        # Collect generated files
        generated_files = []
        for f in Path(output_dir).glob("*.html"):
            generated_files.append(f.name)
        for f in Path(output_dir).glob("*.png"):
            generated_files.append(f.name)

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


class ActivityClusteringStep(PipelineStep):
    """Visualization step for activity data clustering."""

    @property
    def name(self) -> str:
        return "cluster_activity"

    @property
    def display_name(self) -> str:
        return "Activity Clustering"

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

            feature_types = config.get("feature_types", ["firings", "avg_S"])
            method = config.get("method", "dbscan")

            log.info(f"Running activity clustering with {method}")
            logs.append(f"Clustering activity data with {method}")

            result = run_activity_clustering(
                data_path=data_path,
                output_dir=str(step_dir),
                feature_types=feature_types,
                method=method,
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
