"""
Synaptic connections analysis visualization step.

Wraps synaptic_connections_analysis.py to analyze network connectivity clusters.
"""

import logging
import subprocess
import sys
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


def run_synaptic_analysis(
    network_states_dir: str,
    output_dir: str,
    method: str = "louvain",
    n_clusters: int = 5,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run synaptic connections analysis.

    Args:
        network_states_dir: Path to directory with network state exports
        output_dir: Output directory for plots
        method: Clustering method (louvain, spectral, hierarchical)
        n_clusters: Number of clusters for spectral method
        logger: Optional logger

    Returns:
        Dict with status and generated file paths
    """
    log = logger or logging.getLogger(__name__)

    try:
        import os

        os.makedirs(output_dir, exist_ok=True)

        script_path = (
            Path(__file__).parent.parent.parent / "synaptic_connections_analysis.py"
        )

        cmd = [
            sys.executable,
            str(script_path),
            network_states_dir,
            "--output",
            output_dir,
            "--method",
            method,
            "--n-clusters",
            str(n_clusters),
        ]

        log.info(f"Running synaptic analysis: {' '.join(cmd)}")

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

        # Collect generated files recursively
        generated_files = []
        out_path_obj = Path(output_dir)
        for ext in ["*.png", "*.html", "*.svg", "*.pdf"]:
            for f in out_path_obj.rglob(ext):
                generated_files.append(str(f.relative_to(out_path_obj)))

        return {"success": True, "files": list(set(generated_files))}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Analysis timed out", "files": []}
    except Exception as e:
        import traceback

        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "files": [],
        }


class SynapticAnalysisStep(PipelineStep):
    """Visualization step for synaptic connectivity analysis."""

    @property
    def name(self) -> str:
        return "synaptic_analysis"

    @property
    def display_name(self) -> str:
        return "Synaptic Connectivity Analysis"

    def run(self, context: StepContext) -> StepResult:
        start_time = datetime.now()
        logs = []

        try:
            config = context.config
            log = context.logger or logging.getLogger(__name__)

            # Get network states directory from activity recording
            activity_dir = context.output_dir / "activity_recording"
            network_states_dir = activity_dir / "network_states"

            if not network_states_dir.exists():
                # Check if export_network_states was enabled
                raise ValueError(
                    "Network states not found. Enable 'export_network_states: true' "
                    "in activity_recording config to use synaptic analysis."
                )

            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)

            method = config.get("method", "louvain") if config else "louvain"
            n_clusters = config.get("n_clusters", 5) if config else 5

            log.info(f"Running synaptic analysis with {method} method")
            logs.append(f"Analyzing synaptic connectivity with {method}")

            result = run_synaptic_analysis(
                network_states_dir=str(network_states_dir),
                output_dir=str(step_dir),
                method=method,
                n_clusters=n_clusters,
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

            logs.append(f"Generated {len(artifacts)} analysis files")

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
