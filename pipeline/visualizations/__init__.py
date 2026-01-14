"""
Main visualization step for the pipeline.

Orchestrates all visualization types based on config.
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

# Import visualization sub-steps
from pipeline.visualizations.activity_dataset import (
    ActivityDatasetVisualizationStep,
    run_activity_dataset_visualization,
    AVAILABLE_PLOTS as ACTIVITY_DATASET_PLOTS,
)
from pipeline.visualizations.network_activity import (
    NetworkActivityVisualizationStep,
    run_network_activity_visualization,
    AVAILABLE_PLOTS as NETWORK_ACTIVITY_PLOTS,
)
from pipeline.visualizations.activity_3d import (
    Activity3DVisualizationStep,
    run_3d_visualization,
)
from pipeline.visualizations.cluster_neurons import (
    NeuronClusteringStep,
    run_neuron_clustering,
)
from pipeline.visualizations.cluster_activity import (
    ActivityClusteringStep,
    run_activity_clustering,
)
from pipeline.visualizations.concept_hierarchy import (
    ConceptHierarchyStep,
    run_concept_hierarchy,
)
from pipeline.visualizations.synaptic_analysis import (
    SynapticAnalysisStep,
    run_synaptic_analysis,
)


# Registry of available visualization types and their plots
VISUALIZATION_TYPES = {
    "activity_dataset": {
        "step_class": ActivityDatasetVisualizationStep,
        "run_func": run_activity_dataset_visualization,
        "available_plots": ACTIVITY_DATASET_PLOTS,
        "description": "Activity dataset visualizations (firing rates, S values, time series)",
    },
    "network_activity": {
        "step_class": NetworkActivityVisualizationStep,
        "run_func": run_network_activity_visualization,
        "available_plots": NETWORK_ACTIVITY_PLOTS,
        "description": "Network activity visualizations (heatmaps, spike rasters, correlations)",
    },
    "activity_3d": {
        "step_class": Activity3DVisualizationStep,
        "run_func": run_3d_visualization,
        "available_plots": [],
        "description": "3D brain visualization with UMAP clustering",
    },
    "cluster_neurons": {
        "step_class": NeuronClusteringStep,
        "run_func": run_neuron_clustering,
        "available_plots": [],
        "description": "Neuron clustering analysis and visualization",
    },
    "cluster_activity": {
        "step_class": ActivityClusteringStep,
        "run_func": run_activity_clustering,
        "available_plots": [],
        "description": "Activity data clustering analysis",
    },
    "concept_hierarchy": {
        "step_class": ConceptHierarchyStep,
        "run_func": run_concept_hierarchy,
        "available_plots": [],
        "description": "Concept hierarchy dendrograms from activity data",
    },
    "synaptic_analysis": {
        "step_class": SynapticAnalysisStep,
        "run_func": run_synaptic_analysis,
        "available_plots": [],
        "description": "Synaptic connectivity cluster analysis",
    },
}


@StepRegistry.register
class VisualizationsStep(PipelineStep):
    """Main visualization step that runs all configured visualizations."""

    @property
    def name(self) -> str:
        return "visualizations"

    @property
    def display_name(self) -> str:
        return "Visualizations"

    def run(self, context: StepContext) -> StepResult:
        start_time = datetime.now()
        logs = []
        all_artifacts = []

        try:
            config = context.config
            log = context.logger or logging.getLogger(__name__)

            # Check if visualizations are enabled
            if not config or not config.enabled:
                log.info("Visualizations disabled in config")
                return StepResult(
                    status=StepStatus.SKIPPED,
                    start_time=start_time,
                    end_time=datetime.now(),
                    logs=["Visualizations disabled in config"],
                )

            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)

            # Get artifacts from previous steps
            network_artifact = context.get_artifact("network", "network.json")
            activity_artifacts = context.previous_artifacts.get(
                "activity_recording", []
            )

            if not activity_artifacts:
                raise ValueError("Activity recording artifact not found")

            activity_path = str(activity_artifacts[0].path)
            network_path = str(network_artifact.path) if network_artifact else None

            # Process each visualization type
            for viz_config in config.types:
                viz_name = viz_config.name
                plots = viz_config.plots
                params = viz_config.params

                if viz_name not in VISUALIZATION_TYPES:
                    log.warning(f"Unknown visualization type: {viz_name}")
                    logs.append(f"Unknown visualization type: {viz_name}")
                    continue

                viz_info = VISUALIZATION_TYPES[viz_name]
                viz_dir = step_dir / viz_name
                viz_dir.mkdir(parents=True, exist_ok=True)

                log.info(f"Running visualization: {viz_name}")
                logs.append(f"Running {viz_name}")

                try:
                    if viz_name == "activity_dataset":
                        result = run_activity_dataset_visualization(
                            data_path=activity_path,
                            output_dir=str(viz_dir),
                            plot_types=plots or ACTIVITY_DATASET_PLOTS[:3],
                            logger=log,
                        )

                    elif viz_name == "network_activity":
                        result = run_network_activity_visualization(
                            data_path=activity_path,
                            output_dir=str(viz_dir),
                            plot_types=plots or NETWORK_ACTIVITY_PLOTS[:3],
                            logger=log,
                        )

                    elif viz_name == "activity_3d":
                        if network_path:
                            result = run_3d_visualization(
                                network_path=network_path,
                                output_dir=str(viz_dir),
                                dataset=params.get("dataset", "mnist"),
                                samples_per_class=params.get("samples_per_class", 5),
                                ticks=params.get("ticks", 20),
                                clustering_method=params.get(
                                    "clustering_method", "firings"
                                ),
                                logger=log,
                            )
                        else:
                            result = {
                                "success": False,
                                "error": "No network artifact",
                                "files": [],
                            }

                    elif viz_name == "cluster_neurons":
                        result = run_neuron_clustering(
                            data_path=activity_path,
                            output_dir=str(viz_dir),
                            method=params.get("method", "hierarchical"),
                            num_clusters=params.get("num_clusters"),
                            logger=log,
                        )

                    elif viz_name == "cluster_activity":
                        result = run_activity_clustering(
                            data_path=activity_path,
                            output_dir=str(viz_dir),
                            feature_types=params.get(
                                "feature_types", ["firings", "avg_S"]
                            ),
                            method=params.get("method", "dbscan"),
                            logger=log,
                        )

                    elif viz_name == "concept_hierarchy":
                        result = run_concept_hierarchy(
                            data_path=activity_path,
                            output_dir=str(viz_dir),
                            logger=log,
                        )

                    elif viz_name == "synaptic_analysis":
                        network_states_dir = (
                            context.output_dir / "activity_recording" / "network_states"
                        )
                        if network_states_dir.exists():
                            result = run_synaptic_analysis(
                                network_states_dir=str(network_states_dir),
                                output_dir=str(viz_dir),
                                method=params.get("method", "louvain"),
                                n_clusters=params.get("n_clusters", 5),
                                logger=log,
                            )
                        else:
                            result = {
                                "success": False,
                                "error": "Network states not found. Enable export_network_states.",
                                "files": [],
                            }

                    else:
                        result = {
                            "success": False,
                            "error": f"Not implemented: {viz_name}",
                            "files": [],
                        }

                    # Collect artifacts
                    for fname in result.get("files", []):
                        fpath = viz_dir / fname
                        if fpath.exists():
                            # Use absolute path for registration to ensure orchestrator finds it
                            all_artifacts.append(
                                Artifact(
                                    name=f"{viz_name}/{fname}",
                                    path=fpath.absolute(),
                                    artifact_type="visualization",
                                    size_bytes=fpath.stat().st_size,
                                )
                            )

                    if result.get("success"):
                        logs.append(f"  Generated {len(result.get('files', []))} files")
                    else:
                        logs.append(
                            f"  Warning: {result.get('error', 'Unknown error')}"
                        )

                except Exception as e:
                    log.warning(f"Visualization {viz_name} failed: {e}")
                    logs.append(f"  Error: {e}")

            logs.append(f"Total visualizations generated: {len(all_artifacts)}")

            return StepResult(
                status=StepStatus.COMPLETED,
                artifacts=all_artifacts,
                metrics={"total_visualizations": len(all_artifacts)},
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


def list_available_visualizations() -> Dict[str, Dict[str, Any]]:
    """List all available visualization types and their plots."""
    return {
        name: {
            "description": info["description"],
            "available_plots": info["available_plots"],
        }
        for name, info in VISUALIZATION_TYPES.items()
    }
