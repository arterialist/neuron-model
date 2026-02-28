"""
Network builder step for the pipeline.

Builds or loads neural networks based on configuration.
Uses snn_classification_realtime.network_builder for building.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuron.network_config import NetworkConfig as NeuronNetworkConfig

from pipeline.config import NetworkConfig
from pipeline.steps.base import (
    PipelineStep,
    StepContext,
    StepResult,
    StepStatus,
    Artifact,
    StepRegistry,
    StepCancelledException,
)
from snn_classification_realtime.network_builder import build_network


@StepRegistry.register
class NetworkBuilderStep(PipelineStep):
    """Pipeline step for building or loading neural networks."""

    @property
    def name(self) -> str:
        return "network"

    @property
    def display_name(self) -> str:
        return "Network Builder"

    def run(self, context: StepContext) -> StepResult:
        start_time = datetime.now()
        logs: list[str] = []

        try:
            config: NetworkConfig = context.config
            log = context.logger or logging.getLogger(__name__)

            # Create output directory for this step
            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)

            if config.source == "file":
                # Load existing network (path is required when source is "file")
                assert config.path is not None
                network_path = Path(config.path)
                if not network_path.is_absolute():
                    # Try relative to project root
                    project_root = Path(__file__).parent.parent.parent
                    network_path = project_root / config.path

                if not network_path.exists():
                    raise FileNotFoundError(f"Network file not found: {network_path}")

                log.info(f"Loading network from {network_path}")
                logs.append(f"Loading network from {network_path}")

                # Copy network to output directory
                output_path = step_dir / "network.json"
                import shutil

                shutil.copy(network_path, output_path)

                network = NeuronNetworkConfig.load_network_config(str(output_path))
                num_neurons = len(network.network.neurons)
                num_connections = len(network.network.connections)

            else:  # source == "build"
                log.info("Building network from configuration")
                logs.append("Building network from configuration")

                network = build_network(config.build_config, logger=log)
                output_path = step_dir / "network.json"
                NeuronNetworkConfig.save_network_config(network, str(output_path))

                num_neurons = len(network.network.neurons)
                num_connections = len(network.network.connections)

            logs.append(
                f"Network has {num_neurons} neurons and {num_connections} connections"
            )
            log.info(f"Network saved to {output_path}")
            logs.append(f"Network saved to {output_path}")

            # Create artifact
            artifact = Artifact(
                name="network.json",
                path=output_path,
                artifact_type="model",
                size_bytes=output_path.stat().st_size,
                metadata={
                    "num_neurons": num_neurons,
                    "num_connections": num_connections,
                    "source": config.source,
                },
            )

            return StepResult(
                status=StepStatus.COMPLETED,
                artifacts=[artifact],
                metrics={
                    "num_neurons": num_neurons,
                    "num_connections": num_connections,
                },
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs,
            )

        except StepCancelledException:
            raise
        except Exception as e:
            return StepResult(
                status=StepStatus.FAILED,
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs + [f"ERROR: {e}"],
            )
