"""
Network builder step for the pipeline.

Builds or loads neural networks based on configuration.
Refactored from interactive_training.py to be config-driven.
"""

import logging
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuron.network import NeuronNetwork
from neuron.neuron import Neuron, NeuronParameters
from neuron.network_config import NetworkConfig as NeuronNetworkConfig

from pipeline.config import (
    NetworkConfig,
    NetworkBuildConfig,
    LayerConfig,
    LayerType,
    DatasetType,
)
from pipeline.steps.base import (
    PipelineStep,
    StepContext,
    StepResult,
    StepStatus,
    Artifact,
    StepRegistry,
)


def get_dataset_dimensions(dataset: DatasetType) -> tuple[int, int, int]:
    """Get (channels, height, width) for a dataset.

    Returns:
        Tuple of (channels, height, width)
    """
    if dataset in (DatasetType.MNIST, DatasetType.FASHION_MNIST, DatasetType.USPS):
        return (1, 28, 28)
    elif dataset in (
        DatasetType.CIFAR10,
        DatasetType.CIFAR10_COLOR,
        DatasetType.CIFAR100,
        DatasetType.SVHN,
    ):
        return (3, 32, 32)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def build_network_from_config(
    build_config: NetworkBuildConfig, logger: logging.Logger | None = None
) -> NeuronNetwork:
    """Build a neural network from configuration.

    Args:
        build_config: Network building configuration
        logger: Optional logger for output

    Returns:
        Constructed NeuronNetwork instance
    """
    log = logger or logging.getLogger(__name__)

    # Get dataset dimensions
    channels, height, width = get_dataset_dimensions(build_config.dataset)
    is_colored = build_config.dataset == DatasetType.CIFAR10_COLOR
    rgb_separate = build_config.rgb_separate_neurons

    log.info(
        f"Building network for {build_config.dataset.value} "
        f"(C={channels}, H={height}, W={width})"
    )

    network_sim = NeuronNetwork(num_neurons=0, synapses_per_neuron=0)
    net_topology = network_sim.network
    all_layers: list[list[int]] = []

    # Tracking for conv layers
    prev_channels = channels
    prev_h = height
    prev_w = width
    prev_coord_to_id: dict[tuple[int, ...], int] | None = None
    conv_layer_idx = 0

    def create_neuron(
        layer_idx: int,
        layer_name: str,
        num_synapses: int,
        num_terminals: int = 10,
        metadata_extra: dict | None = None,
        synapse_distance_fn=None,
    ) -> int:
        """Create a neuron with defaults."""
        neuron_id = random.randint(0, 2**36 - 1)
        params = NeuronParameters(
            num_inputs=num_synapses,
            num_neuromodulators=2,
            r_base=np.random.uniform(0.9, 1.3),
            b_base=np.random.uniform(1.1, 1.5),
            c=10,
            lambda_param=20.0,
            p=1.0,
            delta_decay=0.96,
            beta_avg=0.999,
            eta_post=0.005,
            eta_retro=0.002,
            gamma=np.array([0.99, 0.995]),
            w_r=np.array([-0.2, 0.05]),
            w_b=np.array([-0.2, 0.05]),
            w_tref=np.array([-20.0, 10.0]),
        )
        metadata = {"layer": layer_idx, "layer_name": layer_name}
        if metadata_extra:
            metadata.update(metadata_extra)
        neuron = Neuron(
            neuron_id,
            params,
            log_level="CRITICAL",
            metadata=metadata,
        )
        for s_id in range(num_synapses):
            distance = (
                synapse_distance_fn(s_id)
                if synapse_distance_fn
                else random.randint(2, 8)
            )
            neuron.add_synapse(s_id, distance_to_hillock=distance)
        for t_id in range(num_terminals):
            neuron.add_axon_terminal(t_id, distance_from_hillock=random.randint(2, 8))
        net_topology.neurons[neuron_id] = neuron
        return neuron_id

    # Build conv or dense input layer based on first layer type
    if build_config.layers and build_config.layers[0].type == LayerType.CONV:
        # Conv-first network - build conv layers iteratively
        pass  # Will be handled in the loop below
    else:
        # Dense-first network - build explicit input layer
        input_size = 100  # Default, could be configurable
        if is_colored:
            pixels_per_image = height * width
            if rgb_separate:
                synapses_per_input = math.ceil(pixels_per_image / (input_size * 3))
                actual_input_neurons = input_size * 3
            else:
                synapses_per_input = math.ceil(pixels_per_image / input_size) * 3
                actual_input_neurons = input_size
        else:
            vector_size = channels * height * width
            synapses_per_input = math.ceil(vector_size / input_size)
            actual_input_neurons = input_size

        input_layer: list[int] = []
        for i in range(actual_input_neurons):
            metadata_extra = {}
            if is_colored and rgb_separate:
                metadata_extra = {
                    "color_channel": i % 3,
                    "spatial_idx": i // 3,
                }
            nid = create_neuron(
                0, "input", synapses_per_input, metadata_extra=metadata_extra
            )
            input_layer.append(nid)
        all_layers.append(input_layer)
        log.info(f"Created input layer with {len(input_layer)} neurons")

    # Build configured layers
    for li, cfg in enumerate(build_config.layers):
        if cfg.type == LayerType.CONV:
            k = cfg.kernel_size
            s = cfg.stride
            filters = cfg.filters
            connectivity = cfg.connectivity

            # Compute output dimensions
            out_h = int(math.floor((prev_h - k) / s) + 1)
            out_w = int(math.floor((prev_w - k) / s) + 1)

            if out_h <= 0 or out_w <= 0:
                raise ValueError(
                    f"Conv layer {li} produces invalid output: out_h={out_h}, out_w={out_w}"
                )

            # Synapses per conv neuron
            in_channels = prev_channels
            synapses_per_neuron = in_channels * k * k

            layer_neurons: list[int] = []
            coord_to_id: dict[tuple[int, ...], int] = {}

            for f_idx in range(filters):
                for y in range(out_h):
                    for x in range(out_w):

                        def conv_distance(s_id: int, k=k) -> int:
                            ky = (s_id // k) % k
                            kx = s_id % k
                            return max(1, k - max(ky, kx))

                        nid = create_neuron(
                            layer_idx=conv_layer_idx,
                            layer_name=f"conv_{conv_layer_idx}",
                            num_synapses=synapses_per_neuron,
                            metadata_extra={
                                "kernel_size": k,
                                "stride": s,
                                "in_channels": in_channels,
                                "y": y,
                                "x": x,
                                "filter": f_idx,
                            },
                            synapse_distance_fn=conv_distance,
                        )
                        layer_neurons.append(nid)
                        coord_to_id[(f_idx, y, x)] = nid

            all_layers.append(layer_neurons)
            log.info(
                f"Created conv layer {conv_layer_idx}: {len(layer_neurons)} neurons "
                f"(filters={filters}, out_h={out_h}, out_w={out_w})"
            )

            # Mark external inputs for first conv layer
            if conv_layer_idx == 0:
                for nid in layer_neurons:
                    neuron = net_topology.neurons[nid]
                    for s_id in neuron.postsynaptic_points:
                        net_topology.external_inputs[(nid, s_id)] = {
                            "info": 0.0,
                            "mod": np.array([0.0, 0.0]),
                        }

            # Connect to previous layer if exists
            if prev_coord_to_id is not None and connectivity > 0:
                for coord, nid in coord_to_id.items():
                    target_neuron = net_topology.neurons[nid]
                    # Connect from overlapping positions in previous layer
                    f_idx, y, x = coord
                    for prev_coord, prev_nid in prev_coord_to_id.items():
                        if random.random() < connectivity:
                            source_neuron = net_topology.neurons[prev_nid]
                            source_terminal = random.randint(
                                0, len(source_neuron.presynaptic_points) - 1
                            )
                            target_synapse = random.randint(
                                0, len(target_neuron.postsynaptic_points) - 1
                            )
                            connection = (
                                prev_nid,
                                source_terminal,
                                nid,
                                target_synapse,
                            )
                            if connection not in net_topology.connections:
                                net_topology.connections.append(connection)

            prev_coord_to_id = coord_to_id
            prev_channels = filters
            prev_h = out_h
            prev_w = out_w
            conv_layer_idx += 1

        else:  # Dense layer
            size = cfg.size
            connectivity = cfg.connectivity

            # Determine synapses per neuron
            if cfg.synapses_per is not None:
                synapses_per = cfg.synapses_per
            elif prev_coord_to_id is not None:
                # Transitioning from conv to dense
                synapses_per = min(len(prev_coord_to_id), 256)
            else:
                synapses_per = 10

            layer_neurons = []
            for _ in range(size):
                nid = create_neuron(
                    layer_idx=len(all_layers),
                    layer_name=f"dense_{len(all_layers)}",
                    num_synapses=synapses_per,
                )
                layer_neurons.append(nid)

            all_layers.append(layer_neurons)
            log.info(f"Created dense layer: {len(layer_neurons)} neurons")

            # Connect from previous layer
            if len(all_layers) > 1:
                prev_layer = all_layers[-2]
                for target_nid in layer_neurons:
                    target_neuron = net_topology.neurons[target_nid]
                    for source_nid in prev_layer:
                        if random.random() < connectivity:
                            source_neuron = net_topology.neurons[source_nid]
                            source_terminal = random.randint(
                                0, len(source_neuron.presynaptic_points) - 1
                            )
                            target_synapse = random.randint(
                                0, len(target_neuron.postsynaptic_points) - 1
                            )
                            connection = (
                                source_nid,
                                source_terminal,
                                target_nid,
                                target_synapse,
                            )
                            if connection not in net_topology.connections:
                                net_topology.connections.append(connection)

            prev_coord_to_id = None  # No longer in conv mode

    log.info(
        f"Network built with {len(net_topology.neurons)} neurons and "
        f"{len(net_topology.connections)} connections"
    )

    return network_sim


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
                # Load existing network
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

                network = build_network_from_config(config.build_config, logger=log)
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

        except Exception as e:
            return StepResult(
                status=StepStatus.FAILED,
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs + [f"ERROR: {e}"],
            )
