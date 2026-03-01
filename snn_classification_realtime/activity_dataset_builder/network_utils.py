"""Network topology and compatibility utilities."""

import math
import sys
from typing import TYPE_CHECKING

from neuron.network import NeuronNetwork

from snn_classification_realtime.core.network_utils import (
    infer_layers_from_metadata,
    determine_input_mapping,
)

if TYPE_CHECKING:
    from snn_classification_realtime.core.config import DatasetConfig


def ceil_to_nearest_ten(value: int) -> int:
    """Round value up to nearest multiple of 10."""
    return int(math.ceil(value / 10.0) * 10)


def compute_default_ticks_per_image(
    network_sim: NeuronNetwork,
    layers: list[list[int]],
) -> int:
    """Estimate propagation time based on network connectivity."""
    net = network_sim.network
    out_deg: dict[int, int] = {}
    in_deg: dict[int, int] = {}
    for src, _src_term, dst, _dst_syn in net.connections:
        out_deg[src] = out_deg.get(src, 0) + 1
        in_deg[dst] = in_deg.get(dst, 0) + 1

    total = 0
    for layer in layers:
        max_in = max((in_deg.get(nid, 0) for nid in layer), default=0)
        max_out = max((out_deg.get(nid, 0) for nid in layer), default=0)
        total += max_in + max_out
    return max(ceil_to_nearest_ten(max(10, total)), 10)


def _compute_input_capacity(
    sim: NeuronNetwork,
    input_ids: list[int] | None,
) -> tuple[int, list[int], int]:
    """Compute network input capacity from input layer."""
    if not input_ids:
        layers = infer_layers_from_metadata(sim)
        input_ids = (
            layers[0]
            if layers and layers[0]
            else list(sim.network.neurons.keys())[:100]
        )
    syn_counts = [
        len(sim.network.neurons[nid].postsynaptic_points) for nid in input_ids
    ]
    syn_per = min(syn_counts) if syn_counts else 0
    capacity = len(input_ids) * syn_per
    return capacity, input_ids, syn_per


def pre_run_compatibility_check(
    network_sim: NeuronNetwork,
    input_layer_ids: list[int] | None,
    dataset_config: "DatasetConfig",
) -> tuple[NeuronNetwork, list[int], "DatasetConfig"]:
    """Ensure dataset vector size fits network input capacity.

    If incompatible, prints reason and details for the agent, then exits.
    Returns (network_sim, input_layer_ids, dataset_config).
    """
    if not input_layer_ids:
        layers = infer_layers_from_metadata(network_sim)
        input_layer_ids = (
            layers[0]
            if layers and layers[0]
            else list(network_sim.network.neurons.keys())[:100]
        )

    capacity, input_layer_ids_eff, syn_per = _compute_input_capacity(
        network_sim, input_layer_ids
    )
    vec = dataset_config.image_vector_size
    if capacity >= vec and syn_per > 0 and len(input_layer_ids_eff) > 0:
        return network_sim, input_layer_ids_eff, dataset_config

    print("Dataset/network input compatibility check failed.")
    print(
        f"  Reason: dataset vector size ({vec}) exceeds network input capacity ({capacity})"
    )
    print(
        f"  Details: input_layer_neurons={len(input_layer_ids_eff)}, "
        f"synapses_per_neuron={syn_per}, capacity={len(input_layer_ids_eff)} × {syn_per} = {capacity}"
    )
    print(f"  Dataset: {dataset_config.dataset_name}, image_vector_size={vec}")
    print(
        "  Fix: use a network with larger input capacity, or a dataset with smaller image size."
    )
    sys.exit(1)
