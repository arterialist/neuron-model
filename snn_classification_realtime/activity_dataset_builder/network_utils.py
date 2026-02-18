"""Network topology and compatibility utilities."""

import math
from typing import TYPE_CHECKING

from neuron.network import NeuronNetwork
from neuron.network_config import NetworkConfig

from snn_classification_realtime.core.network_utils import (
    infer_layers_from_metadata,
    determine_input_mapping,
)
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    select_and_load_dataset,
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

    Optionally prompts to change dataset or network.
    Returns (network_sim, input_layer_ids, dataset_config).
    """
    run_check = (
        input("Run dataset/network input compatibility check? (y/n) [y]: ")
        .strip()
        .lower()
        or "y"
    )
    if run_check == "n":
        if not input_layer_ids:
            layers = infer_layers_from_metadata(network_sim)
            input_layer_ids = (
                layers[0]
                if layers and layers[0]
                else list(network_sim.network.neurons.keys())[:100]
            )
        return network_sim, input_layer_ids, dataset_config

    while True:
        capacity, input_layer_ids_eff, syn_per = _compute_input_capacity(
            network_sim, input_layer_ids
        )
        vec = dataset_config.image_vector_size
        if capacity >= vec and syn_per > 0 and len(input_layer_ids_eff) > 0:
            print(
                f"Compatibility OK: capacity={capacity} (inputs={len(input_layer_ids_eff)} × synapses={syn_per}) >= vector={vec}"
            )
            return network_sim, input_layer_ids_eff, dataset_config

        print(
            f"Incompatible: dataset vector={vec} > network capacity={capacity} "
            f"(inputs={len(input_layer_ids_eff)} × synapses={syn_per})."
        )
        choice = (
            input(
                "Choose action: [d] change dataset, [n] change network, [i] ignore and proceed [n]: "
            )
            .strip()
            .lower()
            or "n"
        )
        if choice == "i":
            print("Proceeding with mismatch; many pixels may be dropped or aliased.")
            return network_sim, input_layer_ids_eff, dataset_config
        if choice == "d":
            try:
                dataset_config = select_and_load_dataset()
            except Exception as e:
                print(f"Failed to load dataset: {e}")
                continue
            continue
        if choice == "n":
            sub = (
                input("[l] load network file, [b] rebuild network [l]: ")
                .strip()
                .lower()
                or "l"
            )
            if sub == "l":
                path = input("Enter network file path: ").strip()
                try:
                    network_sim = NetworkConfig.load_network_config(path)
                except Exception as e:
                    print(f"Failed to load network: {e}")
                    continue
            else:
                print(
                    "Rebuild not supported in this script; please load a compatible network."
                )
            input_layer_ids = None
            continue
