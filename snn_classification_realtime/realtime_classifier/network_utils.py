"""Network topology utilities for real-time classification."""

from neuron.network import NeuronNetwork


def infer_layers_from_metadata(network_sim: NeuronNetwork) -> list[list[int]]:
    """Group neurons by their 'layer' metadata."""
    net = network_sim.network
    layer_to_neurons: dict[int, list[int]] = {}
    for nid, neuron in net.neurons.items():
        layer_idx = int(neuron.metadata.get("layer", 0))
        layer_to_neurons.setdefault(layer_idx, []).append(nid)
    return [layer_to_neurons[k] for k in sorted(layer_to_neurons.keys())]


def determine_input_mapping(
    network_sim: NeuronNetwork,
    layers: list[list[int]],
) -> tuple[list[int], int]:
    """Determine input layer and synapses per neuron."""
    if not layers or not layers[0]:
        raise ValueError("Cannot determine input layer from network metadata.")
    input_layer_ids = layers[0]
    first_input_neuron = network_sim.network.neurons[input_layer_ids[0]]
    return input_layer_ids, len(first_input_neuron.postsynaptic_points)
