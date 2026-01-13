"""
Shared utility functions for pipeline steps.
Extracted from step5_evaluate.py to reduce code duplication.
"""
import torch
import numpy as np
from typing import List, Tuple
from neuron.network import NeuronNetwork


def infer_layers_from_metadata(network_sim: NeuronNetwork) -> List[List[int]]:
    """Group neurons by their 'layer' metadata if present; otherwise fall back to a single layer list in ID order."""
    net = network_sim.network
    layer_to_neurons: dict[int, List[int]] = {}
    for nid, neuron in net.neurons.items():
        layer_idx = int(neuron.metadata.get("layer", 0))
        layer_to_neurons.setdefault(layer_idx, []).append(nid)
    return [layer_to_neurons[k] for k in sorted(layer_to_neurons.keys())]


def determine_input_mapping(network_sim: NeuronNetwork, layers: List[List[int]]) -> Tuple[List[int], int]:
    """Return (input_layer_ids, input_synapses_per_neuron). Assumes first layer is inputs."""
    if not layers or not layers[0]:
        raise ValueError("Cannot determine input layer from network metadata.")
    input_layer_ids = layers[0]
    first_input_neuron = network_sim.network.neurons[input_layer_ids[0]]
    return input_layer_ids, len(first_input_neuron.postsynaptic_points)


def image_to_signals(
    image_tensor: torch.Tensor,
    network_sim: NeuronNetwork,
    input_layer_ids: List[int],
    synapses_per_neuron: int
) -> List[Tuple[int, int, float]]:
    """Map an image to (neuron_id, synapse_id, strength) signals.
    
    Supports legacy dense input and CNN-style input (conv layer at index 0).
    Signals are normalized to [0, 1] range.
    """
    first_neuron = network_sim.network.neurons[input_layer_ids[0]]
    meta = getattr(first_neuron, "metadata", {}) or {}
    is_cnn_input = meta.get("layer_type") == "conv" and meta.get("layer", 0) == 0

    if not is_cnn_input:
        # Simplified dense mapping
        img_vec = image_tensor.view(-1).numpy()
        img_vec = (img_vec + 1.0) * 0.5
        signals = []
        num_input_neurons = len(input_layer_ids)
        for i, pixel_value in enumerate(img_vec):
            neuron_idx = i % num_input_neurons
            synapse_idx = i // num_input_neurons
            if synapse_idx < synapses_per_neuron:
                neuron_id = input_layer_ids[neuron_idx]
                signals.append((neuron_id, synapse_idx, float(pixel_value)))
        return signals

    # CNN mapping
    arr = image_tensor.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]

    signals = []
    for neuron_id in input_layer_ids:
        neuron = network_sim.network.neurons[neuron_id]
        m = getattr(neuron, "metadata", {}) or {}
        k = int(m.get("kernel_size", 1))
        s = int(m.get("stride", 1))
        in_c = int(m.get("in_channels", arr.shape[0]))
        y_out = int(m.get("y", 0))
        x_out = int(m.get("x", 0))
        for c in range(in_c):
            for ky in range(k):
                for kx in range(k):
                    in_y = y_out * s + ky
                    in_x = x_out * s + kx
                    if in_y >= arr.shape[1] or in_x >= arr.shape[2]:
                        continue
                    syn_id = (c * k + ky) * k + kx
                    strength = (float(arr[c, in_y, in_x]) + 1.0) * 0.5
                    signals.append((neuron_id, syn_id, strength))
    return signals
