"""Map image tensors to neural network input signals."""

from typing import Any

import numpy as np
import torch

from neuron.network import NeuronNetwork

from activity_dataset_builder.config import DatasetConfig


def image_to_signals(
    image_tensor: torch.Tensor,
    input_layer_ids: list[int],
    input_synapses_per_neuron: int,
    network_sim: NeuronNetwork,
    dataset_config: DatasetConfig,
) -> list[tuple[int, int, float]]:
    """Map an image to (neuron_id, synapse_id, strength) signals.

    Supports legacy dense input, colored CIFAR-10, and CNN-style input.
    Signals are normalized to [0, 1] range.
    """
    first_neuron = network_sim.network.neurons[input_layer_ids[0]]
    meta = getattr(first_neuron, "metadata", {}) or {}
    is_cnn_input = meta.get("layer_type") == "conv" and meta.get("layer", 0) == 0

    if not is_cnn_input:
        if dataset_config.is_colored_cifar10:
            return _image_to_signals_colored_cifar10(
                image_tensor,
                input_layer_ids,
                input_synapses_per_neuron,
                network_sim,
                dataset_config,
            )
        return _image_to_signals_dense(
            image_tensor,
            input_layer_ids,
            input_synapses_per_neuron,
        )

    return _image_to_signals_cnn(
        image_tensor,
        input_layer_ids,
        network_sim,
    )


def _image_to_signals_dense(
    image_tensor: torch.Tensor,
    input_layer_ids: list[int],
    input_synapses_per_neuron: int,
) -> list[tuple[int, int, float]]:
    """Legacy dense mapping: one pixel per synapse."""
    img_vec = image_tensor.view(-1).numpy().astype(np.float32)
    img_vec = (img_vec + 1.0) * 0.5  # Normalize from [-1, 1] to [0, 1]
    num_input_neurons = len(input_layer_ids)
    signals: list[tuple[int, int, float]] = []
    for pixel_index, pixel_value in enumerate(img_vec):
        target_neuron_index = pixel_index % num_input_neurons
        target_synapse_index = pixel_index // num_input_neurons
        target_synapse_index = min(
            target_synapse_index, input_synapses_per_neuron - 1
        )
        neuron_id = input_layer_ids[target_neuron_index]
        strength = float(pixel_value)
        signals.append((neuron_id, target_synapse_index, strength))
    return signals


def _image_to_signals_colored_cifar10(
    image_tensor: torch.Tensor,
    input_layer_ids: list[int],
    input_synapses_per_neuron: int,
    network_sim: NeuronNetwork,
    dataset_config: DatasetConfig,
) -> list[tuple[int, int, float]]:
    """Colored CIFAR-10: 3 synapses per pixel (RGB) or separate neurons per channel."""
    arr = image_tensor.detach().cpu().numpy().astype(np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        return _image_to_signals_dense(
            image_tensor, input_layer_ids, input_synapses_per_neuron
        )

    h, w = arr.shape[1], arr.shape[2]
    signals: list[tuple[int, int, float]] = []
    norm_factor = dataset_config.cifar10_color_normalization_factor

    first_neuron = network_sim.network.neurons[input_layer_ids[0]]
    meta = getattr(first_neuron, "metadata", {}) or {}
    separate_neurons_per_color = "color_channel" in meta

    if separate_neurons_per_color:
        total_spatial_positions = h * w
        neurons_per_color = len(input_layer_ids) // 3
        for y in range(h):
            for x in range(w):
                pixel_index = y * w + x
                for c in range(3):
                    spatial_neuron_idx = pixel_index % neurons_per_color
                    global_neuron_idx = c * neurons_per_color + spatial_neuron_idx
                    if global_neuron_idx >= len(input_layer_ids):
                        continue
                    pixels_per_neuron = total_spatial_positions // neurons_per_color
                    synapse_index = pixel_index // neurons_per_color
                    neuron_id = input_layer_ids[global_neuron_idx]
                    pixel_value = arr[c, y, x]
                    strength = (float(pixel_value) + 1.0) * norm_factor
                    signals.append((neuron_id, synapse_index, strength))
    else:
        for y in range(h):
            for x in range(w):
                for c in range(3):
                    pixel_index = y * w + x
                    target_neuron_index = pixel_index % len(input_layer_ids)
                    pixels_per_neuron = (h * w) // len(input_layer_ids)
                    if pixel_index // len(input_layer_ids) >= pixels_per_neuron:
                        continue
                    base_synapse_index = (pixel_index // len(input_layer_ids)) * 3
                    synapse_index = base_synapse_index + c
                    if synapse_index >= input_synapses_per_neuron:
                        continue
                    neuron_id = input_layer_ids[target_neuron_index]
                    pixel_value = arr[c, y, x]
                    strength = (float(pixel_value) + 1.0) * norm_factor
                    signals.append((neuron_id, synapse_index, strength))
    return signals


def _image_to_signals_cnn(
    image_tensor: torch.Tensor,
    input_layer_ids: list[int],
    network_sim: NeuronNetwork,
) -> list[tuple[int, int, float]]:
    """CNN input: one neuron per kernel position; synapses map to receptive field."""
    arr = image_tensor.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape for CNN input: {arr.shape}")

    signals: list[tuple[int, int, float]] = []
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
