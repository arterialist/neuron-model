import os
import sys
import time
import argparse
import json
import torch
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
from typing import Dict, Any, List

# Disable loguru logging (used by neuron modules)
from loguru import logger

logger.remove()  # Remove all handlers
logger.add(lambda msg: None)  # Add a no-op handler

# Ensure local imports resolve
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron.nn_core import NNCore
from neuron.network import NeuronNetwork
from neuron.network_config import NetworkConfig
from snn_classification_realtime.fusion_classifier import (
    SNN_HIDDEN_SIZE,
    FusionSNNClassifier,
)


def apply_scaling_to_snapshot(
    snapshot: List[float], scaler_state: Dict[str, Any]
) -> List[float]:
    """Applies saved FeatureScaler state (from prepare_activity_data) to a 1D snapshot.

    The scaler_state is a dict with keys: method, eps, and optionally mean/std, min/max, max_abs.
    """
    if not scaler_state:
        return snapshot
    method = scaler_state.get("method", "none")
    if method == "none":
        return snapshot

    x = torch.tensor(snapshot, dtype=torch.float32)
    eps = float(scaler_state.get("eps", 1e-8))

    if method == "standard":
        mean = scaler_state.get("mean")
        std = scaler_state.get("std")
        if mean is None or std is None:
            return snapshot
        mean_t = mean.to(dtype=torch.float32)
        std_t = std.to(dtype=torch.float32)
        denom = std_t + eps
        y = (x - mean_t) / denom
        return y.tolist()

    if method == "minmax":
        vmin = scaler_state.get("min")
        vmax = scaler_state.get("max")
        if vmin is None or vmax is None:
            return snapshot
        vmin_t = vmin.to(dtype=torch.float32)
        vmax_t = vmax.to(dtype=torch.float32)
        scale = vmax_t - vmin_t
        scale = torch.where(scale == 0, torch.full_like(scale, 1.0), scale)
        y = (x - vmin_t) / (scale + eps)
        return y.tolist()

    if method == "maxabs":
        max_abs = scaler_state.get("max_abs")
        if max_abs is None:
            return snapshot
        max_abs_t = max_abs.to(dtype=torch.float32)
        denom = max_abs_t + eps
        y = x / denom
        return y.tolist()

    return snapshot


# --- Global settings ---
# These will be populated by command-line arguments and data loading
SELECTED_DATASET = None
CURRENT_IMAGE_VECTOR_SIZE = 0
CURRENT_NUM_CLASSES = 10
# Global flag: when True, indicates colored CIFAR-10 with 3 synapses per pixel
IS_COLORED_CIFAR10 = False
# Normalization factor for each color channel in colored CIFAR-10 (default 0.5 for [0,1] range)
CIFAR10_COLOR_NORMALIZATION_FACTOR = 0.5


def select_and_load_dataset(dataset_name: str, cifar10_color_upper_bound: float = 1.0):
    """Loads a specified dataset and sets global variables."""
    global \
        SELECTED_DATASET, \
        CURRENT_IMAGE_VECTOR_SIZE, \
        CURRENT_NUM_CLASSES, \
        IS_COLORED_CIFAR10, \
        CIFAR10_COLOR_NORMALIZATION_FACTOR

    transform_mnist = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    transform_cifar = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset_map = {
        "mnist": (datasets.MNIST, transform_mnist, 10, False),
        "cifar10": (datasets.CIFAR10, transform_cifar, 10, False),
        "cifar10_color": (datasets.CIFAR10, transform_cifar, 10, True),
        "cifar100": (datasets.CIFAR100, transform_cifar, 100, False),
    }

    if dataset_name.lower() not in dataset_map:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Choose from {list(dataset_map.keys())}."
        )

    loader, transform, num_classes, is_colored = dataset_map[dataset_name.lower()]
    CIFAR10_COLOR_NORMALIZATION_FACTOR = (
        cifar10_color_upper_bound / 2.0
    )  # Convert upper bound to normalization factor

    try:
        ds = loader(root="./data", train=False, download=True, transform=transform)
        SELECTED_DATASET = ds
        img0, _ = SELECTED_DATASET[0]
        if is_colored:
            # For colored CIFAR-10, vector size is pixels * 3 (RGB channels)
            CURRENT_IMAGE_VECTOR_SIZE = int(img0.shape[1] * img0.shape[2] * 3)
            IS_COLORED_CIFAR10 = True
            normalization_range = cifar10_color_upper_bound
            print(
                f"Successfully loaded {dataset_name} dataset (colored, {img0.shape[1]}x{img0.shape[2]} pixels × 3 channels = {CURRENT_IMAGE_VECTOR_SIZE} synapses)."
            )
            print(
                f"Each color channel normalized to [0, {normalization_range:.3f}] range"
            )
        else:
            CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
            IS_COLORED_CIFAR10 = False
            print(f"Successfully loaded {dataset_name} dataset.")
        CURRENT_NUM_CLASSES = num_classes
    except Exception as e:
        raise RuntimeError(f"Failed to load {dataset_name}: {e}")


def infer_layers_from_metadata(network_sim: NeuronNetwork) -> list[list[int]]:
    """Groups neurons by their 'layer' metadata."""
    net = network_sim.network
    layer_to_neurons: dict[int, list[int]] = {}
    for nid, neuron in net.neurons.items():
        layer_idx = int(neuron.metadata.get("layer", 0))
        layer_to_neurons.setdefault(layer_idx, []).append(nid)
    return [layer_to_neurons[k] for k in sorted(layer_to_neurons.keys())]


def determine_input_mapping(
    network_sim: NeuronNetwork, layers: list[list[int]]
) -> tuple[list[int], int]:
    """Determines the input layer and synapses per neuron."""
    if not layers or not layers[0]:
        raise ValueError("Cannot determine input layer from network metadata.")
    input_layer_ids = layers[0]
    first_input_neuron = network_sim.network.neurons[input_layer_ids[0]]
    return input_layer_ids, len(first_input_neuron.postsynaptic_points)


def detect_network_color_mode(
    network_sim: NeuronNetwork, layers: list[list[int]]
) -> tuple[bool, bool]:
    """
    Detects the color mode of a network based on its structure.

    Returns:
        tuple: (is_colored, separate_neurons_per_color)
            - is_colored: True if network expects RGB input
            - separate_neurons_per_color: True if network uses separate neurons per color channel
    """
    if not layers or not layers[0]:
        return False, False

    input_layer_ids = layers[0]
    first_neuron = network_sim.network.neurons[input_layer_ids[0]]
    meta = getattr(first_neuron, "metadata", {}) or {}

    # Check for CNN-style input with color_channel metadata
    if "color_channel" in meta:
        return True, True  # RGB with separate neurons per color

    # Check for layer_type == "conv" which indicates CNN structure
    is_cnn = meta.get("layer_type") == "conv"

    if not is_cnn:
        return False, False  # Grayscale

    # Check in_channels to determine if it's RGB
    in_channels = int(meta.get("in_channels", 1))
    if in_channels == 3:
        return True, False  # RGB with 3 synapses per neuron
    return False, False  # Grayscale


def image_to_signals_for_network(
    image_tensor: torch.Tensor,
    network_sim: NeuronNetwork,
    input_layer_ids: list[int],
    synapses_per_neuron: int,
    is_colored: bool,
    separate_neurons_per_color: bool,
    normalization_factor: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Maps an image tensor to (neuron_id, synapse_id, strength) signals for a specific network.

    Args:
        image_tensor: The input image tensor
        network_sim: The network simulation
        input_layer_ids: List of input neuron IDs
        synapses_per_neuron: Number of synapses per input neuron
        is_colored: Whether the network expects colored (RGB) input
        separate_neurons_per_color: Whether the network uses separate neurons per color channel
        normalization_factor: Normalization factor for color channels
    """
    # Detect CNN-style input: first input neuron has layer_type == "conv"
    first_neuron = network_sim.network.neurons[input_layer_ids[0]]
    meta = getattr(first_neuron, "metadata", {}) or {}
    is_cnn_input = meta.get("layer_type") == "conv" and meta.get("layer", 0) == 0

    if not is_cnn_input:
        if is_colored:
            arr = image_tensor.detach().cpu().numpy().astype(np.float32)
            if arr.ndim == 3 and arr.shape[0] == 3:  # CHW format
                h, w = arr.shape[1], arr.shape[2]
                signals = []

                if separate_neurons_per_color:
                    # Architecture 2: One neuron per spatial kernel per RGB channel
                    neurons_per_color = len(input_layer_ids) // 3

                    for y in range(h):
                        for x in range(w):
                            pixel_index = y * w + x
                            for c in range(3):  # RGB channels
                                # Calculate which spatial neuron handles this pixel for this color
                                spatial_neuron_idx = pixel_index % neurons_per_color
                                # Calculate global neuron index: color_offset + spatial_idx
                                global_neuron_idx = (
                                    c * neurons_per_color + spatial_neuron_idx
                                )

                                if global_neuron_idx >= len(input_layer_ids):
                                    continue  # Skip if neuron index exceeds available neurons

                                # Each neuron handles one color channel for multiple pixels
                                synapse_index = pixel_index // neurons_per_color

                                neuron_id = input_layer_ids[global_neuron_idx]
                                # Normalize from [-1, 1] to [0, X] where X is the specified upper bound
                                pixel_value = arr[c, y, x]
                                strength = (
                                    float(pixel_value) + 1.0
                                ) * normalization_factor
                                signals.append((neuron_id, synapse_index, strength))
                else:
                    # Architecture 1: One neuron per spatial kernel, 3 synapses per RGB channel
                    for y in range(h):
                        for x in range(w):
                            for c in range(3):  # RGB channels
                                # Calculate which input neuron handles this pixel
                                pixel_index = y * w + x
                                target_neuron_index = pixel_index % len(input_layer_ids)
                                # Each neuron handles multiple pixels, each pixel has 3 synapses
                                pixels_per_neuron = (h * w) // len(input_layer_ids)
                                if (
                                    pixel_index // len(input_layer_ids)
                                    >= pixels_per_neuron
                                ):
                                    continue  # This pixel doesn't fit in the network
                                base_synapse_index = (
                                    pixel_index // len(input_layer_ids)
                                ) * 3
                                synapse_index = base_synapse_index + c
                                if synapse_index >= synapses_per_neuron:
                                    continue  # Skip if synapse index exceeds available synapses

                                neuron_id = input_layer_ids[target_neuron_index]
                                # Normalize from [-1, 1] to [0, X] where X is the specified upper bound
                                pixel_value = arr[c, y, x]
                                strength = (
                                    float(pixel_value) + 1.0
                                ) * normalization_factor
                                signals.append((neuron_id, synapse_index, strength))
                return signals
        else:
            # Grayscale: Convert RGB to grayscale by averaging channels
            arr = image_tensor.detach().cpu().numpy().astype(np.float32)
            if arr.ndim == 3 and arr.shape[0] == 3:
                # Average RGB channels to get grayscale
                img_vec = np.mean(arr, axis=0).flatten()
            else:
                img_vec = arr.flatten()
            img_vec = (img_vec + 1.0) * 0.5  # Normalize from [-1, 1] to [0, 1]

            signals = []
            num_input_neurons = len(input_layer_ids)
            for i, pixel_value in enumerate(img_vec):
                neuron_idx = i % num_input_neurons
                synapse_idx = i // num_input_neurons
                if synapse_idx < synapses_per_neuron:
                    neuron_id = input_layer_ids[neuron_idx]
                    signals.append((neuron_id, synapse_idx, float(pixel_value)))
            return signals

    # CNN mapping: each input neuron is a kernel (receptive field)
    arr = image_tensor.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape for CNN input: {arr.shape}")

    # For grayscale CNN, convert to grayscale first
    if not is_colored and arr.shape[0] == 3:
        arr = np.mean(arr, axis=0, keepdims=True)

    signals = []
    for neuron_id in input_layer_ids:
        neuron = network_sim.network.neurons[neuron_id]
        m = getattr(neuron, "metadata", {}) or {}
        k = int(m.get("kernel_size", 1))
        s = int(m.get("stride", 1))
        in_c = int(m.get("in_channels", arr.shape[0]))
        y_out = int(m.get("y", 0))
        x_out = int(m.get("x", 0))
        color_channel = m.get("color_channel", None)

        if color_channel is not None:
            # This neuron handles only one color channel
            c = color_channel
            for ky in range(k):
                for kx in range(k):
                    in_y = y_out * s + ky
                    in_x = x_out * s + kx
                    if in_y >= arr.shape[1] or in_x >= arr.shape[2]:
                        continue
                    syn_id = ky * k + kx
                    strength = (float(arr[c, in_y, in_x]) + 1.0) * normalization_factor
                    signals.append((neuron_id, syn_id, strength))
        else:
            # Standard CNN: all channels per neuron
            for c in range(in_c):
                for ky in range(k):
                    for kx in range(k):
                        in_y = y_out * s + ky
                        in_x = x_out * s + kx
                        if in_y >= arr.shape[1] or in_x >= arr.shape[2]:
                            continue
                        syn_id = (c * k + ky) * k + kx
                        strength = (
                            float(arr[c, in_y, in_x]) + 1.0
                        ) * 0.5  # [-1,1] -> [0,1]
                        signals.append((neuron_id, syn_id, strength))
    return signals


def collect_features_consistently(
    network_sim: NeuronNetwork, layers: list[list[int]], feature_types: list[str]
) -> list:
    """
    Collects features using the exact same method as in prepare_activity_data.py
    This ensures consistency between training and inference data collection.
    """
    # Create a mock record structure that matches the JSON format used in training
    mock_record = {"layers": []}

    # Populate the mock record with current network state
    for layer_ids in layers:
        layer_data = {"fired": [], "S": [], "t_ref": []}

        for nid in layer_ids:
            neuron = network_sim.network.neurons[nid]
            # Collect firing state (same as in prepare_activity_data.py)
            layer_data["fired"].append(1 if neuron.O > 0 else 0)
            # Collect membrane potential (same as in prepare_activity_data.py)
            layer_data["S"].append(float(neuron.S))
            # Collect average refractory window (t_ref) values per neuron
            layer_data["t_ref"].append(float(neuron.t_ref))

        mock_record["layers"].append(layer_data)

    # Use the same extraction logic as in prepare_activity_data.py
    if len(feature_types) == 1:
        # Single feature extraction (backward compatibility)
        if feature_types[0] == "firings":
            time_series = extract_firings_time_series([mock_record])
        elif feature_types[0] == "avg_S":
            time_series = extract_avg_S_time_series([mock_record])
        elif feature_types[0] == "avg_t_ref":
            time_series = extract_avg_t_ref_time_series([mock_record])
        else:
            raise ValueError(f"Unknown feature type: {feature_types[0]}")
    else:
        # Multi-feature extraction
        time_series = extract_multi_feature_time_series([mock_record], feature_types)

    # Return the first (and only) time step as a list
    if time_series.numel() > 0:
        return time_series[0].tolist()
    return []


def extract_firings_time_series(image_records: List[Dict[str, Any]]) -> torch.Tensor:
    """Extracts a time series of firings from records of a single image presentation."""
    time_series = []
    if not image_records:
        return torch.empty(0)

    for record in image_records:
        tick_firings = []
        for layer in record.get("layers", []):
            tick_firings.extend(layer.get("fired", []))
        time_series.append(tick_firings)

    return torch.tensor(time_series, dtype=torch.float32)


def extract_avg_S_time_series(image_records: List[Dict[str, Any]]) -> torch.Tensor:
    """Extracts a time series of average membrane potentials (S) from records."""
    time_series = []
    if not image_records:
        return torch.empty(0)

    for record in image_records:
        tick_s_values = []
        for layer in record.get("layers", []):
            tick_s_values.extend(layer.get("S", []))
        time_series.append(tick_s_values)

    return torch.tensor(time_series, dtype=torch.float32)


def extract_avg_t_ref_time_series(image_records: List[Dict[str, Any]]) -> torch.Tensor:
    """Extracts a time series of average refractory window (t_ref) values per neuron."""
    time_series = []
    if not image_records:
        return torch.empty(0)

    for record in image_records:
        tick_tref_values = []
        for layer in record.get("layers", []):
            tick_tref_values.extend(layer.get("t_ref", []))
        time_series.append(tick_tref_values)

    return torch.tensor(time_series, dtype=torch.float32)


def extract_multi_feature_time_series(
    image_records: List[Dict[str, Any]], feature_types: List[str]
) -> torch.Tensor:
    """Extracts a time series combining multiple features from records of a single image presentation."""
    if not image_records:
        return torch.empty(0)

    # Extract each feature type separately
    feature_series = {}
    for feature_type in feature_types:
        if feature_type == "firings":
            feature_series[feature_type] = extract_firings_time_series(image_records)
        elif feature_type == "avg_S":
            feature_series[feature_type] = extract_avg_S_time_series(image_records)
        elif feature_type == "avg_t_ref":
            feature_series[feature_type] = extract_avg_t_ref_time_series(image_records)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    # Concatenate features along the feature dimension (dim=1)
    # Each feature contributes its own dimension to the feature vector
    combined_series = torch.cat(list(feature_series.values()), dim=1)
    return combined_series


def main():
    parser = argparse.ArgumentParser(
        description="Real-time FUSION classification using 3 networks simultaneously."
    )
    parser.add_argument(
        "--fusion-mode",
        action="store_true",
        help="Enable fusion mode with 3 networks.",
    )
    parser.add_argument(
        "--fusion-model-path",
        type=str,
        required=True,
        help="Path to the trained fusion model file.",
    )
    parser.add_argument(
        "--network-a-path",
        type=str,
        required=True,
        help="Path to Network A (Texture) JSON file.",
    )
    parser.add_argument(
        "--network-b-path",
        type=str,
        required=True,
        help="Path to Network B (Geometry) JSON file.",
    )
    parser.add_argument(
        "--network-c-path",
        type=str,
        required=True,
        help="Path to Network C (Color) JSON file.",
    )
    parser.add_argument(
        "--network-d-path",
        type=str,
        required=True,
        help="Path to Network D (4th Network) JSON file.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="cifar10",
        choices=["mnist", "cifar10", "cifar10_color", "cifar100"],
        help="Original dataset for simulation.",
    )
    parser.add_argument(
        "--ticks-per-image",
        type=int,
        default=50,
        help="Number of simulation ticks to present each image.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Number of ticks to use as input for the fusion classifier.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=100,
        help="Number of samples to test in evaluation mode (default: 100).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference ('cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps', etc.). If not specified, will auto-detect the best available device.",
    )
    parser.add_argument(
        "--think-longer",
        action="store_true",
        help="Enable 'think longer' mode: extend simulation time if predictions are incorrect.",
    )
    parser.add_argument(
        "--max-thinking-multiplier",
        type=float,
        default=3.0,
        help="Maximum multiplier for thinking time extension (default: 3.0x base time).",
    )
    parser.add_argument(
        "--bistability-rescue",
        action="store_true",
        help="Enable bistability rescue: consider prediction correct if in top1, or in top2 with confidence difference < 5 percent.",
    )
    parser.add_argument(
        "--cifar10-color-upper-bound",
        type=float,
        default=1,
        help="For colored CIFAR-10, upper bound for each color channel normalization range [0, X] (default: 1).",
    )
    parser.add_argument(
        "--filename-prefix",
        type=str,
        default=None,
        help="Custom prefix for output filenames. If not specified, uses model-based prefix.",
    )
    parser.add_argument(
        "--enable-web-server",
        action="store_true",
        help="Enable the web visualization server (disabled by default for memory efficiency).",
    )
    args = parser.parse_args()

    # Configure device
    if args.device is not None:
        # Use user-specified device
        try:
            DEVICE = torch.device(args.device)

            # Validate device availability
            if DEVICE.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this system")
            elif DEVICE.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available on this system")

            print(f"Using specified device: {DEVICE}")
        except RuntimeError as e:
            print(f"Warning: Failed to use specified device '{args.device}': {e}")
            print("Falling back to auto-detection...")
            DEVICE = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            print(f"Using auto-detected device: {DEVICE}")
    else:
        # Auto-detect best available device
        DEVICE = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using auto-detected device: {DEVICE}")

    # Additional device info
    if DEVICE.type == "cuda":
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(DEVICE)}")
    elif DEVICE.type == "mps":
        print("Using Metal Performance Shaders (MPS) on macOS")

    # 1. Load Dataset
    # For fusion with RGB networks, always use cifar10_color internally
    dataset_name = args.dataset_name
    if dataset_name == "cifar10":
        dataset_name = "cifar10_color"  # Force color mode for RGB networks
    select_and_load_dataset(dataset_name, args.cifar10_color_upper_bound)

    # 2. Load all four networks
    print("\nFUSION MODE: Loading 4 networks simultaneously")

    print(f"Loading Network A (Texture) from {args.network_a_path}...")
    network_a = NetworkConfig.load_network_config(args.network_a_path)
    layers_a = infer_layers_from_metadata(network_a)
    input_layer_ids_a, synapses_per_neuron_a = determine_input_mapping(
        network_a, layers_a
    )
    is_colored_a, separate_neurons_a = detect_network_color_mode(network_a, layers_a)
    num_neurons_a = len(network_a.network.neurons)
    print(
        f"  Network A: {num_neurons_a} neurons, {len(layers_a)} layers, colored={is_colored_a}, separate={separate_neurons_a}"
    )

    print(f"Loading Network B (Geometry) from {args.network_b_path}...")
    network_b = NetworkConfig.load_network_config(args.network_b_path)
    layers_b = infer_layers_from_metadata(network_b)
    input_layer_ids_b, synapses_per_neuron_b = determine_input_mapping(
        network_b, layers_b
    )
    is_colored_b, separate_neurons_b = detect_network_color_mode(network_b, layers_b)
    num_neurons_b = len(network_b.network.neurons)
    print(
        f"  Network B: {num_neurons_b} neurons, {len(layers_b)} layers, colored={is_colored_b}, separate={separate_neurons_b}"
    )

    print(f"Loading Network C (Color) from {args.network_c_path}...")
    network_c = NetworkConfig.load_network_config(args.network_c_path)
    layers_c = infer_layers_from_metadata(network_c)
    input_layer_ids_c, synapses_per_neuron_c = determine_input_mapping(
        network_c, layers_c
    )
    is_colored_c, separate_neurons_c = detect_network_color_mode(network_c, layers_c)
    num_neurons_c = len(network_c.network.neurons)
    print(
        f"  Network C: {num_neurons_c} neurons, {len(layers_c)} layers, colored={is_colored_c}, separate={separate_neurons_c}"
    )

    print(f"Loading Network D (4th Network) from {args.network_d_path}...")
    network_d = NetworkConfig.load_network_config(args.network_d_path)
    layers_d = infer_layers_from_metadata(network_d)
    input_layer_ids_d, synapses_per_neuron_d = determine_input_mapping(
        network_d, layers_d
    )
    is_colored_d, separate_neurons_d = detect_network_color_mode(network_d, layers_d)
    num_neurons_d = len(network_d.network.neurons)
    print(
        f"  Network D: {num_neurons_d} neurons, {len(layers_d)} layers, colored={is_colored_d}, separate={separate_neurons_d}"
    )

    # Create NNCore instances for each network
    nn_core_a = NNCore()
    nn_core_a.neural_net = network_a
    nn_core_a.set_log_level("CRITICAL")

    nn_core_b = NNCore()
    nn_core_b.neural_net = network_b
    nn_core_b.set_log_level("CRITICAL")

    nn_core_c = NNCore()
    nn_core_c.neural_net = network_c
    nn_core_c.set_log_level("CRITICAL")

    nn_core_d = NNCore()
    nn_core_d.neural_net = network_d
    nn_core_d.set_log_level("CRITICAL")

    # 3. Set up fusion parameters
    feature_types_a = ["avg_S", "firings"]
    feature_types_b = ["avg_S", "firings"]
    feature_types_c = ["avg_S", "firings"]
    feature_types_d = ["avg_S", "firings"]

    # 4. Calculate expected input size for fusion model
    # Each network contributes: num_neurons * num_features * window_size
    features_per_tick_a = num_neurons_a * len(feature_types_a)
    features_per_tick_b = num_neurons_b * len(feature_types_b)
    features_per_tick_c = num_neurons_c * len(feature_types_c)
    features_per_tick_d = num_neurons_d * len(feature_types_d)

    expected_input_size = (
        features_per_tick_a
        + features_per_tick_b
        + features_per_tick_c
        + features_per_tick_d
    )
    print(f"\nExpected fusion input size: {expected_input_size}")
    print(
        f"  Network A: {num_neurons_a} neurons × {len(feature_types_a)} features ({feature_types_a}) = {features_per_tick_a}"
    )
    print(
        f"  Network B: {num_neurons_b} neurons × {len(feature_types_b)} features ({feature_types_b}) = {features_per_tick_b}"
    )
    print(
        f"  Network C: {num_neurons_c} neurons × {len(feature_types_c)} features ({feature_types_c}) = {features_per_tick_c}"
    )
    print(
        f"  Network D: {num_neurons_d} neurons × {len(feature_types_d)} features ({feature_types_d}) = {features_per_tick_d}"
    )
    print(f"  Total: {expected_input_size} input features")

    # 5. Load Fusion Classifier
    print(f"\nLoading fusion classifier from {args.fusion_model_path}...")

    fusion_model = FusionSNNClassifier(
        input_size=expected_input_size,
        hidden_size=SNN_HIDDEN_SIZE,
        output_size=CURRENT_NUM_CLASSES,
    ).to(DEVICE)

    # Load checkpoint
    checkpoint = torch.load(args.fusion_model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        fusion_model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"  Loaded from checkpoint (epoch {checkpoint.get('epoch', 'N/A')}, acc: {checkpoint.get('accuracy', 'N/A'):.2f}%)"
        )
    else:
        fusion_model.load_state_dict(checkpoint)

    fusion_model.eval()
    print("Fusion classifier loaded successfully.")

    # 5. Real-time Simulation and Inference Loop
    fusion_window_size = args.window_size
    print(f"Starting fusion evaluation mode with {args.eval_samples} samples...")

    # Initialize evaluation tracking
    eval_results = []
    label_errors = {i: 0 for i in range(CURRENT_NUM_CLASSES)}
    label_errors_second = {i: 0 for i in range(CURRENT_NUM_CLASSES)}
    label_errors_third = {i: 0 for i in range(CURRENT_NUM_CLASSES)}
    label_errors_second_strict = {i: 0 for i in range(CURRENT_NUM_CLASSES)}
    label_errors_third_strict = {i: 0 for i in range(CURRENT_NUM_CLASSES)}
    label_totals = {i: 0 for i in range(CURRENT_NUM_CLASSES)}

    # Set up streaming result writer
    timestamp = int(time.time())

    # Generate consistent filename prefix
    if args.filename_prefix:
        model_dir_name = args.filename_prefix
    else:
        model_dir = os.path.dirname(args.fusion_model_path)
        model_dir_name = os.path.basename(model_dir)
        # Handle case where fusion model path has no directory
        if not model_dir_name:
            # Use model filename without extension as prefix
            model_name = os.path.splitext(os.path.basename(args.fusion_model_path))[0]
            model_dir_name = f"fusion_{model_name}"

    results_filename = f"{model_dir_name}_eval_{timestamp}.jsonl"
    results_file = open(results_filename, "w", buffering=1)
    print(f"Streaming evaluation results to: {results_filename}")

    # Limit samples to dataset size
    num_samples = min(args.eval_samples, len(SELECTED_DATASET))  # type: ignore

    # Main progress bar for fusion evaluation
    main_pbar = tqdm(
        total=num_samples, desc="Fusion Evaluation", position=0, leave=True
    )

    try:
        for i in range(num_samples):
            image_tensor, actual_label = SELECTED_DATASET[i]  # type: ignore
            actual_label = int(actual_label)
            label_totals[actual_label] += 1

            network_a = NetworkConfig.load_network_config(args.network_a_path)
            network_b = NetworkConfig.load_network_config(args.network_b_path)
            network_c = NetworkConfig.load_network_config(args.network_c_path)
            network_d = NetworkConfig.load_network_config(args.network_d_path)

            # Recreate NNCore instances
            nn_core_a = NNCore()
            nn_core_a.neural_net = network_a
            nn_core_a.set_log_level("CRITICAL")
            nn_core_a.neural_net.reset_simulation()

            nn_core_b = NNCore()
            nn_core_b.neural_net = network_b
            nn_core_b.set_log_level("CRITICAL")
            nn_core_b.neural_net.reset_simulation()

            nn_core_c = NNCore()
            nn_core_c.neural_net = network_c
            nn_core_c.set_log_level("CRITICAL")
            nn_core_c.neural_net.reset_simulation()

            nn_core_d = NNCore()
            nn_core_d.neural_net = network_d
            nn_core_d.set_log_level("CRITICAL")
            nn_core_d.neural_net.reset_simulation()

            # Build signals for each network based on its color mode
            signals_a = image_to_signals_for_network(
                image_tensor,
                network_a,
                input_layer_ids_a,
                synapses_per_neuron_a,
                is_colored_a,
                separate_neurons_a,
                0.6,
            )
            signals_b = image_to_signals_for_network(
                image_tensor,
                network_b,
                input_layer_ids_b,
                synapses_per_neuron_b,
                is_colored_b,
                separate_neurons_b,
                0.6,
            )
            signals_c = image_to_signals_for_network(
                image_tensor,
                network_c,
                input_layer_ids_c,
                synapses_per_neuron_c,
                is_colored_c,
                separate_neurons_c,
                0.6,
            )
            signals_d = image_to_signals_for_network(
                image_tensor,
                network_d,
                input_layer_ids_d,
                synapses_per_neuron_d,
                is_colored_d,
                separate_neurons_d,
                0.33,
            )

            # Activity buffers for each network (rolling window)
            activity_buffer_a = []
            activity_buffer_b = []
            activity_buffer_c = []
            activity_buffer_d = []

            # Initialize tracking variables
            final_prediction = None
            final_confidence = 0.0
            final_second_prediction = None
            final_second_confidence = 0.0
            final_third_prediction = None
            final_third_confidence = 0.0

            first_correct_tick = None
            base_time_prediction = None
            base_time_correct = False

            mem1 = fusion_model.lif1.init_leaky()
            mem2 = fusion_model.lif2.init_leaky()
            mem3 = fusion_model.lif3.init_leaky()
            spk_rec = []

            # Think longer variables
            base_ticks_per_image = args.ticks_per_image
            current_ticks_per_image = args.ticks_per_image
            ticks_added = 0
            max_ticks_to_add = int(
                base_ticks_per_image * (args.max_thinking_multiplier - 1.0)
            )
            used_extended_thinking = False
            total_ticks_added = 0
            # Track prediction history for past 5 ticks
            prediction_history = []

            # Tick progress bar for current image
            tick_pbar = tqdm(
                total=current_ticks_per_image,
                desc=f"Image {i + 1}/{num_samples} (Label: {actual_label})",
                position=1,
                leave=False,
            )

            # Track when each prediction level first becomes correct
            first_correct_tick = None
            first_second_correct_tick = None
            first_third_correct_tick = None
            first_correct_appearance_tick = None

            # Main tick loop for fusion
            tick = 0
            max_ticks = base_ticks_per_image + max_ticks_to_add

            outputs = None
            while tick < current_ticks_per_image and tick < max_ticks:
                # Run simulation tick for all networks
                nn_core_a.send_batch_signals(signals_a)
                nn_core_a.do_tick()

                nn_core_b.send_batch_signals(signals_b)
                nn_core_b.do_tick()

                nn_core_c.send_batch_signals(signals_c)
                nn_core_c.do_tick()

                nn_core_d.send_batch_signals(signals_d)
                nn_core_d.do_tick()

                # Collect activity from all networks
                snapshot_a = collect_features_consistently(
                    network_a, layers_a, feature_types_a
                )
                snapshot_b = collect_features_consistently(
                    network_b, layers_b, feature_types_b
                )
                snapshot_c = collect_features_consistently(
                    network_c, layers_c, feature_types_c
                )
                snapshot_d = collect_features_consistently(
                    network_d, layers_d, feature_types_d
                )

                activity_buffer_a.append(snapshot_a)
                activity_buffer_b.append(snapshot_b)
                activity_buffer_c.append(snapshot_c)
                activity_buffer_d.append(snapshot_d)

                # Maintain buffer size (rolling window)
                if len(activity_buffer_a) > fusion_window_size:
                    activity_buffer_a.pop(0)
                    activity_buffer_b.pop(0)
                    activity_buffer_c.pop(0)
                    activity_buffer_d.pop(0)

                # Perform fusion inference when we have enough data (full window)
                if len(activity_buffer_a) > current_ticks_per_image:
                    activity_buffer_a.pop(0)
                    activity_buffer_b.pop(0)
                    activity_buffer_c.pop(0)
                    activity_buffer_d.pop(0)

                with torch.no_grad():
                    # 1. Fuse features per timestep from all networks
                    #    Shape: [window_size, num_features]
                    fused_sequence = []
                    for t in range(len(activity_buffer_a)):
                        # Concatenate features from all networks at timestep t
                        timestep_features = (
                            activity_buffer_a[t]
                            + activity_buffer_b[t]
                            + activity_buffer_c[t]
                            + activity_buffer_d[t]
                        )
                        fused_sequence.append(timestep_features)

                    input_sequence = (
                        torch.tensor(fused_sequence, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(DEVICE)
                    )

                    # 2. Initialize SNN state for THIS sequence, just like in training.
                    mem1 = fusion_model.lif1.init_leaky()
                    mem2 = fusion_model.lif2.init_leaky()
                    mem3 = fusion_model.lif3.init_leaky()
                    spk_rec = []

                    # 3. Process the entire sequence step-by-step.
                    for step in range(
                        input_sequence.shape[1]
                    ):  # Loop over the window_size dimension
                        spk3_step, mem1, mem2, mem3 = fusion_model(
                            input_sequence[:, step, :], mem1, mem2, mem3
                        )
                        spk_rec.append(spk3_step)

                    # 4. Sum spikes over the entire sequence to get a final prediction.
                    spk_rec_tensor = torch.stack(spk_rec, dim=0)
                    spike_counts = spk_rec_tensor.sum(
                        dim=0
                    )  # Sum over the time dimension
                    probabilities = torch.nn.Softmax(dim=1)(spike_counts)

                    # Store predictions
                    top_prob, top_class = probabilities.max(1)
                    final_prediction = top_class.item()
                    final_confidence = top_prob.item()

                    # Record base time prediction
                    if (
                        base_time_prediction is None
                        and tick >= base_ticks_per_image - 1
                    ):
                        base_time_prediction = final_prediction
                        base_time_correct = final_prediction == actual_label

                    # Track prediction history for thinking logic (keep last 5 predictions)
                    prediction_history.append(final_prediction == actual_label)
                    if len(prediction_history) > 5:
                        prediction_history.pop(0)

                    # Get top 3 predictions
                    top3_probs, top3_classes = torch.topk(probabilities[0], 3)
                    if len(top3_probs) > 1:
                        final_second_prediction = top3_classes[1].item()
                        final_second_confidence = top3_probs[1].item()
                    if len(top3_probs) > 2:
                        final_third_prediction = top3_classes[2].item()
                        final_third_confidence = top3_probs[2].item()

                    # Track first correct appearances
                    current_tick = tick + 1
                    if first_correct_tick is None and final_prediction == actual_label:
                        first_correct_tick = current_tick
                    if (
                        first_correct_appearance_tick is None
                        and final_prediction == actual_label
                    ):
                        first_correct_appearance_tick = current_tick
                    if (
                        first_second_correct_tick is None
                        and final_second_prediction == actual_label
                    ):
                        first_second_correct_tick = current_tick
                    if (
                        first_third_correct_tick is None
                        and final_third_prediction == actual_label
                    ):
                        first_third_correct_tick = current_tick

                # Think longer logic
                if (
                    args.think_longer
                    and len(prediction_history) >= 5
                    and not all(
                        prediction_history[-5:]
                    )  # Check if prediction was NOT correct for past 5 ticks
                    and tick == current_ticks_per_image - 1
                    and ticks_added < max_ticks_to_add
                ):
                    used_extended_thinking = True
                    ticks_added += 1
                    total_ticks_added = ticks_added
                    current_ticks_per_image = base_ticks_per_image + ticks_added
                    tick_pbar.total = current_ticks_per_image
                    tick_pbar.set_description(
                        f"Image {i + 1}/{num_samples} (Label: {actual_label}) [Think +{ticks_added}]"
                    )

                # Update tick progress with stats
                postfix_data = {
                    "pred": (
                        final_prediction if final_prediction is not None else "N/A"
                    ),
                    "conf": (
                        f"{final_confidence:.2%}" if final_confidence >= 0 else "N/A"
                    ),
                    "correct": (
                        "✅"
                        if final_prediction == actual_label
                        else "❌"
                        if final_prediction is not None
                        else "⏳"
                    ),
                }

                # Add timing info if available
                if first_correct_tick is not None:
                    postfix_data["1st_tick"] = f"{first_correct_tick}"
                if first_second_correct_tick is not None:
                    postfix_data["2nd_tick"] = f"{first_second_correct_tick}"
                if first_third_correct_tick is not None:
                    postfix_data["3rd_tick"] = f"{first_third_correct_tick}"

                tick_pbar.set_postfix(postfix_data)
                tick_pbar.update(1)
                tick += 1

            tick_pbar.close()

            # Results processing
            is_correct = (
                final_prediction == actual_label
                if final_prediction is not None
                else False
            )
            is_second_correct = (
                final_second_prediction == actual_label
                if final_second_prediction is not None
                else False
            )
            is_third_correct = (
                final_third_prediction == actual_label
                if final_third_prediction is not None
                else False
            )

            is_second_correct_strict = (
                is_second_correct and final_second_confidence > 0.0
            )
            is_third_correct_strict = is_third_correct and final_third_confidence > 0.0

            is_bistability_rescue_correct = is_correct
            if (
                args.bistability_rescue
                and not is_correct
                and final_second_prediction == actual_label
                and final_confidence is not None
                and final_second_confidence is not None
                and (final_confidence - final_second_confidence) < 0.05
            ):
                is_bistability_rescue_correct = True

            had_correct_appearance_but_wrong_final = (
                first_correct_appearance_tick is not None and not is_correct
            )

            # Track errors
            if not is_correct and final_prediction is not None:
                label_errors[actual_label] += 1
            if (
                not is_correct
                and not is_second_correct
                and final_second_prediction is not None
            ):
                label_errors_second[actual_label] += 1
            if (
                not is_correct
                and not is_second_correct
                and not is_third_correct
                and final_third_prediction is not None
            ):
                label_errors_third[actual_label] += 1
            if (
                not is_correct
                and not is_second_correct_strict
                and final_second_prediction is not None
            ):
                label_errors_second_strict[actual_label] += 1
            if (
                not is_correct
                and not is_second_correct_strict
                and not is_third_correct_strict
                and final_third_prediction is not None
            ):
                label_errors_third_strict[actual_label] += 1

            # Create result entry
            result_entry = {
                "image_idx": i,
                "actual_label": actual_label,
                "predicted_label": final_prediction,
                "confidence": final_confidence,
                "correct": is_correct,
                "bistability_rescue_correct": is_bistability_rescue_correct,
                "second_predicted_label": final_second_prediction,
                "second_confidence": final_second_confidence,
                "second_correct": is_second_correct,
                "second_correct_strict": is_second_correct_strict,
                "third_predicted_label": final_third_prediction,
                "third_confidence": final_third_confidence,
                "third_correct": is_third_correct,
                "third_correct_strict": is_third_correct_strict,
                "first_correct_tick": first_correct_tick,
                "first_correct_appearance_tick": first_correct_appearance_tick,
                "first_second_correct_tick": first_second_correct_tick,
                "first_third_correct_tick": first_third_correct_tick,
                "had_correct_appearance_but_wrong_final": had_correct_appearance_but_wrong_final,
                "used_extended_thinking": used_extended_thinking,
                "total_ticks_added": total_ticks_added,
                "base_ticks_per_image": base_ticks_per_image,
                "base_time_prediction": base_time_prediction,
                "base_time_correct": base_time_correct,
                "raw_logits": outputs.tolist() if outputs is not None else None,
                "probabilities": probabilities.tolist()
                if probabilities is not None
                else None,
            }

            # Stream result to file
            json.dump(result_entry, results_file, default=str)
            results_file.write("\n")
            eval_results.append(result_entry)

            # Update progress
            current_accuracy = (
                sum(1 for r in eval_results if r["correct"]) / len(eval_results) * 100
            )
            second_choice_accuracy = (
                sum(1 for r in eval_results if r["correct"] or r["second_correct"])
                / len(eval_results)
                * 100
            )
            third_choice_accuracy = (
                sum(
                    1
                    for r in eval_results
                    if r["correct"] or r["second_correct"] or r["third_correct"]
                )
                / len(eval_results)
                * 100
            )

            # Calculate bistability rescue accuracy if enabled
            current_bistability_rescue_accuracy = (
                (
                    sum(
                        1
                        for r in eval_results
                        if r.get("bistability_rescue_correct", r["correct"])
                    )
                    / len(eval_results)
                    * 100
                )
                if args.bistability_rescue
                else 0.0
            )

            # Calculate current averages for progress display
            current_first_correct_ticks = [
                r["first_correct_tick"]
                for r in eval_results
                if r["first_correct_tick"] is not None
            ]
            current_second_correct_ticks = [
                r["first_second_correct_tick"]
                for r in eval_results
                if r["first_second_correct_tick"] is not None
            ]
            current_third_correct_ticks = [
                r["first_third_correct_tick"]
                for r in eval_results
                if r["first_third_correct_tick"] is not None
            ]

            # Calculate current averages for correct appearance tracking
            current_correct_appearance_ticks = [
                r["first_correct_appearance_tick"]
                for r in eval_results
                if r["first_correct_appearance_tick"] is not None
            ]

            current_appeared_but_wrong_final = [
                r for r in eval_results if r["had_correct_appearance_but_wrong_final"]
            ]

            current_appeared_and_correct_final = [
                r
                for r in eval_results
                if r["first_correct_appearance_tick"] is not None and r["correct"]
            ]

            current_avg_first = (
                sum(current_first_correct_ticks) / len(current_first_correct_ticks)
                if current_first_correct_ticks
                else 0
            )
            current_avg_second = (
                sum(current_second_correct_ticks) / len(current_second_correct_ticks)
                if current_second_correct_ticks
                else 0
            )
            current_avg_third = (
                sum(current_third_correct_ticks) / len(current_third_correct_ticks)
                if current_third_correct_ticks
                else 0
            )
            current_avg_appearance = (
                sum(current_correct_appearance_ticks)
                / len(current_correct_appearance_ticks)
                if current_correct_appearance_ticks
                else 0
            )

            # Calculate current thinking effort for progress display
            current_processed_results = [
                r for r in eval_results if r.get("predicted_label") is not None
            ]
            current_thinking_results = [
                r for r in current_processed_results if r["used_extended_thinking"]
            ]
            current_thinking_ticks = [
                r["total_ticks_added"] for r in current_thinking_results
            ]
            current_avg_thinking = (
                sum(current_thinking_ticks) / len(current_thinking_ticks)
                if current_thinking_ticks
                else 0
            )

            # Calculate current accuracy for progress display (base time vs final)
            current_base_time_correct = [
                r
                for r in current_processed_results
                if r.get("base_time_correct", False)
            ]
            current_final_correct = [
                r for r in current_processed_results if r["correct"]
            ]

            current_base_time_acc = (
                (len(current_base_time_correct) / len(current_processed_results) * 100)
                if current_processed_results
                else 0
            )
            current_final_acc = (
                (len(current_final_correct) / len(current_processed_results) * 100)
                if current_processed_results
                else 0
            )

            main_pbar.set_postfix(
                {
                    "1st_acc": f"{current_accuracy:.1f}%",
                    "2nd_acc": f"{second_choice_accuracy:.1f}%",
                    "3rd_acc": f"{third_choice_accuracy:.1f}%",
                    "1st_time": (
                        f"{current_avg_first:.1f}"
                        if current_first_correct_ticks
                        else "N/A"
                    ),
                    "2nd_time": (
                        f"{current_avg_second:.1f}"
                        if current_second_correct_ticks
                        else "N/A"
                    ),
                    "3rd_time": (
                        f"{current_avg_third:.1f}"
                        if current_third_correct_ticks
                        else "N/A"
                    ),
                    "unstable": f"{len(current_appeared_but_wrong_final)}",
                    "stable": f"{len(current_appeared_and_correct_final)}",
                    "think_ticks": (
                        f"{current_avg_thinking:.1f}"
                        if current_thinking_ticks
                        else "N/A"
                    ),
                    "base_acc": (
                        f"{current_base_time_acc:.1f}%"
                        if current_processed_results
                        else "N/A"
                    ),
                    "final_acc": (
                        f"{current_final_acc:.1f}%"
                        if current_processed_results
                        else "N/A"
                    ),
                    "bistab_acc": (
                        f"{current_bistability_rescue_accuracy:.1f}%"
                        if args.bistability_rescue
                        else "N/A"
                    ),
                    "correct": f"{sum(1 for r in eval_results if r['correct'])}/{len(eval_results)}",
                }
            )
            main_pbar.update(1)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        main_pbar.close()

        # Display evaluation results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        total_correct = sum(1 for r in eval_results if r["correct"])
        total_second_correct = sum(
            1 for r in eval_results if r["correct"] or r["second_correct"]
        )
        total_third_correct = sum(
            1
            for r in eval_results
            if r["correct"] or r["second_correct"] or r["third_correct"]
        )
        total_samples = len(eval_results)
        overall_accuracy = (
            total_correct / total_samples * 100 if total_samples > 0 else 0
        )
        overall_second_accuracy = (
            total_second_correct / total_samples * 100 if total_samples > 0 else 0
        )
        overall_third_accuracy = (
            total_third_correct / total_samples * 100 if total_samples > 0 else 0
        )

        # Calculate strict accuracy metrics (excluding 0% confidence correct predictions)
        total_second_correct_strict = sum(
            1 for r in eval_results if r["correct"] or r["second_correct_strict"]
        )
        total_third_correct_strict = sum(
            1
            for r in eval_results
            if r["correct"] or r["second_correct_strict"] or r["third_correct_strict"]
        )
        overall_second_accuracy_strict = (
            total_second_correct_strict / total_samples * 100
            if total_samples > 0
            else 0
        )
        overall_third_accuracy_strict = (
            total_third_correct_strict / total_samples * 100 if total_samples > 0 else 0
        )

        # Calculate bistability rescue accuracy if enabled
        overall_bistability_rescue_accuracy = None
        bistability_rescue_improvement = None
        if args.bistability_rescue:
            total_bistability_rescue_correct = sum(
                1
                for r in eval_results
                if r.get("bistability_rescue_correct", r["correct"])
            )
            overall_bistability_rescue_accuracy = (
                total_bistability_rescue_correct / total_samples * 100
                if total_samples > 0
                else 0
            )
            bistability_rescue_improvement = (
                overall_bistability_rescue_accuracy - overall_accuracy
            )

        print(
            f"First Choice Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples})"
        )

        if args.bistability_rescue:
            total_bistability_rescue_correct = sum(
                1
                for r in eval_results
                if r.get("bistability_rescue_correct", r["correct"])
            )
            overall_bistability_rescue_accuracy = (
                total_bistability_rescue_correct / total_samples * 100
                if total_samples > 0
                else 0
            )
            bistability_rescue_improvement = (
                overall_bistability_rescue_accuracy - overall_accuracy
            )
            print(
                f"Bistability Rescue Accuracy: {overall_bistability_rescue_accuracy:.2f}% "
                f"({total_bistability_rescue_correct}/{total_samples}) "
                f"[+{bistability_rescue_improvement:.2f}% improvement]"
            )
        print(
            f"Second Choice Accuracy: {overall_second_accuracy:.2f}% ({total_second_correct}/{total_samples})"
        )
        print(
            f"Third Choice Accuracy: {overall_third_accuracy:.2f}% ({total_third_correct}/{total_samples})"
        )
        print(f"Total Errors (1st choice): {total_samples - total_correct}")
        print(f"Total Errors (2nd choice): {total_samples - total_second_correct}")
        print(f"Total Errors (3rd choice): {total_samples - total_third_correct}")
        print()
        print("Strict Accuracy (excluding 0% confidence correct predictions):")
        print("-" * 60)
        print(
            f"Strict Second Choice Accuracy: {overall_second_accuracy_strict:.2f}% ({total_second_correct_strict}/{total_samples})"
        )
        print(
            f"Strict Third Choice Accuracy: {overall_third_accuracy_strict:.2f}% ({total_third_correct_strict}/{total_samples})"
        )
        print(
            f"Strict Total Errors (2nd choice): {total_samples - total_second_correct_strict}"
        )
        print(
            f"Strict Total Errors (3rd choice): {total_samples - total_third_correct_strict}"
        )
        print(
            f"Zero-confidence contribution (2nd): {total_second_correct - total_second_correct_strict} samples"
        )
        print(
            f"Zero-confidence contribution (3rd): {total_third_correct - total_third_correct_strict} samples"
        )
        print()

        # Calculate average ticks to correct prediction for each level
        first_correct_ticks = [
            r["first_correct_tick"]
            for r in eval_results
            if r["first_correct_tick"] is not None
        ]
        second_correct_ticks = [
            r["first_second_correct_tick"]
            for r in eval_results
            if r["first_second_correct_tick"] is not None
        ]
        third_correct_ticks = [
            r["first_third_correct_tick"]
            for r in eval_results
            if r["first_third_correct_tick"] is not None
        ]

        avg_first_correct_ticks = (
            sum(first_correct_ticks) / len(first_correct_ticks)
            if first_correct_ticks
            else 0
        )
        avg_second_correct_ticks = (
            sum(second_correct_ticks) / len(second_correct_ticks)
            if second_correct_ticks
            else 0
        )
        avg_third_correct_ticks = (
            sum(third_correct_ticks) / len(third_correct_ticks)
            if third_correct_ticks
            else 0
        )

        print("Average Ticks to Correct Prediction:")
        print("-" * 40)
        print(
            f"1st Choice: {avg_first_correct_ticks:.1f} ticks ({len(first_correct_ticks)}/{total_samples} samples)"
        )
        print(
            f"2nd Choice: {avg_second_correct_ticks:.1f} ticks ({len(second_correct_ticks)}/{total_samples} samples)"
        )
        print(
            f"3rd Choice: {avg_third_correct_ticks:.1f} ticks ({len(third_correct_ticks)}/{total_samples} samples)"
        )
        print()

        # Calculate statistics for correct appearance tracking
        correct_appearance_ticks = [
            r["first_correct_appearance_tick"]
            for r in eval_results
            if r["first_correct_appearance_tick"] is not None
        ]

        # Cases where correct appeared but final was wrong
        appeared_but_wrong_final = [
            r for r in eval_results if r["had_correct_appearance_but_wrong_final"]
        ]

        # Cases where correct appeared and final was correct
        appeared_and_correct_final = [
            r
            for r in eval_results
            if r["first_correct_appearance_tick"] is not None and r["correct"]
        ]

        avg_correct_appearance_ticks = (
            sum(correct_appearance_ticks) / len(correct_appearance_ticks)
            if correct_appearance_ticks
            else 0
        )

        print("Correct Prediction Appearance Analysis:")
        print("-" * 40)
        print(
            f"Avg ticks to first correct appearance: {avg_correct_appearance_ticks:.1f} ticks ({len(correct_appearance_ticks)}/{total_samples} samples)"
        )
        print(
            f"Current avg ticks to sustained correct: {avg_first_correct_ticks:.1f} ticks ({len(first_correct_ticks)}/{total_samples} samples)"
        )
        print()
        print("Correct Appearance vs Final Result:")
        print("-" * 40)
        print(
            f"Correct appeared but final wrong: {len(appeared_but_wrong_final)} samples"
        )
        print(
            f"Correct appeared and final correct: {len(appeared_and_correct_final)} samples"
        )
        print(
            f"Total samples where correct appeared: {len(correct_appearance_ticks)} samples"
        )
        if len(correct_appearance_ticks) > 0:
            stability_rate = (
                len(appeared_and_correct_final) / len(correct_appearance_ticks) * 100
            )
            print(f"Correct prediction stability rate: {stability_rate:.1f}%")
        print()

        # Analyze thinking effort and performance impact
        # Only consider samples that actually had predictions made
        processed_results = [
            r for r in eval_results if r.get("predicted_label") is not None
        ]

        # Samples that were correct at base time (what we would have gotten without thinking)
        base_time_correct_results = [
            r for r in processed_results if r.get("base_time_correct", False)
        ]

        # Samples that were correct at the end (what we actually got with thinking)
        final_correct_results = [r for r in processed_results if r["correct"]]

        base_time_accuracy = (
            (len(base_time_correct_results) / len(processed_results) * 100)
            if processed_results
            else 0
        )
        final_accuracy = (
            (len(final_correct_results) / len(processed_results) * 100)
            if processed_results
            else 0
        )

        # Samples that used extended thinking (for average calculation)
        thinking_results = [r for r in processed_results if r["used_extended_thinking"]]

        avg_ticks_added = (
            sum(r["total_ticks_added"] for r in thinking_results)
            / len(thinking_results)
            if thinking_results
            else 0
        )

        print("Thinking Effort Analysis:")
        print("-" * 40)
        print(f"Average ticks added: {avg_ticks_added:.1f} ticks")
        print(
            f"Accuracy without thinking (base time): {base_time_accuracy:.1f}% ({len(base_time_correct_results)}/{len(processed_results)})"
        )
        print(
            f"Accuracy with thinking (final): {final_accuracy:.1f}% ({len(final_correct_results)}/{len(processed_results)})"
        )
        print(f"Accuracy improvement: {final_accuracy - base_time_accuracy:.1f}%")

        # Analyze by label which ones required thinking
        print()
        print("Labels Requiring Extended Thinking:")
        print("-" * 40)

        # Analyze which labels benefited from extended thinking
        label_base_correct = {}
        label_final_correct = {}
        label_thinking_count = {}

        for result in processed_results:
            label = result["actual_label"]

            # Count base time correct
            if result.get("base_time_correct", False):
                label_base_correct[label] = label_base_correct.get(label, 0) + 1

            # Count final correct
            if result["correct"]:
                label_final_correct[label] = label_final_correct.get(label, 0) + 1

            # Count thinking used
            if result["used_extended_thinking"]:
                label_thinking_count[label] = label_thinking_count.get(label, 0) + 1

        print("Per-Label Thinking Impact:")
        print("-" * 40)
        for label in sorted(
            set(
                list(label_base_correct.keys())
                + list(label_final_correct.keys())
                + list(label_thinking_count.keys())
            )
        ):
            base_correct = label_base_correct.get(label, 0)
            final_correct = label_final_correct.get(label, 0)
            thinking_used = label_thinking_count.get(label, 0)
            total_for_label = len(
                [r for r in processed_results if r["actual_label"] == label]
            )

            if total_for_label > 0:
                base_acc = base_correct / total_for_label * 100
                final_acc = final_correct / total_for_label * 100
                improvement = final_acc - base_acc
                print(
                    f"Label {label:2d}: Base {base_acc:4.1f}% → Final {final_acc:4.1f}% (+{improvement:4.1f}%) | Thinking: {thinking_used:2d}/{total_for_label:2d}"
                )

        print()

        print("First Choice Error Analysis by Label:")
        print("-" * 50)
        for label in range(CURRENT_NUM_CLASSES):
            if label_totals[label] > 0:
                errors = label_errors[label]
                total = label_totals[label]
                error_rate = errors / total * 100
                print(
                    f"Label {label:2d}: {errors:3d}/{total:3d} errors ({error_rate:5.1f}%)"
                )
            else:
                print(f"Label {label:2d}: No samples")

        print()
        print("Second Choice Error Analysis by Label:")
        print("-" * 50)
        for label in range(CURRENT_NUM_CLASSES):
            if label_totals[label] > 0:
                errors = label_errors_second[label]
                total = label_totals[label]
                error_rate = errors / total * 100
                print(
                    f"Label {label:2d}: {errors:3d}/{total:3d} errors ({error_rate:5.1f}%)"
                )
            else:
                print(f"Label {label:2d}: No samples")

        print()
        print("Third Choice Error Analysis by Label:")
        print("-" * 50)
        for label in range(CURRENT_NUM_CLASSES):
            if label_totals[label] > 0:
                errors = label_errors_third[label]
                total = label_totals[label]
                error_rate = errors / total * 100
                print(
                    f"Label {label:2d}: {errors:3d}/{total:3d} errors ({error_rate:5.1f}%)"
                )
            else:
                print(f"Label {label:2d}: No samples")

        print()
        print("Strict Second Choice Error Analysis by Label:")
        print("-" * 50)
        for label in range(CURRENT_NUM_CLASSES):
            if label_totals[label] > 0:
                errors = label_errors_second_strict[label]
                total = label_totals[label]
                error_rate = errors / total * 100
                print(
                    f"Label {label:2d}: {errors:3d}/{total:3d} errors ({error_rate:5.1f}%)"
                )
            else:
                print(f"Label {label:2d}: No samples")

        print()
        print("Strict Third Choice Error Analysis by Label:")
        print("-" * 50)
        for label in range(CURRENT_NUM_CLASSES):
            if label_totals[label] > 0:
                errors = label_errors_third_strict[label]
                total = label_totals[label]
                error_rate = errors / total * 100
                print(
                    f"Label {label:2d}: {errors:3d}/{total:3d} errors ({error_rate:5.1f}%)"
                )
            else:
                print(f"Label {label:2d}: No samples")

        print("\nDetailed Results (First 10 samples):")
        print("-" * 90)
        for i, result in enumerate(eval_results[:10]):
            status_1st = "✅" if result["correct"] else "❌"
            status_2nd = "✅" if result["second_correct"] else "❌"
            status_3rd = "✅" if result["third_correct"] else "❌"
            status_2nd_strict = "✅" if result["second_correct_strict"] else "❌"
            status_3rd_strict = "✅" if result["third_correct_strict"] else "❌"

            # New status for correct appearance tracking
            appearance_status = (
                "🔄" if result["first_correct_appearance_tick"] is not None else "➖"
            )
            final_after_appearance = (
                "😔" if result["had_correct_appearance_but_wrong_final"] else "😊"
            )

            second_pred = (
                result["second_predicted_label"]
                if result["second_predicted_label"] is not None
                else "N/A"
            )
            second_conf = (
                result["second_confidence"] if result["second_confidence"] > 0 else 0.0
            )
            third_pred = (
                result["third_predicted_label"]
                if result["third_predicted_label"] is not None
                else "N/A"
            )
            third_conf = (
                result["third_confidence"] if result["third_confidence"] > 0 else 0.0
            )

            # Show appearance and sustained correct timing
            appearance_tick = result["first_correct_appearance_tick"] or "N/A"
            sustained_tick = result["first_correct_tick"] or "N/A"

            print(
                f"{i + 1:2d}. Label {result['actual_label']} → 1st: {result['predicted_label']} "
                f"({result['confidence']:.2%}) {status_1st} | Appear@{appearance_tick}/Sustained@{sustained_tick} {appearance_status}{final_after_appearance} | 2nd: {second_pred} "
                f"({second_conf:.2%}) {status_2nd}/{status_2nd_strict} | 3rd: {third_pred} "
                f"({third_conf:.2%}) {status_3rd}/{status_3rd_strict}"
            )

        if len(eval_results) > 10:
            print(f"... and {len(eval_results) - 10} more samples")

        # Close streaming results file
        results_file.close()

        # Save evaluation summary to JSON file
        if eval_results:
            summary_filename = f"{model_dir_name}_eval_{timestamp}_summary.json"

            # Prepare summary data structure (without individual results)
            results_data = {
                "evaluation_metadata": {
                    "timestamp": timestamp,
                    "dataset_name": args.dataset_name,
                    "network_a_path": args.network_a_path,
                    "network_b_path": args.network_b_path,
                    "network_c_path": args.network_c_path,
                    "network_d_path": args.network_d_path,
                    "fusion_model_path": args.fusion_model_path,
                    "ticks_per_image": args.ticks_per_image,
                    "window_size": args.window_size,
                    "fusion_window_size": current_ticks_per_image,
                    "eval_samples": len(eval_results),
                    "think_longer_enabled": args.think_longer,
                    "max_thinking_multiplier": args.max_thinking_multiplier,
                    "bistability_rescue_enabled": args.bistability_rescue,
                    "feature_types": {
                        "network_a": feature_types_a,
                        "network_b": feature_types_b,
                        "network_c": feature_types_c,
                        "network_d": feature_types_d,
                    },
                    "num_classes": CURRENT_NUM_CLASSES,
                    "device": str(DEVICE),
                    "results_file": results_filename,
                },
                "calculated_metrics": {
                    "accuracy_metrics": {
                        "first_choice_accuracy": overall_accuracy,
                        "second_choice_accuracy": overall_second_accuracy,
                        "third_choice_accuracy": overall_third_accuracy,
                        "strict_second_choice_accuracy": overall_second_accuracy_strict,
                        "strict_third_choice_accuracy": overall_third_accuracy_strict,
                        "bistability_rescue_accuracy": overall_bistability_rescue_accuracy,
                        "bistability_rescue_improvement": bistability_rescue_improvement,
                        "total_errors_first_choice": total_samples - total_correct,
                        "total_errors_second_choice": total_samples
                        - total_second_correct,
                        "total_errors_third_choice": total_samples
                        - total_third_correct,
                        "strict_total_errors_second_choice": total_samples
                        - total_second_correct_strict,
                        "strict_total_errors_third_choice": total_samples
                        - total_third_correct_strict,
                        "zero_confidence_contribution_second": total_second_correct
                        - total_second_correct_strict,
                        "zero_confidence_contribution_third": total_third_correct
                        - total_third_correct_strict,
                    },
                    "timing_metrics": {
                        "avg_ticks_to_first_correct": avg_first_correct_ticks,
                        "avg_ticks_to_second_correct": avg_second_correct_ticks,
                        "avg_ticks_to_third_correct": avg_third_correct_ticks,
                        "avg_ticks_to_correct_appearance": avg_correct_appearance_ticks,
                        "correct_prediction_stability_rate": (
                            stability_rate if "stability_rate" in locals() else None
                        ),
                    },
                    "thinking_effort_analysis": {
                        "avg_ticks_added": avg_ticks_added,
                        "base_time_accuracy": base_time_accuracy,
                        "final_accuracy": final_accuracy,
                        "accuracy_improvement": final_accuracy - base_time_accuracy,
                    },
                    "error_analysis_by_label": {
                        "first_choice_errors": {
                            str(label): {
                                "errors": label_errors[label],
                                "total": label_totals[label],
                                "error_rate": (
                                    label_errors[label] / label_totals[label] * 100
                                    if label_totals[label] > 0
                                    else 0
                                ),
                            }
                            for label in range(CURRENT_NUM_CLASSES)
                        },
                        "second_choice_errors": {
                            str(label): {
                                "errors": label_errors_second[label],
                                "total": label_totals[label],
                                "error_rate": (
                                    label_errors_second[label]
                                    / label_totals[label]
                                    * 100
                                    if label_totals[label] > 0
                                    else 0
                                ),
                            }
                            for label in range(CURRENT_NUM_CLASSES)
                        },
                        "third_choice_errors": {
                            str(label): {
                                "errors": label_errors_third[label],
                                "total": label_totals[label],
                                "error_rate": (
                                    label_errors_third[label]
                                    / label_totals[label]
                                    * 100
                                    if label_totals[label] > 0
                                    else 0
                                ),
                            }
                            for label in range(CURRENT_NUM_CLASSES)
                        },
                        "strict_second_choice_errors": {
                            str(label): {
                                "errors": label_errors_second_strict[label],
                                "total": label_totals[label],
                                "error_rate": (
                                    label_errors_second_strict[label]
                                    / label_totals[label]
                                    * 100
                                    if label_totals[label] > 0
                                    else 0
                                ),
                            }
                            for label in range(CURRENT_NUM_CLASSES)
                        },
                        "strict_third_choice_errors": {
                            str(label): {
                                "errors": label_errors_third_strict[label],
                                "total": label_totals[label],
                                "error_rate": (
                                    label_errors_third_strict[label]
                                    / label_totals[label]
                                    * 100
                                    if label_totals[label] > 0
                                    else 0
                                ),
                            }
                            for label in range(CURRENT_NUM_CLASSES)
                        },
                    },
                    "per_label_thinking_impact": {
                        str(label): {
                            "base_correct": label_base_correct.get(label, 0),
                            "final_correct": label_final_correct.get(label, 0),
                            "thinking_used": label_thinking_count.get(label, 0),
                            "total_samples": len(
                                [
                                    r
                                    for r in processed_results
                                    if r["actual_label"] == label
                                ]
                            ),
                            "base_accuracy": (
                                (
                                    label_base_correct.get(label, 0)
                                    / total_for_label
                                    * 100
                                )
                                if (
                                    total_for_label := len(
                                        [
                                            r
                                            for r in processed_results
                                            if r["actual_label"] == label
                                        ]
                                    )
                                )
                                > 0
                                else 0
                            ),
                            "final_accuracy": (
                                (
                                    label_final_correct.get(label, 0)
                                    / total_for_label
                                    * 100
                                )
                                if (
                                    total_for_label := len(
                                        [
                                            r
                                            for r in processed_results
                                            if r["actual_label"] == label
                                        ]
                                    )
                                )
                                > 0
                                else 0
                            ),
                        }
                        for label in sorted(
                            set(
                                list(label_base_correct.keys())
                                + list(label_final_correct.keys())
                                + list(label_thinking_count.keys())
                            )
                        )
                    },
                    "appearance_vs_final_analysis": {
                        "appeared_but_wrong_final_count": len(appeared_but_wrong_final),
                        "appeared_and_correct_final_count": len(
                            appeared_and_correct_final
                        ),
                        "total_correct_appearances": len(correct_appearance_ticks),
                    },
                },
            }

            try:
                with open(summary_filename, "w") as f:
                    json.dump(results_data, f, indent=2, default=str)
                print(f"\nEvaluation summary saved to: {summary_filename}")
                print(f"Individual results streamed to: {results_filename}")
            except Exception as e:
                print(f"Warning: Failed to save evaluation summary to JSON: {e}")

        print("\nExiting evaluation mode.")


if __name__ == "__main__":
    main()
