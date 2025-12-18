import os
import sys
import time
import argparse
import threading
import json
import pickle
import torch
import torch.nn as nn
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
from cli.web_viz.server import NeuralNetworkWebServer
from train_snn_classifier import SNNClassifier


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


def select_and_load_dataset(dataset_name: str):
    """Loads a specified dataset and sets global variables."""
    global SELECTED_DATASET, CURRENT_IMAGE_VECTOR_SIZE, CURRENT_NUM_CLASSES

    transform_mnist = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    transform_cifar = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset_map = {
        "mnist": (datasets.MNIST, transform_mnist, 10),
        "cifar10": (datasets.CIFAR10, transform_cifar, 10),
        "cifar100": (datasets.CIFAR100, transform_cifar, 100),
    }

    if dataset_name.lower() not in dataset_map:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Choose from {list(dataset_map.keys())}."
        )

    loader, transform, num_classes = dataset_map[dataset_name.lower()]

    try:
        ds = loader(root="./data", train=False, download=True, transform=transform)
        SELECTED_DATASET = ds
        img0, _ = SELECTED_DATASET[0]
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
        CURRENT_NUM_CLASSES = num_classes
        print(f"Successfully loaded {dataset_name} dataset.")
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


def image_to_signals(
    image_tensor: torch.Tensor,
    network_sim: NeuronNetwork,
    input_layer_ids: list[int],
    synapses_per_neuron: int,
) -> list[tuple[int, int, float]]:
    """Maps an image tensor to (neuron_id, synapse_id, strength) signals.

    Supports legacy dense input (flattened) and CNN input (conv layer at layer 0).
    """
    # Detect CNN-style input: first input neuron has layer_type == "conv"
    first_neuron = network_sim.network.neurons[input_layer_ids[0]]  # type: ignore
    meta = getattr(first_neuron, "metadata", {}) or {}
    is_cnn_input = meta.get("layer_type") == "conv" and meta.get("layer", 0) == 0

    if not is_cnn_input:
        img_vec = image_tensor.view(-1).numpy()
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

    signals = []
    for neuron_id in input_layer_ids:
        neuron = network_sim.network.neurons[neuron_id]  # type: ignore
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
                    strength = (
                        float(arr[c, in_y, in_x]) + 1.0
                    ) * 0.5  # [-1,1] -> [0,1]
                    signals.append((neuron_id, syn_id, strength))
    return signals


def collect_activity_snapshot(
    network_sim: NeuronNetwork, layers: list[list[int]], feature_type: str
) -> list:
    """Collects a snapshot of the specified feature from the network."""
    snapshot = []
    net = network_sim.network
    for layer_ids in layers:
        if feature_type == "firings":
            snapshot.extend([1 if net.neurons[nid].O > 0 else 0 for nid in layer_ids])
        elif feature_type == "avg_S":
            snapshot.extend([float(net.neurons[nid].S) for nid in layer_ids])
        elif feature_type == "avg_t_ref":
            snapshot.extend([float(net.neurons[nid].t_ref) for nid in layer_ids])
    return snapshot


def collect_multi_feature_snapshot(
    network_sim: NeuronNetwork, layers: list[list[int]], feature_types: list[str]
) -> list:
    """Collects a snapshot combining multiple features from the network."""
    combined_snapshot = []
    for feature_type in feature_types:
        feature_snapshot = collect_activity_snapshot(network_sim, layers, feature_type)
        combined_snapshot.extend(feature_snapshot)
    return combined_snapshot


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
    else:
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


def load_model_config(model_path: str) -> dict:
    """Load model configuration from the config file."""
    config_path = model_path.replace(".pth", "_config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback for models without config (backward compatibility)
        return {
            "feature_types": ["firings"],  # Default assumption
            "num_features": 1,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Real-time classification of neuron network activity using a trained SNN."
    )
    parser.add_argument(
        "--snn-model-path",
        type=str,
        required=True,
        help="Path to the trained snntorch model file.",
    )
    parser.add_argument(
        "--neuron-model-path",
        type=str,
        required=True,
        help="Path to the original neuron network JSON file.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "cifar100"],
        help="Original dataset for simulation.",
    )
    # Note: feature-type is now determined from the model configuration
    # The model config file contains the feature types used during training
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
        help="Number of ticks to use as input for the SNN classifier.",
    )
    parser.add_argument(
        "--evaluation-mode",
        action="store_true",
        help="Enable evaluation mode with automatic testing and progress tracking.",
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
        help="Enable 'think longer' mode: extend simulation time if predictions are incorrect, increasing by 10 ticks each time until correct or max limit reached.",
    )
    parser.add_argument(
        "--max-thinking-multiplier",
        type=float,
        default=3.0,
        help="Maximum multiplier for thinking time extension (default: 3.0x base time, limits total extension to base_time * (multiplier-1)).",
    )
    parser.add_argument(
        "--bistability-rescue",
        action="store_true",
        help="Enable bistability rescue: consider prediction correct if in top1, or in top2 with confidence difference < 5%.",
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
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
            print(f"Using auto-detected device: {DEVICE}")
    else:
        # Auto-detect best available device
        original_device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        DEVICE = original_device
        print(f"Using auto-detected device: {DEVICE}")

    # Additional device info
    if DEVICE.type == "cuda":
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(DEVICE)}")
    elif DEVICE.type == "mps":
        print("Using Metal Performance Shaders (MPS) on macOS")

    # 1. Load Dataset
    select_and_load_dataset(args.dataset_name)

    # 2. Load Neuron Network Simulation
    print(f"Loading neuron network from {args.neuron_model_path}...")
    network_sim = NetworkConfig.load_network_config(args.neuron_model_path)
    nn_core = NNCore()

    # Disable logging in the neural network
    nn_core.set_log_level("CRITICAL")
    nn_core.neural_net = network_sim
    layers = infer_layers_from_metadata(network_sim)
    input_layer_ids, synapses_per_neuron = determine_input_mapping(network_sim, layers)
    num_neurons_total = len(network_sim.network.neurons)
    print(
        f"Neuron network loaded with {num_neurons_total} neurons across {len(layers)} layers."
    )

    # 3. Load Trained SNN Classifier
    print(f"Loading classifier from {args.snn_model_path}...")

    # Load model configuration to determine feature types
    model_config = load_model_config(args.snn_model_path)
    feature_types = model_config.get("feature_types", ["firings"])
    print(f"Model was trained on features: {feature_types}")
    print(f"Architecture: SNN")

    # Set architecture to SNN (only supported option)
    architecture = "snn"

    # Calculate expected input size based on feature types
    # Prefer dataset metadata if present in model_config
    dataset_md = model_config.get("dataset_metadata") or {}
    expected_input_size = dataset_md.get("total_feature_dim")
    if expected_input_size is None:
        neuron_feature_dims = sum(
            1 for ft in feature_types if ft in ("firings", "avg_S", "avg_t_ref")
        )
        expected_input_size = neuron_feature_dims * num_neurons_total
    print(
        f"Expected input size: {expected_input_size} (neurons: {num_neurons_total} × features: {len(feature_types)})"
    )

    snn_model = SNNClassifier(
        input_size=expected_input_size,
        hidden_size=512,
        output_size=CURRENT_NUM_CLASSES,
    ).to(DEVICE)

    snn_model.load_state_dict(torch.load(args.snn_model_path, map_location=DEVICE))

    # Get the actual input size from the loaded model
    actual_input_size = snn_model.fc1.in_features
    print(f"Actual model input size: {actual_input_size}")

    # Validate input size consistency
    if actual_input_size != expected_input_size:
        print(f"WARNING: Input size mismatch!")
        print(
            f"  Expected: {expected_input_size} (neurons: {num_neurons_total} × features: {len(feature_types)})"
        )
        print(f"  Actual: {actual_input_size}")
        print(f"  This mismatch could cause accuracy issues!")
        print(f"  Possible causes:")
        print(f"    - Dataset was prepared with different feature configuration")
        print(f"    - Network structure changed between training and inference")
        print(f"    - Feature extraction method differences")
    else:
        print("✓ Input size matches expected size")

    # Add debugging information
    print(f"\nDebugging Information:")
    print(f"  Network neurons: {num_neurons_total}")
    print(f"  Feature types: {feature_types}")
    print(f"  Number of layers: {len(layers)}")
    print(f"  Layer structure: {[len(layer) for layer in layers]}")
    print(f"  Model input size: {actual_input_size}")
    print(f"  Expected input size: {expected_input_size}")
    print(f"  Feature extraction method: Consistent with training data")

    snn_model.eval()
    print("Classifier loaded successfully.")

    # 3b. Load optional scaler if available (saved by prepare_activity_data.py)
    scaler_state: Dict[str, Any] = {}
    dataset_metadata = model_config.get("dataset_metadata", {})
    scaler_state_file = dataset_metadata.get("scaler_state_file")
    if scaler_state_file and os.path.exists(scaler_state_file):
        try:
            print(f"Loading scaler state from {scaler_state_file}")
            scaler_state = torch.load(scaler_state_file, map_location="cpu")
            print("✓ Scaler state loaded.")
        except Exception as e:
            print(f"Warning: Failed to load scaler state: {e}")
            scaler_state = {}
    else:
        print("No scaler state file found; proceeding without scaling.")

    # 4. Start Web Visualization Server
    print("Starting web visualization server on http://127.0.0.1:5555...")
    web_server = NeuralNetworkWebServer(nn_core, host="127.0.0.1", port=5555)
    server_thread = threading.Thread(target=web_server.run, daemon=True)
    server_thread.start()
    time.sleep(1.5)

    # 5. Real-time Simulation and Inference Loop
    softmax = nn.Softmax(dim=1)
    activity_buffer = []

    if args.evaluation_mode:
        # Evaluation mode: run tests automatically with progress tracking
        print(f"Starting evaluation mode with {args.eval_samples} samples...")

        # Initialize evaluation tracking
        eval_results = []
        label_errors = {i: 0 for i in range(CURRENT_NUM_CLASSES)}
        label_errors_second = {i: 0 for i in range(CURRENT_NUM_CLASSES)}
        label_errors_third = {i: 0 for i in range(CURRENT_NUM_CLASSES)}
        label_errors_second_strict = {i: 0 for i in range(CURRENT_NUM_CLASSES)}
        label_errors_third_strict = {i: 0 for i in range(CURRENT_NUM_CLASSES)}
        label_totals = {i: 0 for i in range(CURRENT_NUM_CLASSES)}

        # Limit samples to dataset size
        num_samples = min(args.eval_samples, len(SELECTED_DATASET))  # type: ignore

        # Main progress bar for overall evaluation
        main_pbar = tqdm(
            total=num_samples, desc="Evaluation Progress", position=0, leave=True
        )

        try:
            for i in range(num_samples):
                image_tensor, actual_label = SELECTED_DATASET[i]  # type: ignore
                actual_label = int(actual_label)
                label_totals[actual_label] += 1

                # Load fresh network instance for each image
                network_sim = NetworkConfig.load_network_config(args.neuron_model_path)
                nn_core.set_log_level("CRITICAL")
                nn_core.neural_net = network_sim

                # Recompute layers and input mapping for consistency
                layers = infer_layers_from_metadata(network_sim)
                input_layer_ids, synapses_per_neuron = determine_input_mapping(
                    network_sim, layers
                )

                signals = image_to_signals(
                    image_tensor, network_sim, input_layer_ids, synapses_per_neuron
                )
                activity_buffer.clear()
                network_sim.reset_simulation()

                # Initialize SNN model state for this image
                mem1 = snn_model.lif1.init_leaky()
                mem2 = snn_model.lif2.init_leaky()
                mem3 = snn_model.lif3.init_leaky()
                mem4 = snn_model.lif4.init_leaky()
                snn_spike_buffer = []  # Buffer to store SNN output spikes

                # Initialize base values for think longer feature
                base_ticks_per_image = args.ticks_per_image
                base_window_size = args.window_size
                current_ticks_per_image = args.ticks_per_image
                current_window_size = args.window_size

                # Tick progress bar for current image (will be updated if extended)
                tick_pbar = tqdm(
                    total=current_ticks_per_image,
                    desc=f"Image {i+1}/{num_samples} (Label: {actual_label})",
                    position=1,
                    leave=False,
                )

                final_prediction = None
                final_confidence = 0.0
                final_second_prediction = None
                final_second_confidence = 0.0
                final_third_prediction = None
                final_third_confidence = 0.0

                # Track when each prediction level first becomes correct
                first_correct_tick = None
                first_second_correct_tick = None
                first_third_correct_tick = None

                # Track when correct prediction first appears in top1 (even if it doesn't stay)
                first_correct_appearance_tick = None

                # Track base time prediction (what would have been the result without thinking)
                base_time_prediction = None
                base_time_correct = False

                # Think longer variables
                ticks_added = 0
                max_ticks_to_add = int(
                    base_ticks_per_image * (args.max_thinking_multiplier - 1.0)
                )
                used_extended_thinking = False
                total_ticks_added = 0

                # Think longer: potentially extend simulation beyond base ticks
                tick = 0
                max_ticks = base_ticks_per_image + max_ticks_to_add

                while tick < current_ticks_per_image and tick < max_ticks:
                    # Run simulation tick
                    nn_core.send_batch_signals(signals)
                    nn_core.do_tick()

                    # Collect activity using consistent method (same as training)
                    snapshot = collect_features_consistently(
                        network_sim, layers, feature_types
                    )

                    # Apply scaling if scaler state is available
                    if scaler_state:
                        snapshot = apply_scaling_to_snapshot(snapshot, scaler_state)

                    activity_buffer.append(snapshot)

                    # Maintain buffer size (use current dynamic window size)
                    if len(activity_buffer) > current_window_size:
                        activity_buffer.pop(0)

                    with torch.no_grad():
                        # 1. Treat the entire buffer as one sequence.
                        #    Shape: [1, window_size, num_features]
                        input_sequence = (
                            torch.tensor(activity_buffer, dtype=torch.float32)
                            .unsqueeze(0)
                            .to(DEVICE)
                        )

                        # 2. Initialize SNN state for THIS sequence, just like in training.
                        mem1 = snn_model.lif1.init_leaky()
                        mem2 = snn_model.lif2.init_leaky()
                        mem3 = snn_model.lif3.init_leaky()
                        mem4 = snn_model.lif4.init_leaky()
                        spk_rec = []

                        # 3. Process the entire sequence step-by-step.
                        for step in range(
                            input_sequence.shape[1]
                        ):  # Loop over the window_size dimension
                            spk2_step, mem1, mem2, mem3, mem4 = snn_model(
                                input_sequence[:, step, :], mem1, mem2, mem3, mem4
                            )
                            spk_rec.append(spk2_step)

                        # 4. Sum spikes over the entire sequence to get a final prediction.
                        spk_rec_tensor = torch.stack(spk_rec, dim=0)
                        spike_counts = spk_rec_tensor.sum(
                            dim=0
                        )  # Sum over the time dimension
                        probabilities = softmax(spike_counts)

                        # Store predictions
                        top_prob, top_class = probabilities.max(1)
                        current_prediction = top_class.item()
                        current_confidence = top_prob.item()

                        # Record base time prediction (first time we reach base_ticks_per_image)
                        if (
                            base_time_prediction is None
                            and tick >= base_ticks_per_image - 1
                        ):
                            base_time_prediction = current_prediction
                            base_time_correct = current_prediction == actual_label

                        # Store final prediction and second-best prediction
                        final_prediction = current_prediction
                        final_confidence = current_confidence

                        # Get second-best and third-best predictions
                        sorted_probs = torch.sort(probabilities[0], descending=True)
                        if len(sorted_probs[0]) > 1:
                            final_second_prediction = sorted_probs[1][1].item()
                            final_second_confidence = sorted_probs[0][1].item()
                        if len(sorted_probs[0]) > 2:
                            final_third_prediction = sorted_probs[1][2].item()
                            final_third_confidence = sorted_probs[0][2].item()

                    # Track when each prediction level first becomes correct
                    # Only record ticks after the classifier has started receiving network activity
                    # (i.e., when confidence is not flat/equal across all labels)
                    current_tick = tick + 1  # Convert to 1-based tick count

                    # Check if classifier has meaningful differentiation (not flat 10% across all labels)
                    # For 10 classes, flat distribution would have max-min diff of ~0
                    prob_diff = (
                        probabilities[0].max().item() - probabilities[0].min().item()
                    )
                    has_meaningful_confidence = (
                        prob_diff > 0.01
                    )  # Small threshold to account for floating point precision

                    if has_meaningful_confidence:
                        if (
                            first_correct_tick is None
                            and final_prediction == actual_label
                            and final_confidence > 0.0
                        ):
                            first_correct_tick = current_tick
                        if (
                            first_second_correct_tick is None
                            and final_second_prediction == actual_label
                            and final_second_confidence > 0.0
                        ):
                            first_second_correct_tick = current_tick
                        if (
                            first_third_correct_tick is None
                            and final_third_prediction == actual_label
                            and final_third_confidence > 0.0
                        ):
                            first_third_correct_tick = current_tick

                        # Track when correct prediction first appears in top1 (even if it doesn't stay)
                        if (
                            first_correct_appearance_tick is None
                            and current_prediction == actual_label
                            and current_confidence > 0.0
                        ):
                            first_correct_appearance_tick = current_tick

                    # Think longer: check if we should extend simulation (only at last tick of current iteration)
                    if (
                        args.think_longer
                        and current_prediction is not None
                        and current_prediction != actual_label
                        and tick
                        == current_ticks_per_image - 1  # Only check at very last tick
                        and ticks_added < max_ticks_to_add
                    ):
                        # Mark that we used extended thinking
                        used_extended_thinking = True
                        total_ticks_added = ticks_added + 10

                        # Extend simulation by 10 more ticks
                        ticks_added += 10
                        current_ticks_per_image = base_ticks_per_image + ticks_added
                        current_window_size = base_window_size + int(
                            ticks_added * (base_window_size / base_ticks_per_image)
                        )

                        # Update progress bar total
                        tick_pbar.total = current_ticks_per_image
                        tick_pbar.set_description(
                            f"Image {i+1}/{num_samples} (Label: {actual_label}) [Think +{ticks_added}]"
                        )

                    # Update tick progress with stats
                    postfix_data = {
                        "pred": (
                            final_prediction if final_prediction is not None else "N/A"
                        ),
                        "conf": (
                            f"{final_confidence:.2%}" if final_confidence > 0 else "N/A"
                        ),
                        "correct": (
                            "✅"
                            if final_prediction == actual_label
                            else "❌" if final_prediction is not None else "⏳"
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

                # Track if correct prediction appeared but final was wrong
                had_correct_appearance_but_wrong_final = (
                    first_correct_appearance_tick is not None
                    and final_prediction != actual_label
                )

                # Record evaluation result
                is_correct = (
                    final_prediction == actual_label
                    if final_prediction is not None
                    else False
                )

                # Regular (lenient) accuracy - counts correct label in top predictions regardless of confidence
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

                # Strict accuracy - only counts correct label if it has > 0% confidence
                is_second_correct_strict = (
                    is_second_correct and final_second_confidence > 0.0
                    if final_second_prediction is not None
                    else False
                )
                is_third_correct_strict = (
                    is_third_correct and final_third_confidence > 0.0
                    if final_third_prediction is not None
                    else False
                )

                # Bistability rescue: correct if in top1, OR in top2 with confidence difference < 5%
                is_bistability_rescue_correct = is_correct  # Start with regular correctness
                if (
                    args.bistability_rescue
                    and not is_correct
                    and final_second_prediction == actual_label
                    and final_second_confidence is not None
                    and final_confidence is not None
                    and (final_confidence - final_second_confidence) < 0.05
                ):
                    is_bistability_rescue_correct = True

                if not is_correct and final_prediction is not None:
                    label_errors[actual_label] += 1

                # Track second-choice errors (only if first choice was wrong)
                if (
                    not is_correct
                    and not is_second_correct
                    and final_second_prediction is not None
                ):
                    label_errors_second[actual_label] += 1

                # Track third-choice errors (only if first and second choices were wrong)
                if (
                    not is_correct
                    and not is_second_correct
                    and not is_third_correct
                    and final_third_prediction is not None
                ):
                    label_errors_third[actual_label] += 1

                # Track strict second-choice errors (only if first choice was wrong and strict second is wrong)
                if (
                    not is_correct
                    and not is_second_correct_strict
                    and final_second_prediction is not None
                ):
                    label_errors_second_strict[actual_label] += 1

                # Track strict third-choice errors (only if first and second choices were wrong and strict third is wrong)
                if (
                    not is_correct
                    and not is_second_correct_strict
                    and not is_third_correct_strict
                    and final_third_prediction is not None
                ):
                    label_errors_third_strict[actual_label] += 1

                eval_results.append(
                    {
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
                    }
                )

                # Update main progress bar with both accuracy metrics
                current_accuracy = (
                    sum(1 for r in eval_results if r["correct"])
                    / len(eval_results)
                    * 100
                )
                # Calculate bistability rescue accuracy if enabled
                current_bistability_rescue_accuracy = (
                    sum(1 for r in eval_results if r.get("bistability_rescue_correct", r["correct"]))
                    / len(eval_results)
                    * 100
                ) if args.bistability_rescue else 0.0
                # Calculate second-choice accuracy (first choice correct OR second choice correct)
                second_choice_accuracy = (
                    sum(1 for r in eval_results if r["correct"] or r["second_correct"])
                    / len(eval_results)
                    * 100
                )
                # Calculate third-choice accuracy (first, second, OR third choice correct)
                third_choice_accuracy = (
                    sum(
                        1
                        for r in eval_results
                        if r["correct"] or r["second_correct"] or r["third_correct"]
                    )
                    / len(eval_results)
                    * 100
                )

                # Calculate strict second-choice accuracy (first choice correct OR second choice correct with >0% confidence)
                strict_second_choice_accuracy = (
                    sum(
                        1
                        for r in eval_results
                        if r["correct"] or r["second_correct_strict"]
                    )
                    / len(eval_results)
                    * 100
                )

                # Calculate strict third-choice accuracy (first, second, OR third choice correct with >0% confidence)
                strict_third_choice_accuracy = (
                    sum(
                        1
                        for r in eval_results
                        if r["correct"]
                        or r["second_correct_strict"]
                        or r["third_correct_strict"]
                    )
                    / len(eval_results)
                    * 100
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
                    r
                    for r in eval_results
                    if r["had_correct_appearance_but_wrong_final"]
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
                    sum(current_second_correct_ticks)
                    / len(current_second_correct_ticks)
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
                    (
                        len(current_base_time_correct)
                        / len(current_processed_results)
                        * 100
                    )
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
                        # "strict_2nd": f"{strict_second_choice_accuracy:.1f}%",
                        # "strict_3rd": f"{strict_third_choice_accuracy:.1f}%",
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
                        "correct": f'{sum(1 for r in eval_results if r["correct"])}/{len(eval_results)}',
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
                if r["correct"]
                or r["second_correct_strict"]
                or r["third_correct_strict"]
            )
            overall_second_accuracy_strict = (
                total_second_correct_strict / total_samples * 100
                if total_samples > 0
                else 0
            )
            overall_third_accuracy_strict = (
                total_third_correct_strict / total_samples * 100
                if total_samples > 0
                else 0
            )

            # Calculate bistability rescue accuracy if enabled
            overall_bistability_rescue_accuracy = None
            bistability_rescue_improvement = None
            if args.bistability_rescue:
                total_bistability_rescue_correct = sum(
                    1 for r in eval_results if r.get("bistability_rescue_correct", r["correct"])
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
                    1 for r in eval_results if r.get("bistability_rescue_correct", r["correct"])
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
                    len(appeared_and_correct_final)
                    / len(correct_appearance_ticks)
                    * 100
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
            thinking_results = [
                r for r in processed_results if r["used_extended_thinking"]
            ]

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
                    "🔄"
                    if result["first_correct_appearance_tick"] is not None
                    else "➖"
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
                    result["second_confidence"]
                    if result["second_confidence"] > 0
                    else 0.0
                )
                third_pred = (
                    result["third_predicted_label"]
                    if result["third_predicted_label"] is not None
                    else "N/A"
                )
                third_conf = (
                    result["third_confidence"]
                    if result["third_confidence"] > 0
                    else 0.0
                )

                # Show appearance and sustained correct timing
                appearance_tick = result["first_correct_appearance_tick"] or "N/A"
                sustained_tick = result["first_correct_tick"] or "N/A"

                print(
                    f"{i+1:2d}. Label {result['actual_label']} → 1st: {result['predicted_label']} "
                    f"({result['confidence']:.2%}) {status_1st} | Appear@{appearance_tick}/Sustained@{sustained_tick} {appearance_status}{final_after_appearance} | 2nd: {second_pred} "
                    f"({second_conf:.2%}) {status_2nd}/{status_2nd_strict} | 3rd: {third_pred} "
                    f"({third_conf:.2%}) {status_3rd}/{status_3rd_strict}"
                )

            if len(eval_results) > 10:
                print(f"... and {len(eval_results) - 10} more samples")

            # Save evaluation results to JSON file
            if eval_results:
                timestamp = int(time.time())
                # Use model path directory name as base for filename
                model_dir = os.path.dirname(args.snn_model_path)
                model_dir_name = os.path.basename(model_dir)
                output_filename = f"{model_dir_name}_eval_{timestamp}.json"

                # Prepare results data structure
                results_data = {
                    "evaluation_metadata": {
                        "timestamp": timestamp,
                        "dataset_name": args.dataset_name,
                        "neuron_model_path": args.neuron_model_path,
                        "snn_model_path": args.snn_model_path,
                        "ticks_per_image": args.ticks_per_image,
                        "window_size": args.window_size,
                        "eval_samples": len(eval_results),
                        "think_longer_enabled": args.think_longer,
                        "max_thinking_multiplier": args.max_thinking_multiplier,
                        "bistability_rescue_enabled": args.bistability_rescue,
                        "feature_types": feature_types,
                        "num_classes": CURRENT_NUM_CLASSES,
                        "device": str(DEVICE),
                    },
                    "evaluation_results": eval_results,
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
                            "appeared_but_wrong_final_count": len(
                                appeared_but_wrong_final
                            ),
                            "appeared_and_correct_final_count": len(
                                appeared_and_correct_final
                            ),
                            "total_correct_appearances": len(correct_appearance_ticks),
                        },
                    },
                }

                try:
                    with open(output_filename, "w") as f:
                        json.dump(results_data, f, indent=2, default=str)
                    print(f"\nEvaluation results saved to: {output_filename}")
                except Exception as e:
                    print(f"Warning: Failed to save evaluation results to JSON: {e}")

            print("\nExiting evaluation mode.")
    else:
        # Interactive mode: original behavior with user-friendly output
        try:
            for i in range(len(SELECTED_DATASET)):  # type: ignore
                image_tensor, actual_label = SELECTED_DATASET[i]  # type: ignore
                print(
                    f"\n--- Presenting Image {i+1}/{len(SELECTED_DATASET)} (Label: {actual_label}) ---"  # type: ignore
                )

                # Load fresh network instance for each image
                print("Loading fresh network instance...")
                network_sim = NetworkConfig.load_network_config(args.neuron_model_path)
                nn_core.set_log_level("CRITICAL")
                nn_core.neural_net = network_sim

                # Recompute layers and input mapping for consistency
                layers = infer_layers_from_metadata(network_sim)
                input_layer_ids, synapses_per_neuron = determine_input_mapping(
                    network_sim, layers
                )

                signals = image_to_signals(
                    image_tensor, network_sim, input_layer_ids, synapses_per_neuron
                )
                activity_buffer.clear()
                network_sim.reset_simulation()

                # Initialize base values for think longer feature (interactive mode)
                base_ticks_per_image_interactive = args.ticks_per_image
                base_window_size_interactive = args.window_size
                current_ticks_per_image_interactive = args.ticks_per_image
                current_window_size_interactive = args.window_size

                # Track base time prediction for interactive mode
                base_time_prediction_interactive = None
                base_time_correct_interactive = False

                # Initialize SNN model state for this image
                mem1 = snn_model.lif1.init_leaky()
                mem2 = snn_model.lif2.init_leaky()
                mem3 = snn_model.lif3.init_leaky()
                mem4 = snn_model.lif4.init_leaky()
                snn_spike_buffer = []  # Buffer to store SNN output spikes

                # Think longer variables for interactive mode
                ticks_added_interactive = 0
                max_ticks_to_add_interactive = int(
                    base_ticks_per_image_interactive
                    * (args.max_thinking_multiplier - 1.0)
                )
                used_extended_thinking_interactive = False
                total_ticks_added_interactive = 0

                # Think longer: potentially extend simulation beyond base ticks (interactive mode)
                tick = 0
                max_ticks_interactive = (
                    base_ticks_per_image_interactive + max_ticks_to_add_interactive
                )

                while (
                    tick < current_ticks_per_image_interactive
                    and tick < max_ticks_interactive
                ):
                    # Run simulation tick
                    nn_core.send_batch_signals(signals)
                    nn_core.do_tick()

                    # Collect activity using consistent method (same as training)
                    snapshot = collect_features_consistently(
                        network_sim, layers, feature_types
                    )

                    # Apply scaling if scaler state is available (interactive mode)
                    if scaler_state:
                        snapshot = apply_scaling_to_snapshot(snapshot, scaler_state)

                    activity_buffer.append(snapshot)

                    # Maintain buffer size (use current dynamic window size)
                    if len(activity_buffer) > current_window_size_interactive:
                        activity_buffer.pop(0)

                    # Perform inference if buffer is full
                    if len(activity_buffer) == current_window_size_interactive:
                        with torch.no_grad():
                            # Get the current time step input
                            current_input = torch.tensor(
                                [activity_buffer[-1]], dtype=torch.float32
                            ).to(DEVICE)

                            # Forward pass with state management
                            spk2, mem1, mem2, mem3, mem4 = snn_model(
                                current_input, mem1, mem2, mem3, mem4
                            )
                            snn_spike_buffer.append(spk2)

                            # Maintain buffer size for SNN spikes
                            if len(snn_spike_buffer) > args.window_size:
                                snn_spike_buffer.pop(0)

                        # Calculate prediction using accumulated spikes
                        if len(snn_spike_buffer) >= 1:
                            spk_rec = torch.stack(snn_spike_buffer, dim=0)

                            # Get output probabilities
                            spike_counts = spk_rec.sum(dim=0)  # Sum spikes over time
                            probabilities = softmax(spike_counts)

                            # Display predictions
                            top_prob, top_class = probabilities.max(1)
                            current_prediction = top_class.item()
                            current_confidence = top_prob.item()

                            # Record base time prediction for interactive mode
                            if (
                                base_time_prediction_interactive is None
                                and tick >= base_ticks_per_image_interactive - 1
                            ):
                                base_time_prediction_interactive = current_prediction
                                base_time_correct_interactive = (
                                    current_prediction == actual_label
                                )

                            probs_list = [
                                f"{label}: {p:.2%}"
                                for label, p in enumerate(probabilities[0])
                            ]
                            mapped_probs_list = [
                                (label, p) for label, p in enumerate(probabilities[0])
                            ]

                            print(
                                f"Tick {tick+1}/{current_ticks_per_image_interactive} | Prediction: {top_class.item()} ({top_prob.item():.2%}) | Certainties: {probs_list}"
                            )

                            # Think longer: check if we should extend simulation (interactive mode)
                            if (
                                args.think_longer
                                and current_prediction != actual_label
                                and tick
                                == current_ticks_per_image_interactive
                                - 1  # Only check at very last tick
                                and ticks_added_interactive
                                < max_ticks_to_add_interactive
                            ):

                                # Mark that we used extended thinking
                                used_extended_thinking_interactive = True
                                total_ticks_added_interactive = (
                                    ticks_added_interactive + 10
                                )

                                # Extend simulation by 10 more ticks
                                ticks_added_interactive += 10
                                current_ticks_per_image_interactive = (
                                    base_ticks_per_image_interactive
                                    + ticks_added_interactive
                                )
                                current_window_size_interactive = (
                                    base_window_size_interactive
                                    + int(
                                        ticks_added_interactive
                                        * (
                                            base_window_size_interactive
                                            / base_ticks_per_image_interactive
                                        )
                                    )
                                )

                                print(
                                    f"  → Extending thinking time by 10 ticks to {current_ticks_per_image_interactive} total (+{ticks_added_interactive} ticks)"
                                )

                            # Check if prediction is correct (for potential early termination)
                            if top_class.item() == actual_label:
                                print("  → Correct prediction made!")
                            print("✅" if top_class.item() == actual_label else "❌")

                            sorted_probabilities = sorted(
                                mapped_probs_list, key=lambda x: x[1], reverse=True
                            )
                            closest_match = sorted_probabilities[1][0]
                            if closest_match == actual_label:
                                print("Next closest match: ", end="")
                                print(
                                    f"{closest_match}: {sorted_probabilities[1][1]:.2%}"
                                )

                    tick += 1
                    time.sleep(
                        0.01
                    )  # Small delay to make output readable and not overwhelm CPU

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        finally:
            print("\nExiting.")


if __name__ == "__main__":
    main()
