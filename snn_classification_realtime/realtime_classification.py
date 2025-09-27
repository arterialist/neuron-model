import os
import sys
import time
import argparse
import threading
import json
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
from train_snn_classifier import SNNClassifier  # Import the SNN model definition

# --- Global settings ---
# These will be populated by command-line arguments and data loading
SELECTED_DATASET = None
CURRENT_IMAGE_VECTOR_SIZE = 0
CURRENT_NUM_CLASSES = 10
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")


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
    image_tensor: torch.Tensor, input_layer_ids: list[int], synapses_per_neuron: int
) -> list[tuple[int, int, float]]:
    """Maps a flattened image tensor to (neuron_id, synapse_id, strength) signals."""
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
        layer_data = {"fired": [], "S": []}

        for nid in layer_ids:
            neuron = network_sim.network.neurons[nid]
            # Collect firing state (same as in prepare_activity_data.py)
            layer_data["fired"].append(1 if neuron.O > 0 else 0)
            # Collect membrane potential (same as in prepare_activity_data.py)
            layer_data["S"].append(float(neuron.S))

        mock_record["layers"].append(layer_data)

    # Use the same extraction logic as in prepare_activity_data.py
    if len(feature_types) == 1:
        # Single feature extraction (backward compatibility)
        if feature_types[0] == "firings":
            time_series = extract_firings_time_series([mock_record])
        elif feature_types[0] == "avg_S":
            time_series = extract_avg_S_time_series([mock_record])
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
    args = parser.parse_args()

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
    print(f"Loading SNN classifier from {args.snn_model_path}...")

    # Load model configuration to determine feature types
    model_config = load_model_config(args.snn_model_path)
    feature_types = model_config.get("feature_types", ["firings"])
    print(f"Model was trained on features: {feature_types}")

    # Calculate expected input size based on feature types
    # Each feature type contributes the total number of neurons
    expected_input_size = num_neurons_total * len(feature_types)
    print(
        f"Expected input size: {expected_input_size} (neurons: {num_neurons_total} × features: {len(feature_types)})"
    )

    # Create model with the expected input size first
    snn_model = SNNClassifier(
        input_size=expected_input_size, hidden_size=128, output_size=CURRENT_NUM_CLASSES
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
    print("SNN classifier loaded.")

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
        label_totals = {i: 0 for i in range(CURRENT_NUM_CLASSES)}

        # Limit samples to dataset size
        num_samples = min(args.eval_samples, len(SELECTED_DATASET))

        # Main progress bar for overall evaluation
        main_pbar = tqdm(
            total=num_samples, desc="Evaluation Progress", position=0, leave=True
        )

        try:
            for i in range(num_samples):
                image_tensor, actual_label = SELECTED_DATASET[i]
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
                    image_tensor, input_layer_ids, synapses_per_neuron
                )
                activity_buffer.clear()
                network_sim.reset_simulation()

                # Initialize SNN model state for this image
                mem1 = snn_model.lif1.init_leaky()
                mem2 = snn_model.lif2.init_leaky()
                mem3 = snn_model.lif2.init_leaky()
                mem4 = snn_model.lif2.init_leaky()
                snn_spike_buffer = []  # Buffer to store SNN output spikes

                # Tick progress bar for current image
                tick_pbar = tqdm(
                    total=args.ticks_per_image,
                    desc=f"Image {i+1}/{num_samples} (Label: {actual_label})",
                    position=1,
                    leave=False,
                )

                final_prediction = None
                final_confidence = 0.0
                final_second_prediction = None
                final_second_confidence = 0.0

                for tick in range(args.ticks_per_image):
                    # Run simulation tick
                    nn_core.send_batch_signals(signals)
                    nn_core.do_tick()

                    # Collect activity using consistent method (same as training)
                    snapshot = collect_features_consistently(
                        network_sim, layers, feature_types
                    )
                    activity_buffer.append(snapshot)

                    # Maintain buffer size
                    if len(activity_buffer) > args.window_size:
                        activity_buffer.pop(0)

                    # Perform inference if buffer is full
                    # Perform inference if buffer is full
                    if len(activity_buffer) == args.window_size:
                        with torch.no_grad():
                            # --- CORRECTED LOGIC ---
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
                            mem3 = snn_model.lif2.init_leaky()
                            mem4 = snn_model.lif2.init_leaky()
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
                            # --- END OF CORRECTION ---

                            # Store final prediction and second-best prediction
                            top_prob, top_class = probabilities.max(1)
                            final_prediction = top_class.item()
                            final_confidence = top_prob.item()

                            # Get second-best prediction
                            sorted_probs = torch.sort(probabilities[0], descending=True)
                            if len(sorted_probs[0]) > 1:
                                final_second_prediction = sorted_probs[1][1].item()
                                final_second_confidence = sorted_probs[0][1].item()

                    # Update tick progress with stats
                    tick_pbar.set_postfix(
                        {
                            "pred": (
                                final_prediction
                                if final_prediction is not None
                                else "N/A"
                            ),
                            "conf": (
                                f"{final_confidence:.2%}"
                                if final_confidence > 0
                                else "N/A"
                            ),
                            "correct": (
                                "✅"
                                if final_prediction == actual_label
                                else "❌" if final_prediction is not None else "⏳"
                            ),
                        }
                    )
                    tick_pbar.update(1)

                tick_pbar.close()

                # Record evaluation result
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

                if not is_correct and final_prediction is not None:
                    label_errors[actual_label] += 1

                # Track second-choice errors (only if first choice was wrong)
                if (
                    not is_correct
                    and not is_second_correct
                    and final_second_prediction is not None
                ):
                    label_errors_second[actual_label] += 1

                eval_results.append(
                    {
                        "image_idx": i,
                        "actual_label": actual_label,
                        "predicted_label": final_prediction,
                        "confidence": final_confidence,
                        "correct": is_correct,
                        "second_predicted_label": final_second_prediction,
                        "second_confidence": final_second_confidence,
                        "second_correct": is_second_correct,
                    }
                )

                # Update main progress bar with both accuracy metrics
                current_accuracy = (
                    sum(1 for r in eval_results if r["correct"])
                    / len(eval_results)
                    * 100
                )
                # Calculate second-choice accuracy (first choice correct OR second choice correct)
                second_choice_accuracy = (
                    sum(1 for r in eval_results if r["correct"] or r["second_correct"])
                    / len(eval_results)
                    * 100
                )
                main_pbar.set_postfix(
                    {
                        "1st_acc": f"{current_accuracy:.1f}%",
                        "2nd_acc": f"{second_choice_accuracy:.1f}%",
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
            total_samples = len(eval_results)
            overall_accuracy = (
                total_correct / total_samples * 100 if total_samples > 0 else 0
            )
            overall_second_accuracy = (
                total_second_correct / total_samples * 100 if total_samples > 0 else 0
            )

            print(
                f"First Choice Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples})"
            )
            print(
                f"Second Choice Accuracy: {overall_second_accuracy:.2f}% ({total_second_correct}/{total_samples})"
            )
            print(f"Total Errors (1st choice): {total_samples - total_correct}")
            print(f"Total Errors (2nd choice): {total_samples - total_second_correct}")
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

            print("\nDetailed Results (First 10 samples):")
            print("-" * 70)
            for i, result in enumerate(eval_results[:10]):
                status_1st = "✅" if result["correct"] else "❌"
                status_2nd = "✅" if result["second_correct"] else "❌"
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
                print(
                    f"{i+1:2d}. Label {result['actual_label']} → 1st: {result['predicted_label']} "
                    f"({result['confidence']:.2%}) {status_1st} | 2nd: {second_pred} "
                    f"({second_conf:.2%}) {status_2nd}"
                )

            if len(eval_results) > 10:
                print(f"... and {len(eval_results) - 10} more samples")

            print("\nExiting evaluation mode.")
    else:
        # Interactive mode: original behavior with user-friendly output
        try:
            for i in range(len(SELECTED_DATASET)):
                image_tensor, actual_label = SELECTED_DATASET[i]
                print(
                    f"\n--- Presenting Image {i+1}/{len(SELECTED_DATASET)} (Label: {actual_label}) ---"
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
                    image_tensor, input_layer_ids, synapses_per_neuron
                )
                activity_buffer.clear()
                network_sim.reset_simulation()

                # Initialize SNN model state for this image
                mem1 = snn_model.lif1.init_leaky()
                mem2 = snn_model.lif2.init_leaky()
                snn_spike_buffer = []  # Buffer to store SNN output spikes

                for tick in range(args.ticks_per_image):
                    # Run simulation tick
                    nn_core.send_batch_signals(signals)
                    nn_core.do_tick()

                    # Collect activity using consistent method (same as training)
                    snapshot = collect_features_consistently(
                        network_sim, layers, feature_types
                    )
                    activity_buffer.append(snapshot)

                    # Maintain buffer size
                    if len(activity_buffer) > args.window_size:
                        activity_buffer.pop(0)

                    # Perform inference if buffer is full
                    if len(activity_buffer) == args.window_size:
                        with torch.no_grad():
                            # Get the current time step input
                            current_input = torch.tensor(
                                [activity_buffer[-1]], dtype=torch.float32
                            ).to(DEVICE)

                            # Forward pass with state management
                            spk2, mem1, mem2 = snn_model(current_input, mem1, mem2)
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

                            probs_list = [
                                f"{label}: {p:.2%}"
                                for label, p in enumerate(probabilities[0])
                            ]
                            mapped_probs_list = [
                                (label, p) for label, p in enumerate(probabilities[0])
                            ]

                            print(
                                f"Tick {tick+1}/{args.ticks_per_image} | Prediction: {top_class.item()} ({top_prob.item():.2%}) | Certainties: {probs_list}"
                            )
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

                    time.sleep(
                        0.01
                    )  # Small delay to make output readable and not overwhelm CPU

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        finally:
            print("\nExiting.")


if __name__ == "__main__":
    main()
