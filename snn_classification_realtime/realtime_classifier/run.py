"""Main orchestration for real-time classification."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

# Ensure workspace root is in path
_parent = Path(__file__).resolve().parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

# Disable loguru logging (used by neuron modules)
from loguru import logger

logger.remove()
logger.add(lambda msg: None)

from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.ablation_registry import get_neuron_class_for_ablation
from cli.web_viz.server import NeuralNetworkWebServer

from snn_classification_realtime.snn_trainer import SNNClassifier
from snn_classification_realtime.realtime_classifier.dataset import (
    select_and_load_dataset,
)
from snn_classification_realtime.realtime_classifier.network_utils import (
    infer_layers_from_metadata,
    determine_input_mapping,
)
from snn_classification_realtime.realtime_classifier.model_config import (
    load_model_config,
)
from snn_classification_realtime.realtime_classifier.evaluation import (
    run_evaluation,
)
from snn_classification_realtime.realtime_classifier.interactive import (
    run_interactive,
)


def _configure_device(args: argparse.Namespace) -> torch.device:
    """Configure and return the torch device."""
    if args.device is not None:
        try:
            device = torch.device(args.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this system")
            if device.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available on this system")
            print(f"Using specified device: {device}")
            return device
        except RuntimeError as e:
            print(f"Warning: Failed to use specified device '{args.device}': {e}")
            print("Falling back to auto-detection...")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using auto-detected device: {device}")
    return device


def run(args: argparse.Namespace) -> None:
    """Run real-time classification (evaluation or interactive mode)."""
    device = _configure_device(args)

    if device.type == "cuda":
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(device)}")
    elif device.type == "mps":
        print("Using Metal Performance Shaders (MPS) on macOS")

    dataset_config = select_and_load_dataset(
        args.dataset_name, args.cifar10_color_upper_bound
    )

    print(f"Loading neuron network from {args.neuron_model_path}...")
    neuron_cls = get_neuron_class_for_ablation(args.ablation)
    if args.ablation:
        print(f"Using ablation neuron model: {args.ablation}")
    network_sim = NetworkConfig.load_network_config(
        args.neuron_model_path, neuron_class=neuron_cls
    )
    nn_core = NNCore()
    nn_core.set_log_level("CRITICAL")
    nn_core.neural_net = network_sim

    layers = infer_layers_from_metadata(network_sim)
    input_layer_ids, synapses_per_neuron = determine_input_mapping(
        network_sim, layers
    )
    num_neurons_total = len(network_sim.network.neurons)
    print(
        f"Neuron network loaded with {num_neurons_total} neurons across "
        f"{len(layers)} layers."
    )

    print(f"Loading classifier from {args.snn_model_path}...")
    model_config = load_model_config(args.snn_model_path)
    feature_types = model_config.get("feature_types", ["firings"])
    print(f"Model was trained on features: {feature_types}")
    print("Architecture: SNN")

    dataset_md = model_config.get("dataset_metadata") or {}
    expected_input_size = dataset_md.get("total_feature_dim")
    if expected_input_size is None:
        neuron_feature_dims = sum(
            1 for ft in feature_types if ft in ("firings", "avg_S", "avg_t_ref")
        )
        expected_input_size = neuron_feature_dims * num_neurons_total
    print(
        f"Expected input size: {expected_input_size} "
        f"(neurons: {num_neurons_total} × features: {len(feature_types)})"
    )

    snn_model = SNNClassifier(
        input_size=expected_input_size,
        hidden_size=512,
        output_size=dataset_config.num_classes,
    ).to(device)
    snn_model.load_state_dict(
        torch.load(args.snn_model_path, map_location=device)
    )

    actual_input_size = snn_model.fc1.in_features
    print(f"Actual model input size: {actual_input_size}")

    if actual_input_size != expected_input_size:
        print("WARNING: Input size mismatch!")
        print(
            f"  Expected: {expected_input_size} "
            f"(neurons: {num_neurons_total} × features: {len(feature_types)})"
        )
        print(f"  Actual: {actual_input_size}")
        print("  This mismatch could cause accuracy issues!")
    else:
        print("✓ Input size matches expected size")

    print("\nDebugging Information:")
    print(f"  Network neurons: {num_neurons_total}")
    print(f"  Feature types: {feature_types}")
    print(f"  Number of layers: {len(layers)}")
    print(f"  Layer structure: {[len(layer) for layer in layers]}")
    print(f"  Model input size: {actual_input_size}")
    print(f"  Expected input size: {expected_input_size}")
    print("  Feature extraction method: Consistent with training data")

    snn_model.eval()
    print("Classifier loaded successfully.")

    scaler_state: dict[str, Any] = {}
    scaler_state_file = dataset_md.get("scaler_state_file")
    if scaler_state_file and os.path.exists(scaler_state_file):
        try:
            print(f"Loading scaler state from {scaler_state_file}")
            scaler_state = torch.load(scaler_state_file, map_location="cpu")
            print("✓ Scaler state loaded.")
        except Exception as e:
            print(f"Warning: Failed to load scaler state: {e}")
    else:
        print("No scaler state file found; proceeding without scaling.")

    web_server = None
    if args.enable_web_server:
        print("Starting web visualization server on http://127.0.0.1:5555...")
        web_server = NeuralNetworkWebServer(nn_core, host="127.0.0.1", port=5555)
        server_thread = threading.Thread(target=web_server.run, daemon=True)
        server_thread.start()
        time.sleep(1.5)
    else:
        print("Web visualization server disabled (use --enable-web-server to enable)")

    if args.evaluation_mode:
        run_evaluation(
            dataset_config=dataset_config,
            network_config_path=args.neuron_model_path,
            neuron_cls=neuron_cls,
            nn_core=nn_core,
            snn_model=snn_model,
            scaler_state=scaler_state,
            feature_types=feature_types,
            device=device,
            args=args,
        )
    else:
        run_interactive(
            dataset_config=dataset_config,
            network_config_path=args.neuron_model_path,
            neuron_cls=neuron_cls,
            nn_core=nn_core,
            snn_model=snn_model,
            scaler_state=scaler_state,
            feature_types=feature_types,
            device=device,
            args=args,
        )


def main() -> None:
    """Entry point for real-time classification CLI."""
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
        choices=["mnist", "fashionmnist", "cifar10", "cifar10_color", "cifar100"],
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
        help="Device for inference ('cpu', 'cuda', 'mps', etc.). Auto-detect if not specified.",
    )
    parser.add_argument(
        "--think-longer",
        action="store_true",
        help="Extend simulation time if predictions incorrect.",
    )
    parser.add_argument(
        "--max-thinking-multiplier",
        type=float,
        default=3.0,
        help="Maximum multiplier for thinking time extension (default: 3.0).",
    )
    parser.add_argument(
        "--bistability-rescue",
        action="store_true",
        help="Consider correct if in top1 or top2 with confidence diff < 5%%.",
    )
    parser.add_argument(
        "--cifar10-color-upper-bound",
        type=float,
        default=1.0,
        help="Upper bound for CIFAR-10 color channel normalization (default: 1.0).",
    )
    parser.add_argument(
        "--enable-web-server",
        action="store_true",
        help="Enable web visualization server.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Ablation name: tref_frozen, retrograde_disabled, etc. Use 'none' for full model.",
    )
    args = parser.parse_args()
    run(args)
