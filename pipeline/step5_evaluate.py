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
import yaml

# Disable loguru logging (used by neuron modules)
from loguru import logger

logger.remove()  # Remove all handlers
logger.add(lambda msg: None)  # Add a no-op handler

# Ensure local imports resolve
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron.nn_core import NNCore
from neuron.network import NeuronNetwork
from neuron.network_config import NetworkConfig
from pipeline.config import EvaluationConfig
from pipeline.step4_train_model import SNNClassifier, SNN_HIDDEN_SIZE
from pipeline.utils import infer_layers_from_metadata, determine_input_mapping, image_to_signals

# --- Helper Functions ---

def apply_scaling_to_snapshot(
    snapshot: List[float], scaler_state: Dict[str, Any]
) -> List[float]:
    """Applies saved FeatureScaler state (from prepare_activity_data) to a 1D snapshot."""
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

# Global settings for dataset
SELECTED_DATASET = None
CURRENT_IMAGE_VECTOR_SIZE = 0
CURRENT_NUM_CLASSES = 10
IS_COLORED_CIFAR10 = False
CIFAR10_COLOR_NORMALIZATION_FACTOR = 0.5

def select_and_load_dataset(dataset_name: str, cifar10_color_upper_bound: float = 1.0):
    global SELECTED_DATASET, CURRENT_IMAGE_VECTOR_SIZE, CURRENT_NUM_CLASSES, IS_COLORED_CIFAR10, CIFAR10_COLOR_NORMALIZATION_FACTOR

    transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_map = {
        "mnist": (datasets.MNIST, transform_mnist, 10, False),
        "fashionmnist": (datasets.FashionMNIST, transform_mnist, 10, False),
        "cifar10": (datasets.CIFAR10, transform_cifar, 10, False),
        "cifar10_color": (datasets.CIFAR10, transform_cifar, 10, True),
        "cifar100": (datasets.CIFAR100, transform_cifar, 100, False),
    }

    if dataset_name.lower() not in dataset_map:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from {list(dataset_map.keys())}.")

    loader, transform, num_classes, is_colored = dataset_map[dataset_name.lower()]
    CIFAR10_COLOR_NORMALIZATION_FACTOR = cifar10_color_upper_bound / 2.0

    try:
        ds = loader(root="./data", train=False, download=True, transform=transform)
        SELECTED_DATASET = ds
        img0, _ = SELECTED_DATASET[0]
        if is_colored:
            CURRENT_IMAGE_VECTOR_SIZE = int(img0.shape[1] * img0.shape[2] * 3)
            IS_COLORED_CIFAR10 = True
        else:
            CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
            IS_COLORED_CIFAR10 = False
        CURRENT_NUM_CLASSES = num_classes
        print(f"Successfully loaded {dataset_name} dataset.")
    except Exception as e:
        raise RuntimeError(f"Failed to load {dataset_name}: {e}")

def collect_features_consistently(network_sim: NeuronNetwork, layers: list[list[int]], feature_types: list[str]) -> list:
    mock_record = {"layers": []}
    for layer_ids in layers:
        layer_data = {"fired": [], "S": [], "t_ref": []}
        for nid in layer_ids:
            neuron = network_sim.network.neurons[nid]
            layer_data["fired"].append(1 if neuron.O > 0 else 0)
            layer_data["S"].append(float(neuron.S))
            layer_data["t_ref"].append(float(neuron.t_ref))
        mock_record["layers"].append(layer_data)

    # Re-use extraction logic
    feature_tensors = []
    for ft in feature_types:
        if ft == "firings":
            ts = []
            for layer in mock_record["layers"]: ts.extend(layer["fired"])
            feature_tensors.append(torch.tensor(ts, dtype=torch.float32))
        elif ft == "avg_S":
             ts = []
             for layer in mock_record["layers"]: ts.extend(layer["S"])
             feature_tensors.append(torch.tensor(ts, dtype=torch.float32))
        elif ft == "avg_t_ref":
             ts = []
             for layer in mock_record["layers"]: ts.extend(layer["t_ref"])
             feature_tensors.append(torch.tensor(ts, dtype=torch.float32))

    combined = torch.cat(feature_tensors)
    return combined.tolist()


def evaluate_model(config: EvaluationConfig, model_dir: str, network_json_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Device setup
    device_str = config.device
    if device_str is None or device_str == "auto":
        device_str = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    DEVICE = torch.device(device_str)
    print(f"Using device: {DEVICE}")

    # 1. Load Dataset
    select_and_load_dataset(config.dataset_name)

    # 2. Load Neuron Network
    print(f"Loading neuron network from {network_json_path}...")
    network_sim = NetworkConfig.load_network_config(network_json_path)
    nn_core = NNCore()
    nn_core.set_log_level("CRITICAL")
    nn_core.neural_net = network_sim
    layers = infer_layers_from_metadata(network_sim)
    input_layer_ids, synapses_per_neuron = determine_input_mapping(network_sim, layers)
    num_neurons_total = len(network_sim.network.neurons)

    # 3. Load Trained SNN
    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Model not found at {model_path}")

    # Load training results to get feature config
    train_res_path = os.path.join(model_dir, "training_results.json")
    if os.path.exists(train_res_path):
        with open(train_res_path, 'r') as f:
            train_results = json.load(f)
            feature_types = train_results.get("feature_types", ["firings"])
    else:
        feature_types = ["firings"] # Fallback

    neuron_feature_dims = len(feature_types)
    expected_input_size = neuron_feature_dims * num_neurons_total

    print(f"Loading SNN model from {model_path} (Expected input: {expected_input_size})")

    snn_model = SNNClassifier(
        input_size=expected_input_size,
        hidden_size=SNN_HIDDEN_SIZE,
        output_size=CURRENT_NUM_CLASSES,
    ).to(DEVICE)

    # Check for dimension mismatch during load
    state_dict = torch.load(model_path, map_location=DEVICE)
    if state_dict['fc1.weight'].shape[1] != expected_input_size:
        print(f"Warning: Model input size {state_dict['fc1.weight'].shape[1]} != Expected {expected_input_size}")
        expected_input_size = state_dict['fc1.weight'].shape[1]
        # Re-init
        snn_model = SNNClassifier(
            input_size=expected_input_size,
            hidden_size=SNN_HIDDEN_SIZE,
            output_size=CURRENT_NUM_CLASSES,
        ).to(DEVICE)

    snn_model.load_state_dict(state_dict)
    snn_model.eval()

    # Scaler
    scaler_state = {}
    # Check if scaler exists in training dir (it should be in Step 3 output, but Step 4 might have copied it?
    # Actually Step 4 takes Step 3 dir as input. The scaler is in Step 3 dir.
    # But here we only receive model_dir (Step 4 output).
    # Ideally the runner should pass the preparation dir or Step 4 should copy scaler.
    # Let's assume scaler might be in model_dir if we modify step 4 to copy it, or we just look for it.
    # For now, we skip scaler if not found, or user can ensure it's there.
    # A better approach: The automated pipeline structure implies we might need access to previous step artifacts.
    # However, let's look for "scaler.pt" in model_dir or assume it's lost for now if not explicitly passed.


    # Evaluation Loop
    softmax = nn.Softmax(dim=1)
    activity_buffer = []

    eval_results = []
    label_errors = {i: 0 for i in range(CURRENT_NUM_CLASSES)}

    num_samples = min(config.eval_samples, len(SELECTED_DATASET))
    print(f"Evaluating on {num_samples} samples...")

    # Streaming results
    timestamp = int(time.time())
    results_filename = os.path.join(output_dir, f"eval_results.jsonl")
    results_file = open(results_filename, "w", buffering=1)

    pbar = tqdm(total=num_samples, desc="Evaluation")

    for i in range(num_samples):
        image_tensor, actual_label = SELECTED_DATASET[i]
        actual_label = int(actual_label)

        # Reset Network
        network_sim = NetworkConfig.load_network_config(network_json_path)
        nn_core.neural_net = network_sim
        layers = infer_layers_from_metadata(network_sim)
        input_layer_ids, synapses_per_neuron = determine_input_mapping(network_sim, layers)

        signals = image_to_signals(image_tensor, network_sim, input_layer_ids, synapses_per_neuron)
        activity_buffer = [] # clear
        network_sim.reset_simulation()

        # Reset SNN
        mem1 = snn_model.lif1.init_leaky()
        mem2 = snn_model.lif2.init_leaky()
        mem3 = snn_model.lif3.init_leaky()
        mem4 = snn_model.lif4.init_leaky()
        snn_spike_buffer = []

        ticks_per_image = config.ticks_per_image

        # Sim Loop
        final_prediction = None
        final_confidence = 0.0

        prediction_history = []

        # Think longer logic
        max_thinking_multiplier = config.max_thinking_multiplier
        base_ticks = ticks_per_image
        max_ticks = int(base_ticks * max_thinking_multiplier) if config.think_longer else base_ticks

        tick = 0
        used_extended_thinking = False

        current_max_ticks = base_ticks

        while tick < current_max_ticks:
            nn_core.send_batch_signals(signals)
            nn_core.do_tick()

            snapshot = collect_features_consistently(network_sim, layers, feature_types)
            if scaler_state:
                snapshot = apply_scaling_to_snapshot(snapshot, scaler_state)

            activity_buffer.append(snapshot)
            if len(activity_buffer) > config.window_size:
                activity_buffer.pop(0)

            # Inference
            with torch.no_grad():
                # Process full buffer as sequence
                input_seq = torch.tensor(activity_buffer, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                # Reset internal mems for this sequence pass?
                # The original script does reset mems for the sequence pass.
                m1, m2, m3, m4 = snn_model.lif1.init_leaky(), snn_model.lif2.init_leaky(), snn_model.lif3.init_leaky(), snn_model.lif4.init_leaky()
                spk_rec = []
                for step in range(input_seq.shape[1]):
                    s2, m1, m2, m3, m4 = snn_model(input_seq[:, step, :], m1, m2, m3, m4)
                    spk_rec.append(s2)

                spk_tensor = torch.stack(spk_rec, dim=0)
                spike_counts = spk_tensor.sum(dim=0)
                probs = softmax(spike_counts)

                top_prob, top_class = probs.max(1)
                curr_pred = top_class.item()
                curr_conf = top_prob.item()

                prediction_history.append(curr_pred == actual_label)
                if len(prediction_history) > 5: prediction_history.pop(0)

                final_prediction = curr_pred
                final_confidence = curr_conf

            # Think longer check
            if config.think_longer and tick == current_max_ticks - 1 and current_max_ticks < max_ticks:
                if len(prediction_history) >= 5 and not all(prediction_history[-5:]):
                    used_extended_thinking = True
                    current_max_ticks += 10 # Extend by 10
                    if current_max_ticks > max_ticks: current_max_ticks = max_ticks

            tick += 1

        # Result
        is_correct = (final_prediction == actual_label)
        result_entry = {
            "image_idx": i,
            "actual_label": actual_label,
            "predicted_label": final_prediction,
            "confidence": final_confidence,
            "correct": is_correct,
            "used_extended_thinking": used_extended_thinking,
            "ticks_used": tick
        }

        json.dump(result_entry, results_file)
        results_file.write("\n")
        eval_results.append(result_entry)

        pbar.update(1)
        pbar.set_postfix({"acc": f"{sum(1 for r in eval_results if r['correct'])/len(eval_results):.2%}"})

    pbar.close()
    results_file.close()

    # Summary
    total_correct = sum(1 for r in eval_results if r['correct'])
    summary = {
        "total_samples": num_samples,
        "correct": total_correct,
        "accuracy": total_correct / num_samples if num_samples else 0,
        "config": config.dict()
    }

    with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation complete. Accuracy: {summary['accuracy']:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--network_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        eval_cfg = EvaluationConfig(**cfg_dict['evaluation'])

    evaluate_model(eval_cfg, args.model_dir, args.network_json, args.output_dir)
