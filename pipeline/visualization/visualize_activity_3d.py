#!/usr/bin/env python3
"""
3D Network Visualization: Activity-Based Clustering in 3D Space

This script creates a 3D spatial map where neurons that activate together for the same
patterns (e.g., "digit 8") cluster together visually, forming activity regions.

Algorithm Overview:
1. Load network and balanced stimulation set (15 images per class)
2. Run activity scan: simulate each image for 50 ticks, capture activation rates
3. Compute preferred labels based on class-specific activity
4. Use UMAP to project neurons into 3D space (correlation metric)
5. Render interactive 3D visualization with plotly

Usage Examples:
    # Basic usage with default settings
    python visualize_brain_3d.py --network networks/my_network.json --output brain.html

    # Custom settings for speed/quality tradeoff
    python visualize_brain_3d.py --network networks/my_network.json \
                                --output brain.html \
                                --samples-per-class 10 \
                                --ticks 25

    # For testing with small datasets
    python visualize_brain_3d.py --network networks/demo.json \
                                --output test_brain.html \
                                --samples-per-class 2 \
                                --ticks 10

Requirements:
- Trained PAULA network JSON file
- MNIST or CIFAR10 dataset (automatically downloaded)
- umap-learn, plotly, torch, torchvision

Output:
- brain.html: Interactive 3D visualization
- brain.png: Static image version
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import plotly.io as pio
import hashlib
import subprocess

# Add local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuron.nn_core import NNCore
from neuron.network import NeuronNetwork
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron

# Add UMAP for dimensionality reduction
try:
    import umap
except ImportError:
    print("UMAP not found. Installing...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn"])
    import umap

# Global rendering and caching configuration
PLOT_IMAGE_SCALE: float = 2.0  # Higher scale => higher DPI for Plotly exports
MATPLOTLIB_DPI: int = 400  # DPI for matplotlib static figures
CACHE_VERSION: str = "v1"


def compute_dataset_hash(file_path: str) -> str:
    """Return a short MD5 hash for the dataset file using shell command.

    Prefer `md5sum` (GNU coreutils). On macOS, fall back to `md5 -q`.
    Returns the first 16 hex chars (lowercase).
    """
    try:
        result = subprocess.run(
            ["md5sum", file_path], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout:
            md5_hex = result.stdout.strip().split()[0]
            return md5_hex.lower()[:16]
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(
            ["md5", "-q", file_path], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout:
            md5_hex = result.stdout.strip().split()[0]
            return md5_hex.lower()[:16]
    except FileNotFoundError:
        pass

    # Fallback to in-Python hashing if shell tools unavailable
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def get_cache_dir(base_output_dir: str) -> str:
    """Return (and create) the cache directory for intermediates."""
    cache_dir = os.path.join(base_output_dir, f"cache_{CACHE_VERSION}")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def format_float_token(value: float, decimals: int = 6) -> str:
    """Return a stable, filename-safe token for a float parameter.

    Uses fixed decimal formatting to avoid scientific notation and rounding
    inconsistencies across runs.
    """
    return f"{float(value):.{decimals}f}"


# Configuration
TICKS_PER_IMAGE = 50
SAMPLES_PER_CLASS = 15
NUM_CLASSES = 10
TOTAL_SAMPLES = SAMPLES_PER_CLASS * NUM_CLASSES

# Class names for different datasets
DATASET_CLASS_NAMES = {
    "mnist": [f"Digit {i}" for i in range(10)],
    "cifar10": [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ],
}


def load_balanced_dataset(
    dataset_name: str = "mnist",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load balanced MNIST dataset with exactly SAMPLES_PER_CLASS images per digit.

    Returns:
        images: (TOTAL_SAMPLES, 784) flattened images
        labels: (TOTAL_SAMPLES,) digit labels
    """
    print(
        f"Loading balanced {dataset_name.upper()} dataset ({SAMPLES_PER_CLASS} samples per class)..."
    )

    if dataset_name.lower() == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ]
        )
        dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset_name.lower() == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize to [-1, 1] for 3 channels
            ]
        )
        dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Collect balanced samples
    images_per_class = {i: [] for i in range(NUM_CLASSES)}

    for img, label in dataset:
        if len(images_per_class[label]) < SAMPLES_PER_CLASS:
            images_per_class[label].append(img)
        # Stop when we have enough samples for all classes
        if all(
            len(samples) >= SAMPLES_PER_CLASS for samples in images_per_class.values()
        ):
            break

    # Verify we got all samples
    for digit in range(NUM_CLASSES):
        if len(images_per_class[digit]) < SAMPLES_PER_CLASS:
            raise ValueError(
                f"Could not find {SAMPLES_PER_CLASS} samples for digit {digit}"
            )

    # Concatenate all samples
    all_images = []
    all_labels = []

    for digit in range(NUM_CLASSES):
        all_images.extend(images_per_class[digit])
        all_labels.extend([digit] * len(images_per_class[digit]))

    # Convert to tensors
    images_tensor = torch.stack(all_images)  # (TOTAL_SAMPLES, C, H, W)
    labels_tensor = torch.tensor(all_labels)  # (TOTAL_SAMPLES,)

    # Flatten images
    images_flat = images_tensor.view(TOTAL_SAMPLES, -1)  # (TOTAL_SAMPLES, C*H*W)

    print(f"Dataset loaded: {TOTAL_SAMPLES} images ({SAMPLES_PER_CLASS} per digit)")
    return images_flat, labels_tensor


def infer_layers_from_metadata(network_sim: NeuronNetwork) -> List[List[int]]:
    """Group neurons by their 'layer' metadata if present; otherwise fall back to a single layer list in ID order."""
    net = network_sim.network
    layer_to_neurons: Dict[int, List[int]] = {}
    for nid, neuron in net.neurons.items():
        layer_idx = (
            int(neuron.metadata.get("layer", 0)) if isinstance(neuron, Neuron) else 0
        )
        layer_to_neurons.setdefault(layer_idx, []).append(nid)
    if not layer_to_neurons:
        return [list(net.neurons.keys())]
    layers_indexed = [layer_to_neurons[k] for k in sorted(layer_to_neurons.keys())]
    return layers_indexed


def determine_input_mapping(
    network: NeuronNetwork, layers: List[List[int]]
) -> Tuple[List[int], int]:
    """Return (input_layer_ids, input_connections_per_neuron). Assumes first layer is inputs."""
    input_layer_ids = layers[0]
    if not input_layer_ids:
        raise ValueError("Input layer appears empty; cannot map images to signals.")
    first_input = network.network.neurons[input_layer_ids[0]]
    input_connections_per_neuron = max(1, len(first_input.postsynaptic_points))
    return input_layer_ids, input_connections_per_neuron


def image_to_signals(
    image_tensor: torch.Tensor,
    input_layer_ids: List[int],
    input_connections_per_neuron: int,
    network_sim: NeuronNetwork,
) -> List[Tuple[int, int, float]]:
    """Map an image to (neuron_id, connection_id, strength) signals.

    Supports legacy dense input and CNN-style input (conv layer at index 0).
    Signals are normalized to [0, 1] range.
    """
    # Detect CNN-style input
    first_neuron = network_sim.network.neurons[input_layer_ids[0]]
    meta = getattr(first_neuron, "metadata", {}) or {}
    is_cnn_input = meta.get("layer_type") == "conv" and meta.get("layer", 0) == 0

    if not is_cnn_input:
        img_vec = image_tensor.view(-1).numpy().astype(np.float32)
        # Normalize from [-1, 1] to [0, 1]
        img_vec = (img_vec + 1.0) * 0.5
        num_input_neurons = len(input_layer_ids)
        signals: List[Tuple[int, int, float]] = []
        for pixel_index, pixel_value in enumerate(img_vec):
            target_neuron_index = pixel_index % num_input_neurons
            target_synapse_index = pixel_index // num_input_neurons
            target_synapse_index = min(
                target_synapse_index, input_connections_per_neuron - 1
            )
            neuron_id = input_layer_ids[target_neuron_index]
            strength = float(pixel_value)
            signals.append((neuron_id, target_synapse_index, strength))
        return signals

    # CNN input: one neuron per kernel position; synapses map to receptive field pixels
    arr = image_tensor.detach().cpu().numpy().astype(np.float32)

    # Handle flattened images - reshape based on expected dimensions
    if arr.ndim == 1:
        if arr.shape[0] == 784:  # MNIST: 1x28x28 = 784
            arr = arr.reshape(1, 28, 28)
        elif arr.shape[0] == 3072:  # CIFAR10: 3x32x32 = 3072
            arr = arr.reshape(3, 32, 32)
        else:
            raise ValueError(f"Unsupported flattened image size: {arr.shape[0]}")
    elif arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape for CNN input: {arr.shape}")

    signals: List[Tuple[int, int, float]] = []
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
                    # Normalize from [-1, 1] to [0, 1]
                    strength = (float(arr[c, in_y, in_x]) + 1.0) * 0.5
                    signals.append((neuron_id, syn_id, strength))
    return signals


def run_fmri_scan(
    network_path: str, images: torch.Tensor, labels: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Run activity scan: simulate network on each image and capture firing rates and average membrane potentials.

    Returns:
        firing_matrix: (TOTAL_SAMPLES, num_neurons) firing rates [0,1]
        avg_S_matrix: (TOTAL_SAMPLES, num_neurons) average membrane potentials
        neuron_ids: List of neuron IDs in order
    """
    print("Running fMRI scan simulation...")

    # Load network
    print(f"Loading network from {network_path}...")
    network_sim = NetworkConfig.load_network_config(network_path)
    nn_core = NNCore()
    nn_core.neural_net = network_sim

    # Disable logging
    nn_core.set_log_level("CRITICAL")

    # Infer network structure
    layers = infer_layers_from_metadata(network_sim)
    input_layer_ids, synapses_per_neuron = determine_input_mapping(network_sim, layers)

    num_neurons = len(network_sim.network.neurons)
    neuron_ids = list(network_sim.network.neurons.keys())

    print(f"Network loaded: {num_neurons} neurons across {len(layers)} layers")
    print(f"Input layer: {len(input_layer_ids)} neurons")

    # Initialize firing rate and avg_S matrices
    firing_matrix = np.zeros((TOTAL_SAMPLES, num_neurons), dtype=np.float32)
    avg_S_matrix = np.zeros((TOTAL_SAMPLES, num_neurons), dtype=np.float32)

    # Track cumulative firing statistics
    total_firings_all_images = 0
    total_possible_firings = 0

    # Run simulation for each image
    with tqdm(range(TOTAL_SAMPLES), desc="Processing images") as pbar:
        for sample_idx in pbar:
            image = images[sample_idx]
            true_label = labels[sample_idx].item()

            # Convert image to signals
            signals = image_to_signals(
                image, input_layer_ids, synapses_per_neuron, network_sim
            )

            # Reset network state
            network_sim.reset_simulation()

            # Initialize firing counters and S accumulators for this image
            firing_counts = np.zeros(num_neurons, dtype=np.int32)
            S_accumulators = np.zeros(num_neurons, dtype=np.float32)

            # Run simulation for TICKS_PER_IMAGE
            image_firings = 0
            for tick in range(TICKS_PER_IMAGE):
                # Send input signals
                nn_core.send_batch_signals(signals)

                # Advance simulation
                nn_core.do_tick()

                # Count firings and accumulate S values for this tick
                for neuron_idx, neuron_id in enumerate(neuron_ids):
                    neuron = network_sim.network.neurons[neuron_id]
                    if neuron.O > 0:  # Neuron fired
                        firing_counts[neuron_idx] += 1
                        image_firings += 1
                    # Accumulate membrane potential (S) for averaging
                    S_accumulators[neuron_idx] += float(neuron.S)

            # Convert to firing rate (0.0 to 1.0) and average S values
            firing_rates = firing_counts / TICKS_PER_IMAGE
            avg_S_values = S_accumulators / TICKS_PER_IMAGE
            firing_matrix[sample_idx] = firing_rates
            avg_S_matrix[sample_idx] = avg_S_values

            # Update cumulative statistics
            total_firings_all_images += image_firings
            total_possible_firings += num_neurons * TICKS_PER_IMAGE

            # Calculate statistics for progress bar
            image_firing_rate = image_firings / (num_neurons * TICKS_PER_IMAGE)
            overall_firing_rate = total_firings_all_images / total_possible_firings
            active_neurons = np.sum(firing_counts > 0)

            # Update progress bar with firing information
            pbar.set_postfix(
                {
                    "label": f"{true_label}",
                    "img_firings": f"{image_firings}",
                    "img_rate": f"{image_firing_rate:.1%}",
                    "active_neurons": f"{active_neurons}/{num_neurons}",
                    "total_rate": f"{overall_firing_rate:.1%}",
                }
            )

    return firing_matrix, avg_S_matrix, neuron_ids


def compute_preferred_labels(
    firing_matrix: np.ndarray, labels: torch.Tensor
) -> np.ndarray:
    """
    Compute preferred label for each neuron based on class-specific activity.

    Args:
        firing_matrix: (TOTAL_SAMPLES, num_neurons) firing rates
        labels: (TOTAL_SAMPLES,) true labels

    Returns:
        preferred_labels: (num_neurons,) preferred class for each neuron
    """
    print("Computing preferred labels...")

    num_neurons = firing_matrix.shape[1]

    # Compute average activity per class for each neuron
    class_activity = np.zeros((NUM_CLASSES, num_neurons))

    for class_idx in range(NUM_CLASSES):
        # Find samples of this class
        class_mask = labels.numpy() == class_idx
        if np.sum(class_mask) > 0:
            # Average firing rate across all samples of this class
            class_activity[class_idx] = np.mean(firing_matrix[class_mask], axis=0)

    # Check for inactive neurons (neurons with no activity across all classes)
    total_activity_per_neuron = np.sum(class_activity, axis=0)
    inactive_threshold = 1e-6  # Very small threshold for "inactive"

    # Preferred label = argmax average activity, but -1 for inactive neurons
    preferred_labels = np.full(num_neurons, -1, dtype=int)  # Default to -1 (inactive)
    active_neurons = total_activity_per_neuron > inactive_threshold

    if np.any(active_neurons):
        # Only assign preferences to active neurons
        active_indices = np.where(active_neurons)[0]
        preferred_labels[active_indices] = np.argmax(
            class_activity[:, active_indices], axis=0
        )

    # Print statistics
    unique_labels, counts = np.unique(preferred_labels, return_counts=True)
    print("Preferred label distribution:")
    total_active = 0
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"  Inactive: {count} neurons")
        else:
            print(f"  Class {label}: {count} neurons")
            total_active += count
    print(f"  Total active neurons: {total_active}")

    return preferred_labels


def apply_umap_projection(
    clustering_matrix: np.ndarray,
    neuron_ids: List[int],
    clustering_method: str = "firings",
) -> Tuple[np.ndarray, List[int]]:
    """
    Apply UMAP to project neurons into 3D space using correlation distance.

    Args:
        clustering_matrix: (TOTAL_SAMPLES, num_neurons) clustering values (firings or avg_S)
        neuron_ids: List of neuron IDs
        clustering_method: Method used for clustering ("firings" or "avg_S")

    Returns:
        positions_3d: (num_neurons, 3) 3D coordinates
    """
    print("Applying UMAP projection...")

    # Transpose: (num_neurons, TOTAL_SAMPLES)
    # Each row represents a neuron's activity across all images
    neuron_activity = clustering_matrix.T

    # Debug: check clustering metric statistics
    clustering_values_per_neuron = np.mean(neuron_activity, axis=1)
    metric_name = (
        "firing rates" if clustering_method == "firings" else "avg membrane potential"
    )
    print(
        f"{metric_name} statistics: min={np.min(clustering_values_per_neuron):.6f}, "
        f"max={np.max(clustering_values_per_neuron):.6f}, "
        f"mean={np.mean(clustering_values_per_neuron):.6f}"
    )

    # Filter out dead neurons (very low activity, not all zeros)
    # Use a small threshold instead of strict > 0
    activity_threshold = 1e-6  # Very small threshold
    active_mask = np.max(neuron_activity, axis=1) > activity_threshold
    active_neurons = neuron_activity[active_mask]
    active_neuron_ids = [
        neuron_ids[i] for i in range(len(neuron_ids)) if active_mask[i]
    ]

    print(
        f"Active neurons (threshold > {activity_threshold}): {len(active_neurons)}/{len(neuron_activity)}"
    )

    if len(active_neurons) == 0:
        print("Warning: No neurons meet the activity threshold. Using all neurons.")
        active_neurons = neuron_activity
        active_neuron_ids = neuron_ids

    # Handle very small datasets that UMAP can't process
    if len(active_neurons) < 5:
        print(
            f"Warning: Only {len(active_neurons)} active neurons - insufficient for UMAP."
        )
        print("Using simple 3D layout instead.")

        # Create a simple 3D layout for very small datasets
        positions_3d = np.zeros((len(active_neurons), 3))
        for i in range(len(active_neurons)):
            # Spread neurons in a simple pattern
            positions_3d[i] = [
                (i % 2) * 2 - 1,  # x: alternate between -1 and 1
                ((i // 2) % 2) * 2 - 1,  # y: alternate in layers
                (i // 4) * 1.0,  # z: stack in layers
            ]
    else:
        # Apply UMAP - adjust parameters based on dataset size
        n_neighbors = min(
            15, len(active_neurons) - 1
        )  # Ensure n_neighbors < dataset size
        n_neighbors = max(2, n_neighbors)  # Minimum of 2 neighbors

        print(f"Using n_neighbors={n_neighbors} for {len(active_neurons)} neurons")

        reducer = umap.UMAP(
            n_components=3,
            metric="correlation",  # Critical: measures synchrony
            n_neighbors=n_neighbors,
            random_state=42,
            min_dist=0.1,
        )

        positions_3d = reducer.fit_transform(active_neurons)

    print("UMAP projection complete")
    print(".3f")

    return positions_3d, active_neuron_ids


def create_brain_visualization(
    positions_3d: np.ndarray,
    preferred_labels: np.ndarray,
    neuron_ids: List[int],
    output_path: str = "brain.html",
    dataset_name: str = "mnist",
):
    """
    Create interactive 3D network visualization using plotly.
    """
    print("Creating 3D brain visualization...")

    # Create hover text
    class_names = DATASET_CLASS_NAMES[dataset_name.lower()]
    hover_text = []
    for i, neuron_id in enumerate(neuron_ids):
        pref_class = preferred_labels[i]
        if pref_class == -1:
            class_name = "Inactive"
        else:
            class_name = class_names[pref_class]
        hover_text.append(f"Neuron ID: {neuron_id}<br>Prefers: {class_name}")

    # Create color map (high contrast)
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Create figure
    fig = go.Figure()

    # Also create cloud version
    fig_cloud = go.Figure()

    # Create 3D grids for density estimation
    x_min, x_max = (
        float(positions_3d[:, 0].min() - 0.5),
        float(positions_3d[:, 0].max() + 0.5),
    )
    y_min, y_max = (
        float(positions_3d[:, 1].min() - 0.5),
        float(positions_3d[:, 1].max() + 0.5),
    )
    z_min, z_max = (
        float(positions_3d[:, 2].min() - 0.5),
        float(positions_3d[:, 2].max() + 0.5),
    )

    # Create grid for density estimation
    grid_size = 20
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    z_grid = np.linspace(z_min, z_max, grid_size)
    xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    # Add traces for inactive neurons first
    inactive_mask = preferred_labels == -1
    inactive_positions = (
        positions_3d[inactive_mask] if np.sum(inactive_mask) > 0 else np.empty((0, 3))
    )
    inactive_hover = (
        [hover_text[i] for i in range(len(inactive_mask)) if inactive_mask[i]]
        if np.sum(inactive_mask) > 0
        else []
    )

    # Add inactive neurons to regular figure
    fig.add_trace(
        go.Scatter3d(
            x=inactive_positions[:, 0],
            y=inactive_positions[:, 1],
            z=inactive_positions[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color="gray",
                opacity=0.4,
                line=dict(width=1, color="darkgray"),
            ),
            name="Inactive",
            text=inactive_hover,
            hovertemplate="%{text}<extra></extra>",
            legendgroup="inactive",
        )
    )

    # Add inactive neurons to cloud figure
    fig_cloud.add_trace(
        go.Scatter3d(
            x=inactive_positions[:, 0],
            y=inactive_positions[:, 1],
            z=inactive_positions[:, 2],
            mode="markers",
            marker=dict(
                size=max(
                    3, min(6, len(inactive_positions) // 2)
                ),  # Scale size with count
                color="gray",
                opacity=0.5,
                line=dict(width=1, color="darkgray"),
            ),
            name=f"Inactive ({len(inactive_positions)} neurons)",
            text=inactive_hover,
            hovertemplate="%{text}<extra></extra>",
            legendgroup="inactive",
        )
    )

    # Add traces for each preferred class
    for class_idx in range(NUM_CLASSES):
        mask = preferred_labels == class_idx
        if np.sum(mask) == 0:
            continue

        class_positions = positions_3d[mask]
        class_hover = [hover_text[i] for i in range(len(mask)) if mask[i]]

        fig.add_trace(
            go.Scatter3d(
                x=class_positions[:, 0],
                y=class_positions[:, 1],
                z=class_positions[:, 2],
                mode="markers",
                marker=dict(
                    size=6,
                    color=colors[class_idx],
                    opacity=0.8,
                    line=dict(width=1, color="black"),
                ),
                name=class_names[class_idx],
                text=class_hover,
                hovertemplate="%{text}<extra></extra>",
                legendgroup=f"class_{class_idx}",
            )
        )

        # Always add scatter points for this class (including small clusters)
        fig_cloud.add_trace(
            go.Scatter3d(
                x=class_positions[:, 0],
                y=class_positions[:, 1],
                z=class_positions[:, 2],
                mode="markers",
                marker=dict(
                    size=max(
                        4, min(8, len(class_positions) * 2)
                    ),  # Scale size with cluster size
                    color=colors[class_idx],
                    opacity=0.7,
                ),
                name=f"{class_names[class_idx]} ({len(class_positions)} neurons)",
                text=class_hover,
                hovertemplate="%{text}<extra></extra>",
                legendgroup=f"class_{class_idx}",
            )
        )

        # Add cloud/density visualization for larger clusters
        try:
            if len(class_positions) >= 3:
                kde = gaussian_kde(class_positions.T, bw_method=0.3)
                positions_flat = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
                density = kde(positions_flat).reshape(xx.shape)
                max_density = np.max(density)
                if max_density > 0:
                    density = density / max_density

                    # Add isosurface cloud to cloud figure
                    fig_cloud.add_trace(
                        go.Isosurface(
                            x=xx.ravel(),
                            y=yy.ravel(),
                            z=zz.ravel(),
                            value=density.ravel(),
                            isomin=0.15,
                            isomax=0.8,
                            surface_count=3,
                            caps=dict(x_show=False, y_show=False, z_show=False),
                            colorscale=[
                                [
                                    0,
                                    f"rgba({int(colors[class_idx][1:3], 16)},{int(colors[class_idx][3:5], 16)},{int(colors[class_idx][5:7], 16)},0.1)",
                                ],
                                [
                                    1,
                                    f"rgba({int(colors[class_idx][1:3], 16)},{int(colors[class_idx][3:5], 16)},{int(colors[class_idx][5:7], 16)},0.6)",
                                ],
                            ],
                            opacity=0.4,
                            name=f"{class_names[class_idx]} Cloud",
                            showscale=False,
                            showlegend=True,
                        )
                    )
        except Exception as e:
            # KDE failed, but points are already added above
            print(f"Warning: Could not create density cloud for class {class_idx}: {e}")

    # Update layout for "brain in void" aesthetic
    fig.update_layout(
        title=dict(
            text="Network Activity: Activity-Based Clustering<br><sub>Neurons clustered by firing synchrony (UMAP projection)</sub>",
            font=dict(size=16, color="white"),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="black",
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        showlegend=True,
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            borderwidth=1,
        ),
        width=1200,
        height=900,
    )

    # Update layout for cloud version
    fig_cloud.update_layout(
        title=dict(
            text="Network Activity: Activity Density Clouds<br><sub>Density-based visualization of activation regions</sub>",
            font=dict(size=16, color="white"),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="black",
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        showlegend=True,
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            borderwidth=1,
        ),
        width=1200,
        height=900,
    )

    # Save interactive HTML
    fig.write_html(output_path)
    print(f"Interactive brain visualization saved to {output_path}")

    # Save JSON for plotly.js
    json_path = output_path.replace(".html", ".json")
    pio.write_json(fig, json_path)
    print(f"JSON export saved to {json_path}")

    # Save cloud version
    cloud_path = output_path.replace(".html", "_cloud.html")
    fig_cloud.write_html(cloud_path)
    print(f"Interactive brain cloud visualization saved to {cloud_path}")

    # Save cloud JSON for plotly.js
    cloud_json_path = cloud_path.replace(".html", ".json")
    pio.write_json(fig_cloud, cloud_json_path)
    print(f"Cloud JSON export saved to {cloud_json_path}")

    # Also save static images
    static_path = output_path.replace(".html", ".png")
    fig.write_image(static_path, width=1200, height=900, scale=2)
    print(f"Static brain visualization saved to {static_path}")

    cloud_static_path = output_path.replace(".html", "_cloud.png")
    fig_cloud.write_image(cloud_static_path, width=1200, height=900, scale=2)
    print(f"Static brain cloud visualization saved to {cloud_static_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create 3D functional brain map from neural activity"
    )
    parser.add_argument(
        "--network",
        type=str,
        required=True,
        help="Path to network JSON configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="neuron_activity_umap_3d",
        help="Base output directory for results",
    )
    parser.add_argument(
        "--ticks", type=int, default=50, help="Number of simulation ticks per image"
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=15,
        help="Number of images to use per digit class",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset to use for brain stimulation (mnist or cifar10)",
    )
    parser.add_argument(
        "--clustering",
        type=str,
        default="firings",
        choices=["firings", "avg_S"],
        help="Clustering method: 'firings' (firing rates) or 'avg_S' (average membrane potential)",
    )

    args = parser.parse_args()

    # Update global constants
    global TICKS_PER_IMAGE, SAMPLES_PER_CLASS, TOTAL_SAMPLES
    TICKS_PER_IMAGE = args.ticks
    SAMPLES_PER_CLASS = args.samples_per_class
    TOTAL_SAMPLES = SAMPLES_PER_CLASS * NUM_CLASSES

    # Create structured output directory
    network_basename = os.path.splitext(os.path.basename(args.network))[0]
    config_suffix = f"dataset_{args.dataset}_ticks{TICKS_PER_IMAGE}_samples{SAMPLES_PER_CLASS}_clustering_{args.clustering}"
    structured_output_dir = os.path.join(
        args.output_dir, network_basename, config_suffix
    )
    os.makedirs(structured_output_dir, exist_ok=True)

    # Prepare cache
    cache_dir = get_cache_dir(structured_output_dir)

    print("=" * 80)
    print("NETWORK ACTIVITY VISUALIZATION: Activity-Based Clustering")
    print("=" * 80)
    print(f"Network: {args.network}")
    print(f"Dataset: {args.dataset}")
    print(f"Clustering method: {args.clustering}")
    print(f"Output directory: {structured_output_dir}")
    print(f"Ticks per image: {TICKS_PER_IMAGE}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(f"Total samples: {TOTAL_SAMPLES}")
    print()

    # Phase 1: Load balanced stimulation set (with caching)
    print("PHASE 1: Loading balanced stimulation set...")
    dataset_cache_path = os.path.join(
        cache_dir, f"{args.dataset}_samples{SAMPLES_PER_CLASS}_balanced.npz"
    )
    if os.path.exists(dataset_cache_path):
        cache = np.load(dataset_cache_path)
        images = torch.from_numpy(cache["images"])
        labels = torch.from_numpy(cache["labels"])
        print(f"Loaded cached dataset from {dataset_cache_path}")
    else:
        images, labels = load_balanced_dataset(args.dataset)
        np.savez_compressed(
            dataset_cache_path, images=images.numpy(), labels=labels.numpy()
        )
        print(f"Saved dataset cache to {dataset_cache_path}")

    # Phase 2: Run fMRI scan (with caching)
    print("\nPHASE 2: Running fMRI scan simulation...")
    network_hash = compute_dataset_hash(args.network)
    fmri_cache_path = os.path.join(
        cache_dir,
        f"fmri_{network_hash}_{args.dataset}_ticks{TICKS_PER_IMAGE}_samples{SAMPLES_PER_CLASS}.npz",
    )
    if os.path.exists(fmri_cache_path):
        cache = np.load(fmri_cache_path, allow_pickle=True)
        firing_matrix = cache["firing_matrix"]
        avg_S_matrix = cache["avg_S_matrix"]
        neuron_ids = cache["neuron_ids"].tolist()
        print(f"Loaded cached fMRI scan from {fmri_cache_path}")
    else:
        firing_matrix, avg_S_matrix, neuron_ids = run_fmri_scan(
            args.network, images, labels
        )
        np.savez_compressed(
            fmri_cache_path,
            firing_matrix=firing_matrix,
            avg_S_matrix=avg_S_matrix,
            neuron_ids=np.array(neuron_ids, dtype=object),
        )
        print(f"Saved fMRI scan cache to {fmri_cache_path}")

    # Phase 3: Compute preferred labels (with caching)
    print("\nPHASE 3: Computing preferred labels...")
    prefs_cache_path = os.path.join(
        cache_dir, f"preferred_labels_{args.dataset}_samples{SAMPLES_PER_CLASS}.npy"
    )
    if os.path.exists(prefs_cache_path):
        preferred_labels = np.load(prefs_cache_path)
        print(f"Loaded cached preferred labels from {prefs_cache_path}")
    else:
        preferred_labels = compute_preferred_labels(firing_matrix, labels)
        np.save(prefs_cache_path, preferred_labels)
        print(f"Saved preferred labels cache to {prefs_cache_path}")

    # Phase 4: Apply UMAP projection (with caching)
    print("\nPHASE 4: Applying UMAP projection...")
    umap_cache_path = os.path.join(
        cache_dir,
        f"umap_3d_{network_hash}_{args.dataset}_ticks{TICKS_PER_IMAGE}_samples{SAMPLES_PER_CLASS}_{args.clustering}.npz",
    )

    # Select clustering matrix based on method
    if args.clustering == "firings":
        clustering_matrix = firing_matrix
    elif args.clustering == "avg_S":
        clustering_matrix = avg_S_matrix
    else:
        raise ValueError(f"Unknown clustering method: {args.clustering}")

    if os.path.exists(umap_cache_path):
        cache = np.load(umap_cache_path, allow_pickle=True)
        positions_3d = cache["positions_3d"]
        active_neuron_ids = cache["active_neuron_ids"].tolist()
        print(f"Loaded cached UMAP projection from {umap_cache_path}")
    else:
        positions_3d, active_neuron_ids = apply_umap_projection(
            clustering_matrix, neuron_ids, args.clustering
        )
        np.savez_compressed(
            umap_cache_path,
            positions_3d=positions_3d,
            active_neuron_ids=np.array(active_neuron_ids, dtype=object),
        )
        print(f"Saved UMAP projection cache to {umap_cache_path}")

    # Update preferred labels for active neurons only
    active_indices = [neuron_ids.index(nid) for nid in active_neuron_ids]
    active_preferred_labels = preferred_labels[active_indices]

    # Phase 5: Create visualization
    print("\nPHASE 5: Rendering brain visualization...")
    output_filename = os.path.join(structured_output_dir, "brain.html")
    create_brain_visualization(
        positions_3d,
        active_preferred_labels,
        active_neuron_ids,
        output_filename,
        args.dataset,
    )

    print("\n✓ Brain mapping complete!")

    # Check if any neurons actually fired
    total_firing = np.sum(firing_matrix)
    if total_firing == 0:
        print("⚠️  WARNING: No neurons fired during the simulation.")
        print("   This could mean:")
        print("   - The network is untrained/not functional")
        print("   - Input mapping needs adjustment")
        print("   - Network requires different stimulation")
        print(
            "   The visualization shows spatial layout but not functional clustering."
        )

    print(f"Open {output_filename} in your browser to explore the neural cortex.")


if __name__ == "__main__":
    main()
