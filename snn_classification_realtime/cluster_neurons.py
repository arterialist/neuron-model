import os
import json
import argparse
import sys
from typing import Dict, Any, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm
import warnings
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # Keep for k-distance graph only
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import hashlib
import subprocess
from torchvision import datasets, transforms

# Import binary dataset support
try:
    from build_activity_dataset import LazyActivityDataset

    BINARY_SUPPORT = True
except ImportError:
    BINARY_SUPPORT = False
    LazyActivityDataset = None

# Suppress numpy warnings for autocorrelation calculations
warnings.filterwarnings(
    "ignore", message="invalid value encountered in divide", category=RuntimeWarning
)


# Global rendering and caching configuration
PLOT_IMAGE_SCALE: float = 2.0  # Higher scale => higher DPI for Plotly exports
MATPLOTLIB_DPI: int = 400  # DPI for matplotlib static figures
CACHE_VERSION: str = "v1"


def log_plot_start(plot_name: str, scope: Optional[str] = None) -> None:
    if scope:
        print(f"\n[Plot] Starting {plot_name} ({scope})...")
    else:
        print(f"\n[Plot] Starting {plot_name}...")


def log_plot_end(plot_name: str, scope: Optional[str] = None) -> None:
    if scope:
        print(f"[Plot] Completed {plot_name} ({scope})")
    else:
        print(f"[Plot] Completed {plot_name}")


def compute_dataset_hash(file_path: str) -> str:
    """Return a short MD5 hash for the dataset file using Python's hashlib.

    Handles both single files (JSON) and directories (binary/HDF5 datasets).
    Returns the first 16 hex characters for stable short tokens.
    """
    import hashlib

    # Determine the actual file to hash
    if os.path.isdir(file_path):
        # Binary dataset: hash the HDF5 file within the directory
        h5_path = os.path.join(file_path, "activity_dataset.h5")
        if not os.path.exists(h5_path):
            raise RuntimeError(
                f"Binary dataset directory {file_path} does not contain activity_dataset.h5"
            )
        actual_path = h5_path
    else:
        # Single file (JSON)
        actual_path = file_path

    try:
        with open(actual_path, "rb") as f:
            file_hash = hashlib.md5()
            # Read file in chunks to handle large files
            while chunk := f.read(8192):
                file_hash.update(chunk)
            return file_hash.hexdigest()[:16]
    except (OSError, IOError) as e:
        raise RuntimeError(f"Failed to compute hash for {actual_path}: {e}")


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


def load_activity_data(path: str):
    """Load binary activity dataset.

    Parameters
    ----------
    path: str
        Path to dataset directory containing activity_dataset.h5

    Returns
    -------
    LazyActivityDataset
        Binary dataset object for lazy loading
    """
    if not BINARY_SUPPORT:
        raise ImportError(
            "Binary dataset support not available. Install required dependencies."
        )

    if not os.path.isdir(path):
        raise ValueError(f"Expected directory path for binary dataset, got: {path}")

    print(f"Loading binary dataset from: {path}")
    return LazyActivityDataset(path)


def group_by_image(
    dataset: "LazyActivityDataset",
    max_ticks: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Groups binary dataset samples by image index.

    Args:
        dataset: LazyActivityDataset (binary format)
        max_ticks: Maximum number of ticks to keep per image. If specified, only first N ticks are kept.

    Returns:
        Dictionary mapping image index to sample metadata
    """
    print("Processing binary dataset")
    buckets: Dict[int, Dict[str, Any]] = {}

    for img_idx in tqdm(range(len(dataset)), desc="Grouping binary dataset by image"):
        # Store reference to dataset and index
        sample = dataset[img_idx]
        buckets[img_idx] = {
            "label": int(sample["label"]),
            "binary_sample": sample,
            "dataset": dataset,
            "max_ticks": max_ticks,
        }

    return buckets


def extract_neuron_features(
    image_buckets: Dict[int, Dict[str, Any]],
    num_classes: int = 10,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Extracts comprehensive feature vectors for each neuron across all images using ALL available metrics.

    For each neuron, we create a feature vector that captures:
    - Per-class statistics for ALL metrics (S, F_avg, t_ref, fired)
    - Overall activity statistics
    - Temporal dynamics

    Parameters
    ----------
    image_buckets: Dict[int, Dict[str, Any]]
        Dictionary mapping image index to binary sample metadata
    num_classes: int
        Number of classes in the dataset

    Returns
    -------
    feature_vectors: (num_neurons, num_features) array
    neuron_ids: List of (layer_idx, neuron_idx) tuples identifying each neuron
    """
    print(
        "Extracting comprehensive neuron-wise features using all available metrics..."
    )

    # Determine network structure from binary data
    sample_bucket = next(iter(image_buckets.values()))
    sample = sample_bucket["binary_sample"]
    num_neurons = sample["u"].shape[1]
    network_structure = [num_neurons]  # Single layer

    print(f"Network structure: {network_structure}")
    print("Processing binary format data")

    # Initialize storage for all neuron metrics per class
    # Structure: [layer][neuron][metric_name][class] = list of values
    neuron_metrics = []
    metric_names = ["S", "F_avg", "t_ref", "fired"]

    for num_neurons in network_structure:
        layer_data = []
        for _ in range(num_neurons):
            neuron_data = {
                metric: {cls: [] for cls in range(num_classes)}
                for metric in metric_names
            }
            layer_data.append(neuron_data)
        neuron_metrics.append(layer_data)

    # Collect all metrics for each neuron per class
    print("Collecting all neuron metrics per class...")
    for img_idx, bucket in tqdm(image_buckets.items(), desc="Processing images"):
        bucket_label = bucket.get("label")
        if bucket_label is None:
            raise ValueError(
                f"Image {img_idx} missing label; ensure dataset includes 'label' field"
            )

        label = int(bucket_label)
        if not 0 <= label < num_classes:
            raise ValueError(
                f"Record for image {img_idx} has label {label}, expected 0 <= label < {num_classes}"
            )

        # Extract from binary tensors
        sample = bucket["binary_sample"]
        max_ticks = bucket.get("max_ticks")

        # Get tensors
        u = sample["u"]  # [ticks, neurons]
        t_ref = sample["t_ref"]
        fr = sample["fr"]
        spikes = sample["spikes"]  # [N, 2] where each row is [tick, neuron_idx]

        # Apply max_ticks limit
        if max_ticks is not None:
            u = u[:max_ticks]
            t_ref = t_ref[:max_ticks]
            fr = fr[:max_ticks]
            # Filter spikes to only include those within max_ticks
            if len(spikes) > 0:
                spikes = spikes[spikes[:, 0] < max_ticks]

        ticks = u.shape[0]
        neurons = u.shape[1]

        # Create fired matrix from spikes
        fired = np.zeros((ticks, neurons), dtype=np.int8)
        if len(spikes) > 0:
            for spike_tick, spike_neuron in spikes:
                if spike_tick < ticks and spike_neuron < neurons:
                    fired[int(spike_tick), int(spike_neuron)] = 1

        # Process each neuron
        for layer_idx in range(len(network_structure)):
            for neuron_idx in range(network_structure[layer_idx]):
                # Extract time series for this neuron
                neuron_u = (
                    u[:, neuron_idx].numpy()
                    if hasattr(u, "numpy")
                    else u[:, neuron_idx]
                )
                neuron_t_ref = (
                    t_ref[:, neuron_idx].numpy()
                    if hasattr(t_ref, "numpy")
                    else t_ref[:, neuron_idx]
                )
                neuron_fr = (
                    fr[:, neuron_idx].numpy()
                    if hasattr(fr, "numpy")
                    else fr[:, neuron_idx]
                )
                neuron_fired = fired[:, neuron_idx]

                # Store metrics
                for metric_name, activity in [
                    ("S", neuron_u),
                    ("t_ref", neuron_t_ref),
                    ("F_avg", neuron_fr),
                    ("fired", neuron_fired),
                ]:
                    if len(activity) > 0:
                        mean_val = np.mean(activity)
                        max_val = np.max(activity)
                        std_val = np.std(activity)
                        min_val = np.min(activity)
                        range_val = max_val - min_val
                        median_val = np.median(activity)

                        # Temporal dynamics
                        if len(activity) > 1:
                            activity_lag1 = activity[:-1]
                            activity_lag2 = activity[1:]

                            if np.var(activity_lag1) == 0 or np.var(activity_lag2) == 0:
                                autocorr = 0.0
                            else:
                                autocorr = np.corrcoef(activity_lag1, activity_lag2)[
                                    0, 1
                                ]
                                if np.isnan(autocorr) or np.isinf(autocorr):
                                    autocorr = 0.0

                            time_points = np.arange(len(activity))
                            unique_values = np.unique(activity)

                            if len(unique_values) <= 1:
                                trend = 0.0
                            else:
                                try:
                                    trend = np.polyfit(time_points, activity, 1)[0]
                                    if np.isnan(trend) or np.isinf(trend):
                                        trend = 0.0
                                except (Warning, ValueError):
                                    trend = 0.0

                            max_indices = np.where(activity == np.max(activity))[0]
                            peak_timing = max_indices[0] / len(activity)
                        else:
                            autocorr = 0.0
                            trend = 0.0
                            peak_timing = 0.0

                        neuron_metrics[layer_idx][neuron_idx][metric_name][
                            label
                        ].append(
                            {
                                "mean": mean_val,
                                "max": max_val,
                                "std": std_val,
                                "min": min_val,
                                "range": range_val,
                                "median": median_val,
                                "autocorr": autocorr,
                                "trend": trend,
                                "peak_timing": peak_timing,
                                "activity_sequence": activity,
                            }
                        )

    # Create comprehensive feature vectors
    print("Creating comprehensive feature vectors...")
    feature_vectors = []
    neuron_ids = []

    for layer_idx, layer_data in enumerate(neuron_metrics):
        for neuron_idx, neuron_data in enumerate(layer_data):
            features = []

            # For each metric, compute per-class statistics
            for metric in metric_names:
                class_means = []
                class_maxs = []
                class_stds = []
                class_mins = []
                class_medians = []
                class_ranges = []
                class_autocorrs = []
                class_trends = []
                class_peak_timings = []

                for cls in range(num_classes):
                    presentations = neuron_data[metric][cls]
                    if presentations:
                        # Average across all presentations of this class for each temporal feature
                        class_means.append(np.mean([p["mean"] for p in presentations]))
                        class_maxs.append(np.mean([p["max"] for p in presentations]))
                        class_stds.append(np.mean([p["std"] for p in presentations]))
                        class_mins.append(np.mean([p["min"] for p in presentations]))
                        class_medians.append(
                            np.mean([p["median"] for p in presentations])
                        )
                        class_ranges.append(
                            np.mean([p["range"] for p in presentations])
                        )
                        class_autocorrs.append(
                            np.mean([p["autocorr"] for p in presentations])
                        )
                        class_trends.append(
                            np.mean([p["trend"] for p in presentations])
                        )
                        class_peak_timings.append(
                            np.mean([p["peak_timing"] for p in presentations])
                        )
                    else:
                        class_means.append(0.0)
                        class_maxs.append(0.0)
                        class_stds.append(0.0)
                        class_mins.append(0.0)
                        class_medians.append(0.0)
                        class_ranges.append(0.0)
                        class_autocorrs.append(0.0)
                        class_trends.append(0.0)
                        class_peak_timings.append(0.0)

                # Add per-class features for this metric (comprehensive temporal statistics)
                features.extend(class_means)
                features.extend(class_maxs)
                features.extend(class_stds)
                features.extend(class_mins)
                features.extend(class_medians)
                features.extend(class_ranges)

                # Add temporal dynamics features
                features.extend(class_autocorrs)
                features.extend(class_trends)
                features.extend(class_peak_timings)

                # Add selectivity and variability metrics
                if max(class_means) + np.mean(class_means) > 0:
                    selectivity = (max(class_means) - np.mean(class_means)) / (
                        max(class_means) + np.mean(class_means)
                    )
                else:
                    selectivity = 0.0
                features.append(selectivity)

                # Add overall variability across all temporal features
                features.append(np.std(class_means))
                features.append(np.std(class_autocorrs))
                features.append(np.std(class_peak_timings))

                # Add dynamic range (max - min across classes)
                features.append(max(class_means) - min(class_means))

            feature_vectors.append(features)
            neuron_ids.append((layer_idx, neuron_idx))

    feature_array = np.array(feature_vectors)
    features_per_metric = (
        num_classes * 6 + 9
    )  # 6 per-class stats + 3 selectivity/variability + 3 autocorr/trend/peak features + dynamic range
    total_features = len(metric_names) * features_per_metric

    print(
        f"Created feature vectors: {feature_array.shape[0]} neurons × {feature_array.shape[1]} features"
    )
    print(
        f"Features per neuron: {len(metric_names)} metrics × {features_per_metric} features per metric"
    )
    print(
        f"  Per metric: {num_classes} classes × 6 temporal stats + 9 derived features = {features_per_metric}"
    )
    print(
        f"Total: {len(metric_names)} × {features_per_metric} = {total_features} features per neuron"
    )

    return feature_array, neuron_ids


def extract_synchrony_features(
    image_buckets: Dict[int, Dict[str, Any]],
    max_ticks: Optional[int] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Extracts binary spike matrix for synchrony-based clustering.

    Creates a matrix M of shape (N_neurons × T_ticks) where each entry is:
    - 1 if the neuron fired at that tick (binary spike)
    - 0 if the neuron was silent

    This focuses on synchrony (when neurons fire together) rather than magnitude.

    Parameters
    ----------
    image_buckets: Dict[int, Dict[str, Any]]
        Dictionary mapping image index to binary sample metadata
    max_ticks: Optional[int]
        Maximum number of ticks to include per image

    Returns
    -------
    spike_matrix: (num_neurons, total_ticks) binary array
    neuron_ids: List of (layer_idx, neuron_idx) tuples identifying each neuron
    """
    print("Extracting binary spike matrix for synchrony-based clustering...")
    print("Processing binary format data")

    # Determine network structure from binary data
    sample_bucket = next(iter(image_buckets.values()))
    sample = sample_bucket["binary_sample"]
    num_neurons = sample["u"].shape[1]
    network_structure = [num_neurons]
    total_neurons = num_neurons

    # Calculate total ticks
    total_ticks = 0
    for bucket in image_buckets.values():
        sample = bucket["binary_sample"]
        ticks = sample["u"].shape[0]
        bucket_max_ticks = bucket.get("max_ticks")
        if bucket_max_ticks is not None:
            total_ticks += min(ticks, bucket_max_ticks)
        else:
            total_ticks += ticks

    print(f"Network structure: {network_structure}")
    print(f"Total neurons: {total_neurons}")
    print(f"Total ticks across all images: {total_ticks}")

    # Initialize spike matrix: (neurons, ticks)
    spike_matrix = np.zeros((total_neurons, total_ticks), dtype=np.int8)

    # Populate spike matrix from binary data
    tick_offset = 0
    for img_idx, bucket in tqdm(image_buckets.items(), desc="Processing images"):
        sample = bucket["binary_sample"]
        spikes = sample["spikes"]  # [N, 2] where each row is [tick, neuron_idx]
        bucket_max_ticks = bucket.get("max_ticks")

        # Determine actual ticks for this sample
        sample_ticks = sample["u"].shape[0]
        if bucket_max_ticks is not None:
            sample_ticks = min(sample_ticks, bucket_max_ticks)

        # Populate spike matrix
        if len(spikes) > 0:
            for spike_tick, spike_neuron in spikes:
                spike_tick = int(spike_tick)
                spike_neuron = int(spike_neuron)
                if spike_tick < sample_ticks and spike_neuron < total_neurons:
                    global_tick_idx = tick_offset + spike_tick
                    spike_matrix[spike_neuron, global_tick_idx] = 1

        tick_offset += sample_ticks

    # Create neuron IDs
    neuron_ids = [(0, i) for i in range(total_neurons)]

    print(
        f"Created spike matrix: {spike_matrix.shape[0]} neurons × {spike_matrix.shape[1]} ticks"
    )
    print(f"Sparsity: {(1 - np.sum(spike_matrix) / spike_matrix.size):.1%} zeros")

    return spike_matrix, neuron_ids


def find_optimal_eps(
    X: np.ndarray, min_samples: int = 5, output_dir: Optional[str] = None
) -> float:
    """Find the optimal eps value for DBSCAN using the k-distance graph method."""
    print("Finding optimal eps value using k-distance graph...")

    k_distances = []
    nbrs = NearestNeighbors(n_neighbors=min_samples + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = distances[:, min_samples]

    sorted_distances = np.sort(k_distances)

    first_derivative = np.gradient(sorted_distances)
    second_derivative = np.gradient(first_derivative)

    elbow_idx = np.argmin(second_derivative)
    optimal_eps = sorted_distances[elbow_idx]

    print(f"Optimal eps value found: {optimal_eps:.4f}")

    if output_dir:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(sorted_distances)), sorted_distances, "b-", linewidth=2)
        plt.axvline(
            x=float(elbow_idx),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Optimal eps = {optimal_eps:.4f}",
        )
        plt.xlabel("Points (sorted by k-distance)")
        plt.ylabel(f"{min_samples}-distance")
        plt.title(f"K-Distance Graph (k={min_samples})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        k_distance_plot_path = os.path.join(output_dir, "k_distance_graph.png")
        plt.tight_layout()
        plt.savefig(k_distance_plot_path, dpi=MATPLOTLIB_DPI)
        plt.close()
        print(f"K-distance graph saved to {k_distance_plot_path}")

    return optimal_eps


def cluster_by_correlation(
    spike_matrix: np.ndarray,
    num_clusters: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Perform hierarchical clustering using correlation distance (1 - Pearson r).

    This measures synchrony between neurons - how similarly they fire over time.
    Correlation distance of 0 means perfect synchrony, 1 means perfect anti-correlation.

    Parameters
    ----------
    spike_matrix: np.ndarray
        Binary spike matrix of shape (num_neurons, num_ticks)
    num_clusters: Optional[int]
        Target number of clusters. If None, will determine automatically
    output_dir: Optional[str]
        Directory to save dendrogram plot

    Returns:
        cluster_labels: np.ndarray
            Cluster labels for each neuron
    """
    print("Performing hierarchical clustering using correlation distance...")

    num_neurons, num_ticks = spike_matrix.shape
    print(f"Spike matrix: {num_neurons} neurons × {num_ticks} ticks")

    # Handle edge case of single neuron
    if num_neurons == 1:
        print("Only 1 neuron found - creating single cluster")
        return np.array([0])

    # Compute correlation matrix (Pearson correlation)
    # Need to handle cases where neurons never fire (constant zero vectors)
    correlation_matrix = np.corrcoef(spike_matrix)

    # Handle NaN values (from constant vectors)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

    # Convert to correlation distance: 1 - |correlation|
    # We use absolute value because we care about synchrony regardless of phase
    distance_matrix = 1 - np.abs(correlation_matrix)

    # Ensure diagonal is 0
    np.fill_diagonal(distance_matrix, 0.0)

    print(f"Correlation distance matrix computed: {distance_matrix.shape}")

    # Perform hierarchical clustering using average linkage
    # 'average' linkage is good for correlation-based clustering
    linkage_matrix = linkage(distance_matrix, method="average", metric="euclidean")

    # Determine number of clusters if not specified
    if num_clusters is None:
        # Use a heuristic: find the largest jump in the dendrogram
        # This is similar to finding the "elbow" in other clustering methods
        heights = linkage_matrix[:, 2]
        if len(heights) > 1:
            # Find the largest gap between consecutive heights
            gaps = np.diff(heights[::-1])  # Reverse so we look from leaves up
            max_gap_idx = np.argmax(gaps)
            # The number of clusters is the number of merges at the height of the max gap
            target_clusters = len(heights) - max_gap_idx
            # Reasonable bounds
            target_clusters = max(2, min(target_clusters, num_neurons // 3))
        else:
            target_clusters = max(2, num_neurons // 5)
    else:
        target_clusters = num_clusters

    print(f"Target number of clusters: {target_clusters}")

    # Cut the dendrogram to get cluster labels
    cluster_labels = fcluster(linkage_matrix, target_clusters, criterion="maxclust")

    # Convert to 0-based indexing and make it a numpy array
    cluster_labels = np.array(cluster_labels - 1, dtype=int)

    # Save dendrogram if output directory provided
    if output_dir:
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=8)
        plt.axhline(
            y=linkage_matrix[-target_clusters + 1, 2],
            color="r",
            linestyle="--",
            label=f"Cut for {target_clusters} clusters",
        )
        plt.title(f"Hierarchical Clustering Dendrogram\n(Correlation Distance)")
        plt.xlabel("Neuron Index")
        plt.ylabel("Distance")
        plt.legend()
        plt.tight_layout()

        dendrogram_path = os.path.join(output_dir, "correlation_dendrogram.png")
        plt.savefig(dendrogram_path, dpi=MATPLOTLIB_DPI)
        plt.close()
        print(f"Dendrogram saved to {dendrogram_path}")

    # Print clustering statistics
    unique_clusters = np.unique(cluster_labels)
    print(f"Found {len(unique_clusters)} clusters")
    for cluster_id in unique_clusters:
        count = np.sum(cluster_labels == cluster_id)
        print(f"  Cluster {cluster_id}: {count} neurons")

    return cluster_labels


def plot_neuron_clusters(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves an interactive scatter plot of neuron clusters."""
    log_plot_start("neuron_clusters_2d", "aggregate")
    # Prepare data for plotly
    df_data = {
        "x": X_2d[:, 0],
        "y": X_2d[:, 1],
        "cluster": cluster_labels.astype(str),
        "preferred_class": preferred_classes.astype(str),
        "layer": [neuron_ids[i][0] for i in range(len(neuron_ids))],
        "neuron_idx": [neuron_ids[i][1] for i in range(len(neuron_ids))],
    }

    # Create hover text with detailed information
    hover_text = []
    for i in range(len(neuron_ids)):
        layer_idx, neuron_idx = neuron_ids[i]
        cluster_id = cluster_labels[i]
        pref_class = preferred_classes[i]
        hover_text.append(
            f"Layer: {layer_idx}<br>"
            f"Neuron: {neuron_idx}<br>"
            f"Cluster: {cluster_id}<br>"
            f"Preferred Class: {pref_class}"
        )
    df_data["hover_text"] = hover_text

    # Map cluster labels to colors
    num_clusters = len(set(cluster_labels) - {-1})
    palette = (
        px.colors.qualitative.Plotly
        if num_clusters <= 10
        else px.colors.qualitative.Dark24
    )

    # Create figure
    fig = go.Figure()

    # Markers for different preferred classes
    # Scatter3d supports a limited symbol set — keep to allowed values
    marker_symbols = [
        "circle",
        "circle-open",
        "cross",
        "diamond",
        "diamond-open",
        "square",
        "square-open",
        "x",
    ]

    # Plot each cluster separately for better control
    for cluster_id in sorted(set(cluster_labels)):
        mask = cluster_labels == cluster_id

        if cluster_id == -1:
            # Noise points
            fig.add_trace(
                go.Scatter(
                    x=X_2d[mask, 0],
                    y=X_2d[mask, 1],
                    mode="markers",
                    name="Noise",
                    marker=dict(
                        size=8,
                        color="lightgray",
                        symbol=[
                            marker_symbols[preferred_classes[i] % len(marker_symbols)]
                            for i in range(len(mask))
                            if mask[i]
                        ],
                        line=dict(width=1, color="gray"),
                    ),
                    text=[hover_text[i] for i in range(len(mask)) if mask[i]],
                    hovertemplate="%{text}<extra></extra>",
                    legendgroup="clusters",
                    legendgrouptitle_text="Cell Assemblies",
                )
            )
        else:
            # Regular clusters
            color_idx = cluster_id % len(palette)
            fig.add_trace(
                go.Scatter(
                    x=X_2d[mask, 0],
                    y=X_2d[mask, 1],
                    mode="markers",
                    name=f"Cluster {cluster_id}",
                    marker=dict(
                        size=12,
                        color=palette[color_idx],
                        symbol=[
                            marker_symbols[preferred_classes[i] % len(marker_symbols)]
                            for i in range(len(mask))
                            if mask[i]
                        ],
                        line=dict(width=1, color="black"),
                    ),
                    text=[hover_text[i] for i in range(len(mask)) if mask[i]],
                    hovertemplate="%{text}<extra></extra>",
                    legendgroup="clusters",
                    legendgrouptitle_text="Cell Assemblies",
                )
            )

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    # Add a separate legend for preferred classes (marker shapes)
    unique_pref_classes = sorted(set(preferred_classes.tolist()))
    # Skip legend if preferred classes were not computed (all -1)
    if unique_pref_classes and unique_pref_classes != [-1]:
        for i, cls in enumerate(unique_pref_classes):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=f"Prefers Class {cls}",
                    marker=dict(
                        size=12,
                        color="rgba(0,0,0,0)",  # transparent fill so shape is emphasized
                        line=dict(width=1.5, color="gray"),
                        symbol=marker_symbols[cls % len(marker_symbols)],
                    ),
                    showlegend=True,
                    legendgroup="preferred",
                    legendgrouptitle_text="Preferred Class",
                )
            )

    # Save interactive HTML
    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = output_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    print(f"Interactive neuron cluster visualization saved to {html_path}")

    # Save static image
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Static neuron cluster visualization saved to {output_path}")
    log_plot_end("neuron_clusters_2d", "aggregate")


def plot_neuron_clusters_cloud(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves an interactive cloud/density version of neuron clusters."""
    log_plot_start("neuron_clusters_2d_cloud", "aggregate")
    num_clusters = len(set(cluster_labels) - {-1})
    palette = (
        px.colors.qualitative.Plotly
        if num_clusters <= 10
        else px.colors.qualitative.Dark24
    )

    # Create figure
    fig = go.Figure()

    # Create a meshgrid for the density plot
    x_min, x_max = X_2d[:, 0].min() - 0.2, X_2d[:, 0].max() + 0.2
    y_min, y_max = X_2d[:, 1].min() - 0.2, X_2d[:, 1].max() + 0.2
    xx, yy = np.mgrid[x_min:x_max:80j, y_min:y_max:80j]

    # Plot each cluster as a colored cloud with contours
    for cluster_id in sorted(set(cluster_labels) - {-1}):
        mask = cluster_labels == cluster_id
        cluster_points = X_2d[mask]

        # Analyze preferred classes in this cluster
        cluster_prefs = preferred_classes[mask]
        unique_prefs, counts = np.unique(cluster_prefs, return_counts=True)
        pref_summary = ", ".join(
            [f"{cls}:{cnt}" for cls, cnt in zip(unique_prefs, counts)]
        )

        print(
            f"  Cluster {cluster_id}: {len(cluster_points)} points, preferred classes: {pref_summary}"
        )

        n_points = len(cluster_points)
        color_idx = cluster_id % len(palette)

        if n_points < 3:
            # Too few points for KDE: show as scatter instead of skipping
            fig.add_trace(
                go.Scatter(
                    x=cluster_points[:, 0],
                    y=cluster_points[:, 1],
                    mode="markers",
                    name=f"Cluster {cluster_id} ({pref_summary})",
                    marker=dict(
                        size=10,
                        color=palette[color_idx],
                        line=dict(width=1, color="black"),
                    ),
                    opacity=0.8,
                )
            )
        else:
            # Create kernel density estimate
            try:
                if n_points > 10:
                    bandwidth = n_points ** (-1 / 6) * 0.5
                else:
                    bandwidth = 0.1

                kde = gaussian_kde(cluster_points.T, bw_method=bandwidth)
                positions = np.vstack([xx.ravel(), yy.ravel()])
                zz = np.reshape(kde(positions), xx.shape)

                # Normalize density
                max_density = np.max(zz)
                if max_density > 0:
                    zz = zz / max_density

                    # Add contour plot for this cluster
                    fig.add_trace(
                        go.Contour(
                            x=xx[0, :],
                            y=yy[:, 0],
                            z=zz,
                            contours=dict(
                                start=0.1,
                                end=1.0,
                                size=0.15,
                            ),
                            colorscale=[
                                [0, "rgba(255,255,255,0)"],
                                [1, palette[color_idx]],
                            ],
                            showscale=False,
                            name=f"Cluster {cluster_id} ({pref_summary})",
                            line=dict(width=2),
                            opacity=0.6,
                            hoverinfo="name",
                        )
                    )

            except (np.linalg.LinAlgError, ValueError):
                # Fallback to scatter if KDE fails
                fig.add_trace(
                    go.Scatter(
                        x=cluster_points[:, 0],
                        y=cluster_points[:, 1],
                        mode="markers",
                        name=f"Cluster {cluster_id} ({pref_summary})",
                        marker=dict(
                            size=10,
                            color=palette[color_idx],
                            line=dict(width=1, color="black"),
                        ),
                        opacity=0.6,
                    )
                )

    # Add noise points
    noise_mask = cluster_labels == -1
    if np.sum(noise_mask) > 0:
        noise_points = X_2d[noise_mask]
        fig.add_trace(
            go.Scatter(
                x=noise_points[:, 0],
                y=noise_points[:, 1],
                mode="markers",
                name="Noise",
                marker=dict(
                    size=6,
                    color="lightgray",
                    symbol="x",
                    line=dict(width=1, color="gray"),
                ),
                opacity=0.4,
            )
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title.replace(
                "Clustering", "Cloud Clustering (with Preferred Classes)"
            ),
            font=dict(size=18),
        ),
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(
            title="Cell Assemblies<br>(Preferred Classes)",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            font=dict(size=10),
        ),
    )

    # Save interactive HTML
    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = output_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    print(f"Interactive neuron cluster cloud visualization saved to {html_path}")

    # Save static image
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Static neuron cluster cloud visualization saved to {output_path}")
    log_plot_end("neuron_clusters_2d_cloud", "aggregate")


def plot_neuron_clusters_3d(
    X_3d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves an interactive 3D scatter plot of neuron clusters."""
    log_plot_start("neuron_clusters_3d", "aggregate")
    # Prepare data for plotly
    hover_text = []
    for i in range(len(neuron_ids)):
        layer_idx, neuron_idx = neuron_ids[i]
        cluster_id = cluster_labels[i]
        pref_class = preferred_classes[i]
        hover_text.append(
            f"Layer: {layer_idx}<br>"
            f"Neuron: {neuron_idx}<br>"
            f"Cluster: {cluster_id}<br>"
            f"Preferred Class: {pref_class}"
        )

    # Map cluster labels to colors
    num_clusters = len(set(cluster_labels) - {-1})
    palette = (
        px.colors.qualitative.Plotly
        if num_clusters <= 10
        else px.colors.qualitative.Dark24
    )

    # Create figure
    fig = go.Figure()

    # Markers for different preferred classes (Scatter3d supports a limited set)
    marker_symbols = [
        "circle",
        "circle-open",
        "cross",
        "diamond",
        "diamond-open",
        "square",
        "square-open",
        "x",
    ]

    # Plot each cluster separately
    for cluster_id in sorted(set(cluster_labels)):
        mask = cluster_labels == cluster_id

        if cluster_id == -1:
            # Noise points
            fig.add_trace(
                go.Scatter3d(
                    x=X_3d[mask, 0],
                    y=X_3d[mask, 1],
                    z=X_3d[mask, 2],
                    mode="markers",
                    name="Noise",
                    marker=dict(
                        size=5,
                        color="lightgray",
                        symbol=[
                            marker_symbols[preferred_classes[i] % len(marker_symbols)]
                            for i in range(len(mask))
                            if mask[i]
                        ],
                        line=dict(width=1, color="gray"),
                    ),
                    text=[hover_text[i] for i in range(len(mask)) if mask[i]],
                    hovertemplate="%{text}<extra></extra>",
                    legendgroup="clusters",
                    legendgrouptitle_text="Cell Assemblies",
                )
            )
        else:
            # Regular clusters
            color_idx = cluster_id % len(palette)
            fig.add_trace(
                go.Scatter3d(
                    x=X_3d[mask, 0],
                    y=X_3d[mask, 1],
                    z=X_3d[mask, 2],
                    mode="markers",
                    name=f"Cluster {cluster_id}",
                    marker=dict(
                        size=8,
                        color=palette[color_idx],
                        symbol=[
                            marker_symbols[preferred_classes[i] % len(marker_symbols)]
                            for i in range(len(mask))
                            if mask[i]
                        ],
                        line=dict(width=1, color="black"),
                    ),
                    text=[hover_text[i] for i in range(len(mask)) if mask[i]],
                    hovertemplate="%{text}<extra></extra>",
                    legendgroup="clusters",
                    legendgrouptitle_text="Cell Assemblies",
                )
            )

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            zaxis_title="t-SNE Component 3",
        ),
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    # Add a separate legend for preferred classes (marker shapes) in 3D
    unique_pref_classes = sorted(set(preferred_classes.tolist()))
    # Skip legend if preferred classes were not computed (all -1)
    if unique_pref_classes and unique_pref_classes != [-1]:
        for i, cls in enumerate(unique_pref_classes):
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    name=f"Prefers Class {cls}",
                    marker=dict(
                        size=8,
                        color="rgba(0,0,0,0)",
                        line=dict(width=1.5, color="gray"),
                        symbol=marker_symbols[cls % len(marker_symbols)],
                    ),
                    showlegend=True,
                    legendgroup="preferred",
                    legendgrouptitle_text="Preferred Class",
                )
            )

    # Save interactive HTML
    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = output_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    print(f"Interactive 3D neuron cluster visualization saved to {html_path}")

    # Save static image
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Static 3D neuron cluster visualization saved to {output_path}")
    log_plot_end("neuron_clusters_3d", "aggregate")


def plot_neuron_clusters_cloud_3d(
    X_3d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves an interactive 3D cloud/volume visualization of neuron clusters."""
    log_plot_start("neuron_clusters_3d_cloud", "aggregate")
    num_clusters = len(set(cluster_labels) - {-1})
    palette = (
        px.colors.qualitative.Plotly
        if num_clusters <= 10
        else px.colors.qualitative.Dark24
    )

    # Create figure
    fig = go.Figure()

    # Create a 3D meshgrid for the density plot
    x_min, x_max = X_3d[:, 0].min() - 0.3, X_3d[:, 0].max() + 0.3
    y_min, y_max = X_3d[:, 1].min() - 0.3, X_3d[:, 1].max() + 0.3
    z_min, z_max = X_3d[:, 2].min() - 0.3, X_3d[:, 2].max() + 0.3

    xx, yy, zz_grid = np.mgrid[x_min:x_max:40j, y_min:y_max:40j, z_min:z_max:40j]

    # Plot each cluster as a colored cloud
    for cluster_id in sorted(set(cluster_labels) - {-1}):
        mask = cluster_labels == cluster_id
        cluster_points = X_3d[mask]

        # Analyze preferred classes in this cluster
        cluster_prefs = preferred_classes[mask]
        unique_prefs, counts = np.unique(cluster_prefs, return_counts=True)
        pref_summary = ", ".join(
            [f"{cls}:{cnt}" for cls, cnt in zip(unique_prefs, counts)]
        )

        print(
            f"  Cluster {cluster_id}: {len(cluster_points)} points, preferred classes: {pref_summary}"
        )

        color_idx = cluster_id % len(palette)

        # ALWAYS show scatter markers so small/sparse clusters are visible
        fig.add_trace(
            go.Scatter3d(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                z=cluster_points[:, 2],
                mode="markers",
                name=f"Cluster {cluster_id} ({pref_summary})",
                marker=dict(size=4, color=palette[color_idx], opacity=0.7),
                showlegend=True,
                legendgroup=f"cluster-{cluster_id}",
            )
        )

        # Try to add a density blob when feasible
        try:
            n_points = len(cluster_points)
            # Use fixed bandwidth to match preferred-class cloud look
            bandwidth = 0.5

            if n_points >= 3:  # KDE needs at least 3 points in 3D to be stable
                kde = gaussian_kde(cluster_points.T, bw_method=bandwidth)
                positions = np.vstack([xx.ravel(), yy.ravel(), zz_grid.ravel()])
                density = np.reshape(kde(positions), xx.shape)

                max_d = np.max(density)
                if max_d > 0:
                    density = density / max_d
                    fig.add_trace(
                        go.Isosurface(
                            x=xx.ravel(),
                            y=yy.ravel(),
                            z=zz_grid.ravel(),
                            value=density.ravel(),
                            isomin=0.2,
                            isomax=0.9,
                            surface_count=3,
                            caps=dict(x_show=False, y_show=False, z_show=False),
                            showscale=False,
                            colorscale=[
                                [0, palette[color_idx]],
                                [1, palette[color_idx]],
                            ],
                            opacity=0.25,
                            name=f"Cluster {cluster_id} density",
                            legendgroup=f"cluster-{cluster_id}",
                            showlegend=False,
                        )
                    )
        except (np.linalg.LinAlgError, ValueError):
            pass

    # Add noise points
    noise_mask = cluster_labels == -1
    if np.sum(noise_mask) > 0:
        noise_points = X_3d[noise_mask]
        fig.add_trace(
            go.Scatter3d(
                x=noise_points[:, 0],
                y=noise_points[:, 1],
                z=noise_points[:, 2],
                mode="markers",
                name="Noise",
                marker=dict(size=5, color="lightgray", symbol="x", opacity=0.4),
            )
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title.replace(
                "Clustering", "3D Cloud Clustering (with Preferred Classes)"
            ),
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            zaxis_title="t-SNE Component 3",
        ),
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(
            title="Cell Assemblies<br>(Preferred Classes)",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            font=dict(size=10),
        ),
    )

    # Save interactive HTML
    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = output_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    print(f"Interactive 3D neuron cluster cloud visualization saved to {html_path}")

    # Save static image
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Static 3D neuron cluster cloud visualization saved to {output_path}")
    log_plot_end("neuron_clusters_3d_cloud", "aggregate")


def plot_brain_region_map(
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    feature_vectors: np.ndarray,
    output_path: str,
):
    """Creates an interactive 'brain region map' showing layer-wise organization of cell assemblies."""
    log_plot_start("brain_region_map", "aggregate")
    # Organize neurons by layer
    layers = {}
    for i, (layer_idx, neuron_idx) in enumerate(neuron_ids):
        if layer_idx not in layers:
            layers[layer_idx] = []
        layers[layer_idx].append(
            {
                "neuron_idx": neuron_idx,
                "cluster": cluster_labels[i],
                "preferred_class": preferred_classes[i],
                "features": feature_vectors[i],
            }
        )

    num_layers = len(layers)
    num_clusters = len(set(cluster_labels) - {-1})
    palette = (
        px.colors.qualitative.Dark24
        if num_clusters > 10
        else px.colors.qualitative.Set1
    )

    # Create subplots - one column per layer
    fig = make_subplots(
        rows=1,
        cols=num_layers,
        subplot_titles=[f"Layer {i}" for i in sorted(layers.keys())],
        horizontal_spacing=0.05,
    )

    # Track which clusters have been added to legend
    legend_clusters = set()

    for col_idx, layer_idx in enumerate(sorted(layers.keys()), start=1):
        layer_neurons = layers[layer_idx]
        layer_neurons.sort(key=lambda x: x["neuron_idx"])

        # Prepare data for this layer
        neuron_indices = [n["neuron_idx"] for n in layer_neurons]
        preferred = [n["preferred_class"] for n in layer_neurons]
        clusters = [n["cluster"] for n in layer_neurons]

        # Create bars colored by cluster
        for neuron in layer_neurons:
            cluster_id = neuron["cluster"]
            neuron_idx = neuron["neuron_idx"]
            pref_class = neuron["preferred_class"]

            if cluster_id == -1:
                color = "lightgray"
                name = "Noise"
            else:
                color = palette[cluster_id % len(palette)]
                name = f"Assembly {cluster_id}"

            # Only show in legend once
            show_legend = cluster_id not in legend_clusters
            if show_legend:
                legend_clusters.add(cluster_id)

            fig.add_trace(
                go.Bar(
                    x=[neuron_idx],
                    y=[1],
                    marker=dict(color=color, line=dict(color="black", width=1)),
                    name=name,
                    text=[f"Prefers: {pref_class}"],
                    hovertemplate=f"Neuron: {neuron_idx}<br>Cluster: {cluster_id}<br>Preferred Class: {pref_class}<extra></extra>",
                    showlegend=show_legend,
                    legendgroup=name,
                ),
                row=1,
                col=col_idx,
            )

        # Update axes for this subplot
        fig.update_xaxes(title_text="Neuron Index", row=1, col=col_idx)
        if col_idx == 1:
            fig.update_yaxes(
                title_text="Cell Assembly", row=1, col=col_idx, showticklabels=False
            )
        else:
            fig.update_yaxes(showticklabels=False, row=1, col=col_idx)

    # Update overall layout
    fig.update_layout(
        title=dict(
            text='Neural Network "Brain Region" Map<br><sub>(Hover for details, colors show cell assemblies)</sub>',
            font=dict(size=18),
            x=0.5,
            xanchor="center",
        ),
        width=max(400 * num_layers, 1200),
        height=600,
        showlegend=True,
        legend=dict(
            title="Cell Assemblies",
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        barmode="overlay",
        hovermode="closest",
    )

    # Save interactive HTML
    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = output_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    print(f"Interactive brain region map saved to {html_path}")

    # Save static image
    fig.write_image(
        output_path,
        width=max(400 * num_layers, 1200),
        height=600,
        scale=PLOT_IMAGE_SCALE,
    )
    print(f"Static brain region map saved to {output_path}")
    log_plot_end("brain_region_map", "aggregate")


def plot_layered_clusters_3d(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """Render a 3D plot where X/Y come from 2D embedding and Z encodes layer index.

    This preserves feature-based clustering in the X/Y plane while arranging neurons
    along Z by their network layer to reflect topology.
    """
    # Prepare hover text
    hover_text = []
    for i in range(len(neuron_ids)):
        layer_idx, neuron_idx = neuron_ids[i]
        cluster_id = cluster_labels[i]
        pref_class = preferred_classes[i]
        hover_text.append(
            f"Layer: {layer_idx}<br>Neuron: {neuron_idx}<br>Cluster: {cluster_id}<br>Preferred Class: {pref_class}"
        )

    # Colors
    num_clusters = len(set(cluster_labels) - {-1})
    palette = (
        px.colors.qualitative.Plotly
        if num_clusters <= 10
        else px.colors.qualitative.Dark24
    )

    # Z coordinate is layer index; slightly jitter within layer to avoid exact overlaps
    z_base = np.array([nid[0] for nid in neuron_ids], dtype=float)
    rng = np.random.default_rng(42)
    z_jitter = rng.normal(0.0, 0.03, size=len(z_base))
    z_coords = z_base + z_jitter

    # Create figure
    fig = go.Figure()

    # Optional: draw translucent planes for each layer
    unique_layers = sorted(set(int(z) for z in z_base))
    x_min, x_max = float(np.min(X_2d[:, 0])), float(np.max(X_2d[:, 0]))
    y_min, y_max = float(np.min(X_2d[:, 1])), float(np.max(X_2d[:, 1]))

    for li in unique_layers:
        fig.add_trace(
            go.Mesh3d(
                x=[x_min, x_max, x_max, x_min],
                y=[y_min, y_min, y_max, y_max],
                z=[li, li, li, li],
                opacity=0.08,
                color="lightgray",
                name=f"Layer {li}",
                hoverinfo="skip",
                showscale=False,
                showlegend=False,
            )
        )

    # Marker symbol set similar to 3D scatter
    marker_symbols = [
        "circle",
        "circle-open",
        "cross",
        "diamond",
        "diamond-open",
        "square",
        "square-open",
        "x",
    ]

    # Plot points by cluster
    for cluster_id in sorted(set(cluster_labels)):
        mask = cluster_labels == cluster_id
        if cluster_id == -1:
            fig.add_trace(
                go.Scatter3d(
                    x=X_2d[mask, 0],
                    y=X_2d[mask, 1],
                    z=z_coords[mask],
                    mode="markers",
                    name="Noise",
                    marker=dict(
                        size=5,
                        color="lightgray",
                        symbol=[
                            marker_symbols[preferred_classes[i] % len(marker_symbols)]
                            for i in range(len(mask))
                            if mask[i]
                        ],
                        line=dict(width=1, color="gray"),
                    ),
                    text=[hover_text[i] for i in range(len(mask)) if mask[i]],
                    hovertemplate="%{text}<extra></extra>",
                    legendgroup="clusters",
                    legendgrouptitle_text="Cell Assemblies",
                )
            )
        else:
            color_idx = cluster_id % len(palette)
            fig.add_trace(
                go.Scatter3d(
                    x=X_2d[mask, 0],
                    y=X_2d[mask, 1],
                    z=z_coords[mask],
                    mode="markers",
                    name=f"Cluster {cluster_id}",
                    marker=dict(
                        size=7,
                        color=palette[color_idx],
                        symbol=[
                            marker_symbols[preferred_classes[i] % len(marker_symbols)]
                            for i in range(len(mask))
                            if mask[i]
                        ],
                        line=dict(width=1, color="black"),
                    ),
                    text=[hover_text[i] for i in range(len(mask)) if mask[i]],
                    hovertemplate="%{text}<extra></extra>",
                    legendgroup="clusters",
                    legendgrouptitle_text="Cell Assemblies",
                )
            )

    # Add legend entries for preferred classes
    unique_pref_classes = sorted(set(preferred_classes.tolist()))
    # Skip legend if preferred classes were not computed (all -1)
    if unique_pref_classes and unique_pref_classes != [-1]:
        for cls in unique_pref_classes:
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    name=f"Prefers Class {cls}",
                    marker=dict(
                        size=8,
                        color="rgba(0,0,0,0)",
                        line=dict(width=1.5, color="gray"),
                        symbol=marker_symbols[cls % len(marker_symbols)],
                    ),
                    showlegend=True,
                    legendgroup="preferred",
                    legendgrouptitle_text="Preferred Class",
                )
            )

    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            zaxis_title="Layer Index (Topology)",
        ),
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    # Save
    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = output_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    print(f"Interactive layered 3D visualization saved to {html_path}")
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Static layered 3D visualization saved to {output_path}")
    log_plot_end("neuron_clusters_layered_3d", "aggregate")


def plot_by_preferred_2d(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """2D scatter colored by preferred class, marker shape by cluster id."""
    log_plot_start("preferred_class_2d", "aggregate")
    # Hover text
    hover_text = []
    for i in range(len(neuron_ids)):
        layer_idx, neuron_idx = neuron_ids[i]
        cluster_id = cluster_labels[i]
        pref_class = preferred_classes[i]
        hover_text.append(
            f"Layer: {layer_idx}<br>Neuron: {neuron_idx}<br>Preferred Class: {pref_class}<br>Cluster: {cluster_id}"
        )

    unique_prefs = sorted(set(preferred_classes.tolist()))
    palette = px.colors.qualitative.Plotly

    # Marker symbols for clusters
    marker_symbols = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
        "star",
        "hexagon",
        "pentagon",
    ]

    fig = go.Figure()

    for c in unique_prefs:
        mask = preferred_classes == c
        color_idx = int(c) % len(palette)
        # symbol per point by cluster id
        symbols = [
            (
                marker_symbols[int(cluster_labels[i]) % len(marker_symbols)]
                if cluster_labels[i] != -1
                else "circle-open"
            )
            for i in range(len(mask))
            if mask[i]
        ]
        fig.add_trace(
            go.Scatter(
                x=X_2d[mask, 0],
                y=X_2d[mask, 1],
                mode="markers",
                name=f"Prefers Class {int(c)}",
                marker=dict(
                    size=12,
                    color=palette[color_idx],
                    symbol=symbols,
                    line=dict(width=1, color="black"),
                ),
                text=[hover_text[i] for i in range(len(mask)) if mask[i]],
                hovertemplate="%{text}<extra></extra>",
                legendgroup="preferred_color",
                legendgrouptitle_text="Preferred Class",
            )
        )

    # Add cluster legend (shapes)
    unique_clusters = sorted(set(cluster_labels.tolist()))
    for cid in unique_clusters:
        name = "Noise" if cid == -1 else f"Cluster {cid}"
        symbol = (
            "circle-open"
            if cid == -1
            else marker_symbols[int(cid) % len(marker_symbols)]
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=name,
                marker=dict(
                    size=12,
                    color="rgba(0,0,0,0)",
                    line=dict(width=1.5, color="gray"),
                    symbol=symbol,
                ),
                showlegend=True,
                legendgroup="clusters_shape",
                legendgrouptitle_text="Clusters (shape)",
            )
        )

    fig.update_layout(
        title=dict(
            text=title.replace("Clustering of Neurons", "2D by Preferred Class"),
            font=dict(size=18),
        ),
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = output_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Preferred-class 2D visualization saved to {html_path} and {output_path}")
    log_plot_end("preferred_class_2d", "aggregate")


def plot_by_preferred_2d_cloud(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """2D cloud colored by preferred class; always overlay scatter for visibility."""
    log_plot_start("preferred_class_2d_cloud", "aggregate")
    palette = px.colors.qualitative.Plotly
    fig = go.Figure()

    # Grid for KDE
    x_min, x_max = X_2d[:, 0].min() - 0.2, X_2d[:, 0].max() + 0.2
    y_min, y_max = X_2d[:, 1].min() - 0.2, X_2d[:, 1].max() + 0.2
    xx, yy = np.mgrid[x_min:x_max:80j, y_min:y_max:80j]

    unique_prefs = sorted(set(preferred_classes.tolist()))
    for c in unique_prefs:
        mask = preferred_classes == c
        pts = X_2d[mask]
        color = palette[int(c) % len(palette)]

        # Always add scatter
        fig.add_trace(
            go.Scatter(
                x=pts[:, 0],
                y=pts[:, 1],
                mode="markers",
                name=f"Class {int(c)}",
                marker=dict(size=6, color=color, line=dict(width=0.5, color="black")),
                opacity=0.6,
            )
        )

        try:
            if len(pts) >= 3:
                kde = gaussian_kde(pts.T, bw_method=0.3)
                grid = np.vstack([xx.ravel(), yy.ravel()])
                den = np.reshape(kde(grid), xx.shape)
                den = den / np.max(den)
                fig.add_trace(
                    go.Contour(
                        x=xx[0, :],
                        y=yy[:, 0],
                        z=den,
                        contours=dict(start=0.1, end=1.0, size=0.15),
                        colorscale=[[0, "rgba(255,255,255,0)"], [1, color]],
                        showscale=False,
                        name=f"Class {int(c)} density",
                        line=dict(width=2),
                        opacity=0.5,
                        hoverinfo="skip",
                    )
                )
        except (np.linalg.LinAlgError, ValueError):
            pass

    fig.update_layout(
        title=dict(
            text=title.replace("Clustering of Neurons", "2D Cloud by Preferred Class"),
            font=dict(size=18),
        ),
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = output_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Preferred-class 2D cloud saved to {html_path} and {output_path}")
    log_plot_end("preferred_class_2d_cloud", "aggregate")


def plot_by_preferred_3d(
    X_3d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """3D scatter colored by preferred class; marker shape by cluster id."""
    log_plot_start("preferred_class_3d", "aggregate")
    hover_text = []
    for i in range(len(neuron_ids)):
        layer_idx, neuron_idx = neuron_ids[i]
        cluster_id = cluster_labels[i]
        pref_class = preferred_classes[i]
        hover_text.append(
            f"Layer: {layer_idx}<br>Neuron: {neuron_idx}<br>Preferred Class: {pref_class}<br>Cluster: {cluster_id}"
        )

    palette = px.colors.qualitative.Plotly
    marker_symbols = [
        "circle",
        "circle-open",
        "cross",
        "diamond",
        "diamond-open",
        "square",
        "square-open",
        "x",
    ]

    fig = go.Figure()
    unique_prefs = sorted(set(preferred_classes.tolist()))
    for c in unique_prefs:
        mask = preferred_classes == c
        color = palette[int(c) % len(palette)]
        symbols = [
            (
                marker_symbols[int(cluster_labels[i]) % len(marker_symbols)]
                if cluster_labels[i] != -1
                else "circle-open"
            )
            for i in range(len(mask))
            if mask[i]
        ]
        fig.add_trace(
            go.Scatter3d(
                x=X_3d[mask, 0],
                y=X_3d[mask, 1],
                z=X_3d[mask, 2],
                mode="markers",
                name=f"Class {int(c)}",
                marker=dict(
                    size=7,
                    color=color,
                    symbol=symbols,
                    line=dict(width=1, color="black"),
                ),
                text=[hover_text[i] for i in range(len(mask)) if mask[i]],
                hovertemplate="%{text}<extra></extra>",
                legendgroup="preferred_color",
                legendgrouptitle_text="Preferred Class",
            )
        )

    # Add cluster legend by shape
    unique_clusters = sorted(set(cluster_labels.tolist()))
    for cid in unique_clusters:
        name = "Noise" if cid == -1 else f"Cluster {cid}"
        symbol = (
            "circle-open"
            if cid == -1
            else marker_symbols[int(cid) % len(marker_symbols)]
        )
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                name=name,
                marker=dict(
                    size=8,
                    color="rgba(0,0,0,0)",
                    line=dict(width=1.5, color="gray"),
                    symbol=symbol,
                ),
                showlegend=True,
                legendgroup="clusters_shape",
                legendgrouptitle_text="Clusters (shape)",
            )
        )

    fig.update_layout(
        title=dict(
            text=title.replace("Clustering of Neurons", "3D by Preferred Class"),
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            zaxis_title="t-SNE Component 3",
        ),
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = output_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Preferred-class 3D visualization saved to {html_path} and {output_path}")
    log_plot_end("preferred_class_3d", "aggregate")


def plot_by_preferred_3d_cloud(
    X_3d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """3D cloud colored by preferred class; always overlay scatter."""
    log_plot_start("preferred_class_3d_cloud", "aggregate")
    palette = px.colors.qualitative.Plotly
    fig = go.Figure()

    # Grid for volume density
    x_min, x_max = X_3d[:, 0].min() - 0.3, X_3d[:, 0].max() + 0.3
    y_min, y_max = X_3d[:, 1].min() - 0.3, X_3d[:, 1].max() + 0.3
    z_min, z_max = X_3d[:, 2].min() - 0.3, X_3d[:, 2].max() + 0.3
    xx, yy, zz_grid = np.mgrid[x_min:x_max:40j, y_min:y_max:40j, z_min:z_max:40j]

    unique_prefs = sorted(set(preferred_classes.tolist()))
    for c in unique_prefs:
        mask = preferred_classes == c
        pts = X_3d[mask]
        color = palette[int(c) % len(palette)]

        # Always add scatter
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=f"Class {int(c)}",
                marker=dict(size=4, color=color, opacity=0.7),
            )
        )

        try:
            if len(pts) >= 3:
                kde = gaussian_kde(pts.T, bw_method=0.3)
                positions = np.vstack([xx.ravel(), yy.ravel(), zz_grid.ravel()])
                density = np.reshape(kde(positions), xx.shape)
                max_d = np.max(density)
                if max_d > 0:
                    density = density / max_d
                    fig.add_trace(
                        go.Isosurface(
                            x=xx.ravel(),
                            y=yy.ravel(),
                            z=zz_grid.ravel(),
                            value=density.ravel(),
                            isomin=0.2,
                            isomax=0.9,
                            surface_count=3,
                            caps=dict(x_show=False, y_show=False, z_show=False),
                            showscale=False,
                            colorscale=[[0, color], [1, color]],
                            opacity=0.25,
                            name=f"Class {int(c)} density",
                            showlegend=False,
                        )
                    )
        except (np.linalg.LinAlgError, ValueError):
            pass

    fig.update_layout(
        title=dict(
            text=title.replace("Clustering of Neurons", "3D Cloud by Preferred Class"),
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            zaxis_title="t-SNE Component 3",
        ),
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = output_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Preferred-class 3D cloud saved to {html_path} and {output_path}")
    log_plot_end("preferred_class_3d_cloud", "aggregate")


def plot_layered_by_preferred_3d(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """Layered 3D view using layer index as Z, colored by preferred class."""
    log_plot_start("preferred_class_layered_3d", "aggregate")
    hover_text = []
    for i in range(len(neuron_ids)):
        layer_idx, neuron_idx = neuron_ids[i]
        cluster_id = cluster_labels[i]
        pref_class = preferred_classes[i]
        hover_text.append(
            f"Layer: {layer_idx}<br>Neuron: {neuron_idx}<br>Preferred Class: {pref_class}<br>Cluster: {cluster_id}"
        )

    palette = px.colors.qualitative.Plotly
    z_base = np.array([nid[0] for nid in neuron_ids], dtype=float)
    rng = np.random.default_rng(42)
    z_coords = z_base + rng.normal(0.0, 0.03, size=len(z_base))

    fig = go.Figure()
    unique_layers = sorted(set(int(z) for z in z_base))
    x_min, x_max = float(np.min(X_2d[:, 0])), float(np.max(X_2d[:, 0]))
    y_min, y_max = float(np.min(X_2d[:, 1])), float(np.max(X_2d[:, 1]))
    for li in unique_layers:
        fig.add_trace(
            go.Mesh3d(
                x=[x_min, x_max, x_max, x_min],
                y=[y_min, y_min, y_max, y_max],
                z=[li, li, li, li],
                opacity=0.08,
                color="lightgray",
                name=f"Layer {li}",
                hoverinfo="skip",
                showscale=False,
                showlegend=False,
            )
        )

    unique_prefs = sorted(set(preferred_classes.tolist()))
    for c in unique_prefs:
        mask = preferred_classes == c
        color = palette[int(c) % len(palette)]
        fig.add_trace(
            go.Scatter3d(
                x=X_2d[mask, 0],
                y=X_2d[mask, 1],
                z=z_coords[mask],
                mode="markers",
                name=f"Class {int(c)}",
                marker=dict(size=6, color=color, line=dict(width=1, color="black")),
                text=[hover_text[i] for i in range(len(mask)) if mask[i]],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=title.replace(
                "Layered 3D Clustering (X/Y=t-SNE, Z=Layer Index)",
                "Layered 3D by Preferred Class",
            ),
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            zaxis_title="Layer Index (Topology)",
        ),
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = output_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Preferred-class layered 3D saved to {html_path} and {output_path}")
    log_plot_end("preferred_class_layered_3d", "aggregate")


def plot_layered_clusters_3d_cloud(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """Topology-aware 3D cloud: X/Y from 2D embedding, Z = layer index; colored by cluster.

    Always overlays scatter markers so small clusters are visible. Adds volumetric
    isosurfaces per cluster when KDE is feasible.
    """
    # Colors
    num_clusters = len(set(cluster_labels) - {-1})
    palette = (
        px.colors.qualitative.Plotly
        if num_clusters <= 10
        else px.colors.qualitative.Dark24
    )

    # Z from layers with slight jitter
    z_base = np.array([nid[0] for nid in neuron_ids], dtype=float)
    rng = np.random.default_rng(42)
    z_coords = z_base + rng.normal(0.0, 0.03, size=len(z_base))

    # Grid for density
    x_min, x_max = float(np.min(X_2d[:, 0])) - 0.3, float(np.max(X_2d[:, 0])) + 0.3
    y_min, y_max = float(np.min(X_2d[:, 1])) - 0.3, float(np.max(X_2d[:, 1])) + 0.3
    z_min, z_max = float(np.min(z_coords)) - 0.15, float(np.max(z_coords)) + 0.15
    xx, yy, zz_grid = np.mgrid[x_min:x_max:40j, y_min:y_max:40j, z_min:z_max:40j]

    # Hover text
    hover_text = []
    for i in range(len(neuron_ids)):
        layer_idx, neuron_idx = neuron_ids[i]
        cluster_id = cluster_labels[i]
        pref_class = preferred_classes[i]
        hover_text.append(
            f"Layer: {layer_idx}<br>Neuron: {neuron_idx}<br>Cluster: {cluster_id}<br>Preferred Class: {pref_class}"
        )

    fig = go.Figure()

    # Layer planes
    unique_layers = sorted(set(int(z) for z in z_base))
    for li in unique_layers:
        fig.add_trace(
            go.Mesh3d(
                x=[x_min, x_max, x_max, x_min],
                y=[y_min, y_min, y_max, y_max],
                z=[li, li, li, li],
                opacity=0.08,
                color="lightgray",
                name=f"Layer {li}",
                hoverinfo="skip",
                showscale=False,
                showlegend=False,
            )
        )

    for cluster_id in sorted(set(cluster_labels)):
        mask = cluster_labels == cluster_id
        pts = np.column_stack((X_2d[mask, 0], X_2d[mask, 1], z_coords[mask]))

        if cluster_id == -1:
            # Noise scatter only
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    name="Noise",
                    marker=dict(size=4, color="lightgray", symbol="x", opacity=0.6),
                )
            )
            continue

        color_idx = cluster_id % len(palette)

        # Scatter overlay for visibility
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=f"Cluster {cluster_id}",
                marker=dict(size=4, color=palette[color_idx], opacity=0.7),
                text=[hover_text[i] for i in range(len(mask)) if mask[i]],
                hovertemplate="%{text}<extra></extra>",
                legendgroup=f"cluster-{cluster_id}",
            )
        )

        # Density isosurface when feasible
        try:
            if len(pts) >= 3:
                kde = gaussian_kde(pts.T, bw_method=0.3)
                positions = np.vstack([xx.ravel(), yy.ravel(), zz_grid.ravel()])
                density = np.reshape(kde(positions), xx.shape)
                max_d = np.max(density)
                if max_d > 0:
                    density = density / max_d
                    fig.add_trace(
                        go.Isosurface(
                            x=xx.ravel(),
                            y=yy.ravel(),
                            z=zz_grid.ravel(),
                            value=density.ravel(),
                            isomin=0.2,
                            isomax=0.9,
                            surface_count=3,
                            caps=dict(x_show=False, y_show=False, z_show=False),
                            showscale=False,
                            colorscale=[
                                [0, palette[color_idx]],
                                [1, palette[color_idx]],
                            ],
                            opacity=0.25,
                            name=f"Cluster {cluster_id} density",
                            legendgroup=f"cluster-{cluster_id}",
                            showlegend=False,
                        )
                    )
        except (np.linalg.LinAlgError, ValueError):
            pass

    fig.update_layout(
        title=dict(
            text=title.replace(
                "Layered 3D Clustering (X/Y=t-SNE, Z=Layer Index)",
                "Layered 3D Cloud (X/Y=t-SNE, Z=Layer Index)",
            ),
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            zaxis_title="Layer Index (Topology)",
        ),
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = output_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Layered 3D cloud saved to {html_path} and {output_path}")


def plot_layered_by_preferred_3d_cloud(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """Topology-aware 3D cloud colored by preferred class; Z = layer index.

    Always overlays scatter; adds isosurfaces per preferred class when KDE allows.
    """
    palette = px.colors.qualitative.Plotly

    z_base = np.array([nid[0] for nid in neuron_ids], dtype=float)
    rng = np.random.default_rng(42)
    z_coords = z_base + rng.normal(0.0, 0.03, size=len(z_base))

    x_min, x_max = float(np.min(X_2d[:, 0])) - 0.3, float(np.max(X_2d[:, 0])) + 0.3
    y_min, y_max = float(np.min(X_2d[:, 1])) - 0.3, float(np.max(X_2d[:, 1])) + 0.3
    z_min, z_max = float(np.min(z_coords)) - 0.15, float(np.max(z_coords)) + 0.15
    xx, yy, zz_grid = np.mgrid[x_min:x_max:40j, y_min:y_max:40j, z_min:z_max:40j]

    fig = go.Figure()

    # Layer planes
    unique_layers = sorted(set(int(z) for z in z_base))
    for li in unique_layers:
        fig.add_trace(
            go.Mesh3d(
                x=[x_min, x_max, x_max, x_min],
                y=[y_min, y_min, y_max, y_max],
                z=[li, li, li, li],
                opacity=0.08,
                color="lightgray",
                name=f"Layer {li}",
                hoverinfo="skip",
                showscale=False,
                showlegend=False,
            )
        )

    for c in sorted(set(preferred_classes.tolist())):
        mask = preferred_classes == c
        pts = np.column_stack((X_2d[mask, 0], X_2d[mask, 1], z_coords[mask]))
        color = palette[int(c) % len(palette)]

        # Scatter overlay
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=f"Class {int(c)}",
                marker=dict(size=4, color=color, opacity=0.7),
            )
        )

        try:
            if len(pts) >= 3:
                kde = gaussian_kde(pts.T, bw_method=0.3)
                positions = np.vstack([xx.ravel(), yy.ravel(), zz_grid.ravel()])
                density = np.reshape(kde(positions), xx.shape)
                max_d = np.max(density)
                if max_d > 0:
                    density = density / max_d
                    fig.add_trace(
                        go.Isosurface(
                            x=xx.ravel(),
                            y=yy.ravel(),
                            z=zz_grid.ravel(),
                            value=density.ravel(),
                            isomin=0.2,
                            isomax=0.9,
                            surface_count=3,
                            caps=dict(x_show=False, y_show=False, z_show=False),
                            showscale=False,
                            colorscale=[[0, color], [1, color]],
                            opacity=0.25,
                            name=f"Class {int(c)} density",
                            showlegend=False,
                        )
                    )
        except (np.linalg.LinAlgError, ValueError):
            pass

    fig.update_layout(
        title=dict(
            text=title.replace(
                "Layered 3D by Preferred Class",
                "Layered 3D Cloud by Preferred Class",
            ),
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            zaxis_title="Layer Index (Topology)",
        ),
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = output_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Preferred-class layered 3D cloud saved to {html_path} and {output_path}")


def analyze_clusters(
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    feature_vectors: np.ndarray,
):
    """Prints detailed analysis of each cluster."""
    print("\n" + "=" * 80)
    print("CELL ASSEMBLY ANALYSIS")
    print("=" * 80)

    for cluster_id in sorted(set(cluster_labels) - {-1}):
        mask = cluster_labels == cluster_id
        cluster_neurons = [neuron_ids[i] for i, m in enumerate(mask) if m]
        cluster_prefs = preferred_classes[mask]
        cluster_features = feature_vectors[mask]

        print(f"\n{'─' * 80}")
        print(f"Cell Assembly {cluster_id} ({len(cluster_neurons)} neurons)")
        print(f"{'─' * 80}")

        # Layer distribution
        layer_counts = {}
        for layer_idx, _ in cluster_neurons:
            layer_counts[layer_idx] = layer_counts.get(layer_idx, 0) + 1
        print(f"  Layer distribution: {dict(sorted(layer_counts.items()))}")

        # Preferred class distribution
        unique_prefs, counts = np.unique(cluster_prefs, return_counts=True)
        print(
            f"  Preferred classes: {dict(zip(unique_prefs.tolist(), counts.tolist()))}"
        )

        # Dominant preference
        dominant_class = unique_prefs[np.argmax(counts)]
        dominant_percentage = (np.max(counts) / len(cluster_prefs)) * 100
        print(
            f"  Dominant preference: Class {dominant_class} ({dominant_percentage:.1f}% of neurons)"
        )

        # Selectivity statistics (last feature is selectivity index)
        selectivity_values = cluster_features[:, -1]
        print(
            f"  Selectivity: mean={np.mean(selectivity_values):.3f}, std={np.std(selectivity_values):.3f}"
        )

    # Analyze noise points if any
    if -1 in cluster_labels:
        mask = cluster_labels == -1
        print(f"\n{'─' * 80}")
        print(f"Noise Points ({np.sum(mask)} neurons)")
        print(f"{'─' * 80}")
        print("  These neurons don't fit into any cell assembly.")

    print("\n" + "=" * 80)


def analyze_clusters_synchrony(
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    spike_matrix: np.ndarray,
):
    """Prints detailed analysis of each cluster for synchrony-based clustering."""
    print("\n" + "=" * 80)
    print("CELL ASSEMBLY ANALYSIS (SYNCHRONY MODE)")
    print("=" * 80)

    for cluster_id in sorted(set(cluster_labels) - {-1}):
        mask = cluster_labels == cluster_id
        cluster_neurons = [neuron_ids[i] for i, m in enumerate(mask) if m]
        cluster_spikes = spike_matrix[mask]

        print(f"\n{'─' * 80}")
        print(f"Cell Assembly {cluster_id} ({len(cluster_neurons)} neurons)")
        print(f"{'─' * 80}")

        # Layer distribution
        layer_counts = {}
        for layer_idx, _ in cluster_neurons:
            layer_counts[layer_idx] = layer_counts.get(layer_idx, 0) + 1
        print(f"  Layer distribution: {dict(sorted(layer_counts.items()))}")

        # Firing rate statistics
        firing_rates = np.mean(cluster_spikes, axis=1)  # Average firing rate per neuron
        print(
            f"  Firing rates: mean={np.mean(firing_rates):.3f}, std={np.std(firing_rates):.3f}, "
            f"range=[{np.min(firing_rates):.3f}, {np.max(firing_rates):.3f}]"
        )

        # Within-cluster synchrony (average correlation)
        if len(cluster_spikes) > 1:
            cluster_corr = np.corrcoef(cluster_spikes)
            # Get upper triangle correlations (excluding diagonal)
            upper_corr = cluster_corr[np.triu_indices_from(cluster_corr, k=1)]
            mean_synchrony = np.mean(np.abs(upper_corr)) if len(upper_corr) > 0 else 0.0
            print(f"  Within-cluster synchrony: {mean_synchrony:.3f}")
        else:
            print("  Within-cluster synchrony: N/A (single neuron)")

    # Analyze noise points if any
    if -1 in cluster_labels:
        mask = cluster_labels == -1
        print(f"\n{'─' * 80}")
        print(f"Noise Points ({np.sum(mask)} neurons)")
        print(f"{'─' * 80}")
        print("  These neurons don't fit into any cell assembly.")

    print("\n" + "=" * 80)


def load_torchvision_dataset(
    name: str,
    root: str,
    split: str,
    limit: int | None,
):
    """Load a torchvision dataset (currently MNIST) and return (images, labels).

    Images returned as float32 in [0, 1], flattened.
    """
    if name.lower() == "mnist":
        is_train = split == "train"
        transform = transforms.Compose([transforms.ToTensor()])
        ds = datasets.MNIST(
            root=root, train=is_train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    if limit is not None:
        limit = min(limit, len(ds))
        ds_subset = [ds[i] for i in range(limit)]
    else:
        ds_subset = [ds[i] for i in range(len(ds))]

    # Convert to numpy arrays
    imgs = []
    labels = []
    for img_t, label in ds_subset:
        # img_t: CxHxW in [0,1]
        np_img = img_t.numpy().astype(np.float32)
        imgs.append(np_img.reshape(-1))
        labels.append(int(label))

    X = np.stack(imgs, axis=0)
    y = np.array(labels, dtype=np.int64)
    return X, y


def cluster_dataset_features(
    X: np.ndarray,
    y: np.ndarray,
    num_target_clusters: int,
    output_dir: str,
    title_prefix: str,
):
    """Cluster raw dataset samples into num_target_clusters and visualize with t-SNE.

    We standardize features, reduce with t-SNE into 3D per Plotly guidance, and
    render colored by dataset clusters, with true labels shown via marker symbol.
    Reference: Plotly t-SNE examples [t-SNE and UMAP projections].
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans on dataset with K equal to number of neuron assemblies
    model = KMeans(n_clusters=num_target_clusters, random_state=42, n_init=10)
    dataset_cluster_labels = model.fit_predict(X_scaled)

    # t-SNE to 3D for visualization
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(X_scaled) - 1))
    proj_3d = tsne.fit_transform(X_scaled)

    # Build plot
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    # Scatter3d supports only this limited set of symbols
    marker_symbols = [
        "circle",
        "circle-open",
        "cross",
        "diamond",
        "diamond-open",
        "square",
        "square-open",
        "x",
    ]

    for cid in sorted(set(dataset_cluster_labels)):
        mask = dataset_cluster_labels == cid
        color = palette[cid % len(palette)]
        # True label as shape (bounded by available symbols)
        symbols = [marker_symbols[int(lbl) % len(marker_symbols)] for lbl in y[mask]]
        fig.add_trace(
            go.Scatter3d(
                x=proj_3d[mask, 0],
                y=proj_3d[mask, 1],
                z=proj_3d[mask, 2],
                mode="markers",
                name=f"Data Cluster {cid}",
                marker=dict(
                    size=5,
                    color=color,
                    symbol=symbols,
                    line=dict(width=0.5, color="black"),
                ),
                hovertemplate="Cluster: %{text}<extra></extra>",
                text=[f"cid={cid}, label={lbl}" for lbl in y[mask]],
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – Dataset Clusters (K={num_target_clusters})",
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            zaxis_title="t-SNE Component 3",
        ),
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dataset_clusters_3d.png")
    html_path = out_path.replace(".png", ".html")
    fig.write_html(html_path)
    json_path = out_path.replace(".png", ".json")
    pio.write_json(fig, json_path)
    fig.write_image(out_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Dataset clustering visualization saved to {out_path}")

    # Create cloud/isosurface version
    fig_cloud = go.Figure()
    # 3D grid for KDE in the projected space
    x_min, x_max = float(proj_3d[:, 0].min()) - 0.3, float(proj_3d[:, 0].max()) + 0.3
    y_min, y_max = float(proj_3d[:, 1].min()) - 0.3, float(proj_3d[:, 1].max()) + 0.3
    z_min, z_max = float(proj_3d[:, 2].min()) - 0.3, float(proj_3d[:, 2].max()) + 0.3
    xx, yy, zz = np.mgrid[x_min:x_max:40j, y_min:y_max:40j, z_min:z_max:40j]

    for cid in sorted(set(dataset_cluster_labels)):
        mask = dataset_cluster_labels == cid
        pts = proj_3d[mask]
        color = palette[cid % len(palette)]

        # Always add scatter for visibility
        fig_cloud.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=f"Data Cluster {cid}",
                marker=dict(size=3, color=color, opacity=0.7),
            )
        )

        # Density blob overlay
        try:
            if len(pts) >= 3:
                kde = gaussian_kde(pts.T, bw_method=0.3)
                positions = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
                density = np.reshape(kde(positions), xx.shape)
                max_d = float(np.max(density))
                if max_d > 0:
                    density = density / max_d
                    fig_cloud.add_trace(
                        go.Isosurface(
                            x=xx.ravel(),
                            y=yy.ravel(),
                            z=zz.ravel(),
                            value=density.ravel(),
                            isomin=0.2,
                            isomax=0.9,
                            surface_count=3,
                            caps=dict(x_show=False, y_show=False, z_show=False),
                            showscale=False,
                            colorscale=[[0, color], [1, color]],
                            opacity=0.25,
                            name=f"Cluster {cid} density",
                            showlegend=False,
                        )
                    )
        except (np.linalg.LinAlgError, ValueError):
            pass

    fig_cloud.update_layout(
        title=dict(
            text=f"{title_prefix} – Dataset Clusters 3D Cloud (K={num_target_clusters})",
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            zaxis_title="t-SNE Component 3",
        ),
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    cloud_path = os.path.join(output_dir, "dataset_clusters_3d_cloud.png")
    cloud_html_path = cloud_path.replace(".png", ".html")
    fig_cloud.write_html(cloud_html_path)
    cloud_json_path = cloud_path.replace(".png", ".json")
    pio.write_json(fig_cloud, cloud_json_path)
    fig_cloud.write_image(cloud_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"Dataset clustering cloud saved to {cloud_path}")

    return dataset_cluster_labels


def main():
    parser = argparse.ArgumentParser(
        description="Cluster neurons to identify cell assemblies."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to dataset directory containing activity_dataset.h5 (binary format).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="neuron_clustering_results",
        help="Directory to save plots and results.",
    )
    parser.add_argument(
        "--clustering-mode",
        type=str,
        default="fixed",
        choices=["fixed", "auto", "synchrony"],
        help="Clustering mode: 'fixed' for K-Means, 'auto' for DBSCAN, 'synchrony' for correlation-based clustering.",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=5,
        help="Number of clusters for 'fixed' mode (K-Means).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes in the dataset.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Eps parameter for DBSCAN clustering.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Min_samples parameter for DBSCAN clustering.",
    )
    parser.add_argument(
        "--dataset-cluster",
        action="store_true",
        help="Also cluster the original dataset samples (e.g., MNIST) to compare with neuron assemblies.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mnist",
        choices=["mnist"],
        help="Torchvision dataset to load for dataset-level clustering.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./data",
        help="Root directory for torchvision datasets (MNIST/CIFAR).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split of the torchvision dataset to use.",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Optional limit on number of dataset samples to process (for speed).",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Maximum number of ticks to include per image presentation. If specified, only the first N ticks will be used.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of image samples to process. If specified, only the first N samples will be used.",
    )
    args = parser.parse_args()

    # Create output directory with dataset name
    # Extract dataset directory name (handle trailing slash)
    dataset_path = args.input_file.rstrip("/")
    dataset_name = os.path.basename(dataset_path)
    structured_output_dir = os.path.join(
        args.output_dir, dataset_name, "all_metrics", args.clustering_mode
    )
    os.makedirs(structured_output_dir, exist_ok=True)

    print(f"Starting neuron clustering analysis")
    print(f"Clustering mode: {args.clustering_mode}")
    if args.clustering_mode == "fixed":
        print(f"Number of clusters: {args.num_clusters}")
    elif args.clustering_mode == "auto":
        print(f"DBSCAN eps: {args.eps or 'auto'}, min_samples: {args.min_samples}")
    elif args.clustering_mode == "synchrony":
        print(f"Correlation-based clustering (synchrony focus)")
    if args.max_ticks is not None:
        print(f"Max ticks per image: {args.max_ticks}")
    if args.max_samples is not None:
        print(f"Max samples to process: {args.max_samples}")
    print(f"Output will be saved in: {structured_output_dir}")

    # 1. Load and process data
    dataset = load_activity_data(args.input_file)
    image_buckets = group_by_image(dataset, max_ticks=args.max_ticks)

    # Prepare cache
    dataset_hash = compute_dataset_hash(args.input_file)
    cache_dir = get_cache_dir(structured_output_dir)

    # Apply max_samples limit if specified
    if args.max_samples is not None:
        image_items = list(image_buckets.items())[: args.max_samples]
        image_buckets = dict(image_items)
        print(f"Limited to {len(image_buckets)} samples (from original dataset)")

    # 2. Extract neuron-wise features (with caching)
    config_suffix = ""
    if args.max_ticks is not None:
        config_suffix += f"_maxticks{args.max_ticks}"
    if args.max_samples is not None:
        config_suffix += f"_maxsamples{args.max_samples}"

    if args.clustering_mode == "synchrony":
        # Use binary spike matrix for synchrony mode
        spike_cache_path = os.path.join(
            cache_dir, f"{dataset_hash}{config_suffix}_spike_matrix.npz"
        )
        if os.path.exists(spike_cache_path):
            cache = np.load(spike_cache_path, allow_pickle=True)
            spike_matrix = cache["spike_matrix"]
            neuron_id_array = cache["neuron_ids"]
            neuron_ids = [(int(x[0]), int(x[1])) for x in neuron_id_array]
            print(f"Loaded cached spike matrix from {spike_cache_path}")
        else:
            spike_matrix, neuron_ids = extract_synchrony_features(
                image_buckets, max_ticks=args.max_ticks
            )
            np.savez_compressed(
                spike_cache_path,
                spike_matrix=spike_matrix,
                neuron_ids=np.array(neuron_ids, dtype=np.int32),
            )
            print(f"Saved spike matrix cache to {spike_cache_path}")
    else:
        # Use comprehensive features for other modes
        features_cache_path = os.path.join(
            cache_dir, f"{dataset_hash}{config_suffix}_features.npz"
        )
        if os.path.exists(features_cache_path):
            cache = np.load(features_cache_path, allow_pickle=True)
            feature_vectors = cache["feature_vectors"]
            neuron_id_array = cache["neuron_ids"]
            neuron_ids = [(int(x[0]), int(x[1])) for x in neuron_id_array]
            print(f"Loaded cached features from {features_cache_path}")
        else:
            feature_vectors, neuron_ids = extract_neuron_features(
                image_buckets, args.num_classes
            )
            np.savez_compressed(
                features_cache_path,
                feature_vectors=feature_vectors,
                neuron_ids=np.array(neuron_ids, dtype=np.int32),
            )
            print(f"Saved features cache to {features_cache_path}")

    # Check if we have data to work with
    data_size = (
        feature_vectors.shape[0]
        if args.clustering_mode != "synchrony"
        else spike_matrix.shape[0]
    )
    if data_size == 0:
        print("No data could be extracted. Exiting.")
        return

    num_neurons = len(neuron_ids)
    if args.clustering_mode == "synchrony":
        num_ticks = spike_matrix.shape[1]
        print(
            f"\nExtracted spike matrix for {num_neurons} neurons ({num_ticks} ticks)."
        )
    else:
        num_features = feature_vectors.shape[1]
        print(
            f"\nExtracted features for {num_neurons} neurons (feature dimension: {num_features})."
        )

    # Check if we have enough neurons for the requested number of clusters
    if (
        (args.clustering_mode in ["fixed", "synchrony"])
        and args.num_clusters is not None
        and args.num_clusters > num_neurons
    ):
        print(
            f"WARNING: Requested {args.num_clusters} clusters but only have {num_neurons} neurons!"
        )
        print(f"Reducing to {max(1, num_neurons // 2)} clusters.")
        args.num_clusters = max(1, num_neurons // 2)
    elif (
        (args.clustering_mode in ["fixed", "synchrony"])
        and args.num_clusters is not None
        and args.num_clusters > num_neurons // 3
    ):
        print(
            f"WARNING: {args.num_clusters} clusters for {num_neurons} neurons may be too many."
        )
        print(
            f"Recommended: {max(2, num_neurons // 5)} to {num_neurons // 2} clusters."
        )

    # Normalize features before clustering (critical for K-means!) — cached
    if args.clustering_mode == "synchrony":
        print("\nUsing binary spike matrix (no scaling needed)...")
        # For synchrony mode, we don't scale the binary spike matrix
    else:
        print("\nStandardizing features (zero mean, unit variance)...")
        scaled_cache_path = os.path.join(
            cache_dir, f"{dataset_hash}{config_suffix}_features_scaled.npy"
        )
        if os.path.exists(scaled_cache_path):
            feature_vectors_scaled = np.load(scaled_cache_path)
            print(f"Loaded cached scaled features from {scaled_cache_path}")
        else:
            scaler = StandardScaler()
            feature_vectors_scaled = scaler.fit_transform(feature_vectors)
            np.save(scaled_cache_path, feature_vectors_scaled)
            print(f"Saved scaled features cache to {scaled_cache_path}")

        print(f"Feature scaling complete.")
        print(
            f"  Before: mean={np.mean(feature_vectors[:, :3], axis=0)}, std={np.std(feature_vectors[:, :3], axis=0)}"
        )
        print(
            f"  After:  mean={np.mean(feature_vectors_scaled[:, :3], axis=0)}, std={np.std(feature_vectors_scaled[:, :3], axis=0)}"
        )

    # Determine preferred class for each neuron — cached
    prefs_cache_path = os.path.join(
        cache_dir, f"{dataset_hash}{config_suffix}_preferred_classes.npy"
    )
    if os.path.exists(prefs_cache_path):
        preferred_classes = np.load(prefs_cache_path)
    else:
        if args.clustering_mode == "synchrony":
            # For synchrony mode, preferred classes are not computed (focus is on synchrony)
            # Use -1 as placeholder
            preferred_classes = np.full(len(neuron_ids), -1, dtype=int)
        else:
            # Use comprehensive features for class preference
            preferred_classes = np.argmax(
                feature_vectors[:, : args.num_classes], axis=1
            )
        np.save(prefs_cache_path, preferred_classes)

    # 3. Perform clustering
    if args.clustering_mode == "fixed":
        print(f"Performing K-Means clustering with K={args.num_clusters}...")
        model = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10)
        title_prefix = f"K-Means (K={args.num_clusters})"
        labels_cache_path = os.path.join(
            cache_dir,
            f"{dataset_hash}{config_suffix}_kmeans_k{args.num_clusters}_labels.npy",
        )
        if os.path.exists(labels_cache_path):
            cluster_labels = np.load(labels_cache_path)
            print(f"Loaded cached K-Means labels from {labels_cache_path}")
        else:
            cluster_labels = model.fit_predict(feature_vectors_scaled)
            np.save(labels_cache_path, cluster_labels)
            print(f"Saved K-Means labels cache to {labels_cache_path}")
    elif args.clustering_mode == "synchrony":
        print(f"Performing correlation-based clustering...")
        title_prefix = f"Correlation Synchrony"
        labels_cache_path = os.path.join(
            cache_dir,
            f"{dataset_hash}{config_suffix}_correlation_labels.npy",
        )
        if os.path.exists(labels_cache_path):
            cluster_labels = np.load(labels_cache_path)
            print(f"Loaded cached correlation labels from {labels_cache_path}")
        else:
            cluster_labels = cluster_by_correlation(
                spike_matrix,
                num_clusters=args.num_clusters,
                output_dir=structured_output_dir,
            )
            np.save(labels_cache_path, cluster_labels)
            print(f"Saved correlation labels cache to {labels_cache_path}")
    else:  # auto
        print("Performing DBSCAN clustering...")

        if args.eps is not None:
            eps_value = args.eps
            print(f"Using provided eps value: {eps_value:.4f}")
        else:
            eps_cache_path = os.path.join(
                cache_dir,
                f"{dataset_hash}{config_suffix}_dbscan_m{args.min_samples}_eps.json",
            )
            if os.path.exists(eps_cache_path):
                with open(eps_cache_path, "r") as f:
                    data = json.load(f)
                    eps_value = float(data.get("eps", 0.5))
                print(f"Loaded cached eps from {eps_cache_path}: {eps_value:.4f}")
            else:
                eps_value = find_optimal_eps(
                    feature_vectors_scaled,
                    min_samples=args.min_samples,
                    output_dir=structured_output_dir,
                )
                with open(eps_cache_path, "w") as f:
                    json.dump({"eps": float(eps_value)}, f)
                print(f"Saved eps cache to {eps_cache_path}")

        model = DBSCAN(eps=eps_value, min_samples=args.min_samples)
        title_prefix = f"DBSCAN (eps={eps_value:.4f}, min_samples={args.min_samples})"
        eps_token = format_float_token(eps_value, decimals=6)
        labels_cache_path = os.path.join(
            cache_dir,
            f"{dataset_hash}{config_suffix}_dbscan_eps{eps_token}_m{args.min_samples}_labels.npy",
        )
        if os.path.exists(labels_cache_path):
            cluster_labels = np.load(labels_cache_path)
            print(f"Loaded cached DBSCAN labels from {labels_cache_path}")
        else:
            cluster_labels = model.fit_predict(feature_vectors_scaled)
            np.save(labels_cache_path, cluster_labels)
            print(f"Saved DBSCAN labels cache to {labels_cache_path}")

    num_found_clusters = len(set(cluster_labels) - {-1})
    print(f"Clustering complete. Found {num_found_clusters} cell assemblies.")

    # 4. Evaluate clustering quality
    if num_found_clusters > 1:
        if args.clustering_mode == "synchrony":
            # For synchrony mode, use spike matrix for silhouette score
            silhouette = silhouette_score(spike_matrix, cluster_labels)
        else:
            silhouette = silhouette_score(feature_vectors_scaled, cluster_labels)
        print(f"Silhouette Score: {silhouette:.4f}")
        print(
            f"Clusters explain {silhouette:.1%} of the variance (higher = more distinct clusters)"
        )

    # 5. Analyze clusters (use original features for interpretability)
    if args.clustering_mode == "synchrony":
        # For synchrony mode, selectivity analysis doesn't apply
        analyze_clusters_synchrony(
            cluster_labels, neuron_ids, preferred_classes, spike_matrix
        )
    else:
        analyze_clusters(cluster_labels, neuron_ids, preferred_classes, feature_vectors)

    # 7. Visualize with t-SNE
    print("Reducing dimensionality with t-SNE for visualization...")

    if args.clustering_mode == "synchrony":
        plot_title = f"{title_prefix} Clustering of Neurons (Binary Spike Synchrony)"
        # Use spike matrix directly for t-SNE (no scaling needed for binary data)
        viz_data = spike_matrix.astype(np.float32)
    else:
        plot_title = f"{title_prefix} Clustering of Neurons (All Metrics: S, F_avg, t_ref, fired)"
        viz_data = feature_vectors_scaled

    # Create 2D visualizations (existing)
    tsne_perplexity = min(30, viz_data.shape[0] - 1)
    tsne_2d_cache_path = os.path.join(
        cache_dir, f"{dataset_hash}{config_suffix}_tsne_2d_p{tsne_perplexity}_rs42.npy"
    )
    if os.path.exists(tsne_2d_cache_path):
        X_2d = np.load(tsne_2d_cache_path)
        print(f"Loaded cached t-SNE 2D from {tsne_2d_cache_path}")
    else:
        tsne_2d = TSNE(
            n_components=2,
            random_state=42,
            perplexity=tsne_perplexity,
        )
        X_2d = tsne_2d.fit_transform(viz_data)
        np.save(tsne_2d_cache_path, X_2d)
        print(f"Saved t-SNE 2D cache to {tsne_2d_cache_path}")

    # Create enhanced scatter plot
    output_filename = os.path.join(structured_output_dir, "neuron_clusters_2d.png")
    plot_neuron_clusters(
        X_2d, cluster_labels, neuron_ids, preferred_classes, plot_title, output_filename
    )

    # Create cloud/heatmap version
    output_filename_cloud = os.path.join(
        structured_output_dir, "neuron_clusters_2d_cloud.png"
    )
    plot_neuron_clusters_cloud(
        X_2d,
        cluster_labels,
        neuron_ids,
        preferred_classes,
        plot_title,
        output_filename_cloud,
    )

    # Create 3D visualizations (new)
    print("Creating 3D t-SNE visualizations...")
    tsne_3d_cache_path = os.path.join(
        cache_dir, f"{dataset_hash}{config_suffix}_tsne_3d_p{tsne_perplexity}_rs42.npy"
    )
    if os.path.exists(tsne_3d_cache_path):
        X_3d = np.load(tsne_3d_cache_path)
        print(f"Loaded cached t-SNE 3D from {tsne_3d_cache_path}")
    else:
        tsne_3d = TSNE(
            n_components=3,
            random_state=42,
            perplexity=tsne_perplexity,
        )
        X_3d = tsne_3d.fit_transform(viz_data)
        np.save(tsne_3d_cache_path, X_3d)
        print(f"Saved t-SNE 3D cache to {tsne_3d_cache_path}")

    # Create 3D enhanced scatter plot
    output_filename_3d = os.path.join(structured_output_dir, "neuron_clusters_3d.png")
    plot_neuron_clusters_3d(
        X_3d,
        cluster_labels,
        neuron_ids,
        preferred_classes,
        plot_title,
        output_filename_3d,
    )

    # Create 3D cloud/volume version
    output_filename_3d_cloud = os.path.join(
        structured_output_dir, "neuron_clusters_3d_cloud.png"
    )
    plot_neuron_clusters_cloud_3d(
        X_3d,
        cluster_labels,
        neuron_ids,
        preferred_classes,
        plot_title,
        output_filename_3d_cloud,
    )

    # 9. Create brain region map
    if args.clustering_mode != "synchrony":
        brain_map_path = os.path.join(structured_output_dir, "brain_region_map.png")
        plot_brain_region_map(
            cluster_labels,
            neuron_ids,
            preferred_classes,
            feature_vectors,
            brain_map_path,
        )

    # 10. Create layered 3D plot (topology-aware)
    layered_title = plot_title.replace(
        "Clustering of Neurons",
        "Layered 3D Clustering (X/Y=t-SNE, Z=Layer Index)",
    )
    layered_path = os.path.join(structured_output_dir, "neuron_clusters_layered_3d.png")
    plot_layered_clusters_3d(
        X_2d,
        cluster_labels,
        neuron_ids,
        preferred_classes,
        layered_title,
        layered_path,
    )

    # 11. Preferred-class views (parallel set)
    pref_2d_path = os.path.join(structured_output_dir, "preferred_class_2d.png")
    plot_by_preferred_2d(
        X_2d,
        cluster_labels,
        neuron_ids,
        preferred_classes,
        plot_title,
        pref_2d_path,
    )

    pref_2d_cloud_path = os.path.join(
        structured_output_dir, "preferred_class_2d_cloud.png"
    )
    plot_by_preferred_2d_cloud(
        X_2d,
        cluster_labels,
        neuron_ids,
        preferred_classes,
        plot_title,
        pref_2d_cloud_path,
    )

    pref_3d_path = os.path.join(structured_output_dir, "preferred_class_3d.png")
    plot_by_preferred_3d(
        X_3d,
        cluster_labels,
        neuron_ids,
        preferred_classes,
        plot_title,
        pref_3d_path,
    )

    pref_3d_cloud_path = os.path.join(
        structured_output_dir, "preferred_class_3d_cloud.png"
    )
    plot_by_preferred_3d_cloud(
        X_3d,
        cluster_labels,
        neuron_ids,
        preferred_classes,
        plot_title,
        pref_3d_cloud_path,
    )

    pref_layered_path = os.path.join(
        structured_output_dir, "preferred_class_layered_3d.png"
    )
    plot_layered_by_preferred_3d(
        X_2d,
        cluster_labels,
        neuron_ids,
        preferred_classes,
        layered_title,
        pref_layered_path,
    )

    # 13. Optional: Cluster original dataset to compare with neuron assemblies
    if args.dataset_cluster:
        num_assemblies = len(set(cluster_labels) - {-1})
        print(
            f"\nClustering original dataset into K={num_assemblies} groups to compare with neuron assemblies..."
        )

        X_raw, y_raw = load_torchvision_dataset(
            name=args.dataset_name,
            root=args.dataset_root,
            split=args.dataset_split,
            limit=args.dataset_limit,
        )

        dataset_out_dir = os.path.join(structured_output_dir, "dataset_clusters")
        cluster_dataset_features(
            X=X_raw,
            y=y_raw,
            num_target_clusters=max(2, num_assemblies),
            output_dir=dataset_out_dir,
            title_prefix=title_prefix,
        )

    # 12. Layered cloud variants
    layered_cloud_path = os.path.join(
        structured_output_dir, "neuron_clusters_layered_3d_cloud.png"
    )
    plot_layered_clusters_3d_cloud(
        X_2d,
        cluster_labels,
        neuron_ids,
        preferred_classes,
        layered_title,
        layered_cloud_path,
    )

    pref_layered_cloud_path = os.path.join(
        structured_output_dir, "preferred_class_layered_3d_cloud.png"
    )
    plot_layered_by_preferred_3d_cloud(
        X_2d,
        cluster_labels,
        neuron_ids,
        preferred_classes,
        layered_title,
        pref_layered_cloud_path,
    )

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
