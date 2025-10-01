import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


def load_activity_data(path: str) -> List[Dict[str, Any]]:
    """Loads the activity dataset from a JSON file."""
    with open(path, "r") as f:
        payload = json.load(f)
    return payload.get("records", [])


def group_by_image_index(
    records: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Groups records by image_index."""
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for rec in tqdm(records, desc="Grouping records by image"):
        img_idx = rec.get("image_index")
        if img_idx is not None:
            buckets.setdefault(int(img_idx), []).append(rec)

    for key in buckets:
        buckets[key].sort(key=lambda r: r.get("tick", 0))
    return buckets


def get_ground_truth_labels(image_indices: List[int], dataset_name: str) -> np.ndarray:
    """Loads the original dataset to get the true labels for the given image indices."""
    print(f"Loading ground truth labels from {dataset_name} dataset...")
    transform = transforms.Compose([transforms.ToTensor()])

    max_idx = max(image_indices)

    if dataset_name == "mnist":
        test_size = 10000
        is_train_split = max_idx >= test_size
        dataset = datasets.MNIST(
            root="./data", train=is_train_split, download=True, transform=transform
        )
        split_name = "train" if is_train_split else "test"
        print(f"Detected {split_name} split (max index: {max_idx})")
    elif dataset_name == "cifar10":
        test_size = 10000
        is_train_split = max_idx >= test_size
        dataset = datasets.CIFAR10(
            root="./data", train=is_train_split, download=True, transform=transform
        )
        split_name = "train" if is_train_split else "test"
        print(f"Detected {split_name} split (max index: {max_idx})")
    else:
        raise ValueError(f"Unsupported dataset for ground truth: {dataset_name}")

    labels = np.array([dataset[i][1] for i in image_indices])
    return labels


def extract_neuron_features(
    image_buckets: Dict[int, List[Dict[str, Any]]],
    true_labels: np.ndarray,
    image_indices: List[int],
    num_classes: int = 10,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Extracts comprehensive feature vectors for each neuron across all images using ALL available metrics.

    For each neuron, we create a feature vector that captures:
    - Per-class statistics for ALL metrics (S, F_avg, t_ref, fired)
    - Overall activity statistics
    - Temporal dynamics

    Returns:
        feature_vectors: (num_neurons, num_features) array
        neuron_ids: List of (layer_idx, neuron_idx) tuples identifying each neuron
    """
    print(
        "Extracting comprehensive neuron-wise features using all available metrics..."
    )

    # First pass: determine network structure
    sample_records = list(image_buckets.values())[0]
    network_structure = []
    for layer in sample_records[0].get("layers", []):
        num_neurons = len(layer.get("S", []))
        network_structure.append(num_neurons)

    print(f"Network structure: {network_structure}")

    # Create mapping from image_index to label
    image_to_label = {
        img_idx: label for img_idx, label in zip(image_indices, true_labels)
    }

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

    # Second pass: collect all metrics for each neuron per class
    print("Collecting all neuron metrics per class...")
    for img_idx, records in tqdm(image_buckets.items(), desc="Processing images"):
        label = image_to_label[img_idx]

        # Collect time series for all metrics
        time_series = {metric: [] for metric in metric_names}

        for record in records:
            for metric in metric_names:
                tick_values = []
                for layer in record.get("layers", []):
                    tick_values.append(layer.get(metric, []))
                time_series[metric].append(tick_values)

        # Calculate statistics for this presentation
        for layer_idx in range(len(network_structure)):
            for neuron_idx in range(network_structure[layer_idx]):
                for metric in metric_names:
                    neuron_activity = [
                        tick[layer_idx][neuron_idx]
                        for tick in time_series[metric]
                        if len(tick) > layer_idx and len(tick[layer_idx]) > neuron_idx
                    ]
                    if neuron_activity:
                        # Store comprehensive temporal statistics for this presentation
                        mean_val = np.mean(neuron_activity)
                        max_val = np.max(neuron_activity)
                        std_val = np.std(neuron_activity)

                        # Add temporal features
                        min_val = np.min(neuron_activity)
                        range_val = max_val - min_val
                        median_val = np.median(neuron_activity)

                        # Temporal dynamics: autocorrelation, trend
                        if len(neuron_activity) > 1:
                            # Simple autocorrelation (lag-1)
                            autocorr = np.corrcoef(
                                neuron_activity[:-1], neuron_activity[1:]
                            )[0, 1]
                            if np.isnan(autocorr):
                                autocorr = 0.0

                            # Linear trend
                            time_points = np.arange(len(neuron_activity))
                            if (
                                len(np.unique(neuron_activity)) > 1
                            ):  # Avoid division by zero
                                trend = np.polyfit(time_points, neuron_activity, 1)[0]
                            else:
                                trend = 0.0

                            # Peak timing (when max occurs)
                            peak_timing = np.argmax(neuron_activity) / len(
                                neuron_activity
                            )  # Normalize to 0-1
                        else:
                            autocorr = 0.0
                            trend = 0.0
                            peak_timing = 0.0

                        neuron_metrics[layer_idx][neuron_idx][metric][label].append(
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
                                "activity_sequence": neuron_activity,  # Keep full sequence for advanced analysis
                            }
                        )

    # Third pass: create comprehensive feature vectors
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


def find_optimal_eps(
    X: np.ndarray, min_samples: int = 5, output_dir: str = None
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
            x=elbow_idx,
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
        plt.savefig(k_distance_plot_path, dpi=300)
        plt.close()
        print(f"K-distance graph saved to {k_distance_plot_path}")

    return optimal_eps


def plot_neuron_clusters(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves a scatter plot of neuron clusters with enhanced visibility."""
    plt.figure(figsize=(16, 12))

    num_clusters = len(set(cluster_labels) - {-1})
    # Use more vivid colors
    palette = sns.color_palette(
        "bright" if num_clusters <= 8 else "tab20", num_clusters
    )

    # Markers for different preferred classes
    markers = ["o", "s", "P", "X", "^", "v", "<", ">", "D", "*"]

    for i in range(X_2d.shape[0]):
        cluster_id = cluster_labels[i]
        preferred_class = preferred_classes[i]

        if cluster_id == -1:
            color = "gray"  # Darker gray for noise points
            size = 35  # Larger size for visibility
            alpha = 0.6
        else:
            color = palette[cluster_id]
            size = 120  # Much larger size for better visibility
            alpha = 0.9  # More opaque

        plt.scatter(
            X_2d[i, 0],
            X_2d[i, 1],
            c=[color],
            s=size,
            marker=markers[preferred_class % len(markers)],
            alpha=alpha,
            edgecolors="black",
            linewidths=1.0,  # Thicker borders
        )

    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)

    # Create custom legends
    cluster_handles = []
    for i in sorted(set(cluster_labels) - {-1}):
        cluster_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"Cluster {i}",
                markerfacecolor=palette[i],
                markersize=10,
            )
        )
    if -1 in cluster_labels:
        cluster_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Noise",
                markerfacecolor="lightgray",
                markersize=8,
            )
        )

    class_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers[i % len(markers)],
            color="gray",
            linestyle="None",
            label=f"Prefers Class {i}",
            markersize=10,
        )
        for i in sorted(set(preferred_classes))
    ]

    legend1 = plt.legend(
        handles=cluster_handles,
        title="Cell Assembly (Cluster)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=10,
    )
    ax = plt.gca()
    ax.add_artist(legend1)
    plt.legend(
        handles=class_handles,
        title="Preferred Class",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Neuron cluster visualization saved to {output_path}")


def plot_neuron_clusters_cloud(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves a cloud/heatmap version of neuron clusters."""
    plt.figure(figsize=(16, 12))

    num_clusters = len(set(cluster_labels) - {-1})
    # Use more vivid colors
    palette = sns.color_palette(
        "bright" if num_clusters <= 8 else "tab20", num_clusters
    )

    # Create a meshgrid for the density plot
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

    # Plot each cluster as a colored cloud
    for cluster_id in sorted(set(cluster_labels) - {-1}):
        mask = cluster_labels == cluster_id
        if np.sum(mask) < 3:  # Skip clusters with too few points
            continue

        cluster_points = X_2d[mask]

        # Create kernel density estimate
        try:
            kde = gaussian_kde(cluster_points.T)
            # Evaluate KDE on meshgrid
            positions = np.vstack([xx.ravel(), yy.ravel()])
            zz = np.reshape(kde(positions), xx.shape)

            # Normalize the density
            zz = zz / zz.max()

            # Plot filled contours
            cs = plt.contourf(
                xx,
                yy,
                zz,
                levels=np.linspace(0.1, 1.0, 8),
                colors=[palette[cluster_id]],
                alpha=0.7,
                extend="both",
            )

        except np.linalg.LinAlgError:
            # Fallback to scatter if KDE fails
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=[palette[cluster_id]],
                s=80,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.5,
            )

    # Add noise points as light scatter
    noise_mask = cluster_labels == -1
    if np.sum(noise_mask) > 0:
        noise_points = X_2d[noise_mask]
        plt.scatter(
            noise_points[:, 0],
            noise_points[:, 1],
            c="lightgray",
            s=40,
            alpha=0.4,
            edgecolors="gray",
            linewidths=0.3,
            marker="x",
        )

    plt.title(
        title.replace("Clustering", "Cloud Clustering"), fontsize=18, fontweight="bold"
    )
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)

    # Create custom legends
    cluster_handles = []
    for i in sorted(set(cluster_labels) - {-1}):
        cluster_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=palette[i], label=f"Cluster {i}")
        )
    if -1 in cluster_labels:
        cluster_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc="lightgray", label="Noise")
        )

    plt.legend(
        handles=cluster_handles,
        title="Cell Assemblies",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Neuron cluster cloud visualization saved to {output_path}")


def plot_brain_region_map(
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    feature_vectors: np.ndarray,
    output_path: str,
):
    """Creates a 'brain region map' showing layer-wise organization of cell assemblies."""
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
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 6))
    if num_layers == 1:
        axes = [axes]

    num_clusters = len(set(cluster_labels) - {-1})
    palette = sns.color_palette("tab20" if num_clusters > 10 else "deep", num_clusters)

    for layer_idx in sorted(layers.keys()):
        ax = axes[layer_idx]
        layer_neurons = layers[layer_idx]

        # Sort by neuron index
        layer_neurons.sort(key=lambda x: x["neuron_idx"])

        # Create a color map for this layer
        colors = []
        for neuron in layer_neurons:
            cluster_id = neuron["cluster"]
            if cluster_id == -1:
                colors.append("lightgray")
            else:
                colors.append(palette[cluster_id])

        # Create bar chart
        neuron_indices = [n["neuron_idx"] for n in layer_neurons]
        preferred = [n["preferred_class"] for n in layer_neurons]

        bars = ax.bar(
            neuron_indices,
            [1] * len(neuron_indices),
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add preferred class labels on top of bars
        for i, (idx, pref) in enumerate(zip(neuron_indices, preferred)):
            ax.text(
                idx,
                1.05,
                str(pref),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_xlabel("Neuron Index", fontsize=12)
        ax.set_ylabel("Cell Assembly", fontsize=12)
        ax.set_title(f"Layer {layer_idx}", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.3)
        ax.set_xlim(-0.5, len(neuron_indices) - 0.5)
        ax.set_yticks([])

    # Add a single legend for all subplots
    cluster_handles = []
    for i in sorted(set(cluster_labels) - {-1}):
        cluster_handles.append(
            plt.Rectangle(
                (0, 0), 1, 1, fc=palette[i], edgecolor="black", label=f"Assembly {i}"
            )
        )
    if -1 in cluster_labels:
        cluster_handles.append(
            plt.Rectangle(
                (0, 0), 1, 1, fc="lightgray", edgecolor="black", label="Noise"
            )
        )

    fig.legend(
        handles=cluster_handles,
        title="Cell Assemblies",
        bbox_to_anchor=(0.5, -0.05),
        loc="upper center",
        ncol=min(6, len(cluster_handles)),
        fontsize=10,
    )

    plt.suptitle(
        'Neural Network "Brain Region" Map\n(Numbers show preferred class)',
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Brain region map saved to {output_path}")


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

        print(f"\n{'─'*80}")
        print(f"Cell Assembly {cluster_id} ({len(cluster_neurons)} neurons)")
        print(f"{'─'*80}")

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
        print(f"\n{'─'*80}")
        print(f"Noise Points ({np.sum(mask)} neurons)")
        print(f"{'─'*80}")
        print("  These neurons don't fit into any cell assembly.")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Cluster neurons to identify cell assemblies."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the unsupervised JSON activity dataset.",
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
        choices=["fixed", "auto"],
        help="Clustering mode: 'fixed' for K-Means, 'auto' for DBSCAN.",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=5,
        help="Number of clusters for 'fixed' mode (K-Means).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Name of the original dataset for fetching ground truth labels.",
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
    args = parser.parse_args()

    # Create output directory
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    structured_output_dir = os.path.join(
        args.output_dir, input_basename, "all_metrics", args.clustering_mode
    )
    os.makedirs(structured_output_dir, exist_ok=True)

    # 1. Load and process data
    records = load_activity_data(args.input_file)
    image_buckets = group_by_image_index(records)
    image_indices = list(image_buckets.keys())
    true_labels = get_ground_truth_labels(image_indices, args.dataset_name)

    # 2. Extract neuron-wise features using ALL available metrics
    feature_vectors, neuron_ids = extract_neuron_features(
        image_buckets, true_labels, image_indices, args.num_classes
    )

    if feature_vectors.shape[0] == 0:
        print("No feature vectors could be extracted. Exiting.")
        return

    num_neurons = feature_vectors.shape[0]
    num_features = feature_vectors.shape[1]
    print(
        f"\nExtracted features for {num_neurons} neurons (feature dimension: {num_features})."
    )

    # Check if we have enough neurons for the requested number of clusters
    if args.clustering_mode == "fixed" and args.num_clusters > num_neurons:
        print(
            f"WARNING: Requested {args.num_clusters} clusters but only have {num_neurons} neurons!"
        )
        print(f"Reducing to {num_neurons // 2} clusters.")
        args.num_clusters = max(2, num_neurons // 2)
    elif args.clustering_mode == "fixed" and args.num_clusters > num_neurons // 3:
        print(
            f"WARNING: {args.num_clusters} clusters for {num_neurons} neurons may be too many."
        )
        print(
            f"Recommended: {max(2, num_neurons // 5)} to {num_neurons // 2} clusters."
        )

    # Normalize features before clustering (critical for K-means!)
    print("\nStandardizing features (zero mean, unit variance)...")
    scaler = StandardScaler()
    feature_vectors_scaled = scaler.fit_transform(feature_vectors)

    print(f"Feature scaling complete.")
    print(
        f"  Before: mean={np.mean(feature_vectors[:, :3], axis=0)}, std={np.std(feature_vectors[:, :3], axis=0)}"
    )
    print(
        f"  After:  mean={np.mean(feature_vectors_scaled[:, :3], axis=0)}, std={np.std(feature_vectors_scaled[:, :3], axis=0)}"
    )

    # Determine preferred class for each neuron (use original unscaled features)
    preferred_classes = np.argmax(feature_vectors[:, : args.num_classes], axis=1)

    # 3. Perform clustering on SCALED features
    if args.clustering_mode == "fixed":
        print(f"Performing K-Means clustering with K={args.num_clusters}...")
        model = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10)
        title_prefix = f"K-Means (K={args.num_clusters})"
        cluster_labels = model.fit_predict(feature_vectors_scaled)
    else:  # auto
        print("Performing DBSCAN clustering...")

        if args.eps is not None:
            eps_value = args.eps
            print(f"Using provided eps value: {eps_value:.4f}")
        else:
            eps_value = find_optimal_eps(
                feature_vectors_scaled,
                min_samples=args.min_samples,
                output_dir=structured_output_dir,
            )

        model = DBSCAN(eps=eps_value, min_samples=args.min_samples)
        title_prefix = f"DBSCAN (eps={eps_value:.4f}, min_samples={args.min_samples})"
        cluster_labels = model.fit_predict(feature_vectors_scaled)

    num_found_clusters = len(set(cluster_labels) - {-1})
    print(f"Clustering complete. Found {num_found_clusters} cell assemblies.")

    # 4. Evaluate clustering quality
    if num_found_clusters > 1:
        silhouette = silhouette_score(feature_vectors_scaled, cluster_labels)
        print(f"Silhouette Score: {silhouette:.4f}")
        print(
            f"Clusters explain {silhouette:.1%} of the variance (higher = more distinct clusters)"
        )

    # 5. Analyze clusters (use original features for interpretability)
    analyze_clusters(cluster_labels, neuron_ids, preferred_classes, feature_vectors)

    # 7. Visualize with t-SNE (use SCALED features for better visualization)
    print("Reducing dimensionality with t-SNE for visualization...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, feature_vectors_scaled.shape[0] - 1),
    )
    X_2d = tsne.fit_transform(feature_vectors_scaled)

    plot_title = (
        f"{title_prefix} Clustering of Neurons (All Metrics: S, F_avg, t_ref, fired)"
    )

    # Create enhanced scatter plot
    output_filename = os.path.join(structured_output_dir, "neuron_clusters_tsne.png")
    plot_neuron_clusters(
        X_2d, cluster_labels, neuron_ids, preferred_classes, plot_title, output_filename
    )

    # Create cloud/heatmap version
    output_filename_cloud = os.path.join(
        structured_output_dir, "neuron_clusters_cloud.png"
    )
    plot_neuron_clusters_cloud(
        X_2d,
        cluster_labels,
        neuron_ids,
        preferred_classes,
        plot_title,
        output_filename_cloud,
    )

    # 9. Create brain region map
    brain_map_path = os.path.join(structured_output_dir, "brain_region_map.png")
    plot_brain_region_map(
        cluster_labels, neuron_ids, preferred_classes, feature_vectors, brain_map_path
    )

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
