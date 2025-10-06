import os
import json
import argparse
from typing import Dict, Any, List, Tuple

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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress numpy warnings for autocorrelation calculations
warnings.filterwarnings(
    "ignore", message="invalid value encountered in divide", category=RuntimeWarning
)


def load_activity_data(path: str) -> List[Dict[str, Any]]:
    """Loads the activity dataset from a JSON file.

    Parameters
    ----------
    path: str
        Path to the JSON file produced by build_activity_dataset.
        The dataset contains a "records" top-level key.
    """
    with open(path, "r") as f:
        payload = json.load(f)
    return payload.get("records", [])


def sort_records_deterministically(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return records sorted by (image_index, tick, label) for deterministic downstream processing."""

    max_tick_fallback = 10**12

    return sorted(
        records,
        key=lambda rec: (
            int(rec.get("image_index", -1)),
            int(rec.get("tick", max_tick_fallback)),
            int(rec.get("label", -1)),
        ),
    )


def group_by_image(
    records: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """Groups records by image id, preserving labels and ordering ticks."""
    buckets: Dict[int, Dict[str, Any]] = {}

    ordered_records = sort_records_deterministically(records)

    for rec in tqdm(ordered_records, desc="Grouping records by image"):
        img_idx = rec.get("image_index")
        if img_idx is None:
            continue

        img_idx = int(img_idx)
        bucket = buckets.setdefault(img_idx, {"label": rec.get("label"), "records": []})

        # Prefer non-None labels if encountered later
        if bucket["label"] is None and rec.get("label") is not None:
            bucket["label"] = rec.get("label")

        bucket["records"].append(rec)

    for bucket in buckets.values():
        bucket["records"].sort(key=lambda r: r.get("tick", 0))

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

    Returns:
        feature_vectors: (num_neurons, num_features) array
        neuron_ids: List of (layer_idx, neuron_idx) tuples identifying each neuron
    """
    print(
        "Extracting comprehensive neuron-wise features using all available metrics..."
    )

    # First pass: determine network structure
    sample_image = next(iter(image_buckets.values()))
    sample_records = sample_image["records"]
    if not sample_records:
        raise ValueError(
            "No records found for sample image; cannot infer network structure"
        )

    network_structure = []
    for layer in sample_records[0].get("layers", []):
        num_neurons = len(layer.get("S", []))
        network_structure.append(num_neurons)

    print(f"Network structure: {network_structure}")

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

        records = bucket["records"]

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
                            # Simple autocorrelation (lag-1) - handle zero variance case
                            activity_lag1 = neuron_activity[:-1]
                            activity_lag2 = neuron_activity[1:]

                            # Check if either array has zero variance
                            if np.var(activity_lag1) == 0 or np.var(activity_lag2) == 0:
                                autocorr = (
                                    0.0  # No correlation possible with constant values
                                )
                            else:
                                autocorr = np.corrcoef(activity_lag1, activity_lag2)[
                                    0, 1
                                ]
                                if np.isnan(autocorr) or np.isinf(autocorr):
                                    autocorr = 0.0

                            # Linear trend - handle zero variance case
                            time_points = np.arange(len(neuron_activity))
                            unique_values = np.unique(neuron_activity)

                            if len(unique_values) <= 1:
                                trend = 0.0  # No trend in constant values
                            else:
                                try:
                                    trend = np.polyfit(time_points, neuron_activity, 1)[
                                        0
                                    ]
                                    if np.isnan(trend) or np.isinf(trend):
                                        trend = 0.0
                                except (np.RankWarning, ValueError):
                                    trend = 0.0  # Handle numerical issues

                            # Peak timing (when max occurs) - handle ties
                            max_indices = np.where(
                                neuron_activity == np.max(neuron_activity)
                            )[0]
                            peak_timing = max_indices[0] / len(
                                neuron_activity
                            )  # Use first occurrence
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
    """Creates and saves an interactive scatter plot of neuron clusters."""
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

    # Save interactive HTML
    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    print(f"Interactive neuron cluster visualization saved to {html_path}")

    # Save static image
    fig.write_image(output_path, width=1400, height=900)
    print(f"Static neuron cluster visualization saved to {output_path}")


def plot_neuron_clusters_cloud(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves an interactive cloud/density version of neuron clusters."""
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
        if np.sum(mask) < 3:  # Skip clusters with too few points
            continue

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

        # Create kernel density estimate
        try:
            n_points = len(cluster_points)
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
                color_idx = cluster_id % len(palette)
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
    print(f"Interactive neuron cluster cloud visualization saved to {html_path}")

    # Save static image
    fig.write_image(output_path, width=1400, height=900)
    print(f"Static neuron cluster cloud visualization saved to {output_path}")


def plot_neuron_clusters_3d(
    X_3d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves an interactive 3D scatter plot of neuron clusters."""
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

    # Save interactive HTML
    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    print(f"Interactive 3D neuron cluster visualization saved to {html_path}")

    # Save static image
    fig.write_image(output_path, width=1400, height=900)
    print(f"Static 3D neuron cluster visualization saved to {output_path}")


def plot_neuron_clusters_cloud_3d(
    X_3d: np.ndarray,
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves an interactive 3D cloud/volume visualization of neuron clusters."""
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
        if np.sum(mask) < 3:  # Skip clusters with too few points
            continue

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

        # Create kernel density estimate
        try:
            n_points = len(cluster_points)
            if n_points > 10:
                bandwidth = n_points ** (-1 / 6) * 0.3
            else:
                bandwidth = 0.08

            kde = gaussian_kde(cluster_points.T, bw_method=bandwidth)
            positions = np.vstack([xx.ravel(), yy.ravel(), zz_grid.ravel()])
            density = np.reshape(kde(positions), xx.shape)

            # Normalize density and render volumetric isosurfaces (blob)
            max_d = np.max(density)
            if max_d > 0:
                density = density / max_d
                color_idx = cluster_id % len(palette)
                # Render a few nested isosurfaces for a cloud-like blob
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
                        colorscale=[[0, palette[color_idx]], [1, palette[color_idx]]],
                        opacity=0.25,
                        name=f"Cluster {cluster_id} ({pref_summary})",
                        legendgroup=f"cluster-{cluster_id}",
                        showlegend=False,
                    )
                )
                # Overlay small points for texture inside the blob
                fig.add_trace(
                    go.Scatter3d(
                        x=cluster_points[:, 0],
                        y=cluster_points[:, 1],
                        z=cluster_points[:, 2],
                        mode="markers",
                        name=f"Cluster {cluster_id} ({pref_summary})",
                        marker=dict(size=3, color=palette[color_idx], opacity=0.6),
                        showlegend=True,
                        legendgroup=f"cluster-{cluster_id}",
                    )
                )

        except (np.linalg.LinAlgError, ValueError):
            # Fallback to scatter if KDE fails
            color_idx = cluster_id % len(palette)
            fig.add_trace(
                go.Scatter3d(
                    x=cluster_points[:, 0],
                    y=cluster_points[:, 1],
                    z=cluster_points[:, 2],
                    mode="markers",
                    name=f"Cluster {cluster_id} ({pref_summary})",
                    marker=dict(size=8, color=palette[color_idx], opacity=0.6),
                )
            )

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
    print(f"Interactive 3D neuron cluster cloud visualization saved to {html_path}")

    # Save static image
    fig.write_image(output_path, width=1400, height=900)
    print(f"Static 3D neuron cluster cloud visualization saved to {output_path}")


def plot_brain_region_map(
    cluster_labels: np.ndarray,
    neuron_ids: List[Tuple[int, int]],
    preferred_classes: np.ndarray,
    feature_vectors: np.ndarray,
    output_path: str,
):
    """Creates an interactive 'brain region map' showing layer-wise organization of cell assemblies."""
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
    print(f"Interactive brain region map saved to {html_path}")

    # Save static image
    fig.write_image(output_path, width=max(400 * num_layers, 1200), height=600)
    print(f"Static brain region map saved to {output_path}")


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
    image_buckets = group_by_image(records)

    # 2. Extract neuron-wise features using ALL available metrics
    feature_vectors, neuron_ids = extract_neuron_features(
        image_buckets, args.num_classes
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

    plot_title = (
        f"{title_prefix} Clustering of Neurons (All Metrics: S, F_avg, t_ref, fired)"
    )

    # Create 2D visualizations (existing)
    tsne_2d = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, feature_vectors_scaled.shape[0] - 1),
    )
    X_2d = tsne_2d.fit_transform(feature_vectors_scaled)

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
    tsne_3d = TSNE(
        n_components=3,
        random_state=42,
        perplexity=min(30, feature_vectors_scaled.shape[0] - 1),
    )
    X_3d = tsne_3d.fit_transform(feature_vectors_scaled)

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
    brain_map_path = os.path.join(structured_output_dir, "brain_region_map.png")
    plot_brain_region_map(
        cluster_labels, neuron_ids, preferred_classes, feature_vectors, brain_map_path
    )

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
