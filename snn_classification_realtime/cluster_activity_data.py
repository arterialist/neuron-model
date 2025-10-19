import os
import json
import argparse
from typing import Dict, Any, Iterable, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import subprocess
from torchvision import datasets, transforms

# Plotting and caching configuration (aligned with cluster_neurons.py)
PLOT_IMAGE_SCALE: float = 2.0
MATPLOTLIB_DPI: int = 400
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
    cache_dir = os.path.join(base_output_dir, f"cache_{CACHE_VERSION}")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_activity_data(path: str) -> List[Dict[str, Any]]:
    """Loads the activity dataset from a JSON file."""
    with open(path, "r") as f:
        payload = json.load(f)
    return payload.get("records", [])


def sort_records_deterministically(
    records: Iterable[Dict[str, Any]],
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

        if bucket["label"] is None and rec.get("label") is not None:
            bucket["label"] = rec.get("label")

        bucket["records"].append(rec)

    for bucket in buckets.values():
        bucket["records"].sort(key=lambda r: r.get("tick", 0))

    return buckets


def extract_feature_vectors(
    image_buckets: Dict[int, Dict[str, Any]],
    feature_types: List[str],
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a single vector per image presentation by aggregating requested metrics across time.
    Supported feature_types include: "firings" -> fired; "avg_S" -> S; "avg_t_ref" -> t_ref.
    For each requested feature, we compute the mean over time per neuron and concatenate.
    """
    feature_vectors: List[np.ndarray] = []
    labels: List[int] = []

    for img_idx, bucket in tqdm(
        image_buckets.items(), desc="Extracting feature vectors"
    ):
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
        if not records:
            continue

        per_feature_vectors: List[np.ndarray] = []

        for ftype in feature_types:
            time_series = []
            for record in records:
                tick_features = []
                for layer in record.get("layers", []):
                    if ftype == "firings":
                        tick_features.extend(layer.get("fired", []))
                    elif ftype == "avg_S":
                        tick_features.extend(layer.get("S", []))
                    elif ftype == "avg_t_ref":
                        tick_features.extend(layer.get("t_ref", []))
                    else:
                        raise ValueError(f"Unsupported feature type: {ftype}")
                time_series.append(tick_features)

            if not time_series or not time_series[0]:
                continue

            ts_array = np.array(time_series, dtype=np.float32)
            per_feature_vectors.append(np.mean(ts_array, axis=0))

        if not per_feature_vectors:
            continue

        feature_vectors.append(np.concatenate(per_feature_vectors, axis=0))
        labels.append(label)

    return np.array(feature_vectors), np.array(labels)


def calculate_k_distance(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Calculate the k-distance for each point in the dataset."""
    print(f"Calculating {k}-distance for each point...")
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    # Return the distance to the k-th nearest neighbor (index k, since index 0 is the point itself)
    return distances[:, k]


def find_optimal_eps(
    X: np.ndarray, min_samples: int = 5, output_dir: Optional[str] = None
) -> float:
    """
    Find the optimal eps value for DBSCAN using the k-distance graph method.

    Args:
        X: Feature vectors
        min_samples: Minimum samples parameter for DBSCAN
        output_dir: Directory to save the k-distance plot

    Returns:
        Optimal eps value
    """
    print("Finding optimal eps value using k-distance graph...")

    # Calculate k-distances
    k_distances = calculate_k_distance(X, min_samples)

    # Sort the distances
    sorted_distances = np.sort(k_distances)

    # Find the elbow point using the second derivative method
    # Calculate first and second derivatives
    first_derivative = np.gradient(sorted_distances)
    second_derivative = np.gradient(first_derivative)

    # Find the point of maximum curvature (elbow)
    # We look for the point where the second derivative is most negative
    elbow_idx = np.argmin(second_derivative)
    optimal_eps = sorted_distances[elbow_idx]

    print(f"Optimal eps value found: {optimal_eps:.4f}")

    # Plot the k-distance graph
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


def plot_clusters(
    X_2d: np.ndarray,
    assigned_labels: np.ndarray,
    true_labels: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves an interactive 2D scatter plot of clusters using Plotly."""
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
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

    for cid in sorted(set(assigned_labels)):
        mask = assigned_labels == cid
        color = "lightgray" if cid == -1 else palette[cid % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=X_2d[mask, 0],
                y=X_2d[mask, 1],
                mode="markers",
                name="Noise" if cid == -1 else f"Cluster {cid}",
                marker=dict(
                    size=10,
                    color=color,
                    symbol=[
                        marker_symbols[int(lbl) % len(marker_symbols)]
                        for lbl in true_labels[mask]
                    ],
                    line=dict(width=1, color="black"),
                ),
                hovertemplate="Cluster: %{text}<extra></extra>",
                text=[f"cid={cid}, label={int(lbl)}" for lbl in true_labels[mask]],
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"2D cluster visualization saved to {output_path}")


def plot_clusters_3d(
    X_3d: np.ndarray,
    assigned_labels: np.ndarray,
    true_labels: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves an interactive 3D scatter plot using Plotly."""
    fig = go.Figure()
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

    for cid in sorted(set(assigned_labels)):
        mask = assigned_labels == cid
        color = "lightgray" if cid == -1 else palette[cid % len(palette)]
        symbols = [
            marker_symbols[int(lbl) % len(marker_symbols)] for lbl in true_labels[mask]
        ]
        fig.add_trace(
            go.Scatter3d(
                x=X_3d[mask, 0],
                y=X_3d[mask, 1],
                z=X_3d[mask, 2],
                mode="markers",
                name="Noise" if cid == -1 else f"Cluster {cid}",
                marker=dict(
                    size=6,
                    color=color,
                    symbol=symbols,
                    line=dict(width=1, color="black"),
                ),
                hovertemplate="Cluster: %{text}<extra></extra>",
                text=[f"cid={cid}, label={int(lbl)}" for lbl in true_labels[mask]],
            )
        )

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

    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"3D cluster visualization saved to {output_path}")


def plot_clusters_cloud_3d(
    X_3d: np.ndarray,
    assigned_labels: np.ndarray,
    true_labels: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves a 3D cloud/volume visualization using Plotly isosurfaces."""
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly

    # Grid for KDE
    x_min, x_max = float(X_3d[:, 0].min()) - 0.3, float(X_3d[:, 0].max()) + 0.3
    y_min, y_max = float(X_3d[:, 1].min()) - 0.3, float(X_3d[:, 1].max()) + 0.3
    z_min, z_max = float(X_3d[:, 2].min()) - 0.3, float(X_3d[:, 2].max()) + 0.3
    xx, yy, zz_grid = np.mgrid[x_min:x_max:40j, y_min:y_max:40j, z_min:z_max:40j]

    for cid in sorted(set(assigned_labels)):
        mask = assigned_labels == cid
        pts = X_3d[mask]
        color = "lightgray" if cid == -1 else palette[cid % len(palette)]

        # Scatter overlay
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name="Noise" if cid == -1 else f"Cluster {cid}",
                marker=dict(size=4, color=color, opacity=0.7),
            )
        )

        if cid == -1:
            continue
        try:
            if len(pts) >= 3:
                kde = gaussian_kde(pts.T, bw_method=0.3)
                positions = np.vstack([xx.ravel(), yy.ravel(), zz_grid.ravel()])
                density = np.reshape(kde(positions), xx.shape)
                max_d = float(np.max(density))
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
                            name=f"Cluster {cid} density",
                            showlegend=False,
                        )
                    )
        except (np.linalg.LinAlgError, ValueError):
            pass

    fig.update_layout(
        title=dict(
            text=title.replace("Clustering", "3D Cloud Clustering"), font=dict(size=18)
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
    fig.write_image(output_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    print(f"3D cluster cloud visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster unsupervised network activity data."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the activity dataset (supervised/unsupervised deprecated wording removed).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="clustering_results",
        help="Directory to save plots and results.",
    )
    parser.add_argument(
        "--feature-types",
        type=str,
        default="firings,avg_S",
        help="Comma-separated list of features to aggregate (e.g., 'firings,avg_S,avg_t_ref').",
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
        default=10,
        help="Number of clusters for 'fixed' mode (K-Means).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Eps parameter for DBSCAN clustering. If not provided, will be automatically determined using k-distance graph.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Min_samples parameter for DBSCAN clustering.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Total number of classes in labels (used for validation).",
    )
    args = parser.parse_args()

    # Create a structured output directory
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    structured_output_dir = os.path.join(
        args.output_dir, input_basename, "multi_features", args.clustering_mode
    )
    os.makedirs(structured_output_dir, exist_ok=True)

    # Caching setup
    dataset_hash = compute_dataset_hash(args.input_file)
    cache_dir = get_cache_dir(structured_output_dir)

    # 1. Load and process data
    records = load_activity_data(args.input_file)
    image_buckets = group_by_image(records)
    feature_type_list = [s.strip() for s in args.feature_types.split(",") if s.strip()]

    features_cache_path = os.path.join(
        cache_dir, f"{dataset_hash}_features_{'-'.join(feature_type_list)}.npz"
    )
    if os.path.exists(features_cache_path):
        cache = np.load(features_cache_path, allow_pickle=True)
        feature_vectors = cache["feature_vectors"]
        true_labels = cache["true_labels"]
        print(f"Loaded cached features from {features_cache_path}")
    else:
        feature_vectors, true_labels = extract_feature_vectors(
            image_buckets, feature_type_list, args.num_classes
        )
        np.savez_compressed(
            features_cache_path,
            feature_vectors=feature_vectors,
            true_labels=true_labels,
        )
        print(f"Saved features cache to {features_cache_path}")

    if feature_vectors.shape[0] == 0:
        print("No feature vectors could be extracted. Exiting.")
        return

    print(
        f"Extracted {feature_vectors.shape[0]} feature vectors of size {feature_vectors.shape[1]}."
    )

    # 2. Perform clustering
    if args.clustering_mode == "fixed":
        print(f"Performing K-Means clustering with K={args.num_clusters}...")
        model = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10)
        title_prefix = f"K-Means (K={args.num_clusters})"
        labels_cache_path = os.path.join(
            cache_dir, f"{dataset_hash}_kmeans_k{args.num_clusters}.npy"
        )
        if os.path.exists(labels_cache_path):
            assigned_labels = np.load(labels_cache_path)
            print(f"Loaded cached KMeans labels from {labels_cache_path}")
        else:
            assigned_labels = model.fit_predict(feature_vectors)
            np.save(labels_cache_path, assigned_labels)
    else:  # auto
        print("Performing DBSCAN clustering...")

        # Determine eps value
        if args.eps is not None:
            eps_value = args.eps
            print(f"Using provided eps value: {eps_value:.4f}")
        else:
            # Find optimal eps value using k-distance graph
            eps_cache_path = os.path.join(
                cache_dir, f"{dataset_hash}_dbscan_m{args.min_samples}_eps.json"
            )
            if os.path.exists(eps_cache_path):
                with open(eps_cache_path, "r") as f:
                    eps_value = float(json.load(f).get("eps", 0.5))
                print(f"Loaded cached eps from {eps_cache_path}: {eps_value:.4f}")
            else:
                eps_value = find_optimal_eps(
                    feature_vectors,
                    min_samples=args.min_samples,
                    output_dir=structured_output_dir,
                )
                with open(eps_cache_path, "w") as f:
                    json.dump({"eps": float(eps_value)}, f)
                print(f"Saved eps cache to {eps_cache_path}")
            print(f"Optimal eps value found: {eps_value:.4f}")

        # Perform DBSCAN clustering
        model = DBSCAN(eps=eps_value, min_samples=args.min_samples)
        title_prefix = f"DBSCAN (eps={eps_value:.4f}, min_samples={args.min_samples})"
        labels_cache_path = os.path.join(
            cache_dir,
            f"{dataset_hash}_dbscan_eps{eps_value:.4f}_m{args.min_samples}.npy",
        )
        if os.path.exists(labels_cache_path):
            assigned_labels = np.load(labels_cache_path)
            print(f"Loaded cached DBSCAN labels from {labels_cache_path}")
        else:
            assigned_labels = model.fit_predict(feature_vectors)
            np.save(labels_cache_path, assigned_labels)

    num_found_clusters = len(set(assigned_labels) - {-1})
    print(f"Clustering complete. Found {num_found_clusters} clusters.")

    # 3. Evaluate results
    ari = adjusted_rand_score(true_labels, assigned_labels)
    nmi = normalized_mutual_info_score(true_labels, assigned_labels)

    print("\n--- Clustering Evaluation ---")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print("-----------------------------")

    # 4. Visualize clusters
    print("Reducing dimensionality with t-SNE for visualization...")

    tsne_perplexity = min(30, feature_vectors.shape[0] - 1)
    tsne_2d_cache = os.path.join(
        cache_dir, f"{dataset_hash}_tsne2d_p{tsne_perplexity}.npy"
    )
    if os.path.exists(tsne_2d_cache):
        X_2d = np.load(tsne_2d_cache)
        print(f"Loaded cached t-SNE 2D from {tsne_2d_cache}")
    else:
        tsne_2d = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
        X_2d = tsne_2d.fit_transform(feature_vectors)
        np.save(tsne_2d_cache, X_2d)

    plot_title = f"{title_prefix} Clustering of Network Activity (multi-features)"
    output_filename = os.path.join(structured_output_dir, "clusters_2d.png")
    plot_clusters(X_2d, assigned_labels, true_labels, plot_title, output_filename)

    # Create 3D visualization (new)
    print("Creating 3D t-SNE visualization...")
    tsne_3d_cache = os.path.join(
        cache_dir, f"{dataset_hash}_tsne3d_p{tsne_perplexity}.npy"
    )
    if os.path.exists(tsne_3d_cache):
        X_3d = np.load(tsne_3d_cache)
        print(f"Loaded cached t-SNE 3D from {tsne_3d_cache}")
    else:
        tsne_3d = TSNE(n_components=3, random_state=42, perplexity=tsne_perplexity)
        X_3d = tsne_3d.fit_transform(feature_vectors)
        np.save(tsne_3d_cache, X_3d)

    plot_title_3d = f"{title_prefix} 3D Clustering of Network Activity (multi-features)"
    output_filename_3d = os.path.join(structured_output_dir, "clusters_3d.png")
    plot_clusters_3d(
        X_3d, assigned_labels, true_labels, plot_title_3d, output_filename_3d
    )

    # Create 3D cloud visualization (new)
    output_filename_3d_cloud = os.path.join(
        structured_output_dir, "clusters_3d_cloud.png"
    )
    plot_clusters_cloud_3d(
        X_3d, assigned_labels, true_labels, plot_title_3d, output_filename_3d_cloud
    )


if __name__ == "__main__":
    main()
