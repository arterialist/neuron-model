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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def load_activity_data(path: str) -> List[Dict[str, Any]]:
    """Loads the activity dataset from a JSON file."""
    with open(path, "r") as f:
        payload = json.load(f)
    return payload.get("records", [])


def group_by_image_index(
    records: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Groups records by image_index for unsupervised datasets."""
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for rec in tqdm(records, desc="Grouping records by image"):
        img_idx = rec.get("image_index")
        if img_idx is not None:
            buckets.setdefault(int(img_idx), []).append(rec)

    for key in buckets:
        buckets[key].sort(key=lambda r: r.get("tick", 0))
    return buckets


def extract_feature_vectors(
    image_buckets: Dict[int, List[Dict[str, Any]]], feature_type: str
) -> Tuple[np.ndarray, List[int]]:
    """
    Extracts a single feature vector for each image presentation by aggregating time-series data.
    The feature vector is the mean value of the feature for each neuron over the presentation time.
    """
    feature_vectors = []
    image_indices = []

    for img_idx, records in tqdm(
        image_buckets.items(), desc="Extracting feature vectors"
    ):
        if not records:
            continue

        time_series = []
        for record in records:
            tick_features = []
            for layer in record.get("layers", []):
                if feature_type == "firings":
                    tick_features.extend(layer.get("fired", []))
                elif feature_type == "avg_S":
                    tick_features.extend(layer.get("S", []))
            time_series.append(tick_features)

        if not time_series or not time_series[0]:
            continue

        # Convert to numpy array and calculate mean over time (axis 0)
        ts_array = np.array(time_series, dtype=np.float32)
        feature_vector = np.mean(ts_array, axis=0)

        feature_vectors.append(feature_vector)
        image_indices.append(img_idx)

    return np.array(feature_vectors), image_indices


def get_ground_truth_labels(image_indices: List[int], dataset_name: str) -> np.ndarray:
    """Loads the original dataset to get the true labels for the given image indices."""
    print(f"Loading ground truth labels from {dataset_name} dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Determine which split to load based on the max index
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

    # The dataset is typically a subset of the full test set, so we need to map indices
    # This assumes the indices in the JSON match the indices in the NON-shuffled dataset object
    labels = np.array([dataset[i][1] for i in image_indices])
    return labels


def calculate_k_distance(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Calculate the k-distance for each point in the dataset."""
    print(f"Calculating {k}-distance for each point...")
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    # Return the distance to the k-th nearest neighbor (index k, since index 0 is the point itself)
    return distances[:, k]


def find_optimal_eps(
    X: np.ndarray, min_samples: int = 5, output_dir: str = None
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


def plot_clusters(
    X_2d: np.ndarray,
    assigned_labels: np.ndarray,
    true_labels: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves a scatter plot of the clusters."""
    plt.figure(figsize=(14, 10))
    palette = sns.color_palette("deep", np.unique(assigned_labels).size)

    # Use cluster assignment for color, and true label for marker style
    markers = ["o", "s", "P", "X", "^", "v", "<", ">", "D", "*"]

    for i in range(X_2d.shape[0]):
        cluster_id = assigned_labels[i]
        true_label_id = true_labels[i]

        # Noise points in DBSCAN are labeled -1
        if cluster_id == -1:
            color = "gray"
            size = 20
        else:
            color = palette[cluster_id]
            size = 50

        plt.scatter(
            X_2d[i, 0],
            X_2d[i, 1],
            c=[color],
            s=size,
            marker=markers[true_label_id % len(markers)],
        )

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Create custom legends
    cluster_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Cluster {i}",
            markerfacecolor=palette[i],
        )
        for i in np.unique(assigned_labels)
        if i != -1
    ]
    if -1 in np.unique(assigned_labels):
        cluster_handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", label="Noise", markerfacecolor="gray"
            )
        )

    true_label_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers[i % len(markers)],
            color="gray",
            linestyle="None",
            label=f"True Label {i}",
        )
        for i in np.unique(true_labels)
    ]

    legend1 = plt.legend(
        handles=cluster_handles,
        title="Assigned Clusters",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax = plt.gca()
    ax.add_artist(legend1)
    plt.legend(
        handles=true_label_handles,
        title="True Labels",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Cluster visualization saved to {output_path}")


def plot_clusters_3d(
    X_3d: np.ndarray,
    assigned_labels: np.ndarray,
    true_labels: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves a 3D scatter plot of the clusters."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    palette = sns.color_palette("deep", np.unique(assigned_labels).size)

    # Use cluster assignment for color, and true label for marker style
    markers = ["o", "s", "P", "X", "^", "v", "<", ">", "D", "*"]

    for i in range(X_3d.shape[0]):
        cluster_id = assigned_labels[i]
        true_label_id = true_labels[i]

        # Noise points in DBSCAN are labeled -1
        if cluster_id == -1:
            color = "gray"
            size = 30
        else:
            color = palette[cluster_id]
            size = 80

        ax.scatter(
            X_3d[i, 0],
            X_3d[i, 1],
            X_3d[i, 2],
            c=[color],
            s=size,
            marker=markers[true_label_id % len(markers)],
            alpha=0.8,
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_zlabel("t-SNE Component 3")

    # Create custom legends
    cluster_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Cluster {i}",
            markerfacecolor=palette[i],
            markersize=8,
        )
        for i in np.unique(assigned_labels)
        if i != -1
    ]
    if -1 in np.unique(assigned_labels):
        cluster_handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", label="Noise", markerfacecolor="gray", markersize=6
            )
        )

    true_label_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers[i % len(markers)],
            color="gray",
            linestyle="None",
            label=f"True Label {i}",
            markersize=8,
        )
        for i in np.unique(true_labels)
    ]

    legend1 = ax.legend(
        handles=cluster_handles,
        title="Assigned Clusters",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=true_label_handles,
        title="True Labels",
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"3D cluster visualization saved to {output_path}")


def plot_clusters_cloud_3d(
    X_3d: np.ndarray,
    assigned_labels: np.ndarray,
    true_labels: np.ndarray,
    title: str,
    output_path: str,
):
    """Creates and saves a 3D cloud/volume visualization of the clusters."""
    from scipy.stats import gaussian_kde

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    palette = sns.color_palette("deep", np.unique(assigned_labels).size)

    # Create a 3D meshgrid for the density plot
    x_min, x_max = X_3d[:, 0].min() - 0.5, X_3d[:, 0].max() + 0.5
    y_min, y_max = X_3d[:, 1].min() - 0.5, X_3d[:, 1].max() + 0.5
    z_min, z_max = X_3d[:, 2].min() - 0.5, X_3d[:, 2].max() + 0.5

    xx, yy, zz_grid = np.mgrid[x_min:x_max:50j, y_min:y_max:50j, z_min:z_max:50j]

    # Plot each cluster as a colored cloud
    for cluster_id in sorted(set(assigned_labels) - {-1}):
        mask = assigned_labels == cluster_id
        if np.sum(mask) < 3:  # Skip clusters with too few points
            continue

        cluster_points = X_3d[mask]

        # Create kernel density estimate
        try:
            kde = gaussian_kde(cluster_points.T)
            # Evaluate KDE on meshgrid
            positions = np.vstack([xx.ravel(), yy.ravel(), zz_grid.ravel()])
            density = np.reshape(kde(positions), xx.shape)

            # Normalize the density
            density = density / density.max()

            # Plot filled contours (using scatter for 3D density)
            # For 3D density visualization, we'll use scatter with varying alpha
            threshold = 0.1
            high_density_mask = density > threshold

            if np.any(high_density_mask):
                scatter_x = xx[high_density_mask]
                scatter_y = yy[high_density_mask]
                scatter_z = zz_grid[high_density_mask]
                scatter_density = density[high_density_mask]

                ax.scatter(
                    scatter_x, scatter_y, scatter_z,
                    c=[palette[cluster_id]],
                    s=20,
                    alpha=scatter_density * 0.6,
                    marker='o'
                )

        except np.linalg.LinAlgError:
            # Fallback to scatter if KDE fails
            ax.scatter(
                cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                c=[palette[cluster_id]], s=60, alpha=0.6, marker='o'
            )

    # Add noise points as light scatter
    noise_mask = assigned_labels == -1
    if np.sum(noise_mask) > 0:
        noise_points = X_3d[noise_mask]
        ax.scatter(
            noise_points[:, 0], noise_points[:, 1], noise_points[:, 2],
            c="gray", s=30, alpha=0.3, marker="x"
        )

    ax.set_title(title.replace("Clustering", "3D Cloud Clustering"), fontsize=16)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_zlabel("t-SNE Component 3")

    # Create custom legends
    cluster_handles = []
    for i in sorted(set(assigned_labels) - {-1}):
        cluster_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=palette[i], label=f"Cluster {i}")
        )
    if -1 in assigned_labels:
        cluster_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc="gray", label="Noise")
        )

    ax.legend(
        handles=cluster_handles,
        title="Cell Assemblies",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"3D cluster cloud visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster unsupervised network activity data."
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
        default="clustering_results",
        help="Directory to save plots and results.",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="firings",
        choices=["firings", "avg_S"],
        help="Temporal characteristic to use for features.",
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
        "--dataset-name",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Name of the original dataset for fetching ground truth labels.",
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
    args = parser.parse_args()

    # Create a structured output directory
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    structured_output_dir = os.path.join(
        args.output_dir, input_basename, args.feature_type, args.clustering_mode
    )
    os.makedirs(structured_output_dir, exist_ok=True)

    # 1. Load and process data
    records = load_activity_data(args.input_file)
    image_buckets = group_by_image_index(records)
    feature_vectors, image_indices = extract_feature_vectors(
        image_buckets, args.feature_type
    )

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
        assigned_labels = model.fit_predict(feature_vectors)
    else:  # auto
        print("Performing DBSCAN clustering...")
        
        # Determine eps value
        if args.eps is not None:
            eps_value = args.eps
            print(f"Using provided eps value: {eps_value:.4f}")
        else:
            # Find optimal eps value using k-distance graph
            eps_value = find_optimal_eps(
                feature_vectors, min_samples=args.min_samples, output_dir=structured_output_dir
            )
            print(f"Optimal eps value found: {eps_value:.4f}")

        # Perform DBSCAN clustering
        model = DBSCAN(eps=eps_value, min_samples=args.min_samples)
        title_prefix = f"DBSCAN (eps={eps_value:.4f}, min_samples={args.min_samples})"
        assigned_labels = model.fit_predict(feature_vectors)

    num_found_clusters = len(set(assigned_labels) - {-1})
    print(f"Clustering complete. Found {num_found_clusters} clusters.")

    # 3. Evaluate results
    true_labels = get_ground_truth_labels(image_indices, args.dataset_name)

    ari = adjusted_rand_score(true_labels, assigned_labels)
    nmi = normalized_mutual_info_score(true_labels, assigned_labels)

    print("\n--- Clustering Evaluation ---")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print("-----------------------------")

    # 4. Visualize clusters
    print("Reducing dimensionality with t-SNE for visualization...")

    # Create 2D visualization (existing)
    tsne_2d = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, feature_vectors.shape[0] - 1),
    )
    X_2d = tsne_2d.fit_transform(feature_vectors)

    plot_title = f"{title_prefix} Clustering of Network Activity ({args.feature_type})"
    output_filename = os.path.join(structured_output_dir, "clusters_2d.png")
    plot_clusters(X_2d, assigned_labels, true_labels, plot_title, output_filename)

    # Create 3D visualization (new)
    print("Creating 3D t-SNE visualization...")
    tsne_3d = TSNE(
        n_components=3,
        random_state=42,
        perplexity=min(30, feature_vectors.shape[0] - 1),
    )
    X_3d = tsne_3d.fit_transform(feature_vectors)

    plot_title_3d = f"{title_prefix} 3D Clustering of Network Activity ({args.feature_type})"
    output_filename_3d = os.path.join(structured_output_dir, "clusters_3d.png")
    plot_clusters_3d(X_3d, assigned_labels, true_labels, plot_title_3d, output_filename_3d)

    # Create 3D cloud visualization (new)
    output_filename_3d_cloud = os.path.join(structured_output_dir, "clusters_3d_cloud.png")
    plot_clusters_cloud_3d(X_3d, assigned_labels, true_labels, plot_title_3d, output_filename_3d_cloud)


if __name__ == "__main__":
    main()
