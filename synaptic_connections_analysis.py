#!/usr/bin/env python3
"""
Synaptic Connectivity Analysis Script

Analyzes network states exported by build_activity_dataset.py to find connectivity
clusters based on synaptic connection weights. Uses graph-based community detection
and clustering to identify densely connected neuron groups.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.cluster import SpectralClustering, DBSCAN
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def load_network_state(filepath: Path) -> Dict[str, Any]:
    """Load a single network state JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_synaptic_weights(
    network_state: Dict[str, Any],
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, int]]:
    """
    Extract synaptic weights from network state.

    Returns:
        weight_matrix: Dict[source_neuron][target_neuron] -> weight
        neuron_layers: Dict[neuron_id] -> layer_index
    """
    # Build connection map: (source, target) -> synapse data
    connections = network_state.get("connections", [])
    synaptic_points = network_state.get("synaptic_points", [])
    neurons = network_state.get("neurons", [])

    # Map neuron_id -> layer from metadata
    neuron_layers: Dict[int, int] = {}
    for neuron in neurons:
        nid = neuron.get("id")
        layer = neuron.get("metadata", {}).get("layer", 0)
        neuron_layers[nid] = layer

    # Index postsynaptic points by (neuron_id, synapse_id)
    synapse_weights: Dict[Tuple[int, int], float] = {}
    for point in synaptic_points:
        if point.get("type") == "postsynaptic":
            neuron_id = point["neuron_id"]
            synapse_id = point["synapse_id"]
            u_i = point.get("u_i", {})
            # Combined weight = info * plast (effective synaptic strength)
            info = u_i.get("info", 1.0)
            plast = u_i.get("plast", 1.0)
            weight = info * plast
            synapse_weights[(neuron_id, synapse_id)] = weight

    # Build weight matrix from connections
    weight_matrix: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for conn in connections:
        source = conn["source_neuron"]
        target = conn["target_neuron"]
        target_synapse = conn["target_synapse"]
        weight = synapse_weights.get((target, target_synapse), 1.0)
        # Accumulate weights if multiple connections exist
        weight_matrix[source][target] += weight

    return dict(weight_matrix), neuron_layers


def build_adjacency_matrix(
    weight_matrix: Dict[int, Dict[int, float]], neuron_ids: List[int]
) -> np.ndarray:
    """Convert weight dict to numpy adjacency matrix."""
    n = len(neuron_ids)
    id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
    adj = np.zeros((n, n), dtype=np.float64)

    for src, targets in weight_matrix.items():
        if src not in id_to_idx:
            continue
        i = id_to_idx[src]
        for tgt, weight in targets.items():
            if tgt not in id_to_idx:
                continue
            j = id_to_idx[tgt]
            adj[i, j] = weight

    return adj


def find_clusters_spectral(adj_matrix: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """Find connectivity clusters using spectral clustering."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required for spectral clustering")

    # Make symmetric for spectral clustering (undirected graph interpretation)
    symmetric = (adj_matrix + adj_matrix.T) / 2
    # Ensure non-negative weights
    symmetric = np.abs(symmetric)

    # Use spectral clustering
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    labels = clustering.fit_predict(symmetric + np.eye(len(symmetric)) * 1e-6)
    return labels


def find_clusters_hierarchical(
    adj_matrix: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, Any]:
    """Find connectivity clusters using hierarchical clustering."""
    # Convert to distance matrix (inverse of weight)
    symmetric = (adj_matrix + adj_matrix.T) / 2
    symmetric = np.abs(symmetric)
    max_weight = symmetric.max() if symmetric.max() > 0 else 1.0
    distance = max_weight - symmetric
    np.fill_diagonal(distance, 0)

    # Hierarchical clustering
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="ward")
    labels = fcluster(linkage_matrix, t=threshold * max_weight, criterion="distance")
    return labels - 1, linkage_matrix  # 0-indexed


def find_communities_louvain(adj_matrix: np.ndarray) -> np.ndarray:
    """Find connectivity communities using Louvain community detection."""
    # Build networkx graph
    n = adj_matrix.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j, weight=adj_matrix[i, j])

    # Convert to undirected for community detection
    G_undirected = G.to_undirected()

    # Use Louvain algorithm
    try:
        communities = nx.community.louvain_communities(  # type: ignore[attr-defined]
            G_undirected, weight="weight", seed=42
        )
        labels = np.zeros(n, dtype=int)
        for idx, comm in enumerate(communities):
            for node in comm:
                labels[node] = idx
        return labels
    except Exception:
        # Fallback to greedy modularity
        communities = nx.community.greedy_modularity_communities(  # type: ignore[attr-defined]
            G_undirected, weight="weight"
        )
        labels = np.zeros(n, dtype=int)
        for idx, comm in enumerate(communities):
            for node in comm:
                labels[node] = idx
        return labels


def aggregate_states_by_label(
    root_dir: Path,
) -> Dict[int, List[Dict[str, Any]]]:
    """Load all network states grouped by label."""
    label_states: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for label_dir in sorted(root_dir.iterdir()):
        if not label_dir.is_dir() or not label_dir.name.startswith("label_"):
            continue
        label = int(label_dir.name.split("_")[1])

        for state_file in sorted(label_dir.glob("*.json")):
            state = load_network_state(state_file)
            state["_source_file"] = str(state_file)
            label_states[label].append(state)

    return dict(label_states)


def compute_mean_weight_matrix(
    states: List[Dict[str, Any]],
) -> Tuple[Dict[int, Dict[int, float]], List[int], Dict[int, int]]:
    """Compute mean synaptic weight matrix across multiple states."""
    all_weights: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    all_neuron_ids: set = set()
    neuron_layers: Dict[int, int] = {}

    for state in states:
        weights, layers = extract_synaptic_weights(state)
        neuron_layers.update(layers)
        for src, targets in weights.items():
            all_neuron_ids.add(src)
            for tgt, w in targets.items():
                all_neuron_ids.add(tgt)
                all_weights[(src, tgt)].append(w)

    # Compute mean
    mean_weights: Dict[int, Dict[int, float]] = defaultdict(dict)
    for (src, tgt), weight_list in all_weights.items():
        mean_weights[src][tgt] = float(np.mean(weight_list))

    neuron_ids = sorted(all_neuron_ids)
    return dict(mean_weights), neuron_ids, neuron_layers


def plot_weight_matrix_heatmap(
    adj_matrix: np.ndarray,
    neuron_ids: List[int],
    labels: np.ndarray,
    title: str,
    output_path: Path,
    neuron_layers: Optional[Dict[int, int]] = None,
):
    """Plot weight matrix as heatmap with cluster annotations."""
    # Sort by cluster label, then by layer
    sort_idx = np.argsort(labels)
    sorted_matrix = adj_matrix[sort_idx][:, sort_idx]
    sorted_labels = labels[sort_idx]
    sorted_ids = [neuron_ids[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Use log scale for better visualization
    plot_data = np.log1p(sorted_matrix)
    im = ax.imshow(plot_data, cmap="viridis", aspect="auto")

    # Add cluster boundaries
    unique_labels = np.unique(sorted_labels)
    boundaries = []
    for lbl in unique_labels[:-1]:
        idx = np.where(sorted_labels == lbl)[0][-1]
        boundaries.append(idx + 0.5)

    for b in boundaries:
        ax.axhline(y=b, color="red", linewidth=1, linestyle="--")
        ax.axvline(x=b, color="red", linewidth=1, linestyle="--")

    ax.set_title(f"{title}\n(log scale, clusters separated by red lines)")
    ax.set_xlabel("Target Neuron Index")
    ax.set_ylabel("Source Neuron Index")

    plt.colorbar(im, ax=ax, label="log(1 + weight)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved heatmap: {output_path}")


def plot_network_graph(
    adj_matrix: np.ndarray,
    neuron_ids: List[int],
    labels: np.ndarray,
    title: str,
    output_path: Path,
    neuron_layers: Optional[Dict[int, int]] = None,
    weight_threshold: float = 0.1,
):
    """Plot network as graph with nodes colored by cluster."""
    n = len(neuron_ids)
    G = nx.DiGraph()

    # Add nodes with attributes
    for i, nid in enumerate(neuron_ids):
        layer = neuron_layers.get(nid, 0) if neuron_layers else 0
        G.add_node(i, neuron_id=nid, cluster=labels[i], layer=layer)

    # Add edges above threshold
    max_weight = adj_matrix.max() if adj_matrix.max() > 0 else 1.0
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] > weight_threshold * max_weight:
                G.add_edge(i, j, weight=adj_matrix[i, j])

    # Layout: use spring layout or layer-based if layers available
    if neuron_layers:
        # Multipartite layout by layer
        layer_map = {i: neuron_layers.get(nid, 0) for i, nid in enumerate(neuron_ids)}
        for i in G.nodes():
            G.nodes[i]["subset"] = layer_map[i]
        try:
            pos = nx.multipartite_layout(G, subset_key="subset")
        except Exception:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Color nodes by cluster
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    node_colors = [cmap(labels[i]) for i in range(n)]

    # Draw edges
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    if edge_weights:
        max_ew = max(edge_weights)
        edge_widths = [0.5 + 2 * w / max_ew for w in edge_weights]
        edge_alphas = [0.3 + 0.5 * w / max_ew for w in edge_weights]
    else:
        edge_widths = []
        edge_alphas = []

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_widths,  # type: ignore[arg-type]
        alpha=0.3,
        edge_color="gray",
        arrows=True,
        arrowsize=8,
        connectionstyle="arc3,rad=0.1",
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,  # type: ignore[arg-type]
        node_size=100,
        alpha=0.9,
    )

    # Legend for clusters
    patches = [
        mpatches.Patch(color=cmap(i), label=f"Cluster {i}")
        for i in range(len(unique_labels))
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8)

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved graph: {output_path}")


def plot_dendrogram(
    linkage_matrix: Any,
    neuron_ids: List[int],
    title: str,
    output_path: Path,
):
    """Plot hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(
        linkage_matrix,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=6,
        labels=[str(nid) for nid in neuron_ids],
    )
    ax.set_title(title)
    ax.set_xlabel("Neuron ID")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved dendrogram: {output_path}")


def plot_cluster_summary(
    labels: np.ndarray,
    neuron_ids: List[int],
    neuron_layers: Dict[int, int],
    title: str,
    output_path: Path,
):
    """Plot summary of clusters: size distribution and layer composition."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cluster size distribution
    ax1 = axes[0]
    sizes = [np.sum(labels == lbl) for lbl in unique_labels]
    ax1.bar(range(n_clusters), sizes, color="steelblue")
    ax1.set_xlabel("Cluster ID")
    ax1.set_ylabel("Number of Neurons")
    ax1.set_title("Cluster Sizes")
    ax1.set_xticks(range(n_clusters))

    # Layer composition per cluster
    ax2 = axes[1]
    unique_layers = sorted(set(neuron_layers.values()))
    layer_counts = np.zeros((n_clusters, len(unique_layers)))

    for i, nid in enumerate(neuron_ids):
        cluster = labels[i]
        layer = neuron_layers.get(nid, 0)
        layer_idx = unique_layers.index(layer)
        cluster_idx = list(unique_labels).index(cluster)
        layer_counts[cluster_idx, layer_idx] += 1

    x = np.arange(n_clusters)
    width = 0.8 / len(unique_layers)
    cmap = plt.cm.get_cmap("Set2", len(unique_layers))

    for i, layer in enumerate(unique_layers):
        offset = (i - len(unique_layers) / 2) * width + width / 2
        ax2.bar(
            x + offset,
            layer_counts[:, i],
            width,
            label=f"Layer {layer}",
            color=cmap(i),
        )

    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Number of Neurons")
    ax2.set_title("Layer Composition per Cluster")
    ax2.set_xticks(range(n_clusters))
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved summary: {output_path}")


def analyze_connectivity_clusters(
    root_dir: Path,
    output_dir: Path,
    method: str = "louvain",
    n_clusters: int = 5,
):
    """Main analysis: find connectivity clusters for each label and compare."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all states
    print(f"Loading network states from: {root_dir}")
    label_states = aggregate_states_by_label(root_dir)

    if not label_states:
        print("No network states found!")
        return

    print(f"Found {len(label_states)} labels with states")

    # Analyze each label
    all_label_clusters: Dict[int, Tuple[np.ndarray, List[int]]] = {}

    for label, states in tqdm(label_states.items(), desc="Analyzing labels"):
        print(f"\n=== Label {label}: {len(states)} samples ===")

        # Compute mean weight matrix
        mean_weights, neuron_ids, neuron_layers = compute_mean_weight_matrix(states)
        adj_matrix = build_adjacency_matrix(mean_weights, neuron_ids)

        print(f"  Neurons: {len(neuron_ids)}, Connections: {np.sum(adj_matrix > 0)}")

        # Find clusters
        if method == "spectral":
            labels_arr = find_clusters_spectral(adj_matrix, n_clusters=n_clusters)
            linkage_matrix = None
        elif method == "hierarchical":
            labels_arr, linkage_matrix = find_clusters_hierarchical(adj_matrix)
        else:  # louvain
            labels_arr = find_communities_louvain(adj_matrix)
            linkage_matrix = None

        n_found = len(np.unique(labels_arr))
        print(f"  Found {n_found} connectivity clusters using {method}")

        all_label_clusters[label] = (labels_arr, neuron_ids)

        # Generate plots
        label_dir = output_dir / f"label_{label}"
        label_dir.mkdir(exist_ok=True)

        plot_weight_matrix_heatmap(
            adj_matrix,
            neuron_ids,
            labels_arr,
            f"Label {label} - Synaptic Weight Matrix",
            label_dir / "weight_matrix_heatmap.png",
            neuron_layers,
        )

        plot_network_graph(
            adj_matrix,
            neuron_ids,
            labels_arr,
            f"Label {label} - Connectivity Graph ({n_found} clusters)",
            label_dir / "connectivity_graph.png",
            neuron_layers,
            weight_threshold=0.05,
        )

        if linkage_matrix is not None:
            plot_dendrogram(
                linkage_matrix,
                neuron_ids,
                f"Label {label} - Hierarchical Clustering",
                label_dir / "dendrogram.png",
            )

        plot_cluster_summary(
            labels_arr,
            neuron_ids,
            neuron_layers,
            f"Label {label} - Cluster Summary",
            label_dir / "cluster_summary.png",
        )

    # Cross-label comparison
    print("\n=== Cross-Label Analysis ===")
    plot_cross_label_comparison(all_label_clusters, output_dir)

    print(f"\nAnalysis complete. Results saved to: {output_dir}")


def plot_cross_label_comparison(
    all_label_clusters: Dict[int, Tuple[np.ndarray, List[int]]],
    output_dir: Path,
):
    """Compare connectivity cluster structures across labels."""
    labels_list = sorted(all_label_clusters.keys())
    n_labels = len(labels_list)

    # Compute cluster counts per label
    cluster_counts = []
    for label in labels_list:
        labels_arr, _ = all_label_clusters[label]
        cluster_counts.append(len(np.unique(labels_arr)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Number of clusters per label
    ax1 = axes[0]
    ax1.bar(labels_list, cluster_counts, color="coral")
    ax1.set_xlabel("Label")
    ax1.set_ylabel("Number of Clusters")
    ax1.set_title("Cluster Count per Label")
    ax1.set_xticks(labels_list)

    # Cluster size distributions (box plot)
    ax2 = axes[1]
    size_data = []
    for label in labels_list:
        labels_arr, _ = all_label_clusters[label]
        unique = np.unique(labels_arr)
        sizes = [np.sum(labels_arr == u) for u in unique]
        size_data.append(sizes)

    ax2.boxplot(size_data, labels=[str(l) for l in labels_list])
    ax2.set_xlabel("Label")
    ax2.set_ylabel("Cluster Size")
    ax2.set_title("Cluster Size Distribution per Label")

    plt.suptitle("Cross-Label Connectivity Comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "cross_label_comparison.png", dpi=200)
    plt.close()
    print(f"Saved cross-label comparison: {output_dir / 'cross_label_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze network states to find connectivity clusters based on synaptic weights"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory containing network state exports (e.g., network_state/<dataset_name>)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: <root_dir>/connectivity_analysis_<method>)",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        choices=["louvain", "spectral", "hierarchical"],
        default="louvain",
        help="Clustering method (default: louvain)",
    )
    parser.add_argument(
        "--n-clusters",
        "-n",
        type=int,
        default=5,
        help="Number of clusters for spectral clustering (default: 5)",
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    output_dir = (
        Path(args.output)
        if args.output
        else root_dir / f"connectivity_analysis_{args.method}"
    )

    analyze_connectivity_clusters(
        root_dir,
        output_dir,
        method=args.method,
        n_clusters=args.n_clusters,
    )


if __name__ == "__main__":
    main()
