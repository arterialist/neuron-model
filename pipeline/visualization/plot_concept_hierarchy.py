import json
import pandas as pd
import numpy as np
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as hierarchy
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import hashlib
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def compute_dataset_hash(file_path: str) -> str:
    """Return a short MD5 hash for the dataset file using shell command.

    Prefer `md5sum` (GNU coreutils). On macOS, fall back to `md5 -q`.
    Returns the first 16 hex chars (lowercase).
    """
    try:
        import subprocess

        result = subprocess.run(
            ["md5sum", file_path], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout:
            md5_hex = result.stdout.strip().split()[0]
            return md5_hex.lower()[:16]
    except FileNotFoundError:
        pass

    try:
        import subprocess

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
    cache_dir = os.path.join(base_output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def is_cifar10_dataset(data: dict) -> bool:
    """Check if the dataset is CIFAR-10 based on metadata."""
    metadata = data.get("evaluation_metadata", {})
    dataset_name = metadata.get("dataset_name", "").lower()
    return "cifar" in dataset_name


def get_class_labels(data: dict) -> list:
    """Get appropriate class labels based on dataset type."""
    if is_cifar10_dataset(data):
        # CIFAR-10 classes
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    else:
        # MNIST or other datasets - use digit labels
        return [str(i) for i in range(10)]


def get_dataset_name(data: dict) -> str:
    """Get dataset name for titles."""
    if is_cifar10_dataset(data):
        return "CIFAR-10"
    else:
        return "MNIST"


def plot_concept_hierarchy(json_file_path, output_dir="concept_hierarchy_output"):
    # Create structured output directory
    json_basename = os.path.splitext(os.path.basename(json_file_path))[0]
    structured_output_dir = os.path.join(output_dir, json_basename)
    os.makedirs(structured_output_dir, exist_ok=True)

    cache_dir = get_cache_dir(structured_output_dir)

    # 1. Load Data (with caching)
    cache_key = compute_dataset_hash(json_file_path)
    data_cache_path = os.path.join(cache_dir, f"data_{cache_key}.json")

    if os.path.exists(data_cache_path):
        print(f"Loading cached data from {data_cache_path}")
        with open(data_cache_path, "r") as f:
            data = json.load(f)
    else:
        print(f"Loading data from {json_file_path}")
        with open(json_file_path, "r") as f:
            data = json.load(f)
        with open(data_cache_path, "w") as f:
            json.dump(data, f)

    df = pd.DataFrame(data["evaluation_results"])

    # Get class labels and dataset info
    class_labels = get_class_labels(data)
    dataset_name = get_dataset_name(data)

    # 2. Build Attractor Matrix (Energy Accumulation)
    attractor_matrix = np.zeros((10, 10))

    for _, row in df.iterrows():
        actual = int(row["actual_label"])

        # Accumulate Top 3 Confidences
        # (We treat confidence as 'Energy' falling into that basin)
        preds = [
            (row["predicted_label"], row["confidence"]),
            (row["second_predicted_label"], row["second_confidence"]),
            (row["third_predicted_label"], row["third_confidence"]),
        ]

        for pred_label, conf in preds:
            if pd.notnull(pred_label):
                attractor_matrix[actual, int(pred_label)] += conf

    # 3. Normalize (Create Probability Distribution per Class)
    # This ensures a class with more samples doesn't dominate
    row_sums = attractor_matrix.sum(axis=1)
    attractor_prob = attractor_matrix / row_sums[:, np.newaxis]

    # 4. PLOT 1: The Dendrogram
    print("Creating dendrogram...")

    # Calculate Correlation Distance (1 - Pearson Correlation)
    # We care about the PATTERN of errors, not the magnitude
    dist_matrix = distance.pdist(attractor_prob, metric="correlation")
    dist_matrix = np.nan_to_num(dist_matrix)  # Handle Safety

    linkage_matrix = hierarchy.linkage(dist_matrix, method="average")

    # Create dendrogram figure
    fig_dendro = go.Figure()

    # Use scipy's dendrogram to get the plot data, then convert to plotly
    from scipy.cluster.hierarchy import dendrogram

    # Create dendrogram with scipy
    dendro_result = dendrogram(
        linkage_matrix,
        labels=class_labels,
        orientation="top",
        distance_sort=True,  # Sort by distance
        show_leaf_counts=False,
        no_plot=True,  # Don't show the plot, just get data
    )

    # Convert scipy dendrogram to plotly traces
    # The dendrogram result contains the x,y coordinates for lines
    icoords = dendro_result["icoord"]  # x coordinates for lines
    dcoords = dendro_result["dcoord"]  # y coordinates for lines

    for i in range(len(icoords)):
        x_coords = icoords[i]
        y_coords = dcoords[i]

        # Create line trace for each branch
        fig_dendro.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(color="black", width=1),
                showlegend=False,
            )
        )

    # Add the leaf labels
    # Position labels at their proper x-coordinates (leaf positions are at 5, 15, 25, ... for n leaves)
    n_leaves = len(dendro_result["ivl"])
    leaf_positions = [5 + i * 10 for i in range(n_leaves)]

    for i, label in enumerate(dendro_result["ivl"]):
        fig_dendro.add_trace(
            go.Scatter(
                x=[leaf_positions[i]],
                y=[0],
                mode="text",
                text=[label],
                textposition="top center",
                showlegend=False,
                textfont=dict(size=12),
            )
        )

    fig_dendro.update_layout(
        title=f"PAULA Concept Hierarchy ({dataset_name} - Attractor Similarity)",
        xaxis_title="Class",
        yaxis_title="Semantic Distance",
        width=1000,
        height=600,
        showlegend=False,
        xaxis=dict(
            showticklabels=False
        ),  # Hide x-axis tick labels since we have text labels
        yaxis=dict(autorange="reversed"),  # Reverse y-axis so root is at top
    )

    # 5. PLOT 2: The Energy Leak Heatmap
    print("Creating heatmap...")

    # We zero out the diagonal to see the "Shadows" clearly
    # (The diagonal is usually huge, washing out the subtle leaks)
    heatmap_data = attractor_prob.copy()
    np.fill_diagonal(heatmap_data, 0)

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_data,
            x=class_labels,
            y=class_labels,
            colorscale="Magma",
            text=[[f"{val:.1%}" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
        )
    )

    fig_heatmap.update_layout(
        title=f"Attractor Leakage Map ({dataset_name} - Where does the energy go?)",
        xaxis_title="Attractor Basin (Predicted)",
        yaxis_title="Input Stimulus (Actual)",
        width=1000,
        height=800,
    )

    # Save plots
    dendro_html_path = os.path.join(
        structured_output_dir, "concept_hierarchy_dendrogram.html"
    )
    dendro_png_path = os.path.join(
        structured_output_dir, "concept_hierarchy_dendrogram.png"
    )
    heatmap_html_path = os.path.join(
        structured_output_dir, "attractor_leakage_heatmap.html"
    )
    heatmap_png_path = os.path.join(
        structured_output_dir, "attractor_leakage_heatmap.png"
    )

    print(f"Saving dendrogram to {dendro_html_path} and {dendro_png_path}")
    fig_dendro.write_html(dendro_html_path)
    dendro_json_path = dendro_html_path.replace(".html", ".json")
    pio.write_json(fig_dendro, dendro_json_path)
    fig_dendro.write_image(dendro_png_path, width=1000, height=600, scale=2)

    print(f"Saving heatmap to {heatmap_html_path} and {heatmap_png_path}")
    fig_heatmap.write_html(heatmap_html_path)
    heatmap_json_path = heatmap_html_path.replace(".html", ".json")
    pio.write_json(fig_heatmap, heatmap_json_path)
    fig_heatmap.write_image(heatmap_png_path, width=1000, height=800, scale=2)

    print(f"Plots saved to {structured_output_dir}")

    return structured_output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create concept hierarchy visualizations from neural network evaluation results"
    )
    parser.add_argument(
        "--json-file",
        type=str,
        required=True,
        help="Path to JSON file containing evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="concept_hierarchy_output",
        help="Base output directory for results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PAULA CONCEPT HIERARCHY VISUALIZATION")
    print("=" * 80)
    print(f"Input file: {args.json_file}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Run the analysis
    output_path = plot_concept_hierarchy(args.json_file, args.output_dir)

    print()
    print("✓ Concept hierarchy analysis complete!")
    print(f"Results saved to: {output_path}")
    print("Files created:")
    print("  - concept_hierarchy_dendrogram.html/png")
    print("  - attractor_leakage_heatmap.html/png")


if __name__ == "__main__":
    main()
