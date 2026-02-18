"""Concept hierarchy visualization logic."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance
from scipy.cluster.hierarchy import dendrogram

from snn_classification_realtime.viz.utils import compute_dataset_hash, get_cache_dir


def _is_cifar10_dataset(data: dict) -> bool:
    """Check if the dataset is CIFAR-10 based on metadata."""
    metadata = data.get("evaluation_metadata", {})
    dataset_name = metadata.get("dataset_name", "").lower()
    return "cifar" in dataset_name


def _get_class_labels(data: dict) -> list[str]:
    """Get appropriate class labels based on dataset type."""
    if _is_cifar10_dataset(data):
        return [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
    return [str(i) for i in range(10)]


def _get_dataset_name(data: dict) -> str:
    """Get dataset name for titles."""
    return "CIFAR-10" if _is_cifar10_dataset(data) else "MNIST"


def run_plot(json_file_path: str, output_dir: str = "concept_hierarchy_output") -> str:
    """Create concept hierarchy visualizations from evaluation JSON."""
    json_basename = os.path.splitext(os.path.basename(json_file_path))[0]
    structured_output_dir = os.path.join(output_dir, json_basename)
    os.makedirs(structured_output_dir, exist_ok=True)

    cache_dir = get_cache_dir(structured_output_dir, cache_version=None)

    cache_key = compute_dataset_hash(json_file_path)
    data_cache_path = os.path.join(cache_dir, f"data_{cache_key}.json")

    if os.path.exists(data_cache_path):
        print(f"Loading cached data from {data_cache_path}")
        with open(data_cache_path) as f:
            data = json.load(f)
    else:
        print(f"Loading data from {json_file_path}")
        with open(json_file_path) as f:
            data = json.load(f)
        with open(data_cache_path, "w") as f:
            json.dump(data, f)

    if "evaluation_results" in data:
        results_list = data["evaluation_results"]
    elif "results" in data:
        results_list = data["results"]
    elif isinstance(data, list):
        results_list = data
    else:
        raise ValueError(
            "Could not find 'evaluation_results' or 'results' key in JSON data"
        )

    df = pd.DataFrame(results_list)
    class_labels = _get_class_labels(data)
    dataset_name = _get_dataset_name(data)

    attractor_matrix = np.zeros((10, 10))
    for _, row in df.iterrows():
        actual = int(row["actual_label"])
        preds = [
            (row["predicted_label"], row["confidence"]),
            (row["second_predicted_label"], row["second_confidence"]),
            (row["third_predicted_label"], row["third_confidence"]),
        ]
        for pred_label, conf in preds:
            if pd.notnull(pred_label):
                attractor_matrix[actual, int(pred_label)] += conf

    row_sums = attractor_matrix.sum(axis=1)
    attractor_prob = attractor_matrix / row_sums[:, np.newaxis]

    print("Creating dendrogram...")
    dist_matrix = distance.pdist(attractor_prob, metric="correlation")
    dist_matrix = np.nan_to_num(dist_matrix)
    linkage_matrix = hierarchy.linkage(dist_matrix, method="average")

    dendro_result = dendrogram(
        linkage_matrix,
        labels=class_labels,
        orientation="top",
        distance_sort=True,
        show_leaf_counts=False,
        no_plot=True,
    )

    fig_dendro = go.Figure()
    for i in range(len(dendro_result["icoord"])):
        fig_dendro.add_trace(
            go.Scatter(
                x=dendro_result["icoord"][i],
                y=dendro_result["dcoord"][i],
                mode="lines",
                line=dict(color="black", width=1),
                showlegend=False,
            )
        )

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
        xaxis=dict(showticklabels=False),
        yaxis=dict(autorange="reversed"),
    )

    print("Creating heatmap...")
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

    dendro_html = os.path.join(structured_output_dir, "concept_hierarchy_dendrogram.html")
    dendro_png = os.path.join(structured_output_dir, "concept_hierarchy_dendrogram.png")
    heatmap_html = os.path.join(structured_output_dir, "attractor_leakage_heatmap.html")
    heatmap_png = os.path.join(structured_output_dir, "attractor_leakage_heatmap.png")

    print(f"Saving dendrogram to {dendro_html} and {dendro_png}")
    fig_dendro.write_html(dendro_html)
    pio.write_json(fig_dendro, dendro_html.replace(".html", ".json"))
    fig_dendro.write_image(dendro_png, width=1000, height=600, scale=2)

    print(f"Saving heatmap to {heatmap_html} and {heatmap_png}")
    fig_heatmap.write_html(heatmap_html)
    pio.write_json(fig_heatmap, heatmap_html.replace(".html", ".json"))
    fig_heatmap.write_image(heatmap_png, width=1000, height=800, scale=2)

    print(f"Plots saved to {structured_output_dir}")
    return structured_output_dir
