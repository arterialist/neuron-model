"""Concept hierarchy visualization logic."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure project root in path when run directly
_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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


def _load_results_jsonl(results_path: str) -> list[dict]:
    """Load per-line JSON results from a JSONL file."""
    results_list: list[dict] = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results_list.append(json.loads(line))
    return results_list


def _resolve_results_path(
    summary_path: str, results_file_ref: str
) -> str | None:
    """Resolve path to results JSONL from summary metadata."""
    # Try as-is (relative to cwd)
    if os.path.isfile(results_file_ref):
        return results_file_ref
    # Try relative to summary's directory
    base = os.path.dirname(os.path.abspath(summary_path))
    candidate = os.path.join(base, os.path.basename(results_file_ref))
    if os.path.isfile(candidate):
        return candidate
    return None


def run_plot(
    json_file_path: str,
    output_dir: str = "concept_hierarchy_output",
    results_file_path: str | None = None,
) -> str:
    """Create concept hierarchy visualizations from evaluation JSON.

    Supports:
    - Single JSON with 'evaluation_results' or 'results' key (legacy)
    - Summary JSON + JSONL: summary has evaluation_metadata.results_file
    - Explicit --results-file path when passing summary
    """
    json_basename = os.path.splitext(os.path.basename(json_file_path))[0]
    structured_output_dir = os.path.join(output_dir, json_basename)
    os.makedirs(structured_output_dir, exist_ok=True)

    cache_dir = get_cache_dir(structured_output_dir, cache_version=None)

    # Load JSON to determine format and compute cache key
    with open(json_file_path) as f:
        data = json.load(f)

    results_path: str | None = None
    if "evaluation_results" in data:
        results_list = data["evaluation_results"]
    elif "results" in data:
        results_list = data["results"]
    elif isinstance(data, list):
        results_list = data
    else:
        # Split format: summary + JSONL
        results_path = results_file_path
        if not results_path:
            ref = (data.get("evaluation_metadata") or {}).get("results_file")
            if ref:
                results_path = _resolve_results_path(json_file_path, ref)
        if results_path and os.path.isfile(results_path):
            print(f"Loading results from {results_path}")
            results_list = _load_results_jsonl(results_path)
            data = {**data, "results": results_list, "evaluation_metadata": data.get("evaluation_metadata", {})}
        else:
            raise ValueError(
                "Could not find 'evaluation_results' or 'results' in JSON. "
                "For split format (summary + JSONL), ensure evaluation_metadata.results_file "
                "points to the JSONL, or pass --results-file explicitly."
            )

    cache_key = compute_dataset_hash(json_file_path)
    if results_path:
        cache_key += "_" + compute_dataset_hash(results_path)
    data_cache_path = os.path.join(cache_dir, f"data_{cache_key}.json")

    if os.path.exists(data_cache_path):
        print(f"Loading cached data from {data_cache_path}")
        with open(data_cache_path) as f:
            cached = json.load(f)
        if isinstance(cached, dict):
            results_list = cached.get("results") or cached.get("evaluation_results") or []
            data = {"evaluation_metadata": cached.get("evaluation_metadata", {}), "results": results_list}
        else:
            results_list = cached if isinstance(cached, list) else []
            data = {"evaluation_metadata": {}, "results": results_list}
    else:
        with open(data_cache_path, "w") as f:
            json.dump({"evaluation_metadata": data.get("evaluation_metadata"), "results": results_list}, f)

    if not results_list:
        raise ValueError("No evaluation results to visualize.")

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


if __name__ == "__main__":
    from snn_classification_realtime.viz.plot_concept_hierarchy import main
    main()
