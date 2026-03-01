"""Concept hierarchy (attractor matrix, leakage, dendrogram)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance

from snn_classification_realtime.eval_analysis.utils import safe_float


def _cluster_node_to_dict(node: Any) -> dict[str, Any] | None:
    if node is None:
        return None
    left = _cluster_node_to_dict(getattr(node, "left", None))
    right = _cluster_node_to_dict(getattr(node, "right", None))
    dist = float(getattr(node, "dist", 0))
    nid = int(getattr(node, "id", -1))
    d: dict[str, Any] = {"id": nid, "dist": round(dist, 6)}
    if left is not None or right is not None:
        d["left"] = left
        d["right"] = right
    return d


def compute_concept_hierarchy(
    results: list[dict[str, Any]],
    num_classes: int,
    class_labels: list[str],
) -> dict[str, Any]:
    df = pd.DataFrame(results)
    if df.empty:
        return {}

    attractor_matrix = np.zeros((num_classes, num_classes))
    for _, row in df.iterrows():
        actual = int(row["actual_label"])
        preds = [
            (row.get("predicted_label"), safe_float(row.get("confidence"))),
            (
                row.get("second_predicted_label"),
                safe_float(row.get("second_confidence")),
            ),
            (
                row.get("third_predicted_label"),
                safe_float(row.get("third_confidence")),
            ),
        ]
        for pred_label, conf in preds:
            if pred_label is not None and not math.isnan(conf):
                p = int(pred_label)
                if 0 <= actual < num_classes and 0 <= p < num_classes:
                    attractor_matrix[actual, p] += conf

    row_sums = attractor_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    attractor_prob = attractor_matrix / row_sums

    leakage_by_label: dict[str, list[dict[str, Any]]] = {}
    for i in range(num_classes):
        row = attractor_prob[i].copy()
        row[i] = 0
        indices = np.argsort(-row)[:num_classes]
        leakage_by_label[class_labels[i] if i < len(class_labels) else str(i)] = [
            {
                "predicted_label": class_labels[j] if j < len(class_labels) else str(j),
                "leakage": round(float(row[j]), 4),
            }
            for j in indices
            if row[j] > 0
        ]

    dist_matrix = distance.pdist(attractor_prob, metric="correlation")
    dist_matrix = np.nan_to_num(dist_matrix)
    linkage_matrix = hierarchy.linkage(dist_matrix, method="average")
    tree = hierarchy.to_tree(linkage_matrix, rd=False)
    dendrogram_dict = _cluster_node_to_dict(tree)

    n = num_classes
    class_distances: list[list[float]] = [[0.0] * n for _ in range(n)]
    condensed_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            class_distances[i][j] = float(dist_matrix[condensed_idx])
            class_distances[j][i] = class_distances[i][j]
            condensed_idx += 1

    pairs: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(
                {
                    "label_a": class_labels[i] if i < len(class_labels) else str(i),
                    "label_b": class_labels[j] if j < len(class_labels) else str(j),
                    "distance": round(float(class_distances[i][j]), 6),
                }
            )
    pairs.sort(key=lambda x: x["distance"])
    most_similar_pairs = pairs[: min(20, len(pairs))]

    return {
        "attractor_matrix": attractor_matrix.tolist(),
        "attractor_prob": attractor_prob.tolist(),
        "leakage_by_label": leakage_by_label,
        "class_distances": class_distances,
        "most_similar_pairs": most_similar_pairs,
        "dendrogram": dendrogram_dict,
    }
