"""Confusion matrix and top error pairs."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_confusion_matrix(
    results: list[dict[str, Any]],
    num_classes: int,
    class_labels: list[str],
    top_k: int = 15,
) -> dict[str, Any]:
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for r in results:
        actual = r.get("actual_label")
        pred = r.get("predicted_label")
        if (
            actual is not None
            and pred is not None
            and 0 <= actual < num_classes
            and 0 <= pred < num_classes
        ):
            cm[actual, pred] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized = (cm / row_sums).tolist()

    errors: list[dict[str, Any]] = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                total_i = cm[i, :].sum()
                rate = float(cm[i, j] / total_i) if total_i > 0 else 0
                errors.append(
                    {
                        "actual_label": class_labels[i]
                        if i < len(class_labels)
                        else str(i),
                        "predicted_label": class_labels[j]
                        if j < len(class_labels)
                        else str(j),
                        "count": int(cm[i, j]),
                        "rate": round(rate, 4),
                    }
                )
    errors.sort(key=lambda x: -x["count"])
    top_errors = errors[:top_k]

    return {
        "counts": cm.tolist(),
        "normalized": normalized,
        "top_errors": top_errors,
    }
