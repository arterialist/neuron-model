"""Specialization (generalist vs specialist) metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from snn_classification_realtime.eval_analysis.utils import safe_float


def gini_coefficient(values: list[float]) -> float:
    """Gini coefficient; 0 = perfectly equal, 1 = maximally unequal."""
    arr = np.array([x for x in values if not math.isnan(x)])
    if len(arr) == 0:
        return float("nan")
    arr = np.sort(arr)
    n = len(arr)
    cumsum = np.cumsum(arr)
    return float(
        (n + 1 - 2 * np.sum(cumsum) / (cumsum[-1] if cumsum[-1] != 0 else 1)) / n
    )


def compute_specialization(
    results: list[dict[str, Any]],
    num_classes: int,
    class_labels: list[str],
    top_k: int = 10,
) -> dict[str, Any]:
    by_label: dict[int, list[dict[str, Any]]] = {i: [] for i in range(num_classes)}
    for r in results:
        al = r.get("actual_label")
        if al is not None and al in by_label:
            by_label[al].append(r)

    accuracies = []
    confidence_means = []
    thinking_means = []
    first_tick_means = []
    for label in range(num_classes):
        rows = by_label[label]
        if not rows:
            accuracies.append(float("nan"))
            confidence_means.append(float("nan"))
            thinking_means.append(float("nan"))
            first_tick_means.append(float("nan"))
            continue
        acc = 100.0 * sum(1 for r in rows if r.get("correct", False)) / len(rows)
        accuracies.append(acc)
        confs = [safe_float(r.get("confidence")) for r in rows]
        confs = [c for c in confs if not math.isnan(c)]
        confidence_means.append(float(np.mean(confs)) if confs else float("nan"))
        ticks_added = [r.get("total_ticks_added", 0) for r in rows]
        thinking_means.append(float(np.mean(ticks_added)))
        ticks_first = [
            r["first_correct_tick"]
            for r in rows
            if r.get("first_correct_tick") is not None
        ]
        first_tick_means.append(
            float(np.mean(ticks_first)) if ticks_first else float("nan")
        )

    accuracies_valid = [a for a in accuracies if not math.isnan(a)]
    thinking_valid = [t for t in thinking_means if not math.isnan(t)]
    first_tick_valid = [t for t in first_tick_means if not math.isnan(t)]

    def std_cv_gini(arr: list[float]) -> tuple[float, float, float]:
        a = np.array(arr)
        if len(a) == 0:
            return float("nan"), float("nan"), float("nan")
        std = float(np.std(a))
        mean = float(np.mean(a))
        cv = std / mean if mean != 0 else float("nan")
        gini = gini_coefficient(arr)
        return std, cv, gini

    acc_std, acc_cv, acc_gini = std_cv_gini(accuracies_valid)
    min_max_gap = (
        max(accuracies_valid) - min(accuracies_valid)
        if accuracies_valid
        else float("nan")
    )

    ranked_acc = [
        (class_labels[i] if i < len(class_labels) else str(i), accuracies[i])
        for i in range(num_classes)
        if not math.isnan(accuracies[i])
    ]
    ranked_acc.sort(key=lambda x: x[1])
    weakest = [{"label": l, "accuracy": round(a, 4)} for l, a in ranked_acc[:top_k]]
    strongest = [
        {"label": l, "accuracy": round(a, 4)} for l, a in reversed(ranked_acc[-top_k:])
    ]

    out: dict[str, Any] = {
        "per_label_accuracy": [
            round(a, 4) if not math.isnan(a) else None for a in accuracies
        ],
        "accuracy_std": round(acc_std, 4) if not math.isnan(acc_std) else None,
        "accuracy_cv": round(acc_cv, 4) if not math.isnan(acc_cv) else None,
        "accuracy_gini": round(acc_gini, 4) if not math.isnan(acc_gini) else None,
        "accuracy_min_max_gap": round(min_max_gap, 4)
        if not math.isnan(min_max_gap)
        else None,
        "weakest_labels": weakest,
        "strongest_labels": strongest,
    }

    if thinking_valid:
        te_std, te_cv, te_gini = std_cv_gini(thinking_valid)
        out["thinking_effort_std"] = round(te_std, 4)
        out["thinking_effort_cv"] = round(te_cv, 4) if not math.isnan(te_cv) else None
        out["thinking_effort_gini"] = (
            round(te_gini, 4) if not math.isnan(te_gini) else None
        )
    if first_tick_valid:
        ft_std, ft_cv, ft_gini = std_cv_gini(first_tick_valid)
        out["first_correct_tick_std"] = round(ft_std, 4)
        out["first_correct_tick_cv"] = (
            round(ft_cv, 4) if not math.isnan(ft_cv) else None
        )
        out["first_correct_tick_gini"] = (
            round(ft_gini, 4) if not math.isnan(ft_gini) else None
        )
    if confidence_means and any(not math.isnan(c) for c in confidence_means):
        conf_valid = [c for c in confidence_means if not math.isnan(c)]
        c_std, c_cv, c_gini = std_cv_gini(conf_valid)
        out["confidence_mean_std"] = round(c_std, 4)
        out["confidence_mean_cv"] = round(c_cv, 4) if not math.isnan(c_cv) else None
        out["confidence_mean_gini"] = (
            round(c_gini, 4) if not math.isnan(c_gini) else None
        )

    return out
