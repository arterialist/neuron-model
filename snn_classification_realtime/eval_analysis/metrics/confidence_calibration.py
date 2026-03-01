"""Confidence calibration (bins, per-label entropy)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from snn_classification_realtime.eval_analysis.utils import safe_float


def compute_confidence_calibration(
    results: list[dict[str, Any]],
    num_classes: int,
    class_labels: list[str],
    num_bins: int = 10,
) -> dict[str, Any]:
    conf_correct = [
        safe_float(r.get("confidence")) for r in results if r.get("correct", False)
    ]
    conf_incorrect = [
        safe_float(r.get("confidence")) for r in results if not r.get("correct", True)
    ]
    conf_correct = [c for c in conf_correct if not math.isnan(c)]
    conf_incorrect = [c for c in conf_incorrect if not math.isnan(c)]

    bins: list[dict[str, Any]] = []
    for i in range(num_bins):
        lo = i / num_bins
        hi = (i + 1) / num_bins
        in_bin = [
            r
            for r in results
            if r.get("predicted_label") is not None
            and lo <= safe_float(r.get("confidence")) < hi
        ]
        if i == num_bins - 1:
            in_bin = [
                r
                for r in results
                if r.get("predicted_label") is not None
                and lo <= safe_float(r.get("confidence")) <= 1.0
            ]
        if not in_bin:
            bins.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "accuracy": None})
            continue
        acc = 100.0 * sum(1 for r in in_bin if r.get("correct", False)) / len(in_bin)
        bins.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": len(in_bin),
                "accuracy": round(acc, 4),
            }
        )

    by_label: dict[str, dict[str, Any]] = {}
    for label in range(num_classes):
        rows = [r for r in results if r.get("actual_label") == label]
        if not rows:
            continue
        name = class_labels[label] if label < len(class_labels) else str(label)
        correct_rows = [r for r in rows if r.get("correct", False)]
        mean_conf_correct = (
            float(np.mean([safe_float(r.get("confidence")) for r in correct_rows]))
            if correct_rows
            else None
        )
        mean_conf_incorrect = (
            float(
                np.mean(
                    [
                        safe_float(r.get("confidence"))
                        for r in rows
                        if not r.get("correct", True)
                    ]
                )
            )
            if rows != correct_rows
            else None
        )
        entropies = []
        for r in rows:
            p = [
                safe_float(r.get("confidence")),
                safe_float(r.get("second_confidence")),
                safe_float(r.get("third_confidence")),
            ]
            p = [x for x in p if not math.isnan(x) and x > 0]
            if p:
                s = sum(p)
                p = [x / s for x in p]
                ent = -sum(x * math.log(x) for x in p if x > 0)
                entropies.append(ent)
        mean_entropy = float(np.mean(entropies)) if entropies else None
        std_entropy = float(np.std(entropies)) if len(entropies) > 1 else None
        by_label[name] = {
            "mean_confidence_correct": round(mean_conf_correct, 4)
            if mean_conf_correct is not None
            else None,
            "mean_confidence_incorrect": round(mean_conf_incorrect, 4)
            if mean_conf_incorrect is not None
            else None,
            "confidence_entropy_mean": round(mean_entropy, 4)
            if mean_entropy is not None
            else None,
            "confidence_entropy_std": round(std_entropy, 4)
            if std_entropy is not None
            else None,
        }

    return {
        "global_mean_confidence_correct": round(float(np.mean(conf_correct)), 4)
        if conf_correct
        else None,
        "global_mean_confidence_incorrect": round(float(np.mean(conf_incorrect)), 4)
        if conf_incorrect
        else None,
        "bins": bins,
        "by_label": by_label,
    }
