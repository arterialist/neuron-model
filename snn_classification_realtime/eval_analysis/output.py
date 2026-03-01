"""Write metrics to JSON and Markdown."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def write_json(metrics: dict[str, Any], path: str) -> None:
    def default(o: Any) -> Any:
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=default)


def write_markdown(
    metrics: dict[str, Any],
    path: str,
    class_labels: list[str],
) -> None:
    lines: list[str] = []
    meta = metrics.get("metadata", {}) or {}
    lines.append("# Evaluation Metrics Report")
    lines.append("")
    lines.append(f"- **Dataset**: {meta.get('dataset_name', 'N/A')}")
    lines.append(f"- **Model**: {meta.get('snn_model_path', 'N/A')}")
    lines.append(f"- **Timestamp**: {meta.get('timestamp', 'N/A')}")
    lines.append(f"- **Samples**: {meta.get('eval_samples', 'N/A')}")
    lines.append("")

    acc = metrics.get("accuracy", {})
    if acc:
        lines.append("## Accuracy")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in acc.items():
            if v is not None:
                lines.append(f"| {k} | {v} |")
        lines.append("")

    per_label = metrics.get("per_label", {})
    if per_label:
        lines.append("## Per-Label Statistics")
        lines.append("")
        lines.append(
            "| Label | Accuracy | Conf Mean | First Tick Mean | Thinking Fraction |"
        )
        lines.append(
            "|-------|----------|-----------|-----------------|-------------------|"
        )
        for name in (
            class_labels
            if class_labels
            else sorted(
                per_label.keys(), key=lambda x: int(x) if str(x).isdigit() else x
            )
        ):
            if name not in per_label:
                continue
            pl = per_label[name]
            acc_mean = pl.get("accuracy", {}).get("mean", "")
            conf_mean = pl.get("confidence", {}).get("mean", "")
            tick_mean = pl.get("timing", {}).get("mean_first_correct_tick", "")
            think_frac = pl.get("thinking", {}).get("fraction_using_thinking", "")
            lines.append(
                f"| {name} | {acc_mean} | {conf_mean} | {tick_mean} | {think_frac} |"
            )
        lines.append("")

    cm = metrics.get("confusion_matrix", {})
    if cm and class_labels:
        lines.append("## Confusion Matrix (normalized)")
        lines.append("")
        header = "| | " + " | ".join(class_labels) + " |"
        sep = "|" + "---|" * (len(class_labels) + 1)
        lines.append(header)
        lines.append(sep)
        norm = cm.get("normalized", [])
        for i, row in enumerate(norm):
            label = class_labels[i] if i < len(class_labels) else str(i)
            cells = [f"{v:.2f}" if isinstance(v, (int, float)) else str(v) for v in row]
            lines.append("| " + label + " | " + " | ".join(cells) + " |")
        lines.append("")
        top_errors = cm.get("top_errors", [])[:10]
        if top_errors:
            lines.append("### Top misclassification pairs")
            lines.append("")
            lines.append("| Actual → Predicted | Count | Rate |")
            lines.append("|--------------------|-------|------|")
            for e in top_errors:
                lines.append(
                    f"| {e['actual_label']} → {e['predicted_label']} | {e['count']} | {e['rate']:.2%} |"
                )
        lines.append("")

    ch = metrics.get("concept_hierarchy", {})
    if ch:
        lines.append("## Concept Hierarchy")
        lines.append("")
        similar = ch.get("most_similar_pairs", [])[:10]
        if similar:
            lines.append("### Most similar class pairs (by attractor distance)")
            lines.append("")
            lines.append("| Label A | Label B | Distance |")
            lines.append("|---------|---------|----------|")
            for p in similar:
                lines.append(
                    f"| {p['label_a']} | {p['label_b']} | {p['distance']:.4f} |"
                )
        lines.append("")

    stab = metrics.get("stability", {})
    if stab:
        lines.append("## Stability")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in stab.items():
            if v is not None:
                lines.append(f"| {k} | {v} |")
        lines.append("")

    spec = metrics.get("specialization", {})
    if spec:
        lines.append("## Specialization (Generalist vs Specialist)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key in [
            "accuracy_std",
            "accuracy_cv",
            "accuracy_gini",
            "accuracy_min_max_gap",
        ]:
            if key in spec and spec[key] is not None:
                lines.append(f"| {key} | {spec[key]} |")
        lines.append("")
        weakest = spec.get("weakest_labels", [])[:5]
        strongest = spec.get("strongest_labels", [])[:5]
        if weakest:
            lines.append("### Weakest labels")
            lines.append("")
            for w in weakest:
                lines.append(f"- **{w['label']}**: {w['accuracy']}%")
            lines.append("")
        if strongest:
            lines.append("### Strongest labels")
            lines.append("")
            for s in strongest:
                lines.append(f"- **{s['label']}**: {s['accuracy']}%")
            lines.append("")

    cal = metrics.get("confidence_calibration", {})
    if cal:
        lines.append("## Confidence calibration")
        lines.append("")
        lines.append(
            f"- Mean confidence (correct): {cal.get('global_mean_confidence_correct')}"
        )
        lines.append(
            f"- Mean confidence (incorrect): {cal.get('global_mean_confidence_incorrect')}"
        )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
