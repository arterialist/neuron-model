import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple
import math

import numpy as np
from tqdm import tqdm

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        Axes3D = None  # type: ignore
except Exception:
    plt = None
    Axes3D = None  # type: ignore

PLOT_TYPES = [
    "firing_rate_per_layer",
    "firing_rate_per_layer_3d",
    "avg_S_per_layer_per_label",
    "avg_S_per_layer_per_label_3d",
    "firings_time_series",
    "firings_time_series_3d",
    "avg_S_time_series",
    "avg_S_time_series_3d",
    "total_fired_cumulative",
    "total_fired_cumulative_3d",
    "network_state_progression",
]


def prompt_plot_type() -> str:
    print("Select plot type:")
    for i, name in enumerate(PLOT_TYPES, start=1):
        print(f"  {i}) {name}")
    resp = input(f"Enter choice [1]: ").strip()
    if resp == "":
        return PLOT_TYPES[0]
    try:
        idx = int(resp)
        if 1 <= idx <= len(PLOT_TYPES):
            return PLOT_TYPES[idx - 1]
    except ValueError:
        pass
    print("Invalid choice; defaulting to option 1.")
    return PLOT_TYPES[0]


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        payload = json.load(f)
    # Support both top-level array and {"records": [...]} formats
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported dataset JSON format")


def group_by_image(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Group by (label, image_index) to collect per-tick records for each image."""
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    labels_present = any(("label" in r) for r in records)
    for rec in records:
        # Some datasets may be unsupervised and not include label
        label = rec.get("label", -1)
        if label == -1 and not labels_present:
            label = 0
        img_idx = rec.get("image_index", -1)
        key = (int(label), int(img_idx))
        buckets.setdefault(key, []).append(rec)
    # Sort per-image records by tick for stability
    for key in buckets:
        buckets[key].sort(key=lambda r: r.get("tick", 0))
    return buckets


def compute_layer_firing_rate_for_image(
    image_records: List[Dict[str, Any]],
) -> List[float]:
    """Compute per-layer firing rate for a given image across all ticks.

    Firing rate per layer = total number of firing events in that layer across all ticks
    divided by (num_neurons_in_layer * num_ticks).
    """
    if not image_records:
        return []
    # Assume layers structure is constant across ticks
    num_layers = len(image_records[0]["layers"])
    num_ticks = len(image_records)
    layer_rates: List[float] = []
    for li in range(num_layers):
        total_fired = 0
        total_neurons = 0
        for rec in image_records:
            layer = rec["layers"][li]
            fired = layer.get("fired", [])
            total_fired += int(np.sum(np.array(fired, dtype=np.int32)))
            if total_neurons == 0:
                total_neurons = len(fired)
        denom = max(1, total_neurons * num_ticks)
        layer_rates.append(total_fired / denom)
    return layer_rates


def plot_firing_rate_per_layer(
    out_dir: str,
    lbl: int,
    layer_rates: List[float],
    title_prefix: str = "",
) -> str:
    if plt is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(layer_rates))
    ax.bar(x, layer_rates, color="#4C78A8")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Firing rate")
    title = f"{title_prefix} Firing rate per layer | label={lbl}"
    ax.set_title(title)
    ax.set_xticks(x)
    _rotate_xtick_labels(ax)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.set_ylim(0.0, max(0.05, max(layer_rates) * 1.2 if layer_rates else 0.05))
    for xi, val in enumerate(layer_rates):
        ax.text(xi, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, f"firing_rate_layer_label_{lbl}", fig, dpi=200, tight=False
    )
    plt.close(fig)
    return out_path


def plot_firing_rate_per_layer_3d(
    out_dir: str, labels: List[int], layers: List[int], values: np.ndarray
) -> str:
    if plt is None or Axes3D is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[attr-defined]
    K = len(labels)
    L = len(layers)
    xpos, ypos = np.meshgrid(np.arange(L), np.arange(K))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos, dtype=float)
    dx = 0.8 * np.ones_like(zpos)
    dy = 0.8 * np.ones_like(zpos)
    dz = values.reshape(K * L)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)  # type: ignore[attr-defined]
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Label")
    ax.set_zlabel("Avg firing rate")  # type: ignore[attr-defined]
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels([str(l) for l in labels])
    ax.set_title("Average firing rate per layer per label (3D)")
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, "firing_rate_per_layer_3d", fig, dpi=200, tight=False
    )
    plt.close(fig)
    return out_path


def compute_avg_S_per_layer_per_label(
    records: List[Dict[str, Any]],
) -> Tuple[List[int], List[int], np.ndarray]:
    """Compute average membrane potential S per layer per label across all ticks and images.

    Returns: (labels_sorted, layer_indices, matrix[label_index, layer_index])
    """
    from collections import defaultdict

    sum_by_label_layer: Dict[int, Dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    count_by_label_layer: Dict[int, Dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    labels_set = set()
    max_layer_index = -1

    labels_present = any(("label" in r) for r in records)

    for rec in records:
        if "label" not in rec:
            if not labels_present:
                label = 0  # single-group for unsupervised datasets
            else:
                continue
        else:
            label = int(rec["label"]) if labels_present else 0
        labels_set.add(label)
        layers = rec.get("layers", [])
        for li, layer in enumerate(layers):
            S = layer.get("S", [])
            if not S:
                continue
            sum_by_label_layer[label][li] += float(
                np.sum(np.array(S, dtype=np.float32))
            )
            count_by_label_layer[label][li] += int(len(S))
            if li > max_layer_index:
                max_layer_index = li

    labels_sorted = sorted(labels_set)
    layer_indices = list(range(max_layer_index + 1))
    if not labels_sorted or not layer_indices:
        return labels_sorted, layer_indices, np.zeros((0, 0), dtype=np.float32)

    mat = np.zeros((len(labels_sorted), len(layer_indices)), dtype=np.float32)
    for li_idx, li in enumerate(layer_indices):
        for lbl_idx, lbl in enumerate(labels_sorted):
            s = sum_by_label_layer[lbl].get(li, 0.0)
            c = count_by_label_layer[lbl].get(li, 0)
            mat[lbl_idx, li_idx] = (s / c) if c > 0 else 0.0
    return labels_sorted, layer_indices, mat


def plot_avg_S_per_layer_per_label(
    out_dir: str, labels: List[int], layers: List[int], values: np.ndarray
) -> str:
    if plt is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(max(10.0, _compute_fig_width_for_ticks(len(time_axis), 12.0, 0.6)), 5)
    )
    num_labels = len(labels)
    num_layers = len(layers)
    x = np.arange(num_layers)
    width = 0.8 / max(1, num_labels)
    cmap = plt.get_cmap("tab10") if hasattr(plt, "get_cmap") else None
    colors = None
    if cmap is not None:
        try:
            # Sample N distinct colors from colormap
            colors = [cmap(i / max(1, num_labels - 1)) for i in range(num_labels)]
        except Exception:
            colors = None
    for i, lbl in enumerate(labels):
        offsets = x - 0.4 + width / 2 + i * width
        color = colors[i % len(colors)] if colors else None
        ax.bar(offsets, values[i, :], width=width, label=str(lbl), color=color)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Average membrane potential S")
    ax.set_title("Average S per layer per label")
    ax.set_xticks(x)
    ax.legend(
        title="Label",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        ncol=1,
        frameon=False,
    )
    ymax = float(np.max(values)) if values.size > 0 else 1.0
    ax.set_ylim(0.0, max(0.05, ymax * 1.2))
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, "avg_S_per_layer_per_label", fig, dpi=200, tight=True
    )
    plt.close(fig)
    return out_path


def plot_avg_S_per_layer_per_label_3d(
    out_dir: str, labels: List[int], layers: List[int], values: np.ndarray
) -> str:
    if plt is None or Axes3D is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[attr-defined]
    K = len(labels)
    L = len(layers)
    xpos, ypos = np.meshgrid(np.arange(L), np.arange(K))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos, dtype=float)
    dx = 0.8 * np.ones_like(zpos)
    dy = 0.8 * np.ones_like(zpos)
    dz = values.reshape(K * L)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)  # type: ignore[attr-defined]
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Label")
    ax.set_zlabel("Average S")  # type: ignore[attr-defined]
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels([str(l) for l in labels])
    ax.set_title("Average S per layer per label (3D)")
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, "avg_S_per_layer_per_label_3d", fig, dpi=200, tight=False
    )
    plt.close(fig)
    return out_path


def compute_firings_time_series_per_label(
    records: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """Compute average fraction of neurons firing per layer over time, averaged across images for each label.

    Returns mapping: label -> { 'layers': List[int], 'time': List[int], 'values': np.ndarray[num_layers, T] }
    where values[li, t] is the average fraction firing in layer li at tick index t (relative within image).
    """
    buckets = group_by_image(records)
    # Organize by label
    label_to_images: Dict[int, List[List[Dict[str, Any]]]] = {}
    labels_present = any(("label" in r) for r in records)
    for (label, img_idx), recs in buckets.items():
        lbl = 0 if (label == -1 and not labels_present) else label
        if lbl == -1:
            continue
        label_to_images.setdefault(lbl, []).append(recs)

    result: Dict[int, Dict[str, Any]] = {}
    for label, images in label_to_images.items():
        if not images:
            continue
        # Determine minimal aligned length across images
        min_T = min(len(recs) for recs in images)
        if min_T == 0:
            continue
        # Determine number of layers from first record
        first_layers = images[0][0].get("layers", [])
        num_layers = len(first_layers)
        sums = np.zeros((num_layers, min_T), dtype=np.float32)
        for recs in images:
            # Align by relative tick within image
            for t in range(min_T):
                layers = recs[t].get("layers", [])
                for li in range(num_layers):
                    fired = layers[li].get("fired", []) if li < len(layers) else []
                    n = len(fired)
                    frac = (
                        (float(np.sum(np.array(fired, dtype=np.int32))) / n)
                        if n > 0
                        else 0.0
                    )
                    sums[li, t] += frac
        # Average across images
        sums /= max(1, len(images))
        result[label] = {
            "layers": list(range(num_layers)),
            "time": list(range(min_T)),
            "values": sums,
        }
    return result


def plot_firings_time_series_for_label(
    out_dir: str,
    label: int,
    layer_indices: List[int],
    time_axis: List[int],
    values: np.ndarray,
) -> str:
    if plt is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(max(10.0, _compute_fig_width_for_ticks(len(time_axis), 12.0, 0.6)), 5)
    )
    cmap = plt.get_cmap("tab20") if hasattr(plt, "get_cmap") else None
    num_layers = len(layer_indices)
    for i, li in enumerate(layer_indices):
        color = cmap(i / max(1, num_layers - 1)) if cmap is not None else None
        ax.plot(
            time_axis, values[i, :], label=f"layer {li}", color=color, linewidth=1.8
        )
        # Overlay horizontal bars via step-draw to emphasize per-tick values
        ax.plot(
            time_axis,
            values[i, :],
            color=color,
            linewidth=1.0,
            alpha=0.35,
            drawstyle="steps-mid",
        )
    ax.set_xlabel("Tick (relative within image)")
    ax.set_ylabel("Fraction firing")
    ax.set_title(f"Firings over time per layer | label={label}")
    ax.set_xlim(time_axis[0], time_axis[-1] if time_axis else 0)
    _rotate_xtick_labels(ax)
    _rotate_xtick_labels(ax)
    ax.set_ylim(0.0, 1.0)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        ncol=1,
        frameon=False,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    # Double axis precision: minor ticks and higher-precision labels
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, f"firings_time_series_label_{label}", fig, dpi=200, tight=True
    )
    plt.close(fig)
    return out_path


def plot_firings_time_series_for_label_3d(
    out_dir: str,
    label: int,
    layer_indices: List[int],
    time_axis: List[int],
    values: np.ndarray,
) -> str:
    if plt is None or Axes3D is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[attr-defined]
    X, Y = np.meshgrid(
        np.array(time_axis, dtype=float), np.array(layer_indices, dtype=float)
    )
    Z = values
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", antialiased=True)  # type: ignore[attr-defined]
    ax.set_xlabel("Tick")
    ax.set_ylabel("Layer index")
    ax.set_zlabel("Fraction firing")  # type: ignore[attr-defined]
    ax.set_title(f"Firings over time per layer (3D) | label={label}")
    fig.colorbar(surf, shrink=0.6, aspect=12)
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, f"firings_time_series_label_{label}_3d", fig, dpi=200, tight=False
    )
    plt.close(fig)
    return out_path


def plot_firings_time_series_all_labels_3d(
    out_dir: str,
    series: Dict[int, Dict[str, Any]],
) -> str:
    if plt is None or Axes3D is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[attr-defined]
    labels_sorted = sorted(series.keys())
    if not labels_sorted:
        return ""
    min_T = min(len(series[l]["time"]) for l in labels_sorted)
    num_layers = len(series[labels_sorted[0]]["layers"])
    gap = 1.0
    for idx, lbl in enumerate(labels_sorted):
        data = series[lbl]
        time_axis = np.array(data["time"][:min_T], dtype=float)
        layers_idx = np.array(data["layers"], dtype=float)
        offset = idx * (num_layers + gap)
        Y_base = layers_idx + offset
        X, Y = np.meshgrid(time_axis, Y_base)
        Z = data["values"][:, :min_T]
        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", antialiased=True)  # type: ignore[attr-defined]
    ax.set_xlabel("Tick")
    ax.set_ylabel("Layer index (stacked by label)")
    ax.set_zlabel("Fraction firing")  # type: ignore[attr-defined]
    ax.set_title("Firings over time per layer (3D) — all labels")
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, "firings_time_series_all_labels_3d", fig, dpi=200, tight=False
    )
    plt.close(fig)
    return out_path


def compute_avg_S_time_series_per_label(
    records: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """Compute average membrane potential S per layer over time, averaged across images for each label.

    Returns mapping: label -> { 'layers': List[int], 'time': List[int], 'values': np.ndarray[num_layers, T] }
    where values[li, t] is the average S in layer li at tick index t (relative within image).
    """
    buckets = group_by_image(records)
    label_to_images: Dict[int, List[List[Dict[str, Any]]]] = {}
    labels_present = any(("label" in r) for r in records)
    for (label, img_idx), recs in buckets.items():
        lbl = 0 if (label == -1 and not labels_present) else label
        if lbl == -1:
            continue
        label_to_images.setdefault(lbl, []).append(recs)

    result: Dict[int, Dict[str, Any]] = {}
    for label, images in label_to_images.items():
        if not images:
            continue
        min_T = min(len(recs) for recs in images)
        if min_T == 0:
            continue
        first_layers = images[0][0].get("layers", [])
        num_layers = len(first_layers)
        sums = np.zeros((num_layers, min_T), dtype=np.float32)
        counts = np.zeros((num_layers, min_T), dtype=np.int32)
        for recs in images:
            for t in range(min_T):
                layers = recs[t].get("layers", [])
                for li in range(num_layers):
                    S = layers[li].get("S", []) if li < len(layers) else []
                    if S:
                        arr = np.array(S, dtype=np.float32)
                        sums[li, t] += float(np.mean(arr))
                        counts[li, t] += 1
        # Average across images (per layer,tick)
        counts_safe = np.maximum(counts, 1)
        avg = sums / counts_safe
        result[label] = {
            "layers": list(range(num_layers)),
            "time": list(range(min_T)),
            "values": avg,
        }
    return result


def plot_avg_S_time_series_for_label(
    out_dir: str,
    label: int,
    layer_indices: List[int],
    time_axis: List[int],
    values: np.ndarray,
) -> str:
    if plt is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("viridis") if hasattr(plt, "get_cmap") else None
    num_layers = len(layer_indices)
    for i, li in enumerate(layer_indices):
        color = cmap(i / max(1, num_layers - 1)) if cmap is not None else None
        ax.plot(
            time_axis, values[i, :], label=f"layer {li}", color=color, linewidth=1.8
        )
        ax.plot(
            time_axis,
            values[i, :],
            color=color,
            linewidth=1.0,
            alpha=0.35,
            drawstyle="steps-mid",
        )
    ax.set_xlabel("Tick (relative within image)")
    ax.set_ylabel("Average S")
    ax.set_title(f"Average S over time per layer | label={label}")
    ax.set_xlim(time_axis[0], time_axis[-1] if time_axis else 0)
    _rotate_xtick_labels(ax)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        ncol=1,
        frameon=False,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, f"avg_S_time_series_label_{label}", fig, dpi=200, tight=True
    )
    plt.close(fig)
    return out_path


def plot_avg_S_time_series_for_label_3d(
    out_dir: str,
    label: int,
    layer_indices: List[int],
    time_axis: List[int],
    values: np.ndarray,
) -> str:
    if plt is None or Axes3D is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[attr-defined]
    X, Y = np.meshgrid(
        np.array(time_axis, dtype=float), np.array(layer_indices, dtype=float)
    )
    Z = values
    surf = ax.plot_surface(X, Y, Z, cmap="plasma", edgecolor="none", antialiased=True)  # type: ignore[attr-defined]
    ax.set_xlabel("Tick")
    ax.set_ylabel("Layer index")
    ax.set_zlabel("Average S")  # type: ignore[attr-defined]
    ax.set_title(f"Average S over time per layer (3D) | label={label}")
    fig.colorbar(surf, shrink=0.6, aspect=12)
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, f"avg_S_time_series_label_{label}_3d", fig, dpi=200, tight=False
    )
    plt.close(fig)
    return out_path


def plot_avg_S_time_series_all_labels_3d(
    out_dir: str,
    series: Dict[int, Dict[str, Any]],
) -> str:
    if plt is None or Axes3D is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[attr-defined]
    labels_sorted = sorted(series.keys())
    if not labels_sorted:
        return ""
    min_T = min(len(series[l]["time"]) for l in labels_sorted)
    num_layers = len(series[labels_sorted[0]]["layers"])
    gap = 1.0
    for idx, lbl in enumerate(labels_sorted):
        data = series[lbl]
        time_axis = np.array(data["time"][:min_T], dtype=float)
        layers_idx = np.array(data["layers"], dtype=float)
        offset = idx * (num_layers + gap)
        Y_base = layers_idx + offset
        X, Y = np.meshgrid(time_axis, Y_base)
        Z = data["values"][:, :min_T]
        ax.plot_surface(X, Y, Z, cmap="plasma", edgecolor="none", antialiased=True)  # type: ignore[attr-defined]
    ax.set_xlabel("Tick")
    ax.set_ylabel("Layer index (stacked by label)")
    ax.set_zlabel("Average S")  # type: ignore[attr-defined]
    ax.set_title("Average S over time per layer (3D) — all labels")
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, "avg_S_time_series_all_labels_3d", fig, dpi=200, tight=False
    )
    plt.close(fig)
    return out_path


def compute_total_fired_cumulative_per_label(
    records: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """Compute cumulative total fired neurons per tick per layer, averaged across images for each label.

    Uses record["cumulative_fires"] if present; otherwise cumulates record["layers"][li]["fired"].

    Returns mapping: label -> { 'layers': List[int], 'time': List[int], 'values': np.ndarray[num_layers, T] }
    where values[li, t] is the average across images of total fires from tick 0..t in that layer.
    """
    buckets = group_by_image(records)
    label_to_images: Dict[int, List[List[Dict[str, Any]]]] = {}
    labels_present = any(("label" in r) for r in records)
    for (label, img_idx), recs in buckets.items():
        lbl = 0 if (label == -1 and not labels_present) else label
        if lbl == -1:
            continue
        label_to_images.setdefault(lbl, []).append(recs)

    result: Dict[int, Dict[str, Any]] = {}
    for label, images in label_to_images.items():
        if not images:
            continue
        min_T = min(len(recs) for recs in images)
        if min_T == 0:
            continue
        first_layers = images[0][0].get("layers", [])
        num_layers = len(first_layers)
        sums = np.zeros((num_layers, min_T), dtype=np.float32)
        for recs in images:
            # per-image cumulative arrays
            per_img = np.zeros((num_layers, min_T), dtype=np.float32)
            for t in range(min_T):
                rec = recs[t]
                # Prefer cumulative_fires if present
                if "cumulative_fires" in rec:
                    cf = rec.get("cumulative_fires", [])
                    for li in range(num_layers):
                        per_layer = cf[li] if li < len(cf) else []
                        per_img[li, t] = float(
                            np.sum(np.array(per_layer, dtype=np.int32))
                        )
                else:
                    layers = rec.get("layers", [])
                    for li in range(num_layers):
                        fired = layers[li].get("fired", []) if li < len(layers) else []
                        per_img[li, t] = float(np.sum(np.array(fired, dtype=np.int32)))
                if t > 0 and "cumulative_fires" not in rec:
                    # Convert to cumulative if we built from instant fired
                    per_img[:, t] += per_img[:, t - 1]
            sums += per_img
        avg = sums / max(1, len(images))
        result[label] = {
            "layers": list(range(num_layers)),
            "time": list(range(min_T)),
            "values": avg,
        }
    return result


def plot_total_fired_cumulative_for_label(
    out_dir: str,
    label: int,
    layer_indices: List[int],
    time_axis: List[int],
    values: np.ndarray,
) -> str:
    if plt is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("plasma") if hasattr(plt, "get_cmap") else None
    num_layers = len(layer_indices)
    for i, li in enumerate(layer_indices):
        color = cmap(i / max(1, num_layers - 1)) if cmap is not None else None
        ax.plot(
            time_axis, values[i, :], label=f"layer {li}", color=color, linewidth=1.8
        )
        ax.plot(
            time_axis,
            values[i, :],
            color=color,
            linewidth=1.0,
            alpha=0.35,
            drawstyle="steps-mid",
        )
    ax.set_xlabel("Tick (relative within image)")
    ax.set_ylabel("Total fired (cumulative)")
    ax.set_title(f"Total fired neurons (cumulative) per layer | label={label}")
    ax.set_xlim(time_axis[0], time_axis[-1] if time_axis else 0)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        ncol=1,
        frameon=False,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, f"total_fired_cumulative_label_{label}", fig, dpi=200, tight=True
    )
    plt.close(fig)
    return out_path


def plot_total_fired_cumulative_for_label_3d(
    out_dir: str,
    label: int,
    layer_indices: List[int],
    time_axis: List[int],
    values: np.ndarray,
) -> str:
    if plt is None or Axes3D is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[attr-defined]
    X, Y = np.meshgrid(
        np.array(time_axis, dtype=float), np.array(layer_indices, dtype=float)
    )
    Z = values
    surf = ax.plot_surface(X, Y, Z, cmap="inferno", edgecolor="none", antialiased=True)  # type: ignore[attr-defined]
    ax.set_xlabel("Tick")
    ax.set_ylabel("Layer index")
    ax.set_zlabel("Total fired (cumulative)")  # type: ignore[attr-defined]
    ax.set_title(f"Total fired cumulative per layer (3D) | label={label}")
    fig.colorbar(surf, shrink=0.6, aspect=12)
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir,
        f"total_fired_cumulative_label_{label}_3d",
        fig,
        dpi=200,
        tight=False,
    )
    plt.close(fig)
    return out_path


def plot_total_fired_cumulative_all_labels_3d(
    out_dir: str,
    series: Dict[int, Dict[str, Any]],
) -> str:
    if plt is None or Axes3D is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[attr-defined]
    labels_sorted = sorted(series.keys())
    if not labels_sorted:
        return ""
    min_T = min(len(series[l]["time"]) for l in labels_sorted)
    num_layers = len(series[labels_sorted[0]]["layers"])
    gap = 1.0
    for idx, lbl in enumerate(labels_sorted):
        data = series[lbl]
        time_axis = np.array(data["time"][:min_T], dtype=float)
        layers_idx = np.array(data["layers"], dtype=float)
        offset = idx * (num_layers + gap)
        Y_base = layers_idx + offset
        X, Y = np.meshgrid(time_axis, Y_base)
        Z = data["values"][:, :min_T]
        ax.plot_surface(X, Y, Z, cmap="inferno", edgecolor="none", antialiased=True)  # type: ignore[attr-defined]
    ax.set_xlabel("Tick")
    ax.set_ylabel("Layer index (stacked by label)")
    ax.set_zlabel("Total fired (cumulative)")  # type: ignore[attr-defined]
    ax.set_title("Total fired cumulative per layer (3D) — all labels")
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, "total_fired_cumulative_all_labels_3d", fig, dpi=200, tight=False
    )
    plt.close(fig)
    return out_path


def _compute_grid(n: int) -> Tuple[int, int]:
    cols = int(math.ceil(math.sqrt(max(1, n))))
    rows = int(math.ceil(n / max(1, cols)))
    return rows, cols


def _save_figure_by_type(
    out_dir: str, base_name: str, fig, dpi: int = 200, tight: bool = False
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for ext in ["png", "svg", "pdf"]:
        subdir = os.path.join(out_dir, ext)
        os.makedirs(subdir, exist_ok=True)
        target_path = os.path.join(subdir, f"{base_name}.{ext}")
        if tight:
            if ext == "png":
                fig.savefig(target_path, dpi=dpi, bbox_inches="tight")
            else:
                fig.savefig(target_path, bbox_inches="tight")
        else:
            if ext == "png":
                fig.savefig(target_path, dpi=dpi)
            else:
                fig.savefig(target_path)
        paths[ext] = target_path
    return paths["png"]


def _rotate_xtick_labels(ax, rotation: float = 45.0) -> None:
    """Rotate x-axis tick labels diagonally to reduce overlap.

    Uses right alignment and anchor rotation mode for better spacing.
    Safe for both 2D and 3D axes.
    """
    try:
        labels = ax.get_xticklabels()
        for lbl in labels:
            lbl.set_rotation(rotation)
            lbl.set_rotation_mode("anchor")
            lbl.set_ha("right")
    except Exception:
        # Be permissive; some backends/axes types may not support this cleanly
        pass


def _compute_fig_width_for_ticks(
    num_ticks: int, min_width: float = 12.0, per_tick: float = 0.6
) -> float:
    """Return a figure width that scales with the number of ticks.

    - min_width: lower bound to keep plots readable for small tick counts
    - per_tick: horizontal inches allocated per tick
    No upper bound is applied to respect the user's request for ample canvas space.
    """
    n = max(1, int(num_ticks))
    return max(min_width, n * per_tick)


def plot_network_state_progression_for_image(
    out_dir: str,
    label: int,
    image_index: int,
    image_records: List[Dict[str, Any]],
) -> str:
    """Plot entire network state progression per neuron over time for one image.

    X-axis: ticks (time)
    Y-axis: membrane potential (S)

    At each tick, neurons are plotted as circles. For that tick:
    - Layers are arranged left→right by small x-offsets relative to the tick
    - Within each layer, all neurons share the same x (vertical line)
    - Circle color encodes S: blue→orange (min→max), overridden to red if fired
    """
    if plt is None:
        return ""
    if not image_records:
        return ""

    # Determine structure
    num_ticks = len(image_records)
    first_layers = image_records[0].get("layers", [])
    num_layers = len(first_layers)
    if num_layers == 0:
        return ""

    # Compute global S range for color normalization
    S_min = float("inf")
    S_max = float("-inf")
    for rec in image_records:
        layers = rec.get("layers", [])
        for layer in layers:
            S_list = layer.get("S", [])
            if S_list:
                arr = np.array(S_list, dtype=np.float32)
                if arr.size:
                    S_min = min(S_min, float(np.min(arr)))
                    S_max = max(S_max, float(np.max(arr)))
    if not np.isfinite(S_min) or not np.isfinite(S_max):
        S_min, S_max = 0.0, 1.0
    if S_max <= S_min:
        S_max = S_min + 1e-6

    # Colormap: blue (#3b82f6) to orange (#f59e0b)
    try:
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        cmap = LinearSegmentedColormap.from_list(
            "blue_orange", ["#3b82f6", "#f59e0b"], N=256
        )
        norm = Normalize(vmin=S_min, vmax=S_max)
    except Exception:
        cmap = None
        norm = None

    # Figure sizing: widen with number of ticks (no upper bound)
    width = _compute_fig_width_for_ticks(num_ticks, min_width=12.0, per_tick=0.6)
    fig, ax = plt.subplots(figsize=(width, 6.0))

    # Precompute per-layer horizontal offsets per tick
    # Layers arranged left→right within each integer tick position, leaving clear gaps between ticks
    # Reduce cluster width so groups at tick t and t+1 do not touch
    cluster_halfwidth = 0.2
    layer_offsets = (
        np.linspace(-cluster_halfwidth, cluster_halfwidth, num_layers, endpoint=True)
        if num_layers > 1
        else np.array([0.0])
    )

    # Draw all points
    default_size = 12.0
    fired_size = default_size * 1.10
    for t, rec in enumerate(image_records):
        layers = rec.get("layers", [])
        for li in range(num_layers):
            layer = layers[li] if li < len(layers) else {}
            S_list = layer.get("S", [])
            fired_list = layer.get("fired", [])
            if not S_list:
                continue
            y_vals = np.array(S_list, dtype=np.float32)
            # Previous-tick S for positioning fired neurons
            prev_y_vals = y_vals
            if t > 0:
                prev_layers = image_records[t - 1].get("layers", [])
                if li < len(prev_layers):
                    prev_S_list = prev_layers[li].get("S", [])
                    if prev_S_list:
                        prev_arr = np.array(prev_S_list, dtype=np.float32)
                        if prev_arr.shape[0] == y_vals.shape[0]:
                            prev_y_vals = prev_arr
                        else:
                            prev_y_vals = np.array(y_vals, copy=True)
                            c = min(prev_arr.shape[0], y_vals.shape[0])
                            if c > 0:
                                prev_y_vals[:c] = prev_arr[:c]

            # Fired mask matching y length
            if fired_list:
                fired_arr_raw = np.array(fired_list, dtype=np.int32)
                if fired_arr_raw.shape[0] == y_vals.shape[0]:
                    fired_mask = fired_arr_raw > 0
                else:
                    fired_mask = np.zeros_like(y_vals, dtype=bool)
                    c = min(fired_arr_raw.shape[0], y_vals.shape[0])
                    if c > 0:
                        fired_mask[:c] = fired_arr_raw[:c] > 0
            else:
                fired_mask = np.zeros_like(y_vals, dtype=bool)

            # Use previous S for y-position when fired, keep highlight
            y_plot = np.where(fired_mask, prev_y_vals, y_vals)

            x_val = float(t + layer_offsets[li])
            x_vals = np.full_like(y_plot, x_val, dtype=float)

            # Base colors by S, override to red if fired
            if cmap is not None and norm is not None:
                colors = cmap(norm(y_plot))
            else:
                # Fallback to grayscale
                colors = np.tile(np.array([0.3, 0.3, 0.3, 0.9]), (y_plot.size, 1))
            if np.any(fired_mask):
                colors[fired_mask] = np.array([0.90, 0.10, 0.10, 1.0])
                sizes = np.where(fired_mask, fired_size, default_size)
            else:
                sizes = np.full(y_plot.shape, default_size, dtype=float)

            ax.scatter(
                x_vals,
                y_plot,
                s=sizes,
                c=colors,
                marker="o",
                linewidths=0.0,
                edgecolors="none",
                alpha=0.9,
                zorder=2 + li,
            )

    # Axes, ticks, labels
    ax.set_xlabel("Tick")
    ax.set_ylabel("Membrane potential S")
    ax.set_title(f"Network state progression | label={label} image={image_index}")
    ax.set_xlim(-0.6, num_ticks - 1 + 0.6)
    # Show integer ticks on x-axis
    ax.set_xticks(list(range(num_ticks)))
    _rotate_xtick_labels(ax)
    # Align lines to bottom by ensuring baseline shows; include 0 if within range
    y_lower = min(0.0, S_min)
    y_upper = S_max + 0.05 * max(1.0, abs(S_max))
    ax.set_ylim(y_lower, y_upper)

    # Vertical separators between tick groups (between clusters at t and t+1)
    for t in range(max(0, num_ticks - 1)):
        xsep = t + 0.5
        ax.vlines(
            xsep,
            y_lower,
            y_upper,
            colors="#9ca3af",
            linestyles="-",
            linewidth=1.0,
            alpha=0.6,
            zorder=1,
        )
    ax.grid(True, which="both", linestyle="--", alpha=0.25)

    # Colorbar for S scale
    if cmap is not None and norm is not None:
        try:
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            cbar = fig.colorbar(mappable, ax=ax, pad=0.01)
            cbar.set_label("Membrane potential S (blue→orange)")
        except Exception:
            pass

    # Legend entry for fired
    try:
        import matplotlib.lines as mlines

        fired_proxy = mlines.Line2D(
            [],
            [],
            color="#e11d48",
            marker="o",
            linestyle="None",
            markersize=6,
            label="fired",
        )
        ax.legend(handles=[fired_proxy], loc="upper right", frameon=False)
    except Exception:
        pass

    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir,
        f"network_state_progression_label_{label}_image_{image_index}",
        fig,
        dpi=200,
        tight=True,
    )
    plt.close(fig)
    return out_path


def plot_firing_rate_per_layer_combined(
    out_dir: str, label_to_rates: Dict[int, List[float]], title_prefix: str = "Average"
) -> str:
    if plt is None:
        return ""
    labels_sorted = sorted(label_to_rates.keys())
    if not labels_sorted:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    rows, cols = _compute_grid(len(labels_sorted))
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(cols * 4.0, rows * 3.2),
        squeeze=False,
        sharex=False,
        sharey=True,
    )
    # Determine common y-limit
    y_max = 0.05
    for lbl in labels_sorted:
        rates = label_to_rates[lbl]
        if rates:
            y_max = max(y_max, max(rates) * 1.2)
    for i, lbl in enumerate(labels_sorted):
        r = i // cols
        c = i % cols
        ax = axs[r][c]
        rates = label_to_rates[lbl]
        x = np.arange(len(rates))
        ax.bar(x, rates, color="#4C78A8")
        ax.set_title(f"{title_prefix} firing rate | label={lbl}")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Firing rate")
        ax.set_ylim(0.0, y_max)
        ax.set_xticks(x)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        for xi, val in enumerate(rates):
            ax.text(xi, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    # Hide any unused axes
    total_axes = rows * cols
    for j in range(len(labels_sorted), total_axes):
        r = j // cols
        c = j % cols
        axs[r][c].axis("off")
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, "firing_rate_per_layer_all_labels", fig, dpi=200, tight=True
    )
    plt.close(fig)
    return out_path


def plot_firings_time_series_combined(
    out_dir: str, series: Dict[int, Dict[str, Any]]
) -> str:
    if plt is None:
        return ""
    labels_sorted = sorted(series.keys())
    if not labels_sorted:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    rows, cols = _compute_grid(len(labels_sorted))
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(cols * 5.0, rows * 3.6),
        squeeze=False,
        sharex=False,
        sharey=True,
    )
    handles = None
    labels_for_legend = None
    for i, lbl in enumerate(labels_sorted):
        data = series[lbl]
        r = i // cols
        c = i % cols
        ax = axs[r][c]
        layer_indices = data["layers"]
        time_axis = data["time"]
        values = data["values"]
        cmap = plt.get_cmap("tab20") if hasattr(plt, "get_cmap") else None
        num_layers = len(layer_indices)
        lines = []
        for j, li in enumerate(layer_indices):
            color = cmap(j / max(1, num_layers - 1)) if cmap is not None else None
            (line,) = ax.plot(
                time_axis, values[j, :], label=f"layer {li}", color=color, linewidth=1.6
            )
            lines.append(line)
        ax.set_title(f"Firings over time | label={lbl}")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Fraction firing")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)
        handles = lines
        labels_for_legend = [f"layer {li}" for li in layer_indices]
    # Hide any unused axes
    total_axes = rows * cols
    for j in range(len(labels_sorted), total_axes):
        r = j // cols
        c = j % cols
        axs[r][c].axis("off")
    if handles and labels_for_legend:
        fig.legend(
            handles,
            labels_for_legend,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            ncol=1,
            frameon=False,
            title="Layers",
        )
    # Rotate x labels for all small subplots (applied to the last used ax)
    _rotate_xtick_labels(ax)
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, "firings_time_series_all_labels", fig, dpi=200, tight=True
    )
    plt.close(fig)
    return out_path


def plot_avg_S_time_series_combined(
    out_dir: str, series: Dict[int, Dict[str, Any]]
) -> str:
    if plt is None:
        return ""
    labels_sorted = sorted(series.keys())
    if not labels_sorted:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    rows, cols = _compute_grid(len(labels_sorted))
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(cols * 5.0, rows * 3.6),
        squeeze=False,
        sharex=False,
        sharey=True,
    )
    # Determine common y-limit
    y_max = 0.05
    for lbl in labels_sorted:
        vals = series[lbl]["values"]
        if vals.size > 0:
            y_max = max(y_max, float(np.max(vals)) * 1.2)
    handles = None
    labels_for_legend = None
    for i, lbl in enumerate(labels_sorted):
        data = series[lbl]
        r = i // cols
        c = i % cols
        ax = axs[r][c]
        layer_indices = data["layers"]
        time_axis = data["time"]
        values = data["values"]
        cmap = plt.get_cmap("viridis") if hasattr(plt, "get_cmap") else None
        num_layers = len(layer_indices)
        lines = []
        for j, li in enumerate(layer_indices):
            color = cmap(j / max(1, num_layers - 1)) if cmap is not None else None
            (line,) = ax.plot(
                time_axis, values[j, :], label=f"layer {li}", color=color, linewidth=1.6
            )
            lines.append(line)
        ax.set_title(f"Average S over time | label={lbl}")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Average S")
        ax.set_ylim(0.0, y_max)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)
        handles = lines
        labels_for_legend = [f"layer {li}" for li in layer_indices]
    # Hide any unused axes
    total_axes = rows * cols
    for j in range(len(labels_sorted), total_axes):
        r = j // cols
        c = j % cols
        axs[r][c].axis("off")
    if handles and labels_for_legend:
        fig.legend(
            handles,
            labels_for_legend,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            ncol=1,
            frameon=False,
            title="Layers",
        )
    _rotate_xtick_labels(ax)
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, "avg_S_time_series_all_labels", fig, dpi=200, tight=True
    )
    plt.close(fig)
    return out_path


def plot_total_fired_cumulative_combined(
    out_dir: str, series: Dict[int, Dict[str, Any]]
) -> str:
    if plt is None:
        return ""
    labels_sorted = sorted(series.keys())
    if not labels_sorted:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    rows, cols = _compute_grid(len(labels_sorted))
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(cols * 5.0, rows * 3.6),
        squeeze=False,
        sharex=False,
        sharey=True,
    )
    # Determine common y-limit
    y_max = 0.05
    for lbl in labels_sorted:
        vals = series[lbl]["values"]
        if vals.size > 0:
            y_max = max(y_max, float(np.max(vals)) * 1.2)
    handles = None
    labels_for_legend = None
    for i, lbl in enumerate(labels_sorted):
        data = series[lbl]
        r = i // cols
        c = i % cols
        ax = axs[r][c]
        layer_indices = data["layers"]
        time_axis = data["time"]
        values = data["values"]
        cmap = plt.get_cmap("plasma") if hasattr(plt, "get_cmap") else None
        num_layers = len(layer_indices)
        lines = []
        for j, li in enumerate(layer_indices):
            color = cmap(j / max(1, num_layers - 1)) if cmap is not None else None
            (line,) = ax.plot(
                time_axis, values[j, :], label=f"layer {li}", color=color, linewidth=1.6
            )
            lines.append(line)
        ax.set_title(f"Total fired (cum) | label={lbl}")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Total fired (cumulative)")
        ax.set_ylim(0.0, y_max)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)
        handles = lines
        labels_for_legend = [f"layer {li}" for li in layer_indices]
    # Hide any unused axes
    total_axes = rows * cols
    for j in range(len(labels_sorted), total_axes):
        r = j // cols
        c = j % cols
        axs[r][c].axis("off")
    if handles and labels_for_legend:
        fig.legend(
            handles,
            labels_for_legend,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            ncol=1,
            frameon=False,
            title="Layers",
        )
    _rotate_xtick_labels(ax)
    plt.tight_layout()
    out_path = _save_figure_by_type(
        out_dir, "total_fired_cumulative_all_labels", fig, dpi=200, tight=True
    )
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize activity dataset")
    parser.add_argument(
        "dataset", type=str, help="Path to dataset JSON (supervised preferred)"
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        choices=PLOT_TYPES,
        help="Type of plot to generate (if omitted, prompts interactively)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="viz",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    if not args.plot:
        args.plot = prompt_plot_type()

    records = load_dataset(args.dataset)
    buckets = group_by_image(records)

    keys = sorted(buckets.keys(), key=lambda k: (k[0], k[1]))

    # Build output directory structure: <out_dir>/<dataset_root>/<plot_type_base>/(2d|3d)
    dataset_root = os.path.splitext(os.path.basename(args.dataset))[0]

    def plot_base_name(name: str) -> str:
        return name[:-3] if name.endswith("_3d") else name

    def ensure_dir(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    if args.plot == "avg_S_per_layer_per_label":
        out_dir = ensure_dir(
            os.path.join(args.out_dir, dataset_root, plot_base_name(args.plot), "2d")
        )
        labels_sorted, layer_indices, mat = compute_avg_S_per_layer_per_label(records)
        out_path = plot_avg_S_per_layer_per_label(
            out_dir, labels_sorted, layer_indices, mat
        )
        if out_path:
            print(f"Saved: {out_path}")
        if "avg_S_per_layer_per_label_3d" in PLOT_TYPES and Axes3D is not None:
            out_dir3d = ensure_dir(
                os.path.join(
                    args.out_dir,
                    dataset_root,
                    plot_base_name("avg_S_per_layer_per_label_3d"),
                    "3d",
                )
            )
            out3d = plot_avg_S_per_layer_per_label_3d(
                out_dir3d, labels_sorted, layer_indices, mat
            )
            if out3d:
                print(f"Saved: {out3d}")
    elif args.plot == "avg_S_per_layer_per_label_3d":
        out_dir3d = ensure_dir(
            os.path.join(args.out_dir, dataset_root, plot_base_name(args.plot), "3d")
        )
        labels_sorted, layer_indices, mat = compute_avg_S_per_layer_per_label(records)
        out3d = plot_avg_S_per_layer_per_label_3d(
            out_dir3d, labels_sorted, layer_indices, mat
        )
        if out3d:
            print(f"Saved: {out3d}")
    elif args.plot == "firings_time_series":
        out_dir = ensure_dir(
            os.path.join(args.out_dir, dataset_root, plot_base_name(args.plot), "2d")
        )
        series = compute_firings_time_series_per_label(records)
        # Combined figure with all labels
        combined_path = plot_firings_time_series_combined(out_dir, series)
        if combined_path:
            print(f"Saved: {combined_path}")
        for lbl in tqdm(sorted(series.keys()), desc="Labels"):
            data = series[lbl]
            out_path = plot_firings_time_series_for_label(
                out_dir, lbl, data["layers"], data["time"], data["values"]
            )
            if out_path:
                print(f"Saved: {out_path}")
            if "firings_time_series_3d" in PLOT_TYPES and Axes3D is not None:
                out_dir3d = ensure_dir(
                    os.path.join(
                        args.out_dir,
                        dataset_root,
                        plot_base_name("firings_time_series_3d"),
                        "3d",
                    )
                )
                out3d = plot_firings_time_series_for_label_3d(
                    out_dir3d,
                    lbl,
                    data["layers"],
                    data["time"],
                    data["values"],
                )
                if out3d:
                    print(f"Saved: {out3d}")
    elif args.plot == "firings_time_series_3d":
        out_dir3d = ensure_dir(
            os.path.join(args.out_dir, dataset_root, plot_base_name(args.plot), "3d")
        )
        series = compute_firings_time_series_per_label(records)
        out3d = plot_firings_time_series_all_labels_3d(out_dir3d, series)
        if out3d:
            print(f"Saved: {out3d}")
    elif args.plot == "avg_S_time_series":
        out_dir = ensure_dir(
            os.path.join(args.out_dir, dataset_root, plot_base_name(args.plot), "2d")
        )
        series = compute_avg_S_time_series_per_label(records)
        # Combined figure with all labels
        combined_path = plot_avg_S_time_series_combined(out_dir, series)
        if combined_path:
            print(f"Saved: {combined_path}")
        for lbl in tqdm(sorted(series.keys()), desc="Labels"):
            data = series[lbl]
            out_path = plot_avg_S_time_series_for_label(
                out_dir, lbl, data["layers"], data["time"], data["values"]
            )
            if out_path:
                print(f"Saved: {out_path}")
            if "avg_S_time_series_3d" in PLOT_TYPES and Axes3D is not None:
                out_dir3d = ensure_dir(
                    os.path.join(
                        args.out_dir,
                        dataset_root,
                        plot_base_name("avg_S_time_series_3d"),
                        "3d",
                    )
                )
                out3d = plot_avg_S_time_series_for_label_3d(
                    out_dir3d, lbl, data["layers"], data["time"], data["values"]
                )
                if out3d:
                    print(f"Saved: {out3d}")
    elif args.plot == "avg_S_time_series_3d":
        out_dir3d = ensure_dir(
            os.path.join(args.out_dir, dataset_root, plot_base_name(args.plot), "3d")
        )
        series = compute_avg_S_time_series_per_label(records)
        out3d = plot_avg_S_time_series_all_labels_3d(out_dir3d, series)
        if out3d:
            print(f"Saved: {out3d}")
    elif args.plot == "total_fired_cumulative":
        out_dir = ensure_dir(
            os.path.join(args.out_dir, dataset_root, plot_base_name(args.plot), "2d")
        )
        series = compute_total_fired_cumulative_per_label(records)
        # Combined figure with all labels
        combined_path = plot_total_fired_cumulative_combined(out_dir, series)
        if combined_path:
            print(f"Saved: {combined_path}")
        for lbl in tqdm(sorted(series.keys()), desc="Labels"):
            data = series[lbl]
            out_path = plot_total_fired_cumulative_for_label(
                out_dir, lbl, data["layers"], data["time"], data["values"]
            )
            if out_path:
                print(f"Saved: {out_path}")
            if "total_fired_cumulative_3d" in PLOT_TYPES and Axes3D is not None:
                out_dir3d = ensure_dir(
                    os.path.join(
                        args.out_dir,
                        dataset_root,
                        plot_base_name("total_fired_cumulative_3d"),
                        "3d",
                    )
                )
                out3d = plot_total_fired_cumulative_for_label_3d(
                    out_dir3d, lbl, data["layers"], data["time"], data["values"]
                )
                if out3d:
                    print(f"Saved: {out3d}")
    elif args.plot == "total_fired_cumulative_3d":
        out_dir3d = ensure_dir(
            os.path.join(args.out_dir, dataset_root, plot_base_name(args.plot), "3d")
        )
        series = compute_total_fired_cumulative_per_label(records)
        out3d = plot_total_fired_cumulative_all_labels_3d(out_dir3d, series)
        if out3d:
            print(f"Saved: {out3d}")
    elif args.plot == "firing_rate_per_layer_3d":
        out_dir3d = ensure_dir(
            os.path.join(args.out_dir, dataset_root, plot_base_name(args.plot), "3d")
        )
        # Compute label → averaged layer rates
        from collections import defaultdict

        sum_rates: Dict[int, List[float]] = defaultdict(list)
        count_rates: Dict[int, int] = defaultdict(int)
        for (label, img_idx), image_records in tqdm(buckets.items(), desc="Images"):
            layer_rates = compute_layer_firing_rate_for_image(image_records)
            if label not in sum_rates or not sum_rates[label]:
                sum_rates[label] = [0.0 for _ in range(len(layer_rates))]
            for i, r in enumerate(layer_rates):
                sum_rates[label][i] += r
            count_rates[label] += 1
        labels_sorted = sorted(sum_rates.keys())
        if labels_sorted:
            num_layers = len(sum_rates[labels_sorted[0]])
            mat = np.zeros((len(labels_sorted), num_layers), dtype=np.float32)
            for li, lbl in enumerate(labels_sorted):
                c = max(1, count_rates[lbl])
                mat[li, :] = np.array([s / c for s in sum_rates[lbl]], dtype=np.float32)
            layer_indices = list(range(num_layers))
            out3d = plot_firing_rate_per_layer_3d(
                out_dir3d, labels_sorted, layer_indices, mat
            )
            if out3d:
                print(f"Saved: {out3d}")
    elif args.plot == "network_state_progression":
        out_dir = ensure_dir(
            os.path.join(args.out_dir, dataset_root, plot_base_name(args.plot), "2d")
        )
        # One plot per image (label, image_index)
        for label, img_idx in tqdm(keys, desc="Images"):
            image_records = buckets[(label, img_idx)]
            out_path = plot_network_state_progression_for_image(
                out_dir, label, img_idx, image_records
            )
            if out_path:
                print(f"Saved: {out_path}")
    else:
        out_dir = ensure_dir(
            os.path.join(args.out_dir, dataset_root, plot_base_name(args.plot), "2d")
        )
        # Compute average firing rate across ALL images per label
        # Aggregate per label → per layer averages
        label_to_rates: Dict[int, List[float]] = {}
        from collections import defaultdict

        sum_rates: Dict[int, List[float]] = defaultdict(list)
        count_rates: Dict[int, int] = defaultdict(int)

        # First compute per-image layer rates, then aggregate by label
        for (label, img_idx), image_records in tqdm(buckets.items(), desc="Images"):
            layer_rates = compute_layer_firing_rate_for_image(image_records)
            if label not in sum_rates or not sum_rates[label]:
                sum_rates[label] = [0.0 for _ in range(len(layer_rates))]
            for i, r in enumerate(layer_rates):
                sum_rates[label][i] += r
            count_rates[label] += 1

        # Average
        for lbl, sums in sum_rates.items():
            c = max(1, count_rates[lbl])
            label_to_rates[lbl] = [s / c for s in sums]

        # Combined figure with all labels
        combined_path = plot_firing_rate_per_layer_combined(out_dir, label_to_rates)
        if combined_path:
            print(f"Saved: {combined_path}")

        # Save one bar chart per label of the averaged rates
        for lbl in tqdm(sorted(label_to_rates.keys()), desc="Labels(avg)"):
            rates = label_to_rates[lbl]
            # Reuse the plotting helper with a synthetic key (lbl, -1)
            out_path = plot_firing_rate_per_layer(
                out_dir, lbl, rates, title_prefix="Average"
            )
            if out_path:
                print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
