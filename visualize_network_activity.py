import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import csv

# Import binary dataset support
from build_activity_dataset import LazyActivityDataset

# Optional: reuse a single Kaleido scope to reduce file descriptors
try:
    from kaleido.scopes.plotly import PlotlyScope  # type: ignore

    KALEIDO_SCOPE: Optional[PlotlyScope] = PlotlyScope()
except Exception:
    KALEIDO_SCOPE = None


# Global rendering and caching configuration
PLOT_IMAGE_SCALE: float = 2.0  # Higher scale => higher DPI for Plotly exports
CACHE_VERSION: str = "v1"
# Control static image exports (PNG/SVG). Set from CLI in main().
SAVE_STATIC_IMAGES: bool = True
_STATIC_EXPORT_DISABLED_ON_ERROR: bool = False


def compute_dataset_hash(file_path: str) -> str:
    """Return a short MD5 hash for the dataset file using Python's hashlib.

    Handles both single files (JSON) and directories (binary/HDF5 datasets).
    Returns the first 16 hex characters for stable short tokens.
    """
    import hashlib

    # Determine the actual file to hash
    if os.path.isdir(file_path):
        # Binary dataset: hash the HDF5 file within the directory
        h5_path = os.path.join(file_path, "activity_dataset.h5")
        if not os.path.exists(h5_path):
            raise RuntimeError(
                f"Binary dataset directory {file_path} does not contain activity_dataset.h5"
            )
        actual_path = h5_path
    else:
        # Single file (JSON)
        actual_path = file_path

    try:
        with open(actual_path, "rb") as f:
            file_hash = hashlib.md5()
            # Read file in chunks to handle large files
            while chunk := f.read(8192):
                file_hash.update(chunk)
            return file_hash.hexdigest()[:16]
    except (OSError, IOError) as e:
        raise RuntimeError(f"Failed to compute hash for {actual_path}: {e}")


def get_cache_dir(base_output_dir: str) -> str:
    """Return (and create) the cache directory for intermediates."""
    cache_dir = os.path.join(base_output_dir, f"cache_{CACHE_VERSION}")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_activity_data(path: str) -> LazyActivityDataset:
    """Load activity dataset from directory (binary format)."""
    if not os.path.isdir(path):
        raise ValueError(f"Dataset path must be a directory: {path}")

    print(f"Loading activity dataset from: {path}")
    return LazyActivityDataset(path)


def group_images_by_label(dataset: LazyActivityDataset) -> Dict[int, List[int]]:
    """Group image indices by label."""
    label_to_indices: Dict[int, List[int]] = {}
    for i in range(len(dataset)):
        # Access label directly if possible without full load, or load sample
        # dataset[i] loads sample.
        # For efficiency, if dataset supports metadata access, use it.
        # But LazyActivityDataset (from build_activity_dataset.py) might not expose metadata separately yet.
        # We'll assume loading sample is fine or we can optimize later.
        sample = dataset[i]
        label = int(sample["label"])
        label_to_indices.setdefault(label, []).append(i)

    # Sort indices
    for lbl in label_to_indices:
        label_to_indices[lbl].sort()
    return label_to_indices


def infer_network_structure(
    dataset: LazyActivityDataset,
) -> Tuple[List[int], List[int], int]:
    """Infer [neurons_per_layer], [layer_offsets], and total_neurons from dataset metadata."""
    if len(dataset) == 0:
        return [], [], 0

    # LazyActivityDataset might have layer_structure attribute
    if hasattr(dataset, "layer_structure"):
        neurons_per_layer = dataset.layer_structure
    else:
        # Fallback: assume single layer with all neurons
        sample = dataset[0]
        total = sample["u"].shape[1]
        neurons_per_layer = [total]

    layer_offsets: List[int] = []
    offset = 0
    for n in neurons_per_layer:
        layer_offsets.append(offset)
        offset += n
    total_neurons = offset
    return neurons_per_layer, layer_offsets, total_neurons


def flatten_neuron_index(
    layer_index: int, neuron_index: int, layer_offsets: List[int]
) -> int:
    return layer_offsets[layer_index] + neuron_index


def ensure_output_dirs(base_out_dir: str, dataset_root: str) -> Tuple[str, str]:
    """Return (fig_dir, cache_dir) for this dataset."""
    fig_dir = os.path.join(base_out_dir, dataset_root)
    os.makedirs(fig_dir, exist_ok=True)
    cache_dir = get_cache_dir(fig_dir)
    return fig_dir, cache_dir


def log_plot_start(plot_name: str, scope: Optional[str] = None) -> None:
    """Log the start of a plot generation (aggregate or per-digit)."""
    if scope:
        print(f"\n[Plot] Starting {plot_name} ({scope})...")
    else:
        print(f"\n[Plot] Starting {plot_name}...")


def log_plot_end(plot_name: str, scope: Optional[str] = None) -> None:
    """Log the completion of a plot generation (aggregate or per-digit)."""
    if scope:
        print(f"[Plot] Completed {plot_name} ({scope})")
    else:
        print(f"[Plot] Completed {plot_name}")


def check_all_caches_exist(
    requested_plots: List[str],
    dataset_hash: str,
    cache_dir: str,
    num_classes: int,
    excluded_plots: Optional[List[str]] = None,
    epoch_field: str = "epoch",
    num_epoch_bins: int = 0,
    tref_box_sample: int = 200000,
    corr_max_neurons: int = 200,
    corr_threshold: float = 0.3,
    raster_gap: int = 1,
    corr_max_images: int = 200,
    animation_frames: int = 50,
    convergence_final_ticks: int = 10,
) -> bool:
    """Check if all cache files for requested plot types exist.

    Returns True if all requested plots have cached results, False otherwise.
    For plots that do not have a cache by design (e.g., spike_raster), this
    will return False to force data loading.

    Runtime parameters are needed to construct accurate cache keys for plots
    that depend on them (e.g., favg_stability, temporal_corr_graph).
    """
    print(f"[Cache Check] Requested plots: {', '.join(requested_plots)}")
    if excluded_plots:
        print(f"[Cache Check] Excluded plots: {', '.join(excluded_plots)}")
    print(f"[Cache Check] Dataset hash: {dataset_hash}")
    print(f"[Cache Check] Cache directory: {cache_dir}")

    cache_files_to_check: List[str] = []

    # Always needed by many plots
    cache_files_to_check.append(f"{dataset_hash}_per_neuron_aggregates.npz")

    # Plots that have dedicated caches
    requires_raw_data = False
    for plot in requested_plots:
        if plot == "s_heatmap_by_class":
            cache_files_to_check.append(f"{dataset_hash}_S_heatmaps_c{num_classes}.npz")
        elif plot == "favg_tref_scatter":
            # uses aggregates cache
            pass
        elif plot == "firing_rate_hist_by_layer":
            # uses aggregates cache
            pass
        elif plot == "tref_timeline":
            cache_files_to_check.append(f"{dataset_hash}_tref_timelines.npz")
        elif plot == "layerwise_s_average":
            cache_files_to_check.append(f"{dataset_hash}_layerwise_S_timeline.npz")
        elif plot == "spike_raster":
            # No cache by design; force raw data load
            print(f"[Cache Check] Plot '{plot}' requires raw data - skipping cache")
            requires_raw_data = True
        elif plot == "phase_portrait":
            cache_files_to_check.append(
                f"{dataset_hash}_phase_portrait_c{num_classes}.npz"
            )
        elif plot == "attractor_landscape" or plot == "attractor_landscape_3d":
            cache_files_to_check.append(f"{dataset_hash}_attractor_landscape_g80.npz")
        elif plot == "tref_bounds_box":
            cache_files_to_check.append(
                f"{dataset_hash}_tref_samples_per_layer_{tref_box_sample}.npz"
            )
        elif plot == "favg_stability":
            cache_files_to_check.append(
                f"{dataset_hash}_favg_stability_{epoch_field}_bins{num_epoch_bins}.npz"
            )
        elif plot == "homeostatic_response":
            # uses aggregates cache
            pass
        elif plot == "affinity_heatmap" or plot == "tref_by_preferred_digit":
            cache_files_to_check.append(f"{dataset_hash}_affinity_c{num_classes}.npz")
        elif plot == "temporal_corr_graph":
            cache_files_to_check.append(
                f"{dataset_hash}_corr_n{corr_max_neurons}_t{corr_threshold:.3f}_g{raster_gap}_m{corr_max_images}.npz"
            )
        elif plot == "s_variance_decay":
            cache_files_to_check.append(f"{dataset_hash}_svar_decay_c{num_classes}.npz")
        elif plot == "attractor_landscape_animated":
            cache_files_to_check.append(
                f"{dataset_hash}_attractor_animated_g80_f{animation_frames}.npz"
            )

    # Attractor landscape plots also need convergence points cache
    if any(
        p in requested_plots
        for p in [
            "attractor_landscape",
            "attractor_landscape_3d",
            "attractor_landscape_animated",
        ]
    ):
        cache_files_to_check.append(
            f"{dataset_hash}_convergence_points_f{convergence_final_ticks}.npz"
        )

    if requires_raw_data:
        print("[Cache Check] Raw data required by at least one plot - will load data")
        return False

    # Verify presence of required cache files
    print(f"[Cache Check] Checking {len(cache_files_to_check)} cache file(s)...")
    missing_files = []
    for cache_file in cache_files_to_check:
        cache_path = os.path.join(cache_dir, cache_file)
        if not os.path.exists(cache_path):
            missing_files.append(cache_file)
            print(f"[Cache Check]   ✗ Missing: {cache_file}")
        else:
            print(f"[Cache Check]   ✓ Found: {cache_file}")

    if missing_files:
        print(
            f"[Cache Check] Result: {len(missing_files)} cache file(s) missing - will load data"
        )
        return False

    print(
        f"[Cache Check] Result: All {len(cache_files_to_check)} cache file(s) found - skipping data load"
    )
    return True


def save_figure(fig: go.Figure, out_dir: str, base_name: str) -> None:
    """Save Plotly figure to HTML, JSON, PNG and SVG."""
    global _STATIC_EXPORT_DISABLED_ON_ERROR
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"{base_name}.html")
    json_path = os.path.join(out_dir, f"{base_name}.json")
    png_path = os.path.join(out_dir, f"{base_name}.png")
    svg_path = os.path.join(out_dir, f"{base_name}.svg")
    fig.write_html(html_path)
    pio.write_json(fig, json_path)
    if not SAVE_STATIC_IMAGES or _STATIC_EXPORT_DISABLED_ON_ERROR:
        return
    try:
        if KALEIDO_SCOPE is not None:
            # Use a shared Kaleido scope to minimize open file descriptors
            png_bytes = KALEIDO_SCOPE.transform(
                fig, format="png", width=1400, height=900, scale=PLOT_IMAGE_SCALE
            )
            with open(png_path, "wb") as f_png:
                f_png.write(png_bytes)
            svg_bytes = KALEIDO_SCOPE.transform(
                fig, format="svg", width=1400, height=900, scale=PLOT_IMAGE_SCALE
            )
            with open(svg_path, "wb") as f_svg:
                f_svg.write(svg_bytes)
        else:
            fig.write_image(png_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
            fig.write_image(svg_path, width=1400, height=900, scale=PLOT_IMAGE_SCALE)
    except Exception as e:
        # Gracefully degrade when kaleido is missing or errors out
        err_msg = (
            f"[warn] Static image export failed for {base_name}: {e}. "
            f"Install/upgrade 'kaleido' (pip install -U kaleido plotly) or use --skip-static-images to disable PNG/SVG."
        )
        print(err_msg)
        # If too many open files, disable further static exports in this run
        if "Too many open files" in str(e):
            print(
                "[warn] Disabling further static image exports for this run due to OS file descriptor limits."
            )
            _STATIC_EXPORT_DISABLED_ON_ERROR = True
    finally:
        # Encourage garbage collection of large figures
        try:
            import gc

            del fig
            gc.collect()
        except Exception:
            pass


def save_csv(
    rows: List[List[Any]], headers: List[str], out_dir: str, base_name: str
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            if headers:
                writer.writerow(headers)
            writer.writerows(rows)
    except Exception as e:
        print(f"[warn] Failed to write CSV {csv_path}: {e}")


def compute_classwise_S_heatmaps(
    label_to_indices: Dict[int, List[int]],
    dataset: LazyActivityDataset,
    neurons_per_layer: List[int],
    layer_offsets: List[int],
    total_neurons: int,
    num_classes: int,
    cache_dir: str,
    dataset_hash: str,
) -> Dict[int, np.ndarray]:
    """Compute class-wise S(t) average heatmaps as arrays [num_neurons, T_min(label)].

    For each label, we average S per neuron at each relative tick across all images of that label.
    Results are cached to reduce repeated work.
    """
    cache_path = os.path.join(
        cache_dir, f"{dataset_hash}_S_heatmaps_c{num_classes}.npz"
    )
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        # New format: positional arrays with labels in arr_0, heatmaps in arr_1..arr_N
        if "arr_0" in data.files and all(k.startswith("arr_") for k in data.files):
            labels = data["arr_0"].astype(np.int32)
            heatmaps = {}
            for i, lbl in enumerate(labels, start=1):
                key = f"arr_{i}"
                if key in data.files:
                    heatmaps[int(lbl)] = data[key]
            return heatmaps
        # Legacy format: per-class keys as strings
        heatmaps = {int(k): data[k] for k in data.files}
        return heatmaps

    heatmaps: Dict[int, np.ndarray] = {}

    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        if not indices:
            continue

        # Check lengths to find min_T
        min_T = float("inf")
        valid_indices = []
        for idx in indices:
            sample = dataset[idx]  # Metadata check only ideally, but we load
            ticks = sample["u"].shape[0]
            if ticks > 0:
                min_T = min(min_T, ticks)
                valid_indices.append(idx)

        if min_T == float("inf") or min_T <= 0 or total_neurons <= 0:
            continue

        min_T = int(min_T)
        sums = np.zeros((total_neurons, min_T), dtype=np.float32)

        for idx in valid_indices:
            sample = dataset[idx]
            u = sample["u"]  # (ticks, neurons)
            # Add valid slice
            sums += u[:min_T, :total_neurons].T

        sums /= max(1, len(valid_indices))
        heatmaps[label] = sums

    # Save cache (positional arrays to satisfy strict type stubs)
    if heatmaps:
        labels_sorted = np.array(sorted(heatmaps.keys()), dtype=np.int32)
        arrays = [heatmaps[int(lbl)] for lbl in labels_sorted]
        # Save as arr_0 = labels, arr_1.. = heatmaps in the same order
        np.savez_compressed(cache_path, labels_sorted, *arrays)
    return heatmaps


def compute_per_neuron_aggregates(
    dataset: LazyActivityDataset,
    neurons_per_layer: List[int],
    layer_offsets: List[int],
    total_neurons: int,
    num_classes: int,
    cache_dir: str,
    dataset_hash: str,
    indices: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """Compute per-neuron aggregates: mean F_avg, mean t_ref, layer index, selectivity (by F_avg).

    Also compute per-class F_avg means per neuron to derive a selectivity index as a proxy for
    contribution/importance.
    """
    cache_path = os.path.join(cache_dir, f"{dataset_hash}_per_neuron_aggregates.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    mean_favg_sum = np.zeros((total_neurons,), dtype=np.float64)
    mean_tref_sum = np.zeros((total_neurons,), dtype=np.float64)
    count_points = np.zeros((total_neurons,), dtype=np.int64)

    # For selectivity by class
    favg_sum_by_class = np.zeros((total_neurons, num_classes), dtype=np.float64)
    favg_count_by_class = np.zeros((total_neurons, num_classes), dtype=np.int64)

    iterator = indices if indices is not None else range(len(dataset))

    # Process all ticks across all images
    for i in tqdm(iterator, desc="Aggregating per-neuron stats"):
        sample = dataset[i]
        label = int(sample["label"])
        if not (0 <= label < num_classes):
            continue

        # sample["fr"] is (ticks, neurons) -> F_avg
        # sample["t_ref"] is (ticks, neurons)
        fr = sample["fr"]
        tr = sample["t_ref"]

        # Sum over time (axis 0)
        # Check shapes
        if fr.shape[1] != total_neurons:
            # Maybe mismatched total_neurons?
            # We trust sample structure matches inferred structure.
            pass

        sum_fr = np.sum(fr, axis=0)
        sum_tr = np.sum(tr, axis=0)
        ticks = fr.shape[0]

        mean_favg_sum += sum_fr
        mean_tref_sum += sum_tr
        count_points += ticks

        favg_sum_by_class[:, label] += sum_fr
        favg_count_by_class[:, label] += ticks

    # Compute means
    count_safe = np.maximum(count_points, 1)
    mean_favg = (mean_favg_sum / count_safe).astype(np.float32)
    mean_tref = (mean_tref_sum / count_safe).astype(np.float32)

    # Compute selectivity index per neuron
    class_means = np.divide(
        favg_sum_by_class,
        np.maximum(favg_count_by_class, 1),
        where=favg_count_by_class > 0,
    )
    max_class_mean = np.max(class_means, axis=1)
    mean_of_means = np.mean(class_means, axis=1)
    denom = max_class_mean + mean_of_means
    with np.errstate(divide="ignore", invalid="ignore"):
        selectivity = np.where(denom > 0, (max_class_mean - mean_of_means) / denom, 0.0)
    selectivity = selectivity.astype(np.float32)

    # Layer index per neuron
    layer_index = np.zeros((total_neurons,), dtype=np.int32)
    for li, n in enumerate(neurons_per_layer):
        base = layer_offsets[li]
        layer_index[base : base + n] = li

    aggregates = {
        "mean_favg": mean_favg,
        "mean_tref": mean_tref,
        "selectivity": selectivity,
        "layer_index": layer_index,
    }

    # Save with explicit kwargs to avoid signature confusion with type checkers
    np.savez_compressed(
        cache_path,
        mean_favg=aggregates["mean_favg"],
        mean_tref=aggregates["mean_tref"],
        selectivity=aggregates["selectivity"],
        layer_index=aggregates["layer_index"],
    )
    return aggregates


def compute_tref_timelines(
    dataset: LazyActivityDataset,
    neurons_per_layer: List[int],
    layer_offsets: List[int],
    total_neurons: int,
    cache_dir: str,
    dataset_hash: str,
    indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, int]:
    """Compute mean t_ref(t) over time across all images for every neuron.

    Returns (timeline_matrix, T_min_all), where timeline_matrix shape is [total_neurons, T_min_all].
    """
    cache_path = os.path.join(cache_dir, f"{dataset_hash}_tref_timelines.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["timelines"], int(data["T_min_all"])  # type: ignore[index]

    if len(dataset) == 0 or total_neurons <= 0:
        return np.zeros((0, 0), dtype=np.float32), 0

    iterator_indices = indices if indices is not None else range(len(dataset))
    num_samples = len(iterator_indices)  # This works for range too? No. range has len.
    # range(N) has len. List has len.
    # But if iterator_indices is range(len(dataset)), len() is fine.
    # Check if indices empty
    if num_samples == 0:
        return np.zeros((0, 0), dtype=np.float32), 0

    # Find T_min_all
    T_min_all = 10**12
    for i in iterator_indices:
        sample = dataset[i]
        T_min_all = min(T_min_all, sample["u"].shape[0])

    if T_min_all <= 0 or T_min_all > 1000000:
        if T_min_all > 1000000:
            T_min_all = 0
        return np.zeros((0, 0), dtype=np.float32), 0

    sums = np.zeros((total_neurons, T_min_all), dtype=np.float64)

    for i in tqdm(iterator_indices, desc="Aggregating t_ref timelines"):
        sample = dataset[i]
        tr = sample["t_ref"]
        if tr.shape[0] >= T_min_all:
            sums += tr[:T_min_all, :total_neurons].T
        else:
            # Should not happen as T_min_all is min
            pass

    sums /= max(1, num_samples)
    timelines = sums.astype(np.float32)
    # Use kwargs to avoid allow_pickle confusion in some type checkers
    np.savez_compressed(cache_path, timelines=timelines, T_min_all=np.array(T_min_all))
    return timelines, T_min_all


def plot_S_heatmap_by_class(
    heatmaps: Dict[int, np.ndarray],
    title_prefix: str,
    out_dir: str,
) -> None:
    """Render a grid of heatmaps (one per class) with x=time, y=neuron id, color=S."""
    if not heatmaps:
        return
    log_plot_start("s_heatmap_by_class", "aggregate")
    labels_sorted = sorted(heatmaps.keys())

    # Determine grid layout (e.g., 2 rows × 5 cols for 10 classes)
    n = len(labels_sorted)
    cols = min(5, n)
    rows = int(np.ceil(n / max(1, cols)))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Digit {label}" for label in labels_sorted],
    )

    for i, lbl in enumerate(labels_sorted):
        r = i // cols + 1
        c = i % cols + 1
        mat = heatmaps[lbl]
        # Heatmap expects z as [y, x] => [neuron, time]
        fig.add_trace(
            go.Heatmap(
                z=mat,
                colorscale="Viridis",
                colorbar=dict(title="S", len=0.8),
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="t", row=r, col=c)
        fig.update_yaxes(title_text="neuron id", row=r, col=c)

    fig.update_layout(
        title=dict(text=f"{title_prefix} – S(t) Heatmaps by Digit", font=dict(size=18)),
        width=max(1400, cols * 500),
        height=max(900, rows * 400),
        hovermode="closest",
    )

    save_figure(fig, out_dir, "s_heatmap_by_class")
    log_plot_end("s_heatmap_by_class", "aggregate")


def plot_favg_vs_tref_scatter(
    aggregates: Dict[str, np.ndarray],
    title_prefix: str,
    out_dir: str,
    theory_c: Optional[float] = None,
    theory_syn_per_layer: Optional[List[int]] = None,
) -> None:
    """Scatter: x = mean F_avg, y = mean t_ref, color by layer."""
    log_plot_start("favg_tref_scatter", "aggregate")
    mean_favg = aggregates["mean_favg"]
    mean_tref = aggregates["mean_tref"]
    layer_index = aggregates["layer_index"]

    fig = go.Figure()
    palette = (
        px.colors.qualitative.Plotly
        if len(set(layer_index.tolist())) <= 10
        else px.colors.qualitative.Dark24
    )

    for li in sorted(set(layer_index.tolist())):
        mask = layer_index == li
        fig.add_trace(
            go.Scatter(
                x=mean_favg[mask],
                y=mean_tref[mask],
                mode="markers",
                name=f"Layer {li}",
                marker=dict(
                    size=8,
                    color=palette[li % len(palette)],
                    line=dict(width=0.5, color="black"),
                ),
                opacity=0.7,
            )
        )

    # Optional theoretical curves per layer
    if (
        theory_c is not None
        and theory_syn_per_layer is not None
        and len(theory_syn_per_layer) > 0
    ):
        f_min = float(np.min(mean_favg)) if mean_favg.size else 0.0
        f_max = float(np.max(mean_favg)) if mean_favg.size else 1.0
        if f_max <= f_min:
            f_max = f_min + 1e-6
        x_line = np.linspace(f_min, f_max, 200)
        for li in sorted(set(layer_index.tolist())):
            lower = 2.0 * float(theory_c)
            upper = float(theory_c) * float(
                theory_syn_per_layer[min(li, len(theory_syn_per_layer) - 1)]
            )
            y_line = upper - (upper - lower) * (x_line * float(theory_c))
            y_line = np.clip(y_line, lower, upper)
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"Theory L{li}",
                    line=dict(color=palette[li % len(palette)], width=2, dash="dash"),
                )
            )

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – F04 vs t_ref (per neuron)", font=dict(size=18)
        ),
        xaxis_title="Mean F04 (firing rate)",
        yaxis_title="Mean t_ref",
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    save_figure(fig, out_dir, "favg_vs_tref_scatter")
    log_plot_end("favg_tref_scatter", "aggregate")
    # CSV output: mean_favg, mean_tref, layer_index
    rows: List[List[Any]] = [
        [
            float(aggregates["mean_favg"][i]),
            float(aggregates["mean_tref"][i]),
            int(aggregates["layer_index"][i]),
        ]
        for i in range(len(aggregates["mean_favg"]))
    ]
    save_csv(
        rows, ["mean_favg", "mean_tref", "layer"], out_dir, "favg_vs_tref_scatter_data"
    )


def plot_firing_rate_hist_by_layer(
    aggregates: Dict[str, np.ndarray],
    title_prefix: str,
    out_dir: str,
    bins: int = 30,
) -> None:
    """Histogram distribution of mean F_avg per layer."""
    log_plot_start("firing_rate_hist_by_layer", "aggregate")
    mean_favg = aggregates["mean_favg"]
    layer_index = aggregates["layer_index"]

    unique_layers = sorted(set(layer_index.tolist()))
    cols = min(5, len(unique_layers)) if unique_layers else 1
    rows = int(np.ceil(len(unique_layers) / max(1, cols)))
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=[f"Layer {li}" for li in unique_layers]
    )

    for i, li in enumerate(unique_layers):
        r = i // cols + 1
        c = i % cols + 1
        mask = layer_index == li
        fig.add_trace(
            go.Histogram(
                x=mean_favg[mask],
                nbinsx=bins,
                marker=dict(color="#4C78A8", line=dict(width=0.5, color="black")),
                opacity=0.8,
                name=f"Layer {li}",
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="Mean F04", row=r, col=c)
        fig.update_yaxes(title_text="Count", row=r, col=c)

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – Firing Rate Histogram by Layer", font=dict(size=18)
        ),
        width=max(1400, cols * 500),
        height=max(900, rows * 400),
        hovermode="closest",
        bargap=0.03,
    )

    save_figure(fig, out_dir, "firing_rate_hist_by_layer")
    log_plot_end("firing_rate_hist_by_layer", "aggregate")


def plot_tref_evolution_timeline(
    timelines: np.ndarray,
    T_min_all: int,
    aggregates: Dict[str, np.ndarray],
    neurons_per_layer: List[int],
    layer_offsets: List[int],
    max_representatives: int,
    title_prefix: str,
    out_dir: str,
) -> None:
    """Line plot of t_ref(t) for representative neurons; color-coded by selectivity proxy.

    Representative selection: choose up to max_representatives neurons with highest selectivity,
    sampling across layers to avoid all coming from a single layer.
    """
    if timelines.size == 0 or T_min_all <= 0:
        return
    log_plot_start("tref_timeline", "aggregate")

    selectivity = aggregates["selectivity"]
    layer_index = aggregates["layer_index"]

    representatives: List[int] = []
    # Per-layer top-k selection
    per_layer_quota = max(1, max_representatives // max(1, len(neurons_per_layer)))
    for li, n in enumerate(neurons_per_layer):
        base = layer_offsets[li]
        if n <= 0:
            continue
        sel_slice = selectivity[base : base + n]
        top_indices_local = np.argsort(-sel_slice)[:per_layer_quota]
        representatives.extend((base + top_indices_local).tolist())
        if len(representatives) >= max_representatives:
            break

    # If still short, fill with global top
    if len(representatives) < max_representatives:
        global_top = np.argsort(-selectivity)[
            : max_representatives - len(representatives)
        ]
        for idx in global_top:
            if int(idx) not in representatives:
                representatives.append(int(idx))

    # Build figure
    fig = go.Figure()
    colorscale = px.colors.sequential.Viridis
    t_axis = list(range(T_min_all))

    for idx in representatives[:max_representatives]:
        li = int(layer_index[idx])
        contrib = float(selectivity[idx])
        color = colorscale[
            int(np.clip(contrib * (len(colorscale) - 1), 0, len(colorscale) - 1))
        ]
        fig.add_trace(
            go.Scatter(
                x=t_axis,
                y=timelines[idx, :],
                mode="lines",
                name=f"n{idx} (L{li}, sel={contrib:.2f})",
                line=dict(color=color, width=2),
                hovertemplate=f"Neuron {idx} (Layer {li})<br>selectivity={contrib:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – t_ref Evolution (representative neurons)",
            font=dict(size=18),
        ),
        xaxis_title="t",
        yaxis_title="t_ref",
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    save_figure(fig, out_dir, "tref_evolution_timeline")
    log_plot_end("tref_timeline", "aggregate")


def compute_layerwise_S_timeline(
    dataset: LazyActivityDataset,
    neurons_per_layer: List[int],
    cache_dir: str,
    dataset_hash: str,
    indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, int]:
    """Compute mean S(t) per layer across all images.

    Returns (timeline_matrix, T_min_all) with shape [num_layers, T_min_all].
    Caches results for reuse.
    """
    cache_path = os.path.join(cache_dir, f"{dataset_hash}_layerwise_S_timeline.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["layer_s"], int(data["T_min_all"])  # type: ignore[index]

    if len(dataset) == 0:
        return np.zeros((0, 0), dtype=np.float32), 0

    num_layers = len(neurons_per_layer)
    if num_layers == 0:
        return np.zeros((0, 0), dtype=np.float32), 0

    iterator_indices = indices if indices is not None else range(len(dataset))
    num_samples = len(iterator_indices)
    if num_samples == 0:
        return np.zeros((0, 0), dtype=np.float32), 0

    # Determine T_min_all safely
    T_min_all = 10**12
    for i in iterator_indices:
        sample = dataset[i]
        T_min_all = min(T_min_all, sample["u"].shape[0])

    if T_min_all <= 0 or T_min_all > 1000000:
        return np.zeros((0, 0), dtype=np.float32), 0

    sums = np.zeros((num_layers, T_min_all), dtype=np.float64)

    # Pre-calculate layer slices
    layer_slices = []
    offset = 0
    for n in neurons_per_layer:
        layer_slices.append(slice(offset, offset + n))
        offset += n

    for i in tqdm(iterator_indices, desc="Aggregating layer-wise S(t)"):
        sample = dataset[i]
        u = sample["u"]  # (ticks, neurons)

        for li, sl in enumerate(layer_slices):
            # u[0:T, sl] -> average over neurons (axis 1)
            layer_u = u[:T_min_all, sl]
            if layer_u.shape[1] > 0:
                sums[li, :] += np.mean(layer_u, axis=1)

    layer_s = (sums / max(1, num_samples)).astype(np.float32)
    np.savez_compressed(cache_path, layer_s=layer_s, T_min_all=np.array(T_min_all))
    return layer_s, T_min_all


def plot_layerwise_S_timeline(
    layer_s: np.ndarray, T_min_all: int, title_prefix: str, out_dir: str
) -> None:
    if layer_s.size == 0 or T_min_all <= 0:
        return
    log_plot_start("layerwise_s_average", "aggregate")
    num_layers = layer_s.shape[0]
    t_axis = list(range(T_min_all))
    fig = go.Figure()
    cmap = (
        px.colors.qualitative.Dark24
        if num_layers > 10
        else px.colors.qualitative.Plotly
    )
    for li in range(num_layers):
        fig.add_trace(
            go.Scatter(
                x=t_axis,
                y=layer_s[li, :],
                mode="lines",
                name=f"Layer {li}",
                line=dict(color=cmap[li % len(cmap)], width=2),
            )
        )
    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – Layer-wise S(t) Average", font=dict(size=18)
        ),
        xaxis_title="t",
        yaxis_title="Mean S",
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )
    save_figure(fig, out_dir, "layerwise_s_timeline")
    log_plot_end("layerwise_s_average", "aggregate")
    # CSV output: t, layer, mean_S
    rows: List[List[Any]] = []
    for li in range(num_layers):
        for t in range(T_min_all):
            rows.append([t, li, float(layer_s[li, t])])
    save_csv(rows, ["t", "layer", "mean_S"], out_dir, "layerwise_s_timeline_data")


def plot_spike_raster_with_tref(
    label_to_indices: Dict[int, List[int]],
    dataset: LazyActivityDataset,
    dataset_hash: str,  # Added argument for consistency if needed, though unused here
    neurons_per_layer: List[int],
    layer_offsets: List[int],
    max_images: int,
    gap: int,
    max_points: int,
    title_prefix: str,
    out_dir: str,
    tref_bounds: Optional[Tuple[float, float]] = None,
) -> None:
    """Render spike raster (neuron id vs time). Color encodes t_ref at spike time.

    Concatenates images along time dimension with a fixed gap.
    """
    if len(dataset) == 0:
        return

    # Order images deterministically by (label, image_index)
    sorted_indices = []
    for lbl in sorted(label_to_indices.keys()):
        sorted_indices.extend(label_to_indices[lbl])

    if max_images is not None and max_images > 0:
        sorted_indices = sorted_indices[:max_images]

    xs: List[int] = []
    ys: List[int] = []
    cs: List[float] = []
    t_cursor = 0

    for idx in tqdm(sorted_indices, desc="Building spike raster"):
        sample = dataset[idx]
        spikes = sample["spikes"]  # (N, 2) -> (tick, neuron)
        tr = sample["t_ref"]  # (ticks, output_neurons)

        ticks = tr.shape[0]

        if spikes.shape[0] > 0:
            # Shift ticks by cursor
            spike_ticks = spikes[:, 0].astype(int)
            spike_neurons = spikes[:, 1].astype(int)

            # Filter invalid neurons?
            # Assuming valid.

            # t_ref at partial ticks
            # We need t_ref values at (tick, neuron)
            # Use advanced indexing
            # Clip ticks just in case
            valid_mask = (spike_ticks < ticks) & (spike_neurons < tr.shape[1])
            spike_ticks = spike_ticks[valid_mask]
            spike_neurons = spike_neurons[valid_mask]

            if len(spike_ticks) > 0:
                tref_values = tr[spike_ticks, spike_neurons]

                xs.extend((spike_ticks + t_cursor).tolist())
                ys.extend(spike_neurons.tolist())
                cs.extend(tref_values.tolist())

        t_cursor += ticks + max(0, int(gap))

    if not xs:
        return

    # Downsample if too many points
    total_points = len(xs)
    if max_points is not None and total_points > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(total_points, size=max_points, replace=False)
        xs = [xs[i] for i in idx]
        ys = [ys[i] for i in idx]
        cs = [cs[i] for i in idx]

    fig = go.Figure()
    cmin = None
    cmax = None
    if tref_bounds is not None:
        cmin, cmax = float(tref_bounds[0]), float(tref_bounds[1])
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=3,
                color=cs,
                colorscale="Viridis",
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                colorbar=dict(title="t_ref"),
                opacity=0.8,
            ),
            name="spikes",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – Spike Raster (color=t_ref)", font=dict(size=18)
        ),
        xaxis_title="t",
        yaxis_title="Neuron id (flattened)",
        width=1400,
        height=900,
        hovermode="closest",
    )
    save_figure(fig, out_dir, "spike_raster")
    # CSV output: t, neuron_id, t_ref_at_spike
    rows: List[List[Any]] = [[int(x), int(y), float(c)] for x, y, c in zip(xs, ys, cs)]
    save_csv(rows, ["t", "neuron_id", "t_ref"], out_dir, "spike_raster_data")


def compute_phase_portrait_series(
    label_to_indices: Dict[int, List[int]],
    dataset: LazyActivityDataset,
    num_classes: int,
    cache_dir: str,
    dataset_hash: str,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Compute per-class trajectories over time for (mean S, mean F_avg, mean t_ref).

    Returns mapping: class -> {"S": S_t, "F": F_t, "T": T_t} with arrays shape [T_class].
    """
    cache_path = os.path.join(
        cache_dir, f"{dataset_hash}_phase_portrait_c{num_classes}.npz"
    )
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        # Positional arrays: arr_0 = labels, then triples per label
        if "arr_0" in data.files and all(k.startswith("arr_") for k in data.files):
            labels = data["arr_0"].astype(np.int32)
            out: Dict[int, Dict[str, np.ndarray]] = {}
            i = 1
            for lbl in labels:
                S_t = data.get(f"arr_{i}")
                F_t = data.get(f"arr_{i + 1}")
                T_t = data.get(f"arr_{i + 2}")
                if S_t is not None and F_t is not None and T_t is not None:
                    out[int(lbl)] = {"S": S_t, "F": F_t, "T": T_t}
                i += 3
            return out
        # Legacy key format
        out: Dict[int, Dict[str, np.ndarray]] = {}
        for k in data.files:
            if k.endswith("_S"):
                lbl = int(k[:-2])
                out.setdefault(lbl, {})["S"] = data[k]
            elif k.endswith("_F"):
                lbl = int(k[:-2])
                out.setdefault(lbl, {})["F"] = data[k]
            elif k.endswith("_T"):
                lbl = int(k[:-2])
                out.setdefault(lbl, {})["T"] = data[k]
        return out

    result: Dict[int, Dict[str, np.ndarray]] = {}

    for lbl in sorted(label_to_indices.keys()):
        indices = label_to_indices[lbl]
        if not indices:
            continue

        T_min = 10**12
        for idx in indices:
            sample = dataset[idx]
            T_min = min(T_min, sample["u"].shape[0])

        if T_min <= 0 or T_min > 1000000:
            continue

        S_series = np.zeros((T_min,), dtype=np.float64)
        F_series = np.zeros((T_min,), dtype=np.float64)
        T_series = np.zeros((T_min,), dtype=np.float64)

        for idx in indices:
            sample = dataset[idx]
            # u, fr, t_ref: (ticks, neurons)
            # Average over all neurons at each tick
            u = sample["u"][:T_min, :]
            fr = sample["fr"][:T_min, :]
            tr = sample["t_ref"][:T_min, :]

            # Mean across neurons (axis 1)
            S_series += np.mean(u, axis=1)
            F_series += np.mean(fr, axis=1)
            T_series += np.mean(tr, axis=1)

        S_series /= max(1, len(indices))
        F_series /= max(1, len(indices))
        T_series /= max(1, len(indices))

        result[int(lbl)] = {
            "S": S_series.astype(np.float32),
            "F": F_series.astype(np.float32),
            "T": T_series.astype(np.float32),
        }

    if result:
        labels_sorted = np.array(sorted(result.keys()), dtype=np.int32)
        arrays: List[np.ndarray] = []
        for lbl in labels_sorted:
            arrays.extend(
                [result[int(lbl)]["S"], result[int(lbl)]["F"], result[int(lbl)]["T"]]
            )
        np.savez_compressed(cache_path, labels_sorted, *arrays)

    return result


def plot_phase_portrait(
    series_by_class: Dict[int, Dict[str, np.ndarray]],
    title_prefix: str,
    out_dir: str,
    theory_c: Optional[float] = None,
    theory_syn_per_layer: Optional[List[int]] = None,
) -> None:
    if not series_by_class:
        return
    log_plot_start("phase_portrait", "aggregate")
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, lbl in enumerate(sorted(series_by_class.keys())):
        s = series_by_class[lbl]["S"]
        f = series_by_class[lbl]["F"]
        t = series_by_class[lbl]["T"]
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter3d(
                x=s,
                y=f,
                z=t,
                mode="lines+markers",
                name=f"Class {lbl}",
                line=dict(color=color, width=3),
                marker=dict(size=3, color=color),
            )
        )
    # Optionally set z-axis bounds if theoretical bounds are available
    zrange = None
    if (
        theory_c is not None
        and theory_syn_per_layer is not None
        and len(theory_syn_per_layer) > 0
    ):
        lower = 2.0 * float(theory_c)
        upper = float(theory_c) * float(
            np.mean(np.asarray(theory_syn_per_layer, dtype=np.float32))
        )
        zrange = [lower, upper]

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – Phase Portrait by Digit (⟨S⟩, ⟨F̄⟩, ⟨t_ref⟩)",
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="⟨S⟩",
            yaxis_title="⟨F̄⟩",
            zaxis_title="⟨t_ref⟩",
            zaxis=dict(range=zrange) if zrange is not None else None,
        ),
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )
    save_figure(fig, out_dir, "phase_portrait_3d")
    log_plot_end("phase_portrait", "aggregate")
    # CSV output: class, t, mean_S, mean_F, mean_tref
    rows: List[List[Any]] = []
    for lbl, series in series_by_class.items():
        S = series["S"]
        F = series["F"]
        T = series["T"]
        for i in range(len(S)):
            rows.append([int(lbl), i, float(S[i]), float(F[i]), float(T[i])])
    save_csv(
        rows,
        ["digit", "t", "mean_S", "mean_F", "mean_tref"],
        out_dir,
        "phase_portrait_data",
    )


def extract_homeostat_from_network(
    network_path: str, expected_layers: int
) -> Tuple[Optional[float], Optional[List[int]]]:
    """Parse a network JSON file to extract c (homeostatic constant) and synapses per layer.

    - c: Prefer global_params.c if present; else average of neuron.params.c
    - synapses per layer: average number of postsynaptic points per neuron in each layer
      (based on neuron metadata.layer). If layer labels are missing or mismatched, return global
      average replicated across expected_layers.
    """
    try:
        with open(network_path, "r") as f:
            cfg = json.load(f)
    except Exception:
        return None, None

    # Extract c
    c_candidates: List[float] = []
    gparams = cfg.get("global_params", {}) or {}
    if isinstance(gparams, dict) and "c" in gparams:
        c_raw = gparams.get("c")
        try:
            c_num = float(c_raw) if c_raw is not None else None
        except Exception:
            c_num = None
        if c_num is not None and np.isfinite(c_num):
            c_candidates.append(float(c_num))
    for neuron in cfg.get("neurons", []) or []:
        params = neuron.get("params", {}) or {}
        if "c" in params:
            c_raw = params.get("c")
            try:
                c_num = float(c_raw) if c_raw is not None else None
            except Exception:
                c_num = None
            if c_num is not None and np.isfinite(c_num):
                c_candidates.append(float(c_num))
    c_value: Optional[float] = None
    if c_candidates:
        # Prefer global if present; else mean of neuron-level
        if "c" in gparams:
            raw = gparams.get("c")
            try:
                c_tmp = float(raw) if raw is not None else None
            except Exception:
                c_tmp = None
            if c_tmp is not None and np.isfinite(c_tmp):
                c_value = float(c_tmp)
            else:
                c_value = float(np.mean(np.asarray(c_candidates, dtype=np.float32)))
        else:
            c_value = float(np.mean(np.asarray(c_candidates, dtype=np.float32)))

    # Build per-neuron synapse counts
    syn_count_by_neuron: Dict[int, int] = {}
    for syn in cfg.get("synaptic_points", []) or []:
        try:
            syn_type = syn.get("type")
        except Exception:
            syn_type = None
        if syn_type == "postsynaptic":
            nid_any = syn.get("neuron_id")
            try:
                nid_int = int(nid_any)
            except Exception:
                continue
            syn_count_by_neuron[nid_int] = syn_count_by_neuron.get(nid_int, 0) + 1

    # Map neuron -> layer
    layer_by_neuron: Dict[int, int] = {}
    for neuron in cfg.get("neurons", []) or []:
        nid_any = neuron.get("id")
        try:
            nid = int(nid_any)
        except Exception:
            continue
        meta = neuron.get("metadata", {}) or {}
        li_any = meta.get("layer", None)
        try:
            li = int(li_any)
            layer_by_neuron[nid] = li
        except Exception:
            pass

    # Compute per-layer averages
    if layer_by_neuron:
        max_layer = max(layer_by_neuron.values()) if layer_by_neuron else -1
        layer_syn_sums = [0] * (max_layer + 1)
        layer_counts = [0] * (max_layer + 1)
        for nid, li in layer_by_neuron.items():
            cnt = syn_count_by_neuron.get(nid, 0)
            if 0 <= li <= max_layer:
                layer_syn_sums[li] += int(cnt)
                layer_counts[li] += 1
        per_layer: List[int] = []
        for li in range(max_layer + 1):
            if layer_counts[li] > 0:
                per_layer.append(int(round(layer_syn_sums[li] / layer_counts[li])))
            else:
                per_layer.append(0)
        # Normalize to expected_layers length
        if expected_layers > 0:
            if len(per_layer) < expected_layers:
                if per_layer:
                    avg_all = int(
                        round(np.mean(np.asarray(per_layer, dtype=np.float32)))
                    )
                else:
                    avg_all = 0
                per_layer.extend(
                    [avg_all for _ in range(expected_layers - len(per_layer))]
                )
            else:
                per_layer = per_layer[:expected_layers]
        syn_per_layer: Optional[List[int]] = per_layer
    else:
        # No layer metadata; use global average
        if syn_count_by_neuron:
            avg_all = int(
                round(
                    np.mean(
                        np.asarray(list(syn_count_by_neuron.values()), dtype=np.float32)
                    )
                )
            )
        else:
            avg_all = 0
        if expected_layers > 0:
            syn_per_layer = [avg_all for _ in range(expected_layers)]
        else:
            syn_per_layer = None

    return c_value, syn_per_layer


def parse_synapses_per_layer(
    arg_value: Optional[str], num_layers: int, default_value: int
) -> List[int]:
    if not arg_value:
        return [default_value for _ in range(num_layers)]
    parts = [p.strip() for p in arg_value.split(",") if p.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(int(p))
        except Exception:
            vals.append(default_value)
    if len(vals) < num_layers:
        vals.extend([default_value for _ in range(num_layers - len(vals))])
    return vals[:num_layers]


def compute_tref_samples_by_layer(
    dataset: LazyActivityDataset,
    neurons_per_layer: List[int],
    sample_limit_per_layer: int,
    cache_dir: str,
    dataset_hash: str,
    indices: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    """Reservoir-sample t_ref values per layer across all ticks and images.

    Returns mapping: layer_idx -> sampled t_ref values (np.ndarray).
    """
    suff = f"_sub{len(indices)}" if indices is not None else ""
    cache_path = os.path.join(
        cache_dir,
        f"{dataset_hash}_tref_samples_per_layer_{sample_limit_per_layer}{suff}.npz",
    )
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        if "arr_0" in data.files and all(k.startswith("arr_") for k in data.files):
            layer_keys = data["arr_0"].astype(np.int32)
            out = {}
            for i, lk in enumerate(layer_keys, start=1):
                key = f"arr_{i}"
                if key in data.files:
                    out[int(lk)] = data[key]
            return out
        # Legacy dict-like save
        return {int(k): data[k] for k in data.files}

    rng = np.random.default_rng(42)

    # Layer slices
    layer_slices = []
    offset = 0
    for n in neurons_per_layer:
        layer_slices.append(slice(offset, offset + n))
        offset += n

    reservoirs: Dict[int, np.ndarray] = {}  # Start with empty

    # We will buffer values and subsample occasionally
    buffers: Dict[int, List[np.ndarray]] = {
        li: [] for li in range(len(neurons_per_layer))
    }

    # Process in chunks to avoid memory explosion
    chunk_size = 100
    iterator = indices if indices is not None else range(len(dataset))

    # We need to wrap iterator in tqdm. If it's a list, we can just wrap it.
    iterable = tqdm(iterator, desc="Sampling t_ref per layer")

    for i in iterable:
        sample = dataset[i]
        tr = sample["t_ref"]  # (ticks, neurons)

        for li, sl in enumerate(layer_slices):
            # Extract layer columns
            sub = tr[:, sl]
            # Flatten
            vals = sub.reshape(-1)
            # Filter NaNs? t_ref usually float.
            buffers[li].append(vals)

        if (i + 1) % chunk_size == 0 or (i + 1) == len(dataset):
            # Compress buffers
            for li in buffers:
                if not buffers[li]:
                    continue
                combined = np.concatenate(buffers[li])
                buffers[li] = []  # Clear buffer

                # Merge with existing reservoir
                if li in reservoirs:
                    combined = np.concatenate([reservoirs[li], combined])

                # Subsample if too large
                if combined.shape[0] > sample_limit_per_layer:
                    idx = rng.choice(
                        combined.shape[0], size=sample_limit_per_layer, replace=False
                    )
                    combined = combined[idx]

                reservoirs[li] = combined

    # Final result
    result = {
        li: reservoirs.get(li, np.array([], dtype=np.float32))
        for li in range(len(neurons_per_layer))
    }

    if result:
        # Save as positional arrays: arr_0 = layer_indices, arr_1.. = values
        layer_keys = np.array(sorted(result.keys()), dtype=np.int32)
        arrays = [result[int(k)] for k in layer_keys]
        np.savez_compressed(cache_path, layer_keys, *arrays)
    return result


def plot_tref_bounds_box(
    tref_samples_by_layer: Dict[int, np.ndarray],
    c_value: float,
    synapses_per_layer: List[int],
    title_prefix: str,
    out_dir: str,
) -> None:
    if not tref_samples_by_layer:
        return
    log_plot_start("tref_bounds_box", "aggregate")
    layers_sorted = sorted(tref_samples_by_layer.keys())
    fig = go.Figure()

    # Box per layer
    for li in layers_sorted:
        vals = tref_samples_by_layer[li]
        fig.add_trace(
            go.Box(
                y=vals,
                name=f"Layer {li}",
                boxpoints=False,
                marker_color=px.colors.qualitative.Plotly[
                    li % len(px.colors.qualitative.Plotly)
                ],
            )
        )

    # Overlay theoretical bounds as lines across layer indices
    x_positions = list(range(1, len(layers_sorted) + 1))
    lower_vals = [2.0 * c_value for _ in layers_sorted]
    upper_vals = [c_value * float(synapses_per_layer[li]) for li in layers_sorted]

    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=lower_vals,
            mode="lines+markers",
            name="Lower bound (2c)",
            line=dict(color="#10b981", width=2, dash="dash"),
            marker=dict(size=6, color="#10b981"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=upper_vals,
            mode="lines+markers",
            name="Upper bound (c×|synapses|)",
            line=dict(color="#ef4444", width=2, dash="dash"),
            marker=dict(size=6, color="#ef4444"),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – t_ref Bounds Compliance", font=dict(size=18)
        ),
        xaxis_title="Layer",
        yaxis_title="t_ref",
        width=1400,
        height=900,
        hovermode="closest",
        boxmode="group",
    )
    save_figure(fig, out_dir, "tref_bounds_box")
    log_plot_end("tref_bounds_box", "aggregate")
    # CSV output: layer, t_ref_sample
    rows: List[List[Any]] = []
    for li, arr in sorted(tref_samples_by_layer.items()):
        for v in arr:
            rows.append([int(li), float(v)])
    save_csv(rows, ["layer", "t_ref"], out_dir, "tref_bounds_box_data")


def compute_favg_stability_over_epochs(
    label_to_indices: Dict[int, List[int]],
    dataset: LazyActivityDataset,
    epoch_field: str,
    num_epoch_bins: int,
    cache_dir: str,
    dataset_hash: str,
    indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean F_avg per epoch.

    If epoch_field exists in sample metadata, group by that.
    Otherwise, if num_epoch_bins > 0, bin images sequentially by label,image_index order.
    Returns (epochs, means) arrays.
    """
    suff = f"_sub{len(indices)}" if indices is not None else ""
    cache_key = (
        f"{dataset_hash}_favg_stability_{epoch_field}_bins{num_epoch_bins}{suff}.npz"
    )
    cache_path = os.path.join(cache_dir, cache_key)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["epochs"], data["means"]

    # Detect epoch field availability in first few samples
    # If indices provided, check those
    check_indices = (
        indices[:10] if indices is not None else range(min(10, len(dataset)))
    )
    has_epoch = False
    for i in check_indices:
        sample = dataset[i]
        if epoch_field in sample:
            has_epoch = True
            break

    epoch_to_sum = {}
    epoch_to_count = {}

    if has_epoch:
        iterator = indices if indices is not None else range(len(dataset))
        for i in tqdm(iterator, desc="Computing F_avg stability (by epoch)"):
            sample = dataset[i]
            val = sample.get(epoch_field)
            try:
                ep = int(val) if val is not None else None
            except Exception:
                ep = None
            if ep is None:
                continue

            # Use mean of "fr"
            fr = sample["fr"]  # (ticks, neurons)
            if fr.size > 0:
                epoch_to_sum[ep] = epoch_to_sum.get(ep, 0.0) + float(np.mean(fr))
                epoch_to_count[ep] = epoch_to_count.get(ep, 0) + 1

    elif num_epoch_bins and num_epoch_bins > 0:
        # Create ordered list of images (label, index)
        if indices is not None:
            sorted_indices = indices
        else:
            sorted_indices = []
            for lbl in sorted(label_to_indices.keys()):
                sorted_indices.extend(label_to_indices[lbl])

        if not sorted_indices:
            epochs_arr = np.zeros((0,), dtype=np.int32)
            means_arr = np.zeros((0,), dtype=np.float32)
            np.savez_compressed(cache_path, epochs=epochs_arr, means=means_arr)
            return epochs_arr, means_arr

        images_per_bin = max(1, int(np.ceil(len(sorted_indices) / num_epoch_bins)))

        for bi in range(num_epoch_bins):
            start = bi * images_per_bin
            stop = min(len(sorted_indices), (bi + 1) * images_per_bin)
            if start >= stop:
                break
            ep = int(bi + 1)

            bin_indices = sorted_indices[start:stop]
            for idx in bin_indices:
                sample = dataset[idx]
                fr = sample["fr"]
                if fr.size > 0:
                    epoch_to_sum[ep] = epoch_to_sum.get(ep, 0.0) + float(np.mean(fr))
                    epoch_to_count[ep] = epoch_to_count.get(ep, 0) + 1
    else:
        # No epoch data available
        epochs_arr = np.zeros((0,), dtype=np.int32)
        means_arr = np.zeros((0,), dtype=np.float32)
        np.savez_compressed(cache_path, epochs=epochs_arr, means=means_arr)
        return epochs_arr, means_arr

    if not epoch_to_sum:
        epochs_arr = np.zeros((0,), dtype=np.int32)
        means_arr = np.zeros((0,), dtype=np.float32)
        np.savez_compressed(cache_path, epochs=epochs_arr, means=means_arr)
        return epochs_arr, means_arr

    epochs_sorted = sorted(epoch_to_sum.keys())
    means = [epoch_to_sum[e] / max(1, epoch_to_count.get(e, 1)) for e in epochs_sorted]
    epochs_arr = np.asarray(epochs_sorted, dtype=np.int32)
    means_arr = np.asarray(means, dtype=np.float32)
    np.savez_compressed(cache_path, epochs=epochs_arr, means=means_arr)
    return epochs_arr, means_arr


def plot_favg_stability(
    epochs: np.ndarray, means: np.ndarray, title_prefix: str, out_dir: str
) -> None:
    if epochs.size == 0:
        # Still log start/end to make it obvious a no-op occurred
        log_plot_start("favg_stability", "aggregate (empty)")
        log_plot_end("favg_stability", "aggregate (empty)")
        return
    log_plot_start("favg_stability", "aggregate")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=means,
            mode="lines+markers",
            name="⟨F̄⟩",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=6, color="#1d4ed8"),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – F̄ Stability Over Training", font=dict(size=18)
        ),
        xaxis_title="Epoch",
        yaxis_title="Mean F̄",
        width=1400,
        height=900,
        hovermode="closest",
    )
    save_figure(fig, out_dir, "favg_stability_over_epochs")
    log_plot_end("favg_stability", "aggregate")
    # CSV output: epoch, mean_F
    rows: List[List[Any]] = [
        [int(epochs[i]), float(means[i])] for i in range(len(epochs))
    ]
    save_csv(rows, ["epoch", "mean_F"], out_dir, "favg_stability_over_epochs_data")


def plot_homeostatic_response_curve(
    aggregates: Dict[str, np.ndarray],
    c_value: float,
    synapses_per_layer: List[int],
    bins: int,
    scope: str,
    title_prefix: str,
    out_dir: str,
) -> None:
    log_plot_start("homeostatic_response", "aggregate")
    mean_favg = aggregates["mean_favg"]
    mean_tref = aggregates["mean_tref"]
    layer_index = aggregates["layer_index"]

    # Compute binned empirical curve
    f_min = float(np.min(mean_favg)) if mean_favg.size else 0.0
    f_max = float(np.max(mean_favg)) if mean_favg.size else 1.0
    if f_max <= f_min:
        f_max = f_min + 1e-6
    edges = np.linspace(f_min, f_max, max(2, bins + 1))
    # Midpoints of consecutive bin edges
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig = go.Figure()

    if scope == "per-layer":
        layers = sorted(set(layer_index.tolist()))
        palette = px.colors.qualitative.Plotly
        for li in layers:
            mask = layer_index == li
            f_vals = mean_favg[mask]
            t_vals = mean_tref[mask]
            if f_vals.size == 0:
                continue
            bin_indices = np.clip(np.digitize(f_vals, edges) - 1, 0, len(edges) - 2)
            sums = np.zeros((len(edges) - 1,), dtype=np.float64)
            counts = np.zeros((len(edges) - 1,), dtype=np.int64)
            for i_b, t_v in zip(bin_indices, t_vals):
                sums[int(i_b)] += float(t_v)
                counts[int(i_b)] += 1
            means = np.divide(sums, np.maximum(counts, 1))
            fig.add_trace(
                go.Scatter(
                    x=centers,
                    y=means,
                    mode="lines+markers",
                    name=f"Empirical L{li}",
                    line=dict(color=palette[li % len(palette)], width=2),
                    marker=dict(size=5),
                )
            )
            # Theoretical curve per layer
            lower = 2.0 * c_value
            upper = c_value * float(synapses_per_layer[li])
            x_line = np.linspace(f_min, f_max, 200)
            y_line = upper - (upper - lower) * (x_line * c_value)
            y_line = np.clip(y_line, lower, upper)
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"Theory L{li}",
                    line=dict(color=palette[li % len(palette)], width=2, dash="dash"),
                )
            )
    else:
        # Global empirical curve
        bin_indices = np.clip(np.digitize(mean_favg, edges) - 1, 0, len(edges) - 2)
        sums = np.zeros((len(edges) - 1,), dtype=np.float64)
        counts = np.zeros((len(edges) - 1,), dtype=np.int64)
        for i_b, t_v in zip(bin_indices, mean_tref):
            sums[int(i_b)] += float(t_v)
            counts[int(i_b)] += 1
        means = np.divide(sums, np.maximum(counts, 1))
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=means,
                mode="lines+markers",
                name="Empirical",
                line=dict(color="#111827", width=3),
                marker=dict(size=6, color="#1f2937"),
            )
        )
        # Global theory curve using average synapses
        avg_syn = (
            float(np.mean(np.asarray(synapses_per_layer, dtype=np.float32)))
            if synapses_per_layer
            else 1.0
        )
        lower = 2.0 * c_value
        upper = c_value * avg_syn
        x_line = np.linspace(f_min, f_max, 200)
        y_line = upper - (upper - lower) * (x_line * c_value)
        y_line = np.clip(y_line, lower, upper)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Theory",
                line=dict(color="#9333ea", width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – Homeostatic Response Curve", font=dict(size=18)
        ),
        xaxis_title="Mean F̄",
        yaxis_title="Mean t_ref",
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )
    save_figure(fig, out_dir, "homeostatic_response_curve")
    log_plot_end("homeostatic_response", "aggregate")
    # CSV output: centers, empirical_mean_tref (global), and/or per-layer series
    rows: List[List[Any]] = []
    if scope == "per-layer":
        # Save per-layer empirical curve values
        # Note: recompute bin means like in the plot for consistency
        mean_favg = aggregates["mean_favg"]
        mean_tref = aggregates["mean_tref"]
        layer_index = aggregates["layer_index"]
        f_min = float(np.min(mean_favg)) if mean_favg.size else 0.0
        f_max = float(np.max(mean_favg)) if mean_favg.size else 1.0
        if f_max <= f_min:
            f_max = f_min + 1e-6
        edges = np.linspace(f_min, f_max, max(2, bins + 1))
        centers = 0.5 * (edges[:-1] + edges[1:])
        for li in sorted(set(layer_index.tolist())):
            mask = layer_index == li
            f_vals = mean_favg[mask]
            t_vals = mean_tref[mask]
            bin_indices = np.clip(np.digitize(f_vals, edges) - 1, 0, len(edges) - 2)
            sums = np.zeros((len(edges) - 1,), dtype=np.float64)
            counts = np.zeros((len(edges) - 1,), dtype=np.int64)
            for i_b, t_v in zip(bin_indices, t_vals):
                sums[int(i_b)] += float(t_v)
                counts[int(i_b)] += 1
            means = np.divide(sums, np.maximum(counts, 1))
            for i_c, c in enumerate(centers):
                rows.append([int(li), float(c), float(means[i_c])])
        save_csv(
            rows,
            ["layer", "F_center", "mean_t_ref"],
            out_dir,
            "homeostatic_response_curve_data",
        )
    else:
        mean_favg = aggregates["mean_favg"]
        mean_tref = aggregates["mean_tref"]
        f_min = float(np.min(mean_favg)) if mean_favg.size else 0.0
        f_max = float(np.max(mean_favg)) if mean_favg.size else 1.0
        if f_max <= f_min:
            f_max = f_min + 1e-6
        edges = np.linspace(f_min, f_max, max(2, bins + 1))
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_indices = np.clip(np.digitize(mean_favg, edges) - 1, 0, len(edges) - 2)
        sums = np.zeros((len(edges) - 1,), dtype=np.float64)
        counts = np.zeros((len(edges) - 1,), dtype=np.int64)
        for i_b, t_v in zip(bin_indices, mean_tref):
            sums[int(i_b)] += float(t_v)
            counts[int(i_b)] += 1
        means = np.divide(sums, np.maximum(counts, 1))
        for i_c, c in enumerate(centers):
            rows.append([float(c), float(means[i_c])])
        save_csv(
            rows, ["F_center", "mean_t_ref"], out_dir, "homeostatic_response_curve_data"
        )


def compute_s_variance_decay(
    label_to_indices: Dict[int, List[int]],
    dataset: LazyActivityDataset,
    neurons_per_layer: List[int],
    layer_offsets: List[int],
    total_neurons: int,
    num_classes: int,
    cache_dir: str,
    dataset_hash: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute S(t) variance across all neurons vs time, per digit class.

    Returns (labels_sorted, series_matrix) with shape [C, T], where T is the
    global minimum number of ticks across all labels/images, and C is the
    number of classes found up to num_classes.
    """
    cache_path = os.path.join(
        cache_dir, f"{dataset_hash}_svar_decay_c{num_classes}.npz"
    )
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["labels"], data["series"]

    labels_sorted = np.array(sorted(label_to_indices.keys()), dtype=np.int32)
    if labels_sorted.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 0), dtype=np.float32)

    # Determine global minimal T across all labels/images
    global_T = 10**12
    found_any = False
    for lbl in labels_sorted:
        indices = label_to_indices[int(lbl)]
        if not indices:
            continue
        for idx in indices:
            sample = dataset[idx]
            global_T = min(global_T, sample["u"].shape[0])
            found_any = True

    if not found_any or global_T <= 0 or global_T > 1000000:
        return labels_sorted, np.zeros((labels_sorted.size, 0), dtype=np.float32)

    global_T = int(global_T)
    # Compute average variance across images for each label at each t
    series = np.zeros((labels_sorted.size, global_T), dtype=np.float32)
    for li_lbl, lbl in enumerate(labels_sorted):
        indices = label_to_indices[int(lbl)]
        if not indices:
            continue

        accum = np.zeros((global_T,), dtype=np.float64)
        count = 0

        for idx in indices:
            sample = dataset[idx]
            u = sample["u"]  # (ticks, neurons)
            if u.shape[0] < global_T:
                continue

            # Compute var at each tick
            # u[:global_T, :] -> (T, neurons)
            # var across neurons (axis 1)
            # We want variance of S across all neurons at time t.
            # u is already float.
            v = np.var(u[:global_T, :total_neurons], axis=1)
            accum += v
            count += 1

        if count > 0:
            series[li_lbl, :] = (accum / float(count)).astype(np.float32)

    np.savez_compressed(cache_path, labels=labels_sorted, series=series)
    return labels_sorted, series


def plot_s_variance_decay(
    labels: np.ndarray,
    series: np.ndarray,
    title_prefix: str,
    out_dir: str,
) -> None:
    if labels.size == 0 or series.size == 0:
        return
    T = series.shape[1]
    t_axis = list(range(T))
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, lbl in enumerate(labels.tolist()):
        fig.add_trace(
            go.Scatter(
                x=t_axis,
                y=series[i, :],
                mode="lines",
                name=f"Digit {int(lbl)}",
                line=dict(color=palette[i % len(palette)], width=2),
            )
        )
    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – S(t) Variance Decay by Digit", font=dict(size=18)
        ),
        xaxis_title="t",
        yaxis_title="Var[S] across neurons",
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )
    save_figure(fig, out_dir, "s_variance_decay")
    # CSV output: digit, t, var_S
    rows: List[List[Any]] = []
    for i, lbl in enumerate(labels.tolist()):
        for t in range(T):
            rows.append([int(lbl), int(t), float(series[i, t])])
    save_csv(rows, ["digit", "t", "var_S"], out_dir, "s_variance_decay_data")


def compute_affinity_matrix(
    label_to_indices: Dict[int, List[int]],
    dataset: LazyActivityDataset,
    neurons_per_layer: List[int],
    layer_offsets: List[int],
    total_neurons: int,
    num_classes: int,
    cache_dir: str,
    dataset_hash: str,
) -> np.ndarray:
    """Compute neuron-digit affinity matrix: rows=neurons, cols=digits, value=mean S when digit shown."""
    cache_path = os.path.join(cache_dir, f"{dataset_hash}_affinity_c{num_classes}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["affinity"]

    sums = np.zeros((total_neurons, num_classes), dtype=np.float64)
    counts = np.zeros((total_neurons, num_classes), dtype=np.int64)

    for lbl in sorted(label_to_indices.keys()):
        if not (0 <= int(lbl) < num_classes):
            continue
        indices = label_to_indices[lbl]
        for idx in indices:
            sample = dataset[idx]
            u = sample["u"]  # (ticks, neurons) -> S?
            # Mean S per neuron for this image
            # u shape: (T, total_neurons) -> mean axis 0 -> (total_neurons,)
            s_mean = np.mean(u, axis=0)  # vector of length total_neurons

            # Add to sums for this label
            sums[:, int(lbl)] += s_mean
            counts[:, int(lbl)] += 1

    counts_safe = np.maximum(counts, 1)
    affinity = (sums / counts_safe).astype(np.float32)
    np.savez_compressed(cache_path, affinity=affinity)
    return affinity


def plot_affinity_heatmap(
    affinity: np.ndarray,
    neurons_per_layer: List[int],
    title_prefix: str,
    out_dir: str,
) -> None:
    if affinity.size == 0:
        return
    log_plot_start("affinity_heatmap", "aggregate")
    # Reorder rows by layer for readability
    order = []
    offset = 0
    for n in neurons_per_layer:
        order.extend(list(range(offset, offset + n)))
        offset += n
    aff_ordered = affinity[order, :]
    fig = go.Figure(
        data=go.Heatmap(z=aff_ordered, colorscale="Viridis", colorbar=dict(title="⟨S⟩"))
    )
    fig.update_layout(
        title=dict(text=f"{title_prefix} – Neuron–Digit Affinity", font=dict(size=18)),
        xaxis_title="Digit",
        yaxis_title="Neuron (by layer)",
        width=1400,
        height=900,
    )
    save_figure(fig, out_dir, "affinity_heatmap")
    log_plot_end("affinity_heatmap", "aggregate")
    # CSV output: neuron_id, digit, mean_S
    rows: List[List[Any]] = []
    for i in range(affinity.shape[0]):
        for d in range(affinity.shape[1]):
            rows.append([int(i), int(d), float(affinity[i, d])])
    save_csv(rows, ["neuron_id", "digit", "mean_S"], out_dir, "affinity_heatmap_data")


def plot_tref_by_preferred_digit(
    affinity: np.ndarray,
    aggregates: Dict[str, np.ndarray],
    title_prefix: str,
    out_dir: str,
) -> None:
    if affinity.size == 0:
        return
    log_plot_start("tref_by_preferred_digit", "aggregate")
    preferred = np.argmax(affinity, axis=1)
    mean_tref = aggregates["mean_tref"]
    # Group t_ref by preferred digit
    # Ensure all digits (columns) appear, even if no neuron prefers a given digit
    num_digits = int(affinity.shape[1]) if affinity.ndim == 2 else 0
    digits = (
        list(range(num_digits)) if num_digits > 0 else sorted(set(preferred.tolist()))
    )
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for d in digits:
        vals = mean_tref[preferred == d]
        if vals.size == 0:
            # Add placeholder trace so category shows up even with zero neurons
            fig.add_trace(
                go.Box(
                    y=[float("nan")],
                    name=f"Digit {int(d)}",
                    boxpoints=False,
                    marker_color=palette[int(d) % len(palette)],
                    showlegend=True,
                )
            )
        else:
            fig.add_trace(
                go.Box(
                    y=vals,
                    name=f"Digit {int(d)}",
                    boxpoints=False,
                    marker_color=palette[int(d) % len(palette)],
                )
            )
    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – t_ref by Preferred Digit", font=dict(size=18)
        ),
        xaxis_title="Preferred Digit",
        yaxis_title="t_ref",
        width=1400,
        height=900,
        boxmode="group",
    )
    save_figure(fig, out_dir, "tref_by_preferred_digit")
    log_plot_end("tref_by_preferred_digit", "aggregate")
    # CSV output: neuron_id, preferred_digit, mean_t_ref
    rows: List[List[Any]] = []
    for i in range(len(preferred)):
        rows.append([int(i), int(preferred[i]), float(mean_tref[i])])
    save_csv(
        rows,
        ["neuron_id", "preferred_digit", "mean_t_ref"],
        out_dir,
        "tref_by_preferred_digit_data",
    )


def compute_temporal_correlation_edges(
    label_to_indices: Dict[int, List[int]],
    dataset: LazyActivityDataset,
    neurons_per_layer: List[int],
    layer_offsets: List[int],
    total_neurons: int,
    max_neurons: int,
    threshold: float,
    gap: int,
    max_images: Optional[int],
    cache_dir: str,
    dataset_hash: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute correlation edges among selected neurons.

    Returns (node_indices, edge_u, edge_v) where edges are pairs of node indices in selected set.
    """
    cache_key = f"{dataset_hash}_corr_n{max_neurons}_t{threshold:.3f}_g{gap}_m{max_images or 0}.npz"
    cache_path = os.path.join(cache_dir, cache_key)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["nodes"], data["u"], data["v"]

    # Select neurons evenly across layers
    selected: List[int] = []
    per_layer_quota = max(1, max_neurons // max(1, len(neurons_per_layer)))
    for li, n in enumerate(neurons_per_layer):
        base = layer_offsets[li]
        take = min(n, per_layer_quota)
        selected.extend(list(range(base, base + take)))
        if len(selected) >= max_neurons:
            break
    selected = selected[:max_neurons]
    if not selected:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    if len(selected) < 2:
        nodes = np.asarray(selected, dtype=np.int32)
        u = np.array([], dtype=np.int32)
        v = np.array([], dtype=np.int32)
        np.savez_compressed(cache_path, nodes=nodes, u=u, v=v)
        return nodes, u, v

    # Build time series by concatenating images
    # Flatten ordering:
    sorted_indices = []
    for lbl in sorted(label_to_indices.keys()):
        sorted_indices.extend(label_to_indices[lbl])

    if max_images is not None and max_images > 0:
        sorted_indices = sorted_indices[:max_images]

    series_list: List[np.ndarray] = []

    for idx in sorted_indices:
        sample = dataset[idx]
        spikes = sample["spikes"]  # (N, 2)
        ticks = sample["u"].shape[0]

        # Create dense binary matrix for this image for selected neurons
        mat = np.zeros((ticks, len(selected)), dtype=np.float32)

        if spikes.shape[0] > 0:
            s_ticks = spikes[:, 0].astype(int)
            s_neurons = spikes[:, 1].astype(int)

            # Map global -> local index
            # This can be slow if done spike-by-spike.
            # Faster: Create lookup array.
            # But we can create it once outside.
            pass  # See optimization below

    # Optimization: Create lookup once
    neuron_to_selected_idx = np.full(total_neurons, -1, dtype=np.int32)
    neuron_to_selected_idx[selected] = np.arange(len(selected), dtype=np.int32)

    for idx in sorted_indices:
        sample = dataset[idx]
        spikes = sample["spikes"]
        ticks = sample["u"].shape[0]
        mat = np.zeros((ticks, len(selected)), dtype=np.float32)

        if spikes.shape[0] > 0:
            s_ticks = spikes[:, 0].astype(int)
            s_neurons = spikes[:, 1].astype(int)

            # Filter
            valid_mask = (s_neurons < total_neurons) & (
                neuron_to_selected_idx[s_neurons] >= 0
            )
            valid_ticks = s_ticks[valid_mask]
            valid_neurons = s_neurons[valid_mask]

            mapped = neuron_to_selected_idx[valid_neurons]

            # Clip ticks
            tick_mask = valid_ticks < ticks
            mat[valid_ticks[tick_mask], mapped[tick_mask]] = 1.0

        series_list.append(mat)
        if gap > 0:
            series_list.append(np.zeros((gap, len(selected)), dtype=np.float32))

    if not series_list:
        return (
            np.asarray(selected, dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    full_series = np.concatenate(series_list, axis=0)  # (TotalTicks, K)

    if full_series.shape[0] < 2:
        return (
            np.asarray(selected, dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    # np.corrcoef expects (N, T) variables as rows
    corr = np.corrcoef(full_series.T)
    np.fill_diagonal(corr, 0.0)

    u_list, v_list = np.where(corr > threshold)
    mask = u_list < v_list
    u_indices = u_list[mask]
    v_indices = v_list[mask]

    nodes = np.asarray(selected, dtype=np.int32)
    u_arr = u_indices.astype(np.int32)
    v_arr = v_indices.astype(np.int32)

    np.savez_compressed(cache_path, nodes=nodes, u=u_arr, v=v_arr)
    return nodes, u_arr, v_arr


def plot_temporal_correlation_graph(
    nodes: np.ndarray,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    neurons_per_layer: List[int],
    layer_offsets: List[int],
    aggregates: Dict[str, np.ndarray],
    title_prefix: str,
    out_dir: str,
) -> None:
    if nodes.size == 0:
        return
    log_plot_start("temporal_corr_graph", "aggregate")
    layer_index = aggregates["layer_index"]
    mean_tref = aggregates["mean_tref"]

    # Node positions: x = layer, y = index within layer normalized
    x_coords = []
    y_coords = []
    for nid in nodes:
        li = int(layer_index[nid])
        # position within layer
        base = layer_offsets[li]
        y = (int(nid) - base) / max(1, neurons_per_layer[li])
        x_coords.append(li)
        y_coords.append(y)

    node_colors = mean_tref[nodes]

    # Build edge line segments
    line_x: List[float] = []
    line_y: List[float] = []
    for a, b in zip(edges_u, edges_v):
        line_x.append(float(x_coords[int(a)]))
        line_x.append(float(x_coords[int(b)]))
        line_x.append(float("nan"))
        line_y.append(float(y_coords[int(a)]))
        line_y.append(float(y_coords[int(b)]))
        line_y.append(float("nan"))

    fig = go.Figure()
    # Edges
    if line_x:
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color="rgba(0,0,0,0.2)", width=1),
                hoverinfo="skip",
                name="correlation",
            )
        )
    # Nodes
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers",
            marker=dict(
                size=8,
                color=node_colors,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="t_ref"),
                line=dict(width=0.5, color="black"),
            ),
            text=[f"Neuron {int(n)} (L{int(layer_index[n])})" for n in nodes],
            hovertemplate="%{text}<extra></extra>",
            name="neurons",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – Temporal Correlation Graph", font=dict(size=18)
        ),
        xaxis_title="Layer",
        yaxis_title="Index within layer",
        width=1400,
        height=900,
        hovermode="closest",
    )

    save_figure(fig, out_dir, "temporal_corr_graph")
    log_plot_end("temporal_corr_graph", "aggregate")
    # CSV output: edges and nodes
    rows_nodes: List[List[Any]] = [
        [int(n), int(aggregates["layer_index"][n]), float(aggregates["mean_tref"][n])]
        for n in nodes
    ]
    save_csv(
        rows_nodes,
        ["neuron_id", "layer", "mean_t_ref"],
        out_dir,
        "temporal_corr_graph_nodes",
    )
    rows_edges: List[List[Any]] = [
        [int(nodes[int(a)]), int(nodes[int(b)])] for a, b in zip(edges_u, edges_v)
    ]
    save_csv(
        rows_edges,
        ["source_neuron", "target_neuron"],
        out_dir,
        "temporal_corr_graph_edges",
    )


def compute_convergence_points(
    label_to_indices: Dict[int, List[int]],
    dataset: LazyActivityDataset,
    num_classes: int,
    final_ticks: int,
    cache_dir: str,
    dataset_hash: str,
) -> Dict[int, List[Tuple[float, float]]]:
    """Compute convergence points (final state) per image per class.

    For each image, takes the average (⟨S⟩, ⟨t_ref⟩) over the last `final_ticks` ticks.
    Returns mapping: class -> list of (s_mean, t_mean) convergence points.
    """
    cache_path = os.path.join(
        cache_dir, f"{dataset_hash}_convergence_points_f{final_ticks}.npz"
    )
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        if "arr_0" in data.files:
            labels = data["arr_0"].astype(np.int32)
            out: Dict[int, List[Tuple[float, float]]] = {}
            idx = 1
            for lbl in labels:
                s_arr = data[f"arr_{idx}"]
                t_arr = data[f"arr_{idx + 1}"]
                out[int(lbl)] = list(zip(s_arr.tolist(), t_arr.tolist()))
                idx += 2
            return out

    convergence_by_class: Dict[int, List[Tuple[float, float]]] = {}

    for label in sorted(label_to_indices.keys()):
        if not (0 <= int(label) < num_classes):
            continue
        indices = label_to_indices[label]
        if not indices:
            continue

        points_local: List[Tuple[float, float]] = []

        for idx in indices:
            sample = dataset[idx]
            u = sample[
                "u"
            ]  # (ticks, neurons) -> S values are implicitly u if using rate?
            # Wait, S is not u. u is membrane potential? NO.
            # In ActivityRecorder, u is usually potential. Fired is binary.
            # LazyActivityDataset: "u" is float potential? Or S?
            # If load_activity_data_binary saves "u", it assumes u.
            # BUT the old buckets had "S" in layers.
            # "S" usually means "Synaptic drive" or "S value" in this model.
            # Check LazyActivityDataset keys.
            # "u", "fired", "t_ref", "fr", "label".
            # Does "u" correspond to "S"?
            # In the original buckets, 'S' was stored. 'u' was apparently separate?
            # Let's check compute_layerwise_S_timeline (895): it uses sample["u"].
            # So "u" is treated as "S" here?
            # "layer_u = u[:T_min_all, sl]" -> sums[li] += mean(layer_u).
            # Output is "Layer-wise S(t) Average".
            # So yes, sample["u"] is likely S.

            # What about t_ref? sample["t_ref"].

            tr = sample["t_ref"]
            ticks = u.shape[0]

            # Take last final_ticks
            start = max(0, ticks - final_ticks)
            if start >= ticks:
                continue

            s_tail = u[start:, :]
            t_tail = tr[start:, :]

            if s_tail.size > 0 and t_tail.size > 0:
                s_mean = float(np.mean(s_tail))
                t_mean = float(np.mean(t_tail))
                points_local.append((s_mean, t_mean))

        if points_local:
            convergence_by_class[int(label)] = points_local

    # Save cache
    if convergence_by_class:
        labels_sorted = np.array(sorted(convergence_by_class.keys()), dtype=np.int32)
        arrays: List[np.ndarray] = []
        for lbl in labels_sorted:
            pts = convergence_by_class[int(lbl)]
            s_arr = np.array([p[0] for p in pts], dtype=np.float32)
            t_arr = np.array([p[1] for p in pts], dtype=np.float32)
            arrays.extend([s_arr, t_arr])
        np.savez_compressed(cache_path, labels_sorted, *arrays)

    return convergence_by_class


def compute_attractor_landscapes(
    label_to_indices: Dict[int, List[int]],
    dataset: LazyActivityDataset,
    num_classes: int,
    cache_dir: str,
    dataset_hash: str,
    grid_size: int = 80,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """Compute per-digit 2D energy landscapes over (⟨S⟩, ⟨t_ref⟩).

    Returns (x_edges, y_edges, energy_by_class) where each surface is shape [y, x].
    Energy is defined as -log(normalized density + eps).
    """
    cache_path = os.path.join(
        cache_dir, f"{dataset_hash}_attractor_landscape_g{int(grid_size)}.npz"
    )
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        # Positional arrays: arr_0 = labels, arr_1 = x_edges, arr_2 = y_edges, arr_3.. = per-label surfaces
        if "arr_0" in data.files and all(k.startswith("arr_") for k in data.files):
            labels = data["arr_0"].astype(np.int32)
            x_edges = data["arr_1"].astype(np.float32)
            y_edges = data["arr_2"].astype(np.float32)
            out: Dict[int, np.ndarray] = {}
            for i, lbl in enumerate(labels, start=3):
                key = f"arr_{i}"
                if key in data.files:
                    out[int(lbl)] = data[key]
            return x_edges, y_edges, out

    # Gather per-tick (⟨S⟩, ⟨t_ref⟩) points per class
    points_by_class: Dict[int, List[Tuple[float, float]]] = {}
    s_all: List[float] = []
    t_all: List[float] = []

    for label in sorted(label_to_indices.keys()):
        if not (0 <= int(label) < num_classes):
            continue
        indices = label_to_indices[label]
        for idx in indices:
            sample = dataset[idx]
            u = sample["u"]  # (ticks, neurons) -> S?
            tr = sample["t_ref"]  # (ticks, neurons)

            # Average over neurons (axis 1) -> (ticks,)
            s_t = np.mean(u, axis=1)
            t_t = np.mean(tr, axis=1)

            # Combine to (S, T) pairs
            count = len(s_t)
            if count > 0:
                # Flatten
                for k in range(count):
                    s_val = float(s_t[k])
                    t_val = float(t_t[k])
                    points_by_class.setdefault(int(label), []).append((s_val, t_val))
                    s_all.append(s_val)
                    t_all.append(t_val)

    if not s_all or not t_all:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), {}

    # Define global grid edges (shared across classes)
    s_min = float(np.min(np.asarray(s_all, dtype=np.float64)))
    s_max = float(np.max(np.asarray(s_all, dtype=np.float64)))
    t_min = float(np.min(np.asarray(t_all, dtype=np.float64)))
    t_max = float(np.max(np.asarray(t_all, dtype=np.float64)))
    # Add small margins
    s_pad = 1e-6 if s_max == s_min else 0.02 * (s_max - s_min)
    t_pad = 1e-6 if t_max == t_min else 0.02 * (t_max - t_min)
    x_edges = np.linspace(
        s_min - s_pad, s_max + s_pad, int(grid_size) + 1, dtype=np.float32
    )
    y_edges = np.linspace(
        t_min - t_pad, t_max + t_pad, int(grid_size) + 1, dtype=np.float32
    )

    def smooth_counts(counts: np.ndarray) -> np.ndarray:
        # Simple edge-aware 3x3 smoothing kernel
        k = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 1]], dtype=np.float32)
        h, w = counts.shape
        out_sm = np.zeros_like(counts, dtype=np.float32)
        for i in range(h):
            i0 = max(0, i - 1)
            i1 = min(h, i + 2)
            for j in range(w):
                j0 = max(0, j - 1)
                j1 = min(w, j + 2)
                window = counts[i0:i1, j0:j1]
                ki0 = 1 - (i - i0)
                ki1 = 2 + (i1 - (i + 1))
                kj0 = 1 - (j - j0)
                kj1 = 2 + (j1 - (j + 1))
                kern = k[ki0:ki1, kj0:kj1]
                denom = float(np.sum(kern)) if np.sum(kern) > 0 else 1.0
                out_sm[i, j] = float(np.sum(window * kern) / denom)
        return out_sm

    energy_by_class: Dict[int, np.ndarray] = {}
    eps = 1e-8
    for lbl, pts in points_by_class.items():
        if not pts:
            continue
        s_vals_arr = np.asarray([p[0] for p in pts], dtype=np.float32)
        t_vals_arr = np.asarray([p[1] for p in pts], dtype=np.float32)
        H, x_ed, y_ed = np.histogram2d(s_vals_arr, t_vals_arr, bins=[x_edges, y_edges])
        # Histogram2d returns H with shape [len(x_edges)-1, len(y_edges)-1]
        counts = H.T.astype(np.float32)
        counts = smooth_counts(counts)
        total = float(np.sum(counts))
        density = counts / total if total > 0 else counts
        energy = -np.log(density + eps)
        energy_by_class[int(lbl)] = energy.astype(np.float32)

    # Save cache (labels, x_edges, y_edges, per-label energy surfaces)
    if energy_by_class:
        labels_sorted = np.array(sorted(energy_by_class.keys()), dtype=np.int32)
        arrays: List[np.ndarray] = [
            x_edges.astype(np.float32),
            y_edges.astype(np.float32),
        ]
        arrays.extend([energy_by_class[int(lbl)] for lbl in labels_sorted])
        np.savez_compressed(cache_path, labels_sorted, *arrays)

    return x_edges, y_edges, energy_by_class


def plot_attractor_landscape_overlay(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    energy_by_class: Dict[int, np.ndarray],
    title_prefix: str,
    out_dir: str,
) -> None:
    if not energy_by_class:
        return
    log_plot_start("attractor_landscape", "aggregate (overlay)")
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, lbl in enumerate(sorted(energy_by_class.keys())):
        z = energy_by_class[int(lbl)]
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Contour(
                x=x_centers,
                y=y_centers,
                z=z,
                contours=dict(coloring="lines", showlines=True),
                line=dict(color=color, width=2),
                showscale=False,
                name=f"Digit {int(lbl)}",
                hoverinfo="skip",
                opacity=0.9,
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – Attractor Landscape Overlay (⟨S⟩ vs ⟨t_ref⟩)",
            font=dict(size=18),
        ),
        xaxis_title="⟨S⟩",
        yaxis_title="⟨t_ref⟩",
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    save_figure(fig, out_dir, "attractor_landscape_overlay")
    log_plot_end("attractor_landscape", "aggregate (overlay)")


def _energy_to_density(z_energy: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Invert energy surface back to normalized density used during caching.

    Energy was computed as -log(density + eps). We recover density by
    exp(-energy) - eps, then clamp to [0, 1].
    """
    with np.errstate(over="ignore", invalid="ignore"):
        den = np.exp(-np.asarray(z_energy, dtype=np.float32)) - float(eps)
    den = np.clip(den, 0.0, 1.0).astype(np.float32)
    return den


def plot_attractor_density_overlay(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    energy_by_class: Dict[int, np.ndarray],
    title_prefix: str,
    out_dir: str,
) -> None:
    if not energy_by_class:
        return
    log_plot_start("attractor_landscape_density", "aggregate (overlay)")
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, lbl in enumerate(sorted(energy_by_class.keys())):
        z_energy = energy_by_class[int(lbl)]
        z_density = _energy_to_density(z_energy)
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Contour(
                x=x_centers,
                y=y_centers,
                z=z_density,
                contours=dict(coloring="lines", showlines=True),
                line=dict(color=color, width=2),
                showscale=False,
                name=f"Digit {int(lbl)}",
                hoverinfo="skip",
                opacity=0.9,
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – Attractor Density Overlay (⟨S⟩ vs ⟨t_ref⟩)",
            font=dict(size=18),
        ),
        xaxis_title="⟨S⟩",
        yaxis_title="⟨t_ref⟩",
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(size=10)),
    )

    save_figure(fig, out_dir, "attractor_landscape_density_overlay")
    log_plot_end("attractor_landscape_density", "aggregate (overlay)")


def plot_attractor_landscape_per_digit(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    energy_by_class: Dict[int, np.ndarray],
    title_prefix: str,
    out_dir: str,
    convergence_by_class: Optional[Dict[int, List[Tuple[float, float]]]] = None,
) -> None:
    if not energy_by_class:
        return
    log_plot_start("attractor_landscape", "per-digit heatmaps")
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    for lbl, z in sorted(energy_by_class.items()):
        subdir = os.path.join(out_dir, f"per_digit/digit_{int(lbl)}")
        os.makedirs(subdir, exist_ok=True)

        # Create base heatmap figure (for static export)
        fig_static = go.Figure(
            data=go.Heatmap(
                x=x_centers,
                y=y_centers,
                z=z,
                colorscale="Viridis",
                colorbar=dict(title="Energy"),
            )
        )
        fig_static.update_layout(
            title=dict(
                text=f"{title_prefix} – Attractor Landscape (Digit {int(lbl)})",
                font=dict(size=18),
            ),
            xaxis_title="⟨S⟩",
            yaxis_title="⟨t_ref⟩",
            width=1400,
            height=900,
        )
        save_figure(fig_static, subdir, "attractor_landscape")

        # Create HTML-only figure with toggleable convergence points
        if convergence_by_class and int(lbl) in convergence_by_class:
            conv_pts = convergence_by_class[int(lbl)]
            if conv_pts:
                fig_with_conv = go.Figure()
                fig_with_conv.add_trace(
                    go.Heatmap(
                        x=x_centers,
                        y=y_centers,
                        z=z,
                        colorscale="Viridis",
                        colorbar=dict(title="Energy"),
                        name="Energy Basin",
                        showlegend=True,
                    )
                )
                conv_s = [p[0] for p in conv_pts]
                conv_t = [p[1] for p in conv_pts]
                fig_with_conv.add_trace(
                    go.Scatter(
                        x=conv_s,
                        y=conv_t,
                        mode="markers",
                        marker=dict(
                            size=8,
                            color="red",
                            symbol="x",
                            line=dict(width=1, color="white"),
                        ),
                        name="Convergence Points (click to toggle)",
                        hovertemplate="⟨S⟩=%{x:.3f}<br>⟨t_ref⟩=%{y:.3f}<extra>Convergence</extra>",
                    )
                )
                fig_with_conv.update_layout(
                    title=dict(
                        text=f"{title_prefix} – Attractor Landscape with Convergence (Digit {int(lbl)})",
                        font=dict(size=18),
                    ),
                    xaxis_title="⟨S⟩",
                    yaxis_title="⟨t_ref⟩",
                    width=1400,
                    height=900,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.01,
                        font=dict(size=11),
                        itemclick="toggle",
                        itemdoubleclick="toggleothers",
                    ),
                )
                html_path = os.path.join(
                    subdir, "attractor_landscape_with_convergence.html"
                )
                fig_with_conv.write_html(html_path)
                json_path = html_path.replace(".html", ".json")
                pio.write_json(fig_with_conv, json_path)

                # Also create separate convergence-only scatter plot
                fig_conv = go.Figure(
                    data=go.Scatter(
                        x=conv_s,
                        y=conv_t,
                        mode="markers",
                        marker=dict(
                            size=10,
                            color="red",
                            symbol="x",
                            line=dict(width=1, color="black"),
                        ),
                        name="Convergence",
                        hovertemplate="⟨S⟩=%{x:.3f}<br>⟨t_ref⟩=%{y:.3f}<extra></extra>",
                    )
                )
                fig_conv.update_layout(
                    title=dict(
                        text=f"{title_prefix} – Convergence Points (Digit {int(lbl)})",
                        font=dict(size=18),
                    ),
                    xaxis_title="⟨S⟩",
                    yaxis_title="⟨t_ref⟩",
                    width=1400,
                    height=900,
                )
                save_figure(fig_conv, subdir, "convergence_points")

        # CSV export: x_center, y_center, energy
        rows: List[List[Any]] = []
        for yi in range(len(y_centers)):
            for xi in range(len(x_centers)):
                rows.append(
                    [float(x_centers[xi]), float(y_centers[yi]), float(z[yi, xi])]
                )
        save_csv(
            rows,
            ["S_center", "t_ref_center", "energy"],
            subdir,
            "attractor_landscape_data",
        )

        # CSV export for convergence points
        if convergence_by_class and int(lbl) in convergence_by_class:
            conv_rows = [
                [float(p[0]), float(p[1])] for p in convergence_by_class[int(lbl)]
            ]
            save_csv(
                conv_rows,
                ["S_convergence", "t_ref_convergence"],
                subdir,
                "convergence_points_data",
            )
    log_plot_end("attractor_landscape", "per-digit heatmaps")


def plot_attractor_density_per_digit(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    energy_by_class: Dict[int, np.ndarray],
    title_prefix: str,
    out_dir: str,
    convergence_by_class: Optional[Dict[int, List[Tuple[float, float]]]] = None,
) -> None:
    if not energy_by_class:
        return
    log_plot_start("attractor_landscape_density", "per-digit heatmaps")
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    for lbl, z_energy in sorted(energy_by_class.items()):
        z_density = _energy_to_density(z_energy)
        subdir = os.path.join(out_dir, f"per_digit/digit_{int(lbl)}")
        os.makedirs(subdir, exist_ok=True)

        # Create base heatmap figure (for static export)
        fig_static = go.Figure(
            data=go.Heatmap(
                x=x_centers,
                y=y_centers,
                z=z_density,
                colorscale="Viridis",
                colorbar=dict(title="Density"),
                zmin=0.0,
                zmax=1.0,
            )
        )
        fig_static.update_layout(
            title=dict(
                text=f"{title_prefix} – Attractor Density (Digit {int(lbl)})",
                font=dict(size=18),
            ),
            xaxis_title="⟨S⟩",
            yaxis_title="⟨t_ref⟩",
            width=1400,
            height=900,
        )
        save_figure(fig_static, subdir, "attractor_landscape_density")

        # Create HTML-only figure with toggleable convergence points
        if convergence_by_class and int(lbl) in convergence_by_class:
            conv_pts = convergence_by_class[int(lbl)]
            if conv_pts:
                fig_with_conv = go.Figure()
                fig_with_conv.add_trace(
                    go.Heatmap(
                        x=x_centers,
                        y=y_centers,
                        z=z_density,
                        colorscale="Viridis",
                        colorbar=dict(title="Density"),
                        zmin=0.0,
                        zmax=1.0,
                        name="Density Basin",
                        showlegend=True,
                    )
                )
                conv_s = [p[0] for p in conv_pts]
                conv_t = [p[1] for p in conv_pts]
                fig_with_conv.add_trace(
                    go.Scatter(
                        x=conv_s,
                        y=conv_t,
                        mode="markers",
                        marker=dict(
                            size=8,
                            color="red",
                            symbol="x",
                            line=dict(width=1, color="white"),
                        ),
                        name="Convergence Points (click to toggle)",
                        hovertemplate="⟨S⟩=%{x:.3f}<br>⟨t_ref⟩=%{y:.3f}<extra>Convergence</extra>",
                    )
                )
                fig_with_conv.update_layout(
                    title=dict(
                        text=f"{title_prefix} – Attractor Density with Convergence (Digit {int(lbl)})",
                        font=dict(size=18),
                    ),
                    xaxis_title="⟨S⟩",
                    yaxis_title="⟨t_ref⟩",
                    width=1400,
                    height=900,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.01,
                        font=dict(size=11),
                        itemclick="toggle",
                        itemdoubleclick="toggleothers",
                    ),
                )
                html_path = os.path.join(
                    subdir, "attractor_density_with_convergence.html"
                )
                fig_with_conv.write_html(html_path)
                json_path = html_path.replace(".html", ".json")
                pio.write_json(fig_with_conv, json_path)

        # CSV export: x_center, y_center, density
        rows: List[List[Any]] = []
        for yi in range(len(y_centers)):
            for xi in range(len(x_centers)):
                rows.append(
                    [
                        float(x_centers[xi]),
                        float(y_centers[yi]),
                        float(z_density[yi, xi]),
                    ]
                )
        save_csv(
            rows,
            ["S_center", "t_ref_center", "density"],
            subdir,
            "attractor_density_data",
        )
    log_plot_end("attractor_landscape_density", "per-digit heatmaps")


def plot_attractor_landscape_3d_overlay(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    energy_by_class: Dict[int, np.ndarray],
    title_prefix: str,
    out_dir: str,
) -> None:
    if not energy_by_class:
        return
    log_plot_start("attractor_landscape", "aggregate 3D overlay")
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Global z range for consistent color scaling
    z_min = None
    z_max = None
    for z in energy_by_class.values():
        zmin_local = float(np.nanmin(z)) if np.size(z) else 0.0
        zmax_local = float(np.nanmax(z)) if np.size(z) else 0.0
        z_min = zmin_local if z_min is None else min(z_min, zmin_local)
        z_max = zmax_local if z_max is None else max(z_max, zmax_local)
    if (
        z_min is None
        or z_max is None
        or not np.isfinite(z_min)
        or not np.isfinite(z_max)
    ):
        z_min, z_max = 0.0, 1.0

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly

    for i, lbl in enumerate(sorted(energy_by_class.keys())):
        z = energy_by_class[int(lbl)]
        color = palette[i % len(palette)]
        group = f"digit_{int(lbl)}"
        # Surface for this digit
        fig.add_trace(
            go.Surface(
                x=x_centers,
                y=y_centers,
                z=z,
                colorscale="Viridis",
                cmin=z_min,
                cmax=z_max,
                opacity=0.7,
                showscale=False,
                name=f"Digit {int(lbl)}",
                legendgroup=group,
                showlegend=False,
            )
        )
        # Proxy legend item enabling toggling via legend group
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=6, color=color, opacity=0.9),
                name=f"Digit {int(lbl)}",
                legendgroup=group,
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – Attractor Landscape 3D Overlay (⟨S⟩, ⟨t_ref⟩ → Energy)",
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="⟨S⟩",
            yaxis_title="⟨t_ref⟩",
            zaxis_title="Energy",
        ),
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            font=dict(size=10),
            groupclick="togglegroup",
        ),
    )

    save_figure(fig, out_dir, "attractor_landscape_3d_overlay")
    log_plot_end("attractor_landscape", "aggregate 3D overlay")


def plot_attractor_density_3d_overlay(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    energy_by_class: Dict[int, np.ndarray],
    title_prefix: str,
    out_dir: str,
) -> None:
    if not energy_by_class:
        return
    log_plot_start("attractor_landscape_density", "aggregate 3D overlay")
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly

    # Use a consistent z range for normalized density
    z_min, z_max = 0.0, 1.0

    for i, lbl in enumerate(sorted(energy_by_class.keys())):
        z_energy = energy_by_class[int(lbl)]
        z_density = _energy_to_density(z_energy)
        color = palette[i % len(palette)]
        group = f"digit_{int(lbl)}"
        fig.add_trace(
            go.Surface(
                x=x_centers,
                y=y_centers,
                z=z_density,
                colorscale="Viridis",
                cmin=z_min,
                cmax=z_max,
                opacity=0.7,
                showscale=False,
                name=f"Digit {int(lbl)}",
                legendgroup=group,
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=6, color=color, opacity=0.9),
                name=f"Digit {int(lbl)}",
                legendgroup=group,
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – Attractor Density 3D Overlay (⟨S⟩, ⟨t_ref⟩ → Density)",
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="⟨S⟩",
            yaxis_title="⟨t_ref⟩",
            zaxis_title="Density (normalized)",
        ),
        width=1400,
        height=900,
        hovermode="closest",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            font=dict(size=10),
            groupclick="togglegroup",
        ),
    )

    save_figure(fig, out_dir, "attractor_landscape_density_3d_overlay")
    log_plot_end("attractor_landscape_density", "aggregate 3D overlay")


def plot_attractor_landscape_3d_per_digit(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    energy_by_class: Dict[int, np.ndarray],
    title_prefix: str,
    out_dir: str,
) -> None:
    if not energy_by_class:
        return
    log_plot_start("attractor_landscape", "per-digit 3D")
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    for lbl, z in sorted(energy_by_class.items()):
        subdir = os.path.join(out_dir, f"per_digit/digit_{int(lbl)}")
        fig = go.Figure(
            data=go.Surface(
                x=x_centers,
                y=y_centers,
                z=z,
                colorscale="Viridis",
                colorbar=dict(title="Energy"),
                showscale=True,
                opacity=0.9,
            )
        )
        fig.update_layout(
            title=dict(
                text=f"{title_prefix} – Attractor Landscape 3D (Digit {int(lbl)})",
                font=dict(size=18),
            ),
            scene=dict(
                xaxis_title="⟨S⟩",
                yaxis_title="⟨t_ref⟩",
                zaxis_title="Energy",
            ),
            width=1400,
            height=900,
        )
        save_figure(fig, subdir, "attractor_landscape_3d")

        # CSV export: x_center, y_center, energy
        rows: List[List[Any]] = []
        for yi in range(len(y_centers)):
            for xi in range(len(x_centers)):
                rows.append(
                    [float(x_centers[xi]), float(y_centers[yi]), float(z[yi, xi])]
                )
        save_csv(
            rows,
            ["S_center", "t_ref_center", "energy"],
            subdir,
            "attractor_landscape_3d_data",
        )
    log_plot_end("attractor_landscape", "per-digit 3D")


def plot_attractor_density_3d_per_digit(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    energy_by_class: Dict[int, np.ndarray],
    title_prefix: str,
    out_dir: str,
) -> None:
    if not energy_by_class:
        return
    log_plot_start("attractor_landscape_density", "per-digit 3D")
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    for lbl, z_energy in sorted(energy_by_class.items()):
        z_density = _energy_to_density(z_energy)
        subdir = os.path.join(out_dir, f"per_digit/digit_{int(lbl)}")
        fig = go.Figure(
            data=go.Surface(
                x=x_centers,
                y=y_centers,
                z=z_density,
                colorscale="Viridis",
                colorbar=dict(title="Density"),
                showscale=True,
                opacity=0.9,
                cmin=0.0,
                cmax=1.0,
            )
        )
        fig.update_layout(
            title=dict(
                text=f"{title_prefix} – Attractor Density 3D (Digit {int(lbl)})",
                font=dict(size=18),
            ),
            scene=dict(
                xaxis_title="⟨S⟩",
                yaxis_title="⟨t_ref⟩",
                zaxis_title="Density (normalized)",
            ),
            width=1400,
            height=900,
        )
        save_figure(fig, subdir, "attractor_landscape_density_3d")

        # CSV export: x_center, y_center, density
        rows: List[List[Any]] = []
        for yi in range(len(y_centers)):
            for xi in range(len(x_centers)):
                rows.append(
                    [
                        float(x_centers[xi]),
                        float(y_centers[yi]),
                        float(z_density[yi, xi]),
                    ]
                )
        save_csv(
            rows,
            ["S_center", "t_ref_center", "density"],
            subdir,
            "attractor_density_3d_data",
        )
    log_plot_end("attractor_landscape_density", "per-digit 3D")


def compute_attractor_landscapes_animated(
    label_to_indices: Dict[int, List[int]],
    dataset: LazyActivityDataset,
    num_classes: int,
    num_frames: int,
    cache_dir: str,
    dataset_hash: str,
    grid_size: int = 80,
    convergence_final_ticks: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[np.ndarray]], int]:
    """Compute per-digit animated attractor landscapes showing evolution over time.

    For each digit, collects (⟨S⟩, ⟨t_ref⟩) points at each tick and creates
    cumulative density frames from tick 0 to tick T.

    Also computes and caches convergence points (final state per image) as a side effect.

    Returns (x_edges, y_edges, frames_by_class, max_ticks) where frames_by_class
    maps digit -> list of density arrays (one per frame), and max_ticks is the
    global minimum number of ticks across all images.
    """
    cache_path = os.path.join(
        cache_dir,
        f"{dataset_hash}_attractor_animated_g{int(grid_size)}_f{int(num_frames)}.npz",
    )
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        if "arr_0" in data.files:
            labels = data["arr_0"].astype(np.int32)
            x_edges = data["arr_1"].astype(np.float32)
            y_edges = data["arr_2"].astype(np.float32)
            max_ticks = int(data["arr_3"])
            out: Dict[int, List[np.ndarray]] = {}
            idx = 4
            for lbl in labels:
                frames_count = int(data[f"arr_{idx}"])
                idx += 1
                frames = []
                for _ in range(frames_count):
                    frames.append(data[f"arr_{idx}"])
                    idx += 1
                out[int(lbl)] = frames
            return x_edges, y_edges, out, max_ticks

    # Gather per-tick (⟨S⟩, ⟨t_ref⟩) points per class with tick index
    # Structure: class -> list of (tick_idx, s_mean, t_mean)
    points_by_class: Dict[int, List[Tuple[int, float, float]]] = {}
    # Also track per-image points for convergence computation
    # Structure: (label, img_idx) -> list of (tick_idx, s_mean, t_mean)
    points_by_image: Dict[Tuple[int, int], List[Tuple[int, float, float]]] = {}
    s_all: List[float] = []
    t_all: List[float] = []
    max_ticks_global: Optional[int] = None

    # First pass: determine max ticks across all images
    # We want max_ticks to be the LIMIT for animation, usually the MINIMUM duration across samples?
    # Original code took min(max_ticks_global, T).
    # This means we only animate up to the shortest sample duration.
    # Let's replicate this.

    # Check max ticks
    # We can just iterate once.

    T_min = 10**12
    found_any = False

    iterator_labels = sorted(label_to_indices.keys())
    for label in iterator_labels:
        if not (0 <= int(label) < num_classes):
            continue
        for idx in label_to_indices[label]:
            sample = dataset[idx]
            T = sample["u"].shape[0]
            T_min = min(T_min, T)
            found_any = True

    if not found_any or T_min > 1000000:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), {}, 0

    max_ticks_global = T_min

    # Second pass: collect points with tick indices
    for label in iterator_labels:
        if not (0 <= int(label) < num_classes):
            continue
        indices = label_to_indices[label]
        for img_idx_local, idx in enumerate(
            indices
        ):  # img_idx_local is arbitrary index for dict key
            sample = dataset[idx]
            u = sample["u"]
            tr = sample["t_ref"]

            # Crop to max_ticks_global
            u_crop = u[:max_ticks_global, :]
            t_crop = tr[:max_ticks_global, :]

            # Per tick averages
            # Mean over neurons (axis 1)
            s_t = np.mean(u_crop, axis=1)
            t_t = np.mean(t_crop, axis=1)

            # Vectorized add
            for tick_idx in range(len(s_t)):
                s_val = float(s_t[tick_idx])
                t_val = float(t_t[tick_idx])

                points_by_class.setdefault(int(label), []).append(
                    (tick_idx, s_val, t_val)
                )
                # Use dataset index as unique image identifier? Or (label, img_idx_local)
                points_by_image.setdefault((int(label), int(idx)), []).append(
                    (tick_idx, s_val, t_val)
                )
                s_all.append(s_val)
                t_all.append(t_val)

    if not s_all or not t_all:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), {}, 0

    # Compute and cache convergence points (final state per image)
    convergence_by_class: Dict[int, List[Tuple[float, float]]] = {}
    for (label, img_idx), pts in points_by_image.items():
        if not pts:
            continue
        # Sort by tick and take last N ticks
        pts_sorted = sorted(pts, key=lambda x: x[0])
        final_pts = (
            pts_sorted[-convergence_final_ticks:]
            if len(pts_sorted) >= convergence_final_ticks
            else pts_sorted
        )
        if final_pts:
            s_conv = float(np.mean([p[1] for p in final_pts]))
            t_conv = float(np.mean([p[2] for p in final_pts]))
            convergence_by_class.setdefault(int(label), []).append((s_conv, t_conv))

    # Save convergence points cache
    if convergence_by_class:
        conv_cache_path = os.path.join(
            cache_dir,
            f"{dataset_hash}_convergence_points_f{convergence_final_ticks}.npz",
        )
        labels_sorted_conv = np.array(
            sorted(convergence_by_class.keys()), dtype=np.int32
        )
        conv_arrays: List[np.ndarray] = []
        for lbl in labels_sorted_conv:
            conv_pts = convergence_by_class[int(lbl)]
            s_arr = np.array([p[0] for p in conv_pts], dtype=np.float32)
            t_arr = np.array([p[1] for p in conv_pts], dtype=np.float32)
            conv_arrays.extend([s_arr, t_arr])
        np.savez_compressed(conv_cache_path, labels_sorted_conv, *conv_arrays)

    # Define global grid edges (shared across classes and frames)
    s_min = float(np.min(np.asarray(s_all, dtype=np.float64)))
    s_max = float(np.max(np.asarray(s_all, dtype=np.float64)))
    t_min = float(np.min(np.asarray(t_all, dtype=np.float64)))
    t_max = float(np.max(np.asarray(t_all, dtype=np.float64)))
    s_pad = 1e-6 if s_max == s_min else 0.02 * (s_max - s_min)
    t_pad = 1e-6 if t_max == t_min else 0.02 * (t_max - t_min)
    x_edges = np.linspace(
        s_min - s_pad, s_max + s_pad, int(grid_size) + 1, dtype=np.float32
    )
    y_edges = np.linspace(
        t_min - t_pad, t_max + t_pad, int(grid_size) + 1, dtype=np.float32
    )

    # Determine frame tick boundaries
    frame_ticks = np.linspace(0, max_ticks_global, num_frames + 1, dtype=np.int32)
    frame_ticks = np.unique(frame_ticks)  # Remove duplicates for small tick counts

    def smooth_counts(counts: np.ndarray) -> np.ndarray:
        k = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 1]], dtype=np.float32)
        h, w = counts.shape
        out_sm = np.zeros_like(counts, dtype=np.float32)
        for i in range(h):
            i0 = max(0, i - 1)
            i1 = min(h, i + 2)
            for j in range(w):
                j0 = max(0, j - 1)
                j1 = min(w, j + 2)
                window = counts[i0:i1, j0:j1]
                ki0 = 1 - (i - i0)
                ki1 = 2 + (i1 - (i + 1))
                kj0 = 1 - (j - j0)
                kj1 = 2 + (j1 - (j + 1))
                kern = k[ki0:ki1, kj0:kj1]
                denom = float(np.sum(kern)) if np.sum(kern) > 0 else 1.0
                out_sm[i, j] = float(np.sum(window * kern) / denom)
        return out_sm

    frames_by_class: Dict[int, List[np.ndarray]] = {}

    for lbl, pts in tqdm(
        points_by_class.items(), desc="Computing animated attractor frames"
    ):
        if not pts:
            continue
        # Sort points by tick index for cumulative processing
        pts_sorted = sorted(pts, key=lambda x: x[0])
        frames: List[np.ndarray] = []

        for frame_idx in range(1, len(frame_ticks)):
            tick_end = int(frame_ticks[frame_idx])
            # Cumulative: use all points up to tick_end
            filtered_pts = [(s, t) for (tick, s, t) in pts_sorted if tick < tick_end]
            if not filtered_pts:
                # Empty frame - use zeros
                frames.append(
                    np.zeros((int(grid_size), int(grid_size)), dtype=np.float32)
                )
                continue

            s_vals_arr = np.asarray([p[0] for p in filtered_pts], dtype=np.float32)
            t_vals_arr = np.asarray([p[1] for p in filtered_pts], dtype=np.float32)
            H, _, _ = np.histogram2d(s_vals_arr, t_vals_arr, bins=[x_edges, y_edges])
            counts = H.T.astype(np.float32)
            counts = smooth_counts(counts)
            total = float(np.sum(counts))
            density = counts / total if total > 0 else counts
            frames.append(density.astype(np.float32))

        frames_by_class[int(lbl)] = frames

    # Save cache
    if frames_by_class:
        labels_sorted = np.array(sorted(frames_by_class.keys()), dtype=np.int32)
        arrays: List[np.ndarray] = [
            x_edges.astype(np.float32),
            y_edges.astype(np.float32),
            np.array(max_ticks_global, dtype=np.int32),
        ]
        for lbl in labels_sorted:
            frames = frames_by_class[int(lbl)]
            arrays.append(np.array(len(frames), dtype=np.int32))
            arrays.extend(frames)
        np.savez_compressed(cache_path, labels_sorted, *arrays)

    return x_edges, y_edges, frames_by_class, max_ticks_global


def _create_animated_attractor_figure(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    frames: List[np.ndarray],
    frame_ticks: np.ndarray,
    max_ticks: int,
    lbl: int,
    title_prefix: str,
    mode: str,  # "density" or "energy"
    convergence_points: Optional[List[Tuple[float, float]]] = None,
) -> go.Figure:
    """Helper to create an animated attractor landscape figure.

    Args:
        mode: "density" for normalized density, "energy" for -log(density + eps)
        convergence_points: Optional list of (s, t_ref) final state points to overlay
    """
    eps = 1e-8

    def transform_frame(f: np.ndarray) -> np.ndarray:
        if mode == "energy":
            with np.errstate(divide="ignore", invalid="ignore"):
                return -np.log(f + eps).astype(np.float32)
        return f

    # Transform all frames
    transformed_frames = [transform_frame(f) for f in frames]

    # Determine global z range for consistent color scaling
    if mode == "density":
        z_min = 0.0
        z_max = max(float(np.max(f)) for f in frames if f.size > 0) if frames else 1.0
        z_max = max(z_max, 1e-6)
        colorbar_title = "Density"
        title_type = "Attractor Density"
    else:  # energy
        # Energy is -log(density), so higher density -> lower energy
        z_vals = [f for f in transformed_frames if f.size > 0]
        if z_vals:
            z_min = min(float(np.nanmin(f)) for f in z_vals)
            z_max = max(float(np.nanmax(f)) for f in z_vals)
            # Clamp extreme values
            if not np.isfinite(z_min):
                z_min = 0.0
            if not np.isfinite(z_max):
                z_max = 20.0
        else:
            z_min, z_max = 0.0, 20.0
        colorbar_title = "Energy"
        title_type = "Attractor Energy"

    # Prepare convergence scatter trace if provided
    conv_scatter = None
    if convergence_points:
        conv_s = [p[0] for p in convergence_points]
        conv_t = [p[1] for p in convergence_points]
        conv_scatter = go.Scatter(
            x=conv_s,
            y=conv_t,
            mode="markers",
            marker=dict(
                size=8,
                color="red",
                symbol="x",
                line=dict(width=1, color="white"),
            ),
            name="Convergence",
            hovertemplate="⟨S⟩=%{x:.3f}<br>⟨t_ref⟩=%{y:.3f}<extra>Convergence</extra>",
        )

    # Create initial frame data
    initial_data = [
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=(
                transformed_frames[0]
                if transformed_frames
                else np.zeros((len(y_centers), len(x_centers)))
            ),
            colorscale="Viridis",
            colorbar=dict(title=colorbar_title),
            zmin=z_min,
            zmax=z_max,
        )
    ]
    if conv_scatter is not None:
        initial_data.append(conv_scatter)  # type: ignore

    fig = go.Figure(data=initial_data)

    # Create animation frames
    # Only update the heatmap (trace 0), leave convergence points (trace 1) static
    animation_frames = []
    for i, frame_data in enumerate(transformed_frames):
        tick_label = int(frame_ticks[i + 1]) if i + 1 < len(frame_ticks) else max_ticks
        animation_frames.append(
            go.Frame(
                data=[
                    go.Heatmap(
                        x=x_centers,
                        y=y_centers,
                        z=frame_data,
                        colorscale="Viridis",
                        colorbar=dict(title=colorbar_title),
                        zmin=z_min,
                        zmax=z_max,
                    )
                ],
                traces=[0],  # Only update trace 0 (heatmap), not trace 1 (convergence)
                name=str(i),
                layout=go.Layout(
                    title=dict(
                        text=f"{title_prefix} – {title_type} (Digit {int(lbl)}) – Tick 0-{tick_label}",
                        font=dict(size=18),
                    )
                ),
            )
        )

    fig.frames = animation_frames

    # Create slider steps
    steps = []
    for i in range(len(transformed_frames)):
        tick_label = int(frame_ticks[i + 1]) if i + 1 < len(frame_ticks) else max_ticks
        steps.append(
            dict(
                args=[
                    [str(i)],
                    dict(
                        frame=dict(duration=100, redraw=True),
                        mode="immediate",
                        transition=dict(duration=50),
                    ),
                ],
                label=f"t={tick_label}",
                method="animate",
            )
        )

    sliders = [
        dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(size=14),
                prefix="Frame: ",
                visible=True,
                xanchor="right",
            ),
            transition=dict(duration=50),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=steps,
        )
    ]

    # Update layout with animation controls
    fig.update_layout(
        title=dict(
            text=f"{title_prefix} – {title_type} (Digit {int(lbl)}) – Tick 0-{int(frame_ticks[1]) if len(frame_ticks) > 1 else 0}",
            font=dict(size=18),
        ),
        xaxis_title="⟨S⟩",
        yaxis_title="⟨t_ref⟩",
        width=1400,
        height=900,
        updatemenus=[
            # Play/Pause controls
            dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=0.1,
                xanchor="right",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=200, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=100),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            ),
            # Speed controls
            dict(
                type="buttons",
                showactive=True,
                active=1,  # Default to 1x speed
                y=1.15,
                x=0.55,
                xanchor="left",
                buttons=[
                    dict(
                        label="0.25x",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=800, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=400),
                            ),
                        ],
                    ),
                    dict(
                        label="0.5x",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=400, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=200),
                            ),
                        ],
                    ),
                    dict(
                        label="1x",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=200, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=100),
                            ),
                        ],
                    ),
                    dict(
                        label="2x",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=100, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=50),
                            ),
                        ],
                    ),
                    dict(
                        label="4x",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=50, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=25),
                            ),
                        ],
                    ),
                    dict(
                        label="8x",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=25, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=12.5),
                            ),
                        ],
                    ),
                    dict(
                        label="16x",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=12.5, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=6.25),
                            ),
                        ],
                    ),
                ],
            ),
        ],
        sliders=sliders,
        # Add annotation for speed label
        annotations=[
            dict(
                text="Speed:",
                x=0.54,
                y=1.14,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    return fig


def plot_attractor_landscape_animated_per_digit(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    frames_by_class: Dict[int, List[np.ndarray]],
    max_ticks: int,
    num_frames: int,
    title_prefix: str,
    out_dir: str,
    convergence_by_class: Optional[Dict[int, List[Tuple[float, float]]]] = None,
) -> None:
    """Create animated attractor landscape heatmaps per digit (HTML only).

    Generates both density and energy animated plots for each digit.
    If convergence_by_class is provided, creates additional versions with toggleable
    convergence point overlay.
    """
    if not frames_by_class:
        return
    log_plot_start("attractor_landscape_animated", "per-digit (density + energy)")
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Determine frame tick boundaries for labels
    frame_ticks = np.linspace(0, max_ticks, num_frames + 1, dtype=np.int32)
    frame_ticks = np.unique(frame_ticks)

    for lbl, frames in sorted(frames_by_class.items()):
        if not frames:
            continue
        subdir = os.path.join(out_dir, f"per_digit/digit_{int(lbl)}")
        os.makedirs(subdir, exist_ok=True)

        # Get convergence points for this digit if available
        conv_pts = None
        if convergence_by_class and int(lbl) in convergence_by_class:
            conv_pts = convergence_by_class[int(lbl)]

        # Create and save density animation (without convergence)
        fig_density = _create_animated_attractor_figure(
            x_centers,
            y_centers,
            frames,
            frame_ticks,
            max_ticks,
            lbl,
            title_prefix,
            "density",
        )
        html_path_density = os.path.join(subdir, "attractor_density_animated.html")
        fig_density.write_html(html_path_density)
        json_path_density = html_path_density.replace(".html", ".json")
        pio.write_json(fig_density, json_path_density)

        # Create and save energy animation (without convergence)
        fig_energy = _create_animated_attractor_figure(
            x_centers,
            y_centers,
            frames,
            frame_ticks,
            max_ticks,
            lbl,
            title_prefix,
            "energy",
        )
        html_path_energy = os.path.join(subdir, "attractor_energy_animated.html")
        fig_energy.write_html(html_path_energy)
        json_path_energy = html_path_energy.replace(".html", ".json")
        pio.write_json(fig_energy, json_path_energy)

        # Create versions with convergence overlay if available
        if conv_pts:
            # Density with convergence
            fig_density_conv = _create_animated_attractor_figure(
                x_centers,
                y_centers,
                frames,
                frame_ticks,
                max_ticks,
                lbl,
                title_prefix,
                "density",
                convergence_points=conv_pts,
            )
            # Update legend to indicate toggle capability
            fig_density_conv.update_layout(
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01,
                    font=dict(size=11),
                    itemclick="toggle",
                    itemdoubleclick="toggleothers",
                ),
            )
            html_path_density_conv = os.path.join(
                subdir, "attractor_density_animated_with_convergence.html"
            )
            fig_density_conv.write_html(html_path_density_conv)
            json_path_density_conv = html_path_density_conv.replace(".html", ".json")
            pio.write_json(fig_density_conv, json_path_density_conv)

            # Energy with convergence
            fig_energy_conv = _create_animated_attractor_figure(
                x_centers,
                y_centers,
                frames,
                frame_ticks,
                max_ticks,
                lbl,
                title_prefix,
                "energy",
                convergence_points=conv_pts,
            )
            fig_energy_conv.update_layout(
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01,
                    font=dict(size=11),
                    itemclick="toggle",
                    itemdoubleclick="toggleothers",
                ),
            )
            html_path_energy_conv = os.path.join(
                subdir, "attractor_energy_animated_with_convergence.html"
            )
            fig_energy_conv.write_html(html_path_energy_conv)
            json_path_energy_conv = html_path_energy_conv.replace(".html", ".json")
            pio.write_json(fig_energy_conv, json_path_energy_conv)

    log_plot_end("attractor_landscape_animated", "per-digit (density + energy)")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize network activity with extendable, Plotly-based analyses."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the activity dataset JSON (same schema as other tools).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="viz_network",
        help="Directory to save plots and caches.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes (digits) expected in the dataset.",
    )
    parser.add_argument(
        "--plots",
        type=str,
        nargs="+",
        default=[
            "s_heatmap_by_class",
            "favg_tref_scatter",
            "firing_rate_hist_by_layer",
            "tref_timeline",
            "layerwise_s_average",
            "spike_raster",
            "phase_portrait",
        ],
        choices=[
            "s_heatmap_by_class",
            "favg_tref_scatter",
            "firing_rate_hist_by_layer",
            "tref_timeline",
            "layerwise_s_average",
            "spike_raster",
            "phase_portrait",
            "attractor_landscape",
            "attractor_landscape_3d",
            "attractor_landscape_animated",
            "tref_bounds_box",
            "favg_stability",
            "homeostatic_response",
            "affinity_heatmap",
            "tref_by_preferred_digit",
            "temporal_corr_graph",
            "s_variance_decay",
            "all",
        ],
        help="Which plots to generate.",
    )
    parser.add_argument(
        "--exclude-plots",
        type=str,
        nargs="+",
        default=[],
        choices=[
            "s_heatmap_by_class",
            "favg_tref_scatter",
            "firing_rate_hist_by_layer",
            "tref_timeline",
            "layerwise_s_average",
            "spike_raster",
            "phase_portrait",
            "attractor_landscape",
            "attractor_landscape_3d",
            "attractor_landscape_animated",
            "tref_bounds_box",
            "favg_stability",
            "homeostatic_response",
            "affinity_heatmap",
            "tref_by_preferred_digit",
            "temporal_corr_graph",
            "s_variance_decay",
        ],
        help="Plots to exclude when using --plots all (or any other selection).",
    )
    parser.add_argument(
        "--max-representative-neurons",
        type=int,
        default=10,
        help="Max number of neurons to show in the t_ref timeline plot.",
    )
    parser.add_argument(
        "--homeostat-c",
        type=float,
        default=1.0,
        help="Homeostatic constant c used for theoretical bounds/curves.",
    )
    parser.add_argument(
        "--network-config",
        type=str,
        default=None,
        help="Path to network JSON config to auto-extract c and synapses per layer.",
    )
    parser.add_argument(
        "--synapses-per-layer",
        type=str,
        default=None,
        help="Comma-separated |synapses| per layer for theoretical bounds. If omitted, uses --synapses-default for all layers.",
    )
    parser.add_argument(
        "--synapses-default",
        type=int,
        default=100,
        help="Default |synapses| value when --synapses-per-layer is not provided.",
    )
    parser.add_argument(
        "--epoch-field",
        type=str,
        default="epoch",
        help="Record field name that stores training epoch. If absent, use --num-epoch-bins to approximate.",
    )
    parser.add_argument(
        "--num-epoch-bins",
        type=int,
        default=0,
        help="If no epoch field is present, split images into this many sequential bins to approximate epochs.",
    )
    parser.add_argument(
        "--tref-box-sample",
        type=int,
        default=200000,
        help="Max number of t_ref samples per layer for box plot (reservoir sampling).",
    )
    parser.add_argument(
        "--response-bins",
        type=int,
        default=20,
        help="Number of bins for homeostatic response empirical curve.",
    )
    parser.add_argument(
        "--response-scope",
        type=str,
        default="global",
        choices=["global", "per-layer"],
        help="Plot response curve aggregated globally or per layer.",
    )
    parser.add_argument(
        "--skip-static-images",
        action="store_true",
        help="Skip PNG/SVG exports and save only interactive HTML (useful if kaleido is unavailable).",
    )
    parser.add_argument(
        "--raster-max-images",
        type=int,
        default=50,
        help="Max number of images to include in spike raster (concatenated).",
    )
    parser.add_argument(
        "--raster-gap",
        type=int,
        default=1,
        help="Gap (ticks) inserted between images in spike raster.",
    )
    parser.add_argument(
        "--raster-max-points",
        type=int,
        default=200000,
        help="Max points to plot in spike raster (randomly subsample if exceeded).",
    )
    parser.add_argument(
        "--corr-max-neurons",
        type=int,
        default=200,
        help="Max neurons to include in temporal correlation graph (sampled across layers).",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.3,
        help="Correlation threshold for edges in temporal correlation graph.",
    )
    parser.add_argument(
        "--corr-max-images",
        type=int,
        default=200,
        help="Max images to concatenate for correlation time series.",
    )
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=50,
        help="Number of frames for animated attractor landscape plots.",
    )
    parser.add_argument(
        "--convergence-final-ticks",
        type=int,
        default=10,
        help="Number of final ticks to average for convergence point computation.",
    )
    parser.add_argument(
        "--legacy-json",
        action="store_true",
        help="Force loading as legacy JSON format instead of binary",
    )

    args = parser.parse_args()
    dataset_root = os.path.splitext(os.path.basename(args.input_file))[0]
    fig_dir, cache_dir = ensure_output_dirs(args.output_dir, dataset_root)

    dataset_hash = compute_dataset_hash(args.input_file)

    # Expand plots for cache evaluation (do not depend on data)
    all_plots_expanded = [
        "s_heatmap_by_class",
        "favg_tref_scatter",
        "firing_rate_hist_by_layer",
        "tref_timeline",
        "layerwise_s_average",
        "spike_raster",
        "phase_portrait",
        "attractor_landscape",
        "attractor_landscape_3d",
        "attractor_landscape_animated",
        "tref_bounds_box",
        "favg_stability",
        "homeostatic_response",
        "affinity_heatmap",
        "tref_by_preferred_digit",
        "temporal_corr_graph",
        "s_variance_decay",
    ]
    plots_requested = list(args.plots)
    if "all" in plots_requested:
        plots_requested = all_plots_expanded
        args.plots = plots_requested

    # Apply exclusions before cache check
    excluded_plots = []
    if args.exclude_plots:
        excluded_plots = [p for p in args.plots if p in args.exclude_plots]
        args.plots = [p for p in args.plots if p not in args.exclude_plots]
        if excluded_plots:
            print(
                f"\n[Exclusion] Excluding {len(excluded_plots)} plot(s): {', '.join(excluded_plots)}"
            )
            print(
                f"[Exclusion] Remaining {len(args.plots)} plot(s): {', '.join(args.plots)}\n"
            )

    print("Checking if all requested plots are cached...")
    all_cached = check_all_caches_exist(
        args.plots,
        dataset_hash,
        cache_dir,
        args.num_classes,
        excluded_plots,
        epoch_field=args.epoch_field,
        num_epoch_bins=args.num_epoch_bins,
        tref_box_sample=args.tref_box_sample,
        corr_max_neurons=args.corr_max_neurons,
        corr_threshold=args.corr_threshold,
        raster_gap=args.raster_gap,
        corr_max_images=args.corr_max_images,
        animation_frames=args.animation_frames,
        convergence_final_ticks=args.convergence_final_ticks,
    )

    if all_cached:
        print("All requested plots are cached. Skipping data loading and computation.")
        # Reconstruct minimal network structure from aggregates cache
        aggregates_cache_path = os.path.join(
            cache_dir, f"{dataset_hash}_per_neuron_aggregates.npz"
        )
        # Initialize placeholders for structures we may not need when cached
        buckets: Dict[int, List[int]] = {}  # Now label_to_indices
        dataset: Optional[LazyActivityDataset] = None
        neurons_per_layer: List[int] = []
        layer_offsets: List[int] = []
        total_neurons: int = 0
        aggregates: Dict[str, np.ndarray]
        if os.path.exists(aggregates_cache_path):
            data = np.load(aggregates_cache_path, allow_pickle=True)
            aggregates = {k: data[k] for k in data.files}  # type: ignore[assignment]
            layer_index = aggregates["layer_index"]
            neurons_per_layer = []
            layer_offsets = []
            offset = 0
            for li in sorted(set(layer_index.tolist())):
                count = int(np.sum(layer_index == li))
                neurons_per_layer.append(count)
                layer_offsets.append(offset)
                offset += count
            total_neurons = offset
        else:
            print(
                "Warning: Could not load network structure from cache. Loading data anyway."
            )
            all_cached = False

    if not all_cached:
        print("Some caches are missing. Loading data and computing...")
        # Load and prepare
        dataset = load_activity_data(args.input_file)
        print(f"Loaded {len(dataset)} records from {args.input_file}")

        # Group indices
        buckets = group_images_by_label(dataset)

        neurons_per_layer, layer_offsets, total_neurons = infer_network_structure(
            dataset
        )

        if total_neurons == 0:
            print("No neurons inferred from dataset. Exiting.")
            return

    title_prefix = "Network Activity"

    # Optional: derive c and synapses per layer from network config
    auto_c: Optional[float] = None
    auto_syn_per_layer: Optional[List[int]] = None
    if args.network_config:
        # extract_homeostat calls are unchanged if they just parse JSON file
        auto_c, auto_syn_per_layer = extract_homeostat_from_network(
            args.network_config, len(neurons_per_layer)
        )

    # Configure static export behavior from CLI
    global SAVE_STATIC_IMAGES
    SAVE_STATIC_IMAGES = not args.skip_static_images

    # Note: plots list and exclusions have already been processed earlier before cache check

    # Precompute/load aggregates depending on cache status
    # Precompute/load aggregates depending on cache status
    if not all_cached and dataset is not None:
        aggregates = compute_per_neuron_aggregates(
            dataset,
            neurons_per_layer,
            layer_offsets,
            total_neurons,
            args.num_classes,
            cache_dir,
            dataset_hash,
        )
    else:
        # Already loaded from cache above
        assert "aggregates" in locals()

    # S(t) Heatmap by Class
    if "s_heatmap_by_class" in args.plots:
        heatmaps = compute_classwise_S_heatmaps(
            buckets,
            dataset,  # Added dataset
            neurons_per_layer,
            layer_offsets,
            total_neurons,
            args.num_classes,
            cache_dir,
            dataset_hash,
        )
        plot_S_heatmap_by_class(heatmaps, title_prefix, fig_dir)
        # Per-digit subplots stored under subdirectories
        if heatmaps:
            for lbl, mat in heatmaps.items():
                log_plot_start("s_heatmap_by_class", f"digit {int(lbl)}")
                subdir = os.path.join(fig_dir, f"per_digit/digit_{int(lbl)}")
                # Plot single heatmap for this digit
                fig = go.Figure(
                    data=go.Heatmap(
                        z=mat, colorscale="Viridis", colorbar=dict(title="S")
                    )
                )
                fig.update_layout(
                    title=dict(
                        text=f"{title_prefix} – S(t) Heatmap (Digit {int(lbl)})",
                        font=dict(size=18),
                    ),
                    xaxis_title="t",
                    yaxis_title="Neuron",
                    width=1400,
                    height=900,
                )
                save_figure(fig, subdir, "s_heatmap")
                log_plot_end("s_heatmap_by_class", f"digit {int(lbl)}")

    # F_avg vs t_ref scatter
    if "favg_tref_scatter" in args.plots:
        plot_favg_vs_tref_scatter(
            aggregates,
            title_prefix,
            fig_dir,
            theory_c=auto_c,
            theory_syn_per_layer=auto_syn_per_layer,
        )
        # Per-digit: recompute per-neuron aggregates using only records of that digit
        for lbl in sorted(buckets.keys()):
            indices = buckets[lbl]
            if not indices:
                continue
            sub_aggs = compute_per_neuron_aggregates(
                dataset,
                neurons_per_layer,
                layer_offsets,
                total_neurons,
                args.num_classes,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
                indices=indices,
            )
            subdir = os.path.join(fig_dir, f"per_digit/digit_{int(lbl)}")
            plot_favg_vs_tref_scatter(
                sub_aggs,
                title_prefix,
                subdir,
                theory_c=auto_c,
                theory_syn_per_layer=auto_syn_per_layer,
            )

    # Firing rate histogram by layer
    if "firing_rate_hist_by_layer" in args.plots:
        plot_firing_rate_hist_by_layer(aggregates, title_prefix, fig_dir)
        # Per-digit: recompute aggregates using only records of that digit
        for lbl in sorted(buckets.keys()):
            indices = buckets[lbl]
            if not indices:
                continue
            sub_aggs = compute_per_neuron_aggregates(
                dataset,
                neurons_per_layer,
                layer_offsets,
                total_neurons,
                args.num_classes,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
                indices=indices,
            )
            subdir = os.path.join(fig_dir, f"per_digit/digit_{int(lbl)}")
            plot_firing_rate_hist_by_layer(sub_aggs, title_prefix, subdir)

    # t_ref evolution timeline for representative neurons
    if "tref_timeline" in args.plots:
        timelines, T_min_all = compute_tref_timelines(
            dataset,
            neurons_per_layer,
            layer_offsets,
            total_neurons,
            cache_dir,
            dataset_hash,
        )
        plot_tref_evolution_timeline(
            timelines,
            T_min_all,
            aggregates,
            neurons_per_layer,
            layer_offsets,
            args.max_representative_neurons,
            title_prefix,
            fig_dir,
        )
        # Per-digit: restrict records/buckets to that digit before computing timelines
        for lbl in sorted(buckets.keys()):
            indices = buckets[lbl]
            tl_sub, T_sub = compute_tref_timelines(
                dataset,
                neurons_per_layer,
                layer_offsets,
                total_neurons,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
                indices=indices,
            )
            subdir = os.path.join(fig_dir, f"per_digit/digit_{int(lbl)}")
            plot_tref_evolution_timeline(
                tl_sub,
                T_sub,
                aggregates,
                neurons_per_layer,
                layer_offsets,
                args.max_representative_neurons,
                title_prefix,
                subdir,
            )

    # Layer-wise S(t) Average
    if "layerwise_s_average" in args.plots:
        layer_s, T_min_all = compute_layerwise_S_timeline(
            dataset, neurons_per_layer, cache_dir, dataset_hash
        )
        plot_layerwise_S_timeline(layer_s, T_min_all, title_prefix, fig_dir)
        # Per-digit layer-wise S timelines
        for lbl in sorted(buckets.keys()):
            indices = buckets[lbl]
            layer_s_sub, T_sub = compute_layerwise_S_timeline(
                dataset,
                neurons_per_layer,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
                indices=indices,
            )
            subdir = os.path.join(fig_dir, f"per_digit/digit_{int(lbl)}")
            plot_layerwise_S_timeline(layer_s_sub, T_sub, title_prefix, subdir)

    # Spike raster with t_ref color
    if "spike_raster" in args.plots:
        tref_bounds = None
        if (
            auto_c is not None
            and auto_syn_per_layer is not None
            and len(auto_syn_per_layer) > 0
        ):
            lower = 2.0 * float(auto_c)
            upper = float(auto_c) * float(
                np.mean(np.asarray(auto_syn_per_layer, dtype=np.float32))
            )
            tref_bounds = (lower, upper)
        plot_spike_raster_with_tref(
            dataset,
            neurons_per_layer,
            layer_offsets,
            args.raster_max_images,
            args.raster_gap,
            args.raster_max_points,
            title_prefix,
            fig_dir,
            tref_bounds=tref_bounds,
        )
        # Per-digit rasters
        for lbl in sorted(buckets.keys()):
            indices = buckets[lbl]
            subdir = os.path.join(fig_dir, f"per_digit/digit_{int(lbl)}")
            plot_spike_raster_with_tref(
                dataset,
                neurons_per_layer,
                layer_offsets,
                args.raster_max_images,
                args.raster_gap,
                args.raster_max_points,
                title_prefix,
                subdir,
                tref_bounds=tref_bounds,
                indices=indices,
            )

    # Phase portrait 3D
    if "phase_portrait" in args.plots:
        series_by_class = compute_phase_portrait_series(
            buckets, dataset, args.num_classes, cache_dir, dataset_hash
        )
        plot_phase_portrait(
            series_by_class,
            title_prefix,
            fig_dir,
            theory_c=auto_c,
            theory_syn_per_layer=auto_syn_per_layer,
        )
        # Per-digit portraits
        for lbl in sorted(buckets.keys()):
            # Pass subset dictionary for single label
            sub_b = {lbl: buckets[lbl]}
            series_sub = compute_phase_portrait_series(
                sub_b,
                dataset,
                args.num_classes,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
            )
            subdir = os.path.join(fig_dir, f"per_digit/digit_{int(lbl)}")
            plot_phase_portrait(
                series_sub,
                title_prefix,
                subdir,
                theory_c=auto_c,
                theory_syn_per_layer=auto_syn_per_layer,
            )

    # Affinity heatmap
    if "affinity_heatmap" in args.plots or "tref_by_preferred_digit" in args.plots:
        affinity = compute_affinity_matrix(
            buckets,
            dataset,
            neurons_per_layer,
            layer_offsets,
            total_neurons,
            args.num_classes,
            cache_dir,
            dataset_hash,
        )
        if "affinity_heatmap" in args.plots:
            plot_affinity_heatmap(affinity, neurons_per_layer, title_prefix, fig_dir)
        if "tref_by_preferred_digit" in args.plots:
            plot_tref_by_preferred_digit(affinity, aggregates, title_prefix, fig_dir)

    # Temporal correlation graph
    if "temporal_corr_graph" in args.plots:
        nodes, u, v = compute_temporal_correlation_edges(
            buckets,
            dataset,
            neurons_per_layer,
            layer_offsets,
            total_neurons,
            args.corr_max_neurons,
            args.corr_threshold,
            args.raster_gap,
            args.corr_max_images,
            cache_dir,
            dataset_hash,
        )
        plot_temporal_correlation_graph(
            nodes,
            u,
            v,
            neurons_per_layer,
            layer_offsets,
            aggregates,
            title_prefix,
            fig_dir,
        )

    # Compute convergence points if any attractor landscape plot is requested
    convergence_by_class: Optional[Dict[int, List[Tuple[float, float]]]] = None
    if any(
        p in args.plots
        for p in [
            "attractor_landscape",
            "attractor_landscape_3d",
            "attractor_landscape_animated",
        ]
    ):
        convergence_by_class = compute_convergence_points(
            buckets,
            dataset,
            args.num_classes,
            args.convergence_final_ticks,
            cache_dir,
            dataset_hash,
        )

    # Attractor landscape (energy surfaces over (⟨S⟩, ⟨t_ref⟩))
    if "attractor_landscape" in args.plots:
        x_edges, y_edges, energy_by_class = compute_attractor_landscapes(
            buckets, dataset, args.num_classes, cache_dir, dataset_hash
        )
        plot_attractor_landscape_overlay(
            x_edges, y_edges, energy_by_class, title_prefix, fig_dir
        )
        plot_attractor_landscape_per_digit(
            x_edges,
            y_edges,
            energy_by_class,
            title_prefix,
            fig_dir,
            convergence_by_class=convergence_by_class,
        )
        # Additional normalized density views (reuse energy cache)
        plot_attractor_density_overlay(
            x_edges, y_edges, energy_by_class, title_prefix, fig_dir
        )
        plot_attractor_density_per_digit(
            x_edges,
            y_edges,
            energy_by_class,
            title_prefix,
            fig_dir,
            convergence_by_class=convergence_by_class,
        )

    # Attractor landscape 3D overlay
    if "attractor_landscape_3d" in args.plots:
        x_edges, y_edges, energy_by_class = compute_attractor_landscapes(
            buckets, dataset, args.num_classes, cache_dir, dataset_hash
        )
        plot_attractor_landscape_3d_overlay(
            x_edges, y_edges, energy_by_class, title_prefix, fig_dir
        )
        plot_attractor_landscape_3d_per_digit(
            x_edges, y_edges, energy_by_class, title_prefix, fig_dir
        )
        # Additional normalized density 3D views (reuse energy cache)
        plot_attractor_density_3d_overlay(
            x_edges, y_edges, energy_by_class, title_prefix, fig_dir
        )
        plot_attractor_density_3d_per_digit(
            x_edges, y_edges, energy_by_class, title_prefix, fig_dir
        )

    # Animated attractor landscape per digit
    if "attractor_landscape_animated" in args.plots:
        x_edges_anim, y_edges_anim, frames_by_class, max_ticks = (
            compute_attractor_landscapes_animated(
                buckets,
                dataset,
                args.num_classes,
                args.animation_frames,
                cache_dir,
                dataset_hash,
                convergence_final_ticks=args.convergence_final_ticks,
            )
        )
        # Reload convergence points from cache (computed by animated landscapes)
        if convergence_by_class is None or not convergence_by_class:
            convergence_by_class = compute_convergence_points(
                buckets,
                dataset,
                args.num_classes,
                args.convergence_final_ticks,
                cache_dir,
                dataset_hash,
            )
        plot_attractor_landscape_animated_per_digit(
            x_edges_anim,
            y_edges_anim,
            frames_by_class,
            max_ticks,
            args.animation_frames,
            title_prefix,
            fig_dir,
            convergence_by_class=convergence_by_class,
        )

    # S(t) variance decay per digit
    if "s_variance_decay" in args.plots:
        lbls, svar = compute_s_variance_decay(
            buckets,
            dataset,
            neurons_per_layer,
            layer_offsets,
            total_neurons,
            args.num_classes,
            cache_dir,
            dataset_hash,
        )
        plot_s_variance_decay(lbls, svar, title_prefix, fig_dir)
        # Per-digit lines in subdirectories
        if lbls.size > 0:
            T = svar.shape[1]
            t_axis = list(range(T))
            for i, lbl in enumerate(lbls.tolist()):
                subdir = os.path.join(fig_dir, f"per_digit/digit_{int(lbl)}")
                fig = go.Figure(
                    data=go.Scatter(
                        x=t_axis, y=svar[i, :], mode="lines", name=f"Digit {int(lbl)}"
                    )
                )
                fig.update_layout(
                    title=dict(
                        text=f"{title_prefix} – S(t) Variance Decay (Digit {int(lbl)})",
                        font=dict(size=18),
                    ),
                    xaxis_title="t",
                    yaxis_title="Var[S] across neurons",
                    width=1400,
                    height=900,
                )
                save_figure(fig, subdir, "s_variance_decay")

    # t_ref bounds compliance (box per layer with theoretical bounds)
    if "tref_bounds_box" in args.plots:
        tref_samples_by_layer = compute_tref_samples_by_layer(
            dataset,
            neurons_per_layer,
            args.tref_box_sample,
            cache_dir,
            dataset_hash,
        )
        syn_per_layer = (
            auto_syn_per_layer
            if auto_syn_per_layer is not None
            else parse_synapses_per_layer(
                args.synapses_per_layer, len(neurons_per_layer), args.synapses_default
            )
        )
        plot_tref_bounds_box(
            tref_samples_by_layer,
            (auto_c if (auto_c is not None) else args.homeostat_c),
            syn_per_layer,
            title_prefix,
            fig_dir,
        )
        # Per-digit tref distributions
        for lbl in sorted(buckets.keys()):
            indices = buckets[lbl]
            tref_samples_sub = compute_tref_samples_by_layer(
                dataset,
                neurons_per_layer,
                args.tref_box_sample,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
                indices=indices,
            )
            subdir = os.path.join(fig_dir, f"per_digit/digit_{int(lbl)}")
            plot_tref_bounds_box(
                tref_samples_sub,
                (auto_c if (auto_c is not None) else args.homeostat_c),
                syn_per_layer,
                title_prefix,
                subdir,
            )

    # F̄ stability over training (epoch-wise)
    if "favg_stability" in args.plots:
        epochs, means = compute_favg_stability_over_epochs(
            buckets,
            dataset,
            args.epoch_field,
            args.num_epoch_bins,
            cache_dir,
            dataset_hash,
        )
        plot_favg_stability(epochs, means, title_prefix, fig_dir)
        # Per-digit stability
        for lbl in sorted(buckets.keys()):
            indices = buckets[lbl]
            e_sub, m_sub = compute_favg_stability_over_epochs(
                buckets,
                dataset,
                args.epoch_field,
                args.num_epoch_bins,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
                indices=indices,
            )
            subdir = os.path.join(fig_dir, f"per_digit/digit_{int(lbl)}")
            plot_favg_stability(e_sub, m_sub, title_prefix, subdir)

    # Homeostatic response curve
    if "homeostatic_response" in args.plots:
        syn_per_layer = (
            auto_syn_per_layer
            if auto_syn_per_layer is not None
            else parse_synapses_per_layer(
                args.synapses_per_layer, len(neurons_per_layer), args.synapses_default
            )
        )
        plot_homeostatic_response_curve(
            aggregates,
            (auto_c if (auto_c is not None) else args.homeostat_c),
            syn_per_layer,
            args.response_bins,
            args.response_scope,
            title_prefix,
            fig_dir,
        )
        # Per-digit: recompute aggregates using only records of that digit
        for lbl in sorted(buckets.keys()):
            indices = buckets[lbl]
            if not indices:
                continue
            sub_aggs = compute_per_neuron_aggregates(
                dataset,
                neurons_per_layer,
                layer_offsets,
                total_neurons,
                args.num_classes,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
                indices=indices,
            )
            subdir = os.path.join(fig_dir, f"per_digit/digit_{int(lbl)}")
            plot_homeostatic_response_curve(
                sub_aggs,
                (auto_c if (auto_c is not None) else args.homeostat_c),
                syn_per_layer,
                args.response_bins,
                args.response_scope,
                title_prefix,
                subdir,
            )

    print("\n✓ Visualization complete.")


if __name__ == "__main__":
    main()
