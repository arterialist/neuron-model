import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional, cast, Iterator

import numpy as np
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import csv

# Optional imports for enhanced data loading
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

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
    """Return a short MD5 hash for the dataset file using shell command.

    Prefer `md5sum` (GNU coreutils). On macOS, fall back to `md5 -q` if needed.
    Returns the first 16 hex characters for stable short tokens.
    """
    try:
        # Try GNU md5sum
        result = subprocess.run(
            ["md5sum", file_path], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout:
            md5_hex = result.stdout.strip().split()[0]
            return md5_hex.lower()[:16]
    except FileNotFoundError:
        pass

    # Fallback: macOS `md5 -q`
    try:
        result = subprocess.run(
            ["md5", "-q", file_path], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout:
            md5_hex = result.stdout.strip().split()[0]
            return md5_hex.lower()[:16]
    except FileNotFoundError:
        pass

    raise RuntimeError(
        "Neither 'md5sum' nor 'md5' was found on PATH. Please install coreutils or provide md5."
    )


def get_cache_dir(base_output_dir: str) -> str:
    """Return (and create) the cache directory for intermediates."""
    cache_dir = os.path.join(base_output_dir, f"cache_{CACHE_VERSION}")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_activity_data(path: str, loader: str = "json") -> List[Dict[str, Any]]:
    """Loads the activity dataset from a JSON file.

    Args:
        path: Path to the JSON file
        loader: Loading method - 'json', 'pandas', or 'streaming'

    Supports both top-level array and {"records": [...]} formats.
    """
    if loader == "pandas":
        return load_activity_data_pandas(path)
    elif loader == "streaming":
        return load_activity_data_streaming(path)
    elif loader == "json":
        return load_activity_data_json(path)
    else:
        raise ValueError(f"Unknown loader: {loader}")


def load_activity_data_json(path: str) -> List[Dict[str, Any]]:
    """Loads the activity dataset from a JSON file using standard json module.

    Supports both top-level array and {"records": [...]} formats.
    """
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported dataset JSON format")


def load_activity_data_pandas(path: str) -> List[Dict[str, Any]]:
    """Loads the activity dataset using pandas for potentially better performance.

    Args:
        path: Path to the JSON file

    Returns:
        List of record dictionaries
    """
    if not PANDAS_AVAILABLE:
        print("Warning: pandas not available, falling back to standard JSON loader")
        return load_activity_data_json(path)

    print("Loading data with pandas...")
    try:
        # First detect JSON format
        with open(path, "rb") as fh_probe:
            head = fh_probe.read(2048)
            first_non_ws = None
            for b in head:
                if chr(b) not in [" ", "\n", "\r", "\t"]:
                    first_non_ws = chr(b)
                    break

        if first_non_ws == "{":
            # Object format - need to handle records array
            print("Detected object format, using manual parsing with pandas...")
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "records" in data:
                df = pd.DataFrame(data["records"])
            else:
                raise ValueError("Object format detected but no 'records' array found")
        elif first_non_ws == "[":
            # Array format - standard JSON lines
            print("Detected array format, using standard pandas loading...")
            # Try different chunk sizes for large files
            try:
                df = pd.read_json(
                    path, orient="records", lines=False, chunksize=1000000
                )
                # Concatenate chunks if needed
                if hasattr(df, "__iter__"):
                    df_list = list(df)
                    if len(df_list) > 1:
                        df = pd.concat(df_list, ignore_index=True)
                    else:
                        df = df_list[0]
            except:
                # Fallback to standard loading
                df = pd.read_json(path, orient="records", lines=False)
        else:
            raise ValueError(f"Unsupported JSON format starting with: {first_non_ws}")

        # Convert DataFrame back to list of dictionaries
        records = df.to_dict("records")
        print(f"Loaded {len(records)} records with pandas")

        return records

    except Exception as e:
        print(f"Pandas loading failed: {e}")
        print("Falling back to standard JSON loader")
        return load_activity_data_json(path)


def load_activity_data_streaming(path: str) -> List[Dict[str, Any]]:
    """Loads the activity dataset using streaming JSON parser for memory efficiency.

    Args:
        path: Path to the JSON file

    Returns:
        List of record dictionaries
    """
    try:
        import ijson  # type: ignore
    except ImportError:
        print(
            "Warning: ijson not available for streaming, falling back to standard JSON loader"
        )
        return load_activity_data_json(path)

    print("Loading data with streaming JSON parser...")
    # Detect top-level container by inspecting the first non-whitespace byte
    with open(path, "rb") as fh_probe:
        head = fh_probe.read(2048)
        first_non_ws = None
        for b in head:
            if chr(b) not in [" ", "\n", "\r", "\t"]:
                first_non_ws = chr(b)
                break

    if first_non_ws is None:
        return []  # empty file

    records = []
    try:
        if first_non_ws == "{":
            # Object with records array
            with open(path, "rb") as fh:
                for rec in tqdm(ijson.items(fh, "records.item"), desc="Streaming JSON"):
                    records.append(rec)
        elif first_non_ws == "[":
            # Top-level array
            with open(path, "rb") as fh:
                for rec in tqdm(ijson.items(fh, "item"), desc="Streaming JSON"):
                    records.append(rec)
        else:
            raise ValueError("Unsupported JSON format for streaming")

        print(f"Loaded {len(records)} records with streaming parser")

        return records

    except Exception as e:
        print(f"Streaming JSON loading failed: {e}")
        print("Falling back to standard JSON loader")
        return load_activity_data_json(path)


def validate_labels_or_die(records: List[Dict[str, Any]], num_classes: int) -> None:
    """Ensure every record has a valid label in [0, num_classes-1].

    Intended for MNIST (10 classes), but works for any num_classes.
    Raises ValueError with a concise diagnostic if violations are found.
    """
    if not records:
        raise ValueError("Dataset contains no records.")

    min_label = 0
    max_label = num_classes - 1
    missing = 0
    out_of_range = 0
    non_int = 0
    labels_found: set[int] = set()

    for rec in records:
        if "label" not in rec:
            missing += 1
            continue
        val = rec.get("label")
        try:
            sval = int(val)
        except Exception:
            non_int += 1
            continue
        if not (min_label <= sval <= max_label):
            out_of_range += 1
            continue
        labels_found.add(sval)

    if missing or non_int or out_of_range:
        problems = []
        if missing:
            problems.append(f"missing label: {missing}")
        if non_int:
            problems.append(f"non-integer label: {non_int}")
        if out_of_range:
            problems.append(f"labels outside [{min_label},{max_label}]: {out_of_range}")
        raise ValueError(
            "Invalid labels in dataset (expected supervised labels). "
            + "; ".join(problems)
        )


def sort_records_deterministically(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return records sorted by (image_index, tick, label) for deterministic downstream processing."""
    max_tick_fallback = 10**12
    return sorted(
        records,
        key=lambda rec: (
            int(rec.get("image_index", -1)),
            int(rec.get("tick", max_tick_fallback)),
            int(rec.get("label", -1)),
        ),
    )


def group_by_image(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Group by (label, image_index) to collect per-tick records for each image."""
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    labels_present = any(("label" in r) for r in records)
    for rec in records:
        label = rec.get("label", -1)
        if label == -1 and not labels_present:
            label = 0
        img_idx = rec.get("image_index", -1)
        key = (int(label), int(img_idx))
        buckets.setdefault(key, []).append(rec)
    for key in buckets:
        buckets[key].sort(key=lambda r: r.get("tick", 0))
    return buckets


def available_labels(
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]], num_classes: int
) -> List[int]:
    labels = sorted({int(k[0]) for k in buckets.keys() if 0 <= int(k[0]) < num_classes})
    return labels


def filter_records_by_label(
    records: List[Dict[str, Any]], lbl: int
) -> List[Dict[str, Any]]:
    return [rec for rec in records if int(rec.get("label", -1)) == int(lbl)]


def filter_buckets_by_label(
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]], lbl: int
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    return {k: v for k, v in buckets.items() if int(k[0]) == int(lbl)}


def infer_network_structure(
    records: List[Dict[str, Any]],
) -> Tuple[List[int], List[int], int]:
    """Infer [neurons_per_layer], [layer_offsets], and total_neurons from first record."""
    if not records:
        return [], [], 0
    first_layers = records[0].get("layers", [])
    neurons_per_layer: List[int] = []
    for layer in first_layers:
        num_neurons = len(layer.get("S", []))
        neurons_per_layer.append(num_neurons)
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
    """Save Plotly figure to HTML, PNG and SVG."""
    global _STATIC_EXPORT_DISABLED_ON_ERROR
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"{base_name}.html")
    png_path = os.path.join(out_dir, f"{base_name}.png")
    svg_path = os.path.join(out_dir, f"{base_name}.svg")
    fig.write_html(html_path)
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
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]],
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

    # Build per-class list of images
    class_to_images: Dict[int, List[List[Dict[str, Any]]]] = {}
    for (label, _img_idx), recs in buckets.items():
        if 0 <= label < num_classes:
            class_to_images.setdefault(label, []).append(recs)

    heatmaps: Dict[int, np.ndarray] = {}
    for label in sorted(class_to_images.keys()):
        images = class_to_images[label]
        if not images:
            continue
        min_T = min(len(recs) for recs in images)
        if min_T <= 0 or total_neurons <= 0:
            continue
        sums = np.zeros((total_neurons, min_T), dtype=np.float32)
        for recs in images:
            for t in range(min_T):
                layers = recs[t].get("layers", [])
                for li, layer in enumerate(layers):
                    S = layer.get("S", [])
                    if not S:
                        continue
                    S_arr = np.asarray(S, dtype=np.float32)
                    if S_arr.size == 0:
                        continue
                    base = layer_offsets[li]
                    end = base + min(neurons_per_layer[li], S_arr.shape[0])
                    sums[base:end, t] += S_arr[: end - base]
        sums /= max(1, len(images))
        heatmaps[label] = sums

    # Save cache (positional arrays to satisfy strict type stubs)
    if heatmaps:
        labels_sorted = np.array(sorted(heatmaps.keys()), dtype=np.int32)
        arrays = [heatmaps[int(lbl)] for lbl in labels_sorted]
        # Save as arr_0 = labels, arr_1.. = heatmaps in the same order
        np.savez_compressed(cache_path, labels_sorted, *arrays)
    return heatmaps


def compute_per_neuron_aggregates(
    records: List[Dict[str, Any]],
    neurons_per_layer: List[int],
    layer_offsets: List[int],
    total_neurons: int,
    num_classes: int,
    cache_dir: str,
    dataset_hash: str,
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

    # Process all ticks across all images
    for rec in tqdm(records, desc="Aggregating per-neuron stats"):
        label = int(rec.get("label", -1))
        if not (0 <= label < num_classes):
            continue
        layers = rec.get("layers", [])
        for li, layer in enumerate(layers):
            F_avg = layer.get("F_avg", [])
            t_ref = layer.get("t_ref", [])
            if not F_avg or not t_ref:
                # Skip if either is missing
                continue
            F_arr = np.asarray(F_avg, dtype=np.float32)
            T_arr = np.asarray(t_ref, dtype=np.float32)
            n = min(neurons_per_layer[li], F_arr.shape[0], T_arr.shape[0])
            if n <= 0:
                continue
            base = layer_offsets[li]
            idx_slice = slice(base, base + n)
            mean_favg_sum[idx_slice] += F_arr[:n]
            mean_tref_sum[idx_slice] += T_arr[:n]
            count_points[idx_slice] += 1

            favg_sum_by_class[idx_slice, label] += F_arr[:n]
            favg_count_by_class[idx_slice, label] += 1

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
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]],
    neurons_per_layer: List[int],
    layer_offsets: List[int],
    total_neurons: int,
    cache_dir: str,
    dataset_hash: str,
) -> Tuple[np.ndarray, int]:
    """Compute mean t_ref(t) over time across all images for every neuron.

    Returns (timeline_matrix, T_min_all), where timeline_matrix shape is [total_neurons, T_min_all].
    """
    cache_path = os.path.join(cache_dir, f"{dataset_hash}_tref_timelines.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["timelines"], int(data["T_min_all"])  # type: ignore[index]

    all_images = list(buckets.values())
    if not all_images or total_neurons <= 0:
        return np.zeros((0, 0), dtype=np.float32), 0
    T_min_all = min(len(recs) for recs in all_images)
    if T_min_all <= 0:
        return np.zeros((0, 0), dtype=np.float32), 0

    sums = np.zeros((total_neurons, T_min_all), dtype=np.float64)
    for recs in tqdm(all_images, desc="Aggregating t_ref timelines"):
        for t in range(T_min_all):
            layers = recs[t].get("layers", [])
            for li, layer in enumerate(layers):
                tref = layer.get("t_ref", [])
                if not tref:
                    continue
                T_arr = np.asarray(tref, dtype=np.float32)
                n = min(neurons_per_layer[li], T_arr.shape[0])
                if n <= 0:
                    continue
                base = layer_offsets[li]
                sums[base : base + n, t] += T_arr[:n]
    sums /= max(1, len(all_images))
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
        rows=rows, cols=cols, subplot_titles=[f"Digit {l}" for l in labels_sorted]
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
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]],
    neurons_per_layer: List[int],
    cache_dir: str,
    dataset_hash: str,
) -> Tuple[np.ndarray, int]:
    """Compute mean S(t) per layer across all images.

    Returns (timeline_matrix, T_min_all) with shape [num_layers, T_min_all].
    Caches results for reuse.
    """
    cache_path = os.path.join(cache_dir, f"{dataset_hash}_layerwise_S_timeline.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["layer_s"], int(data["T_min_all"])  # type: ignore[index]

    all_images = list(buckets.values())
    if not all_images:
        return np.zeros((0, 0), dtype=np.float32), 0
    num_layers = len(neurons_per_layer)
    T_min_all = min(len(recs) for recs in all_images)
    if T_min_all <= 0 or num_layers == 0:
        return np.zeros((0, 0), dtype=np.float32), 0

    sums = np.zeros((num_layers, T_min_all), dtype=np.float64)
    counts = np.zeros((num_layers, T_min_all), dtype=np.int64)

    for recs in tqdm(all_images, desc="Aggregating layer-wise S(t)"):
        for t in range(T_min_all):
            layers = recs[t].get("layers", [])
            for li, layer in enumerate(layers[:num_layers]):
                S = layer.get("S", [])
                if not S:
                    continue
                arr = np.asarray(S, dtype=np.float32)
                if arr.size == 0:
                    continue
                sums[li, t] += float(np.mean(arr))
                counts[li, t] += 1

    counts_safe = np.maximum(counts, 1)
    layer_s = (sums / counts_safe).astype(np.float32)
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
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]],
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
    if not buckets:
        return
    # Order images deterministically by (label, image_index)
    keys = sorted(buckets.keys(), key=lambda k: (k[0], k[1]))
    if max_images is not None and max_images > 0:
        keys = keys[:max_images]

    xs: List[int] = []
    ys: List[int] = []
    cs: List[float] = []
    t_cursor = 0

    for key in tqdm(keys, desc="Building spike raster"):
        recs = buckets[key]
        for rec in recs:
            layers = rec.get("layers", [])
            for li, layer in enumerate(layers):
                fired = layer.get("fired", [])
                tref = layer.get("t_ref", [])
                if not fired:
                    continue
                f_arr = np.asarray(fired, dtype=np.int32)
                tref_arr = (
                    np.asarray(tref, dtype=np.float32)
                    if tref
                    else np.zeros_like(f_arr, dtype=np.float32)
                )
                n = (
                    min(neurons_per_layer[li], f_arr.shape[0], tref_arr.shape[0])
                    if tref
                    else min(neurons_per_layer[li], f_arr.shape[0])
                )
                base = layer_offsets[li]
                for ni in range(n):
                    if f_arr[ni] > 0:
                        xs.append(t_cursor)
                        ys.append(base + ni)
                        cs.append(float(tref_arr[ni]) if tref else 0.0)
            t_cursor += 1
        # gap between images
        t_cursor += max(0, int(gap))

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
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]],
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
                F_t = data.get(f"arr_{i+1}")
                T_t = data.get(f"arr_{i+2}")
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

    # Build per-class image list
    class_to_images: Dict[int, List[List[Dict[str, Any]]]] = {}
    for (label, _img_idx), recs in buckets.items():
        if 0 <= label < num_classes:
            class_to_images.setdefault(label, []).append(recs)

    result: Dict[int, Dict[str, np.ndarray]] = {}
    for lbl, images in class_to_images.items():
        if not images:
            continue
        T_min = min(len(recs) for recs in images)
        if T_min <= 0:
            continue
        S_series = np.zeros((T_min,), dtype=np.float64)
        F_series = np.zeros((T_min,), dtype=np.float64)
        T_series = np.zeros((T_min,), dtype=np.float64)
        for recs in images:
            for t in range(T_min):
                layers = recs[t].get("layers", [])
                s_vals = []
                f_vals = []
                tr_vals = []
                for layer in layers:
                    S = layer.get("S", [])
                    if S:
                        s_arr = np.asarray(S, dtype=np.float32)
                        if s_arr.size:
                            s_vals.append(float(np.mean(s_arr)))
                    F = layer.get("F_avg", [])
                    if F:
                        f_arr = np.asarray(F, dtype=np.float32)
                        if f_arr.size:
                            f_vals.append(float(np.mean(f_arr)))
                    Tref = layer.get("t_ref", [])
                    if Tref:
                        t_arr = np.asarray(Tref, dtype=np.float32)
                        if t_arr.size:
                            tr_vals.append(float(np.mean(t_arr)))
                if s_vals:
                    S_series[t] += float(np.mean(s_vals))
                if f_vals:
                    F_series[t] += float(np.mean(f_vals))
                if tr_vals:
                    T_series[t] += float(np.mean(tr_vals))
        S_series /= max(1, len(images))
        F_series /= max(1, len(images))
        T_series /= max(1, len(images))
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
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]],
    neurons_per_layer: List[int],
    sample_limit_per_layer: int,
    cache_dir: str,
    dataset_hash: str,
) -> Dict[int, np.ndarray]:
    """Reservoir-sample t_ref values per layer across all ticks and images.

    Returns mapping: layer_idx -> sampled t_ref values (np.ndarray).
    """
    cache_path = os.path.join(
        cache_dir, f"{dataset_hash}_tref_samples_per_layer_{sample_limit_per_layer}.npz"
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
    reservoirs: Dict[int, List[float]] = {
        li: [] for li in range(len(neurons_per_layer))
    }
    counts: Dict[int, int] = {li: 0 for li in range(len(neurons_per_layer))}

    for recs in tqdm(buckets.values(), desc="Sampling t_ref per layer"):
        for rec in recs:
            layers = rec.get("layers", [])
            for li, layer in enumerate(layers[: len(neurons_per_layer)]):
                tref = layer.get("t_ref", [])
                if not tref:
                    continue
                for val in tref:
                    counts[li] += 1
                    try:
                        v = float(val)
                    except Exception:
                        continue
                    if len(reservoirs[li]) < sample_limit_per_layer:
                        reservoirs[li].append(v)
                    else:
                        j = rng.integers(0, counts[li])
                        if j < sample_limit_per_layer:
                            reservoirs[li][int(j)] = v

    result = {li: np.asarray(reservoirs[li], dtype=np.float32) for li in reservoirs}
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
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]],
    epoch_field: str,
    num_epoch_bins: int,
    cache_dir: str,
    dataset_hash: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean F_avg per epoch.

    If epoch_field exists in any record, group by that.
    Otherwise, if num_epoch_bins > 0, bin images sequentially by label,image_index order.
    Returns (epochs, means) arrays.
    """
    cache_key = f"{dataset_hash}_favg_stability_{epoch_field}_bins{num_epoch_bins}.npz"
    cache_path = os.path.join(cache_dir, cache_key)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["epochs"], data["means"]

    # Detect epoch field availability
    has_epoch = False
    for recs in buckets.values():
        for rec in recs:
            if epoch_field in rec:
                has_epoch = True
                break
        if has_epoch:
            break

    epoch_to_sum = {}
    epoch_to_count = {}

    if has_epoch:
        for recs in buckets.values():
            for rec in recs:
                val = rec.get(epoch_field)
                try:
                    ep = int(val) if val is not None else None
                except Exception:
                    ep = None
                if ep is None:
                    continue
                layers = rec.get("layers", [])
                for layer in layers:
                    F = layer.get("F_avg", [])
                    if F:
                        arr = np.asarray(F, dtype=np.float32)
                        if arr.size:
                            epoch_to_sum[ep] = epoch_to_sum.get(ep, 0.0) + float(
                                np.mean(arr)
                            )
                            epoch_to_count[ep] = epoch_to_count.get(ep, 0) + 1
    elif num_epoch_bins and num_epoch_bins > 0:
        # Create ordered list of images and split into bins
        keys = sorted(buckets.keys(), key=lambda k: (k[0], k[1]))
        if not keys:
            # Save empty cache to avoid recomputation
            epochs_arr = np.zeros((0,), dtype=np.int32)
            means_arr = np.zeros((0,), dtype=np.float32)
            np.savez_compressed(cache_path, epochs=epochs_arr, means=means_arr)
            return epochs_arr, means_arr
        images_per_bin = max(1, int(np.ceil(len(keys) / num_epoch_bins)))
        for bi in range(num_epoch_bins):
            start = bi * images_per_bin
            stop = min(len(keys), (bi + 1) * images_per_bin)
            if start >= stop:
                break
            ep = int(bi + 1)
            for key in keys[start:stop]:
                recs = buckets[key]
                for rec in recs:
                    for layer in rec.get("layers", []):
                        F = layer.get("F_avg", [])
                        if F:
                            arr = np.asarray(F, dtype=np.float32)
                            if arr.size:
                                epoch_to_sum[ep] = epoch_to_sum.get(ep, 0.0) + float(
                                    np.mean(arr)
                                )
                                epoch_to_count[ep] = epoch_to_count.get(ep, 0) + 1
    else:
        # No epoch data available - save empty cache to avoid recomputation
        epochs_arr = np.zeros((0,), dtype=np.int32)
        means_arr = np.zeros((0,), dtype=np.float32)
        np.savez_compressed(cache_path, epochs=epochs_arr, means=means_arr)
        return epochs_arr, means_arr

    if not epoch_to_sum:
        # No valid data found - save empty cache
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
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]],
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

    # Group by label
    class_to_images: Dict[int, List[List[Dict[str, Any]]]] = {}
    for (label, _img_idx), recs in buckets.items():
        if 0 <= label < num_classes:
            class_to_images.setdefault(label, []).append(recs)

    labels_sorted = np.array(sorted(class_to_images.keys()), dtype=np.int32)
    if labels_sorted.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 0), dtype=np.float32)

    # Determine global minimal T across all labels/images
    global_T = None
    for lbl in labels_sorted:
        images = class_to_images[int(lbl)]
        if not images:
            continue
        min_T = min(len(recs) for recs in images)
        global_T = min(min_T, global_T) if global_T is not None else min_T
    if global_T is None or global_T <= 0:
        return labels_sorted, np.zeros((labels_sorted.size, 0), dtype=np.float32)

    # Compute average variance across images for each label at each t
    series = np.zeros((labels_sorted.size, int(global_T)), dtype=np.float32)
    for li_lbl, lbl in enumerate(labels_sorted):
        images = class_to_images[int(lbl)]
        if not images:
            continue
        accum = np.zeros((int(global_T),), dtype=np.float64)
        count = 0
        for recs in images:
            if len(recs) < global_T:
                continue
            for t in range(int(global_T)):
                layers = recs[t].get("layers", [])
                # Collect S from all neurons across all layers
                vals: List[float] = []
                for li, layer in enumerate(layers):
                    S_list = layer.get("S", [])
                    if not S_list:
                        continue
                    arr = np.asarray(S_list, dtype=np.float32)
                    n = min(neurons_per_layer[li], arr.shape[0])
                    if n > 0:
                        vals.extend(arr[:n].tolist())
                if vals:
                    v = float(np.var(np.asarray(vals, dtype=np.float32)))
                else:
                    v = 0.0
                accum[t] += v
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
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]],
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

    # Iterate images grouped by label
    class_to_images: Dict[int, List[List[Dict[str, Any]]]] = {}
    for (label, _img), recs in buckets.items():
        if 0 <= label < num_classes:
            class_to_images.setdefault(label, []).append(recs)

    for lbl, images in class_to_images.items():
        for recs in images:
            for rec in recs:
                layers = rec.get("layers", [])
                for li, layer in enumerate(layers):
                    S_list = layer.get("S", [])
                    if not S_list:
                        continue
                    arr = np.asarray(S_list, dtype=np.float32)
                    n = min(neurons_per_layer[li], arr.shape[0])
                    if n <= 0:
                        continue
                    base = layer_offsets[li]
                    sums[base : base + n, lbl] += arr[:n]
                    counts[base : base + n, lbl] += 1

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
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]],
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

    # If only a single neuron is selected, there can be no pairwise correlations/edges.
    if len(selected) < 2:
        nodes = np.asarray(selected, dtype=np.int32)
        u = np.array([], dtype=np.int32)
        v = np.array([], dtype=np.int32)
        np.savez_compressed(cache_path, nodes=nodes, u=u, v=v)
        return nodes, u, v

    # Build time series by concatenating images
    series: List[List[int]] = [[] for _ in selected]
    keys = sorted(buckets.keys(), key=lambda k: (k[0], k[1]))
    if max_images is not None and max_images > 0:
        keys = keys[:max_images]
    for key in keys:
        recs = buckets[key]
        for rec in recs:
            layers = rec.get("layers", [])
            for li, layer in enumerate(layers):
                fired = layer.get("fired", [])
                if not fired:
                    continue
                f_arr = np.asarray(fired, dtype=np.int32)
                n = min(neurons_per_layer[li], f_arr.shape[0])
                base = layer_offsets[li]
                for idx, g in enumerate(selected):
                    if base <= g < base + n:
                        series[idx].append(int(f_arr[g - base] > 0))
                    else:
                        series[idx].append(0)
        # gap between images
        for idx in range(len(series)):
            series[idx].extend([0] * max(0, int(gap)))

    # Convert to array (N x T)
    max_len = max((len(s) for s in series), default=0)
    if max_len == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
    arr = np.zeros((len(series), max_len), dtype=np.float32)
    for i, s in enumerate(series):
        if s:
            arr[i, : len(s)] = np.asarray(s, dtype=np.float32)

    # Correlation matrix
    if arr.shape[1] < 2:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
    corr = np.corrcoef(arr)
    np.fill_diagonal(corr, 0.0)
    # Threshold edges
    u_list: List[int] = []
    v_list: List[int] = []
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if float(corr[i, j]) >= float(threshold):
                u_list.append(i)
                v_list.append(j)

    nodes = np.asarray(selected, dtype=np.int32)
    u = np.asarray(u_list, dtype=np.int32)
    v = np.asarray(v_list, dtype=np.int32)
    np.savez_compressed(cache_path, nodes=nodes, u=u, v=v)
    return nodes, u, v


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


def compute_attractor_landscapes(
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]],
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

    for (label, _img_idx), recs in buckets.items():
        if not (0 <= int(label) < num_classes):
            continue
        for rec in recs:
            layers = rec.get("layers", [])
            s_vals_local: List[float] = []
            t_vals_local: List[float] = []
            for layer in layers:
                S_list = layer.get("S", [])
                if S_list:
                    arr = np.asarray(S_list, dtype=np.float32)
                    if arr.size:
                        s_vals_local.append(float(np.mean(arr)))
                T_list = layer.get("t_ref", [])
                if T_list:
                    arrt = np.asarray(T_list, dtype=np.float32)
                    if arrt.size:
                        t_vals_local.append(float(np.mean(arrt)))
            if s_vals_local and t_vals_local:
                s_mean = float(np.mean(s_vals_local))
                t_mean = float(np.mean(t_vals_local))
                points_by_class.setdefault(int(label), []).append((s_mean, t_mean))
                s_all.append(s_mean)
                t_all.append(t_mean)

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
) -> None:
    if not energy_by_class:
        return
    log_plot_start("attractor_landscape", "per-digit heatmaps")
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    for lbl, z in sorted(energy_by_class.items()):
        subdir = os.path.join(out_dir, f"per_digit/digit_{int(lbl)}")
        fig = go.Figure(
            data=go.Heatmap(
                x=x_centers,
                y=y_centers,
                z=z,
                colorscale="Viridis",
                colorbar=dict(title="Energy"),
            )
        )
        fig.update_layout(
            title=dict(
                text=f"{title_prefix} – Attractor Landscape (Digit {int(lbl)})",
                font=dict(size=18),
            ),
            xaxis_title="⟨S⟩",
            yaxis_title="⟨t_ref⟩",
            width=1400,
            height=900,
        )
        save_figure(fig, subdir, "attractor_landscape")

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
    log_plot_end("attractor_landscape", "per-digit heatmaps")


def plot_attractor_density_per_digit(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    energy_by_class: Dict[int, np.ndarray],
    title_prefix: str,
    out_dir: str,
) -> None:
    if not energy_by_class:
        return
    log_plot_start("attractor_landscape_density", "per-digit heatmaps")
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    for lbl, z_energy in sorted(energy_by_class.items()):
        z_density = _energy_to_density(z_energy)
        subdir = os.path.join(out_dir, f"per_digit/digit_{int(lbl)}")
        fig = go.Figure(
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
        fig.update_layout(
            title=dict(
                text=f"{title_prefix} – Attractor Density (Digit {int(lbl)})",
                font=dict(size=18),
            ),
            xaxis_title="⟨S⟩",
            yaxis_title="⟨t_ref⟩",
            width=1400,
            height=900,
        )
        save_figure(fig, subdir, "attractor_landscape_density")

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
        "--loader",
        type=str,
        default="json",
        choices=["json", "pandas", "streaming"],
        help=(
            "Data loading method: 'json' (standard), 'pandas' (DataFrame-based), "
            "or 'streaming' (memory-efficient with ijson). "
            "Default: json. Use --show-loaders to see availability."
        ),
    )
    args = parser.parse_args()

    # Check loader availability and inform user
    if args.loader == "pandas" and not PANDAS_AVAILABLE:
        print("Warning: pandas not installed. Install with 'pip install pandas'")
        print("Falling back to standard JSON loader")
    elif args.loader == "streaming":
        try:
            import ijson  # type: ignore
        except ImportError:
            print("Warning: ijson not installed. Install with 'pip install ijson'")
            print("Falling back to standard JSON loader")

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

    print(f"Checking if all requested plots are cached...")
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
    )

    if all_cached:
        print("All requested plots are cached. Skipping data loading and computation.")
        # Reconstruct minimal network structure from aggregates cache
        aggregates_cache_path = os.path.join(
            cache_dir, f"{dataset_hash}_per_neuron_aggregates.npz"
        )
        # Initialize placeholders for structures we may not need when cached
        buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        ordered_records: List[Dict[str, Any]] = []
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
        print(f"Using loader: {args.loader}")
        records = load_activity_data(args.input_file, loader=args.loader)
        print(f"Loaded {len(records)} records from {args.input_file}")
        # Enforce supervised MNIST-style labels 0..num_classes-1
        validate_labels_or_die(records, args.num_classes)
        ordered_records = sort_records_deterministically(records)
        buckets = group_by_image(ordered_records)
        neurons_per_layer, layer_offsets, total_neurons = infer_network_structure(
            ordered_records
        )

        if total_neurons == 0:
            print("No neurons inferred from dataset. Exiting.")
            return

    title_prefix = "Network Activity"

    # Optional: derive c and synapses per layer from network config
    auto_c: Optional[float] = None
    auto_syn_per_layer: Optional[List[int]] = None
    if args.network_config:
        auto_c, auto_syn_per_layer = extract_homeostat_from_network(
            args.network_config, len(neurons_per_layer)
        )

    # Configure static export behavior from CLI
    global SAVE_STATIC_IMAGES
    SAVE_STATIC_IMAGES = not args.skip_static_images

    # Note: plots list and exclusions have already been processed earlier before cache check

    # Precompute/load aggregates depending on cache status
    if not all_cached:
        aggregates = compute_per_neuron_aggregates(
            ordered_records,
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
        for lbl in available_labels(buckets, args.num_classes):
            sub_records = filter_records_by_label(ordered_records, lbl)
            if not sub_records:
                continue
            sub_aggs = compute_per_neuron_aggregates(
                sub_records,
                neurons_per_layer,
                layer_offsets,
                total_neurons,
                args.num_classes,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
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
        for lbl in available_labels(buckets, args.num_classes):
            sub_records = filter_records_by_label(ordered_records, lbl)
            if not sub_records:
                continue
            sub_aggs = compute_per_neuron_aggregates(
                sub_records,
                neurons_per_layer,
                layer_offsets,
                total_neurons,
                args.num_classes,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
            )
            subdir = os.path.join(fig_dir, f"per_digit/digit_{int(lbl)}")
            plot_firing_rate_hist_by_layer(sub_aggs, title_prefix, subdir)

    # t_ref evolution timeline for representative neurons
    if "tref_timeline" in args.plots:
        timelines, T_min_all = compute_tref_timelines(
            buckets,
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
        for lbl in available_labels(buckets, args.num_classes):
            sub_b = filter_buckets_by_label(buckets, lbl)
            tl_sub, T_sub = compute_tref_timelines(
                sub_b,
                neurons_per_layer,
                layer_offsets,
                total_neurons,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
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
            buckets, neurons_per_layer, cache_dir, dataset_hash
        )
        plot_layerwise_S_timeline(layer_s, T_min_all, title_prefix, fig_dir)
        # Per-digit layer-wise S timelines
        for lbl in available_labels(buckets, args.num_classes):
            sub_b = filter_buckets_by_label(buckets, lbl)
            layer_s_sub, T_sub = compute_layerwise_S_timeline(
                sub_b, neurons_per_layer, cache_dir, f"{dataset_hash}_digit{int(lbl)}"
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
            buckets,
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
        for lbl in available_labels(buckets, args.num_classes):
            sub_b = filter_buckets_by_label(buckets, lbl)
            subdir = os.path.join(fig_dir, f"per_digit/digit_{int(lbl)}")
            plot_spike_raster_with_tref(
                sub_b,
                neurons_per_layer,
                layer_offsets,
                args.raster_max_images,
                args.raster_gap,
                args.raster_max_points,
                title_prefix,
                subdir,
                tref_bounds=tref_bounds,
            )

    # Phase portrait 3D
    if "phase_portrait" in args.plots:
        series_by_class = compute_phase_portrait_series(
            buckets, args.num_classes, cache_dir, dataset_hash
        )
        plot_phase_portrait(
            series_by_class,
            title_prefix,
            fig_dir,
            theory_c=auto_c,
            theory_syn_per_layer=auto_syn_per_layer,
        )
        # Per-digit portraits
        for lbl in available_labels(buckets, args.num_classes):
            sub_b = filter_buckets_by_label(buckets, lbl)
            series_sub = compute_phase_portrait_series(
                sub_b, args.num_classes, cache_dir, f"{dataset_hash}_digit{int(lbl)}"
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

    # Attractor landscape (energy surfaces over (⟨S⟩, ⟨t_ref⟩))
    if "attractor_landscape" in args.plots:
        x_edges, y_edges, energy_by_class = compute_attractor_landscapes(
            buckets, args.num_classes, cache_dir, dataset_hash
        )
        plot_attractor_landscape_overlay(
            x_edges, y_edges, energy_by_class, title_prefix, fig_dir
        )
        plot_attractor_landscape_per_digit(
            x_edges, y_edges, energy_by_class, title_prefix, fig_dir
        )
        # Additional normalized density views (reuse energy cache)
        plot_attractor_density_overlay(
            x_edges, y_edges, energy_by_class, title_prefix, fig_dir
        )
        plot_attractor_density_per_digit(
            x_edges, y_edges, energy_by_class, title_prefix, fig_dir
        )

    # Attractor landscape 3D overlay
    if "attractor_landscape_3d" in args.plots:
        x_edges, y_edges, energy_by_class = compute_attractor_landscapes(
            buckets, args.num_classes, cache_dir, dataset_hash
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

    # S(t) variance decay per digit
    if "s_variance_decay" in args.plots:
        lbls, svar = compute_s_variance_decay(
            buckets,
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
            buckets,
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
        for lbl in available_labels(buckets, args.num_classes):
            sub_b = filter_buckets_by_label(buckets, lbl)
            tref_samples_sub = compute_tref_samples_by_layer(
                sub_b,
                neurons_per_layer,
                args.tref_box_sample,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
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
            args.epoch_field,
            args.num_epoch_bins,
            cache_dir,
            dataset_hash,
        )
        plot_favg_stability(epochs, means, title_prefix, fig_dir)
        # Per-digit stability
        for lbl in available_labels(buckets, args.num_classes):
            sub_b = filter_buckets_by_label(buckets, lbl)
            e_sub, m_sub = compute_favg_stability_over_epochs(
                sub_b,
                args.epoch_field,
                args.num_epoch_bins,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
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
        for lbl in available_labels(buckets, args.num_classes):
            sub_records = filter_records_by_label(ordered_records, lbl)
            if not sub_records:
                continue
            sub_aggs = compute_per_neuron_aggregates(
                sub_records,
                neurons_per_layer,
                layer_offsets,
                total_neurons,
                args.num_classes,
                cache_dir,
                f"{dataset_hash}_digit{int(lbl)}",
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
