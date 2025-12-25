import os
import json
import argparse
import sys
from typing import Dict, Any, List, Tuple, Iterable, Iterator, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

# Import binary dataset support
from build_activity_dataset import LazyActivityDataset


# Optimized extraction functions for binary dataset format (neurons dict)
def extract_S_from_layer(layer: Dict[str, Any]) -> List[float]:
    """Extract S values from layer (optimized for neurons dict format)."""
    neurons = layer.get("neurons", [])
    return [neuron.get("S", 0.0) for neuron in neurons]


def extract_fired_from_layer(layer: Dict[str, Any]) -> List[int]:
    """Extract fired values from layer (optimized for neurons dict format)."""
    neurons = layer.get("neurons", [])
    return [neuron.get("fired", 0) for neuron in neurons]


def extract_t_ref_from_layer(layer: Dict[str, Any]) -> List[float]:
    """Extract t_ref values from layer (optimized for neurons dict format)."""
    neurons = layer.get("neurons", [])
    return [neuron.get("t_ref", 0.0) for neuron in neurons]


def load_dataset(
    path: str, use_streaming: bool = False, legacy_json: bool = False
) -> Tuple[Iterator[Dict[str, Any]], Optional[int]]:
    """Load dataset, preferring binary format unless legacy_json is True.

    Supports both binary tensor format and legacy JSON formats.

    Args:
        path: Path to dataset directory (binary) or JSON file
        use_streaming: If True, use ijson for streaming JSON. If False, load entire file into memory.
        legacy_json: Force loading as legacy JSON format instead of binary

    Returns:
        Tuple of (records_iterator, total_count). total_count is None when streaming.
    """

    # Check if it's a directory (binary format)
    if os.path.isdir(path) and not legacy_json:
        print(f"Loading binary dataset from: {path}")
        dataset = LazyActivityDataset(path)

        # Convert to JSON-compatible format for existing processing code
        def binary_records_iterator():
            for i in range(len(dataset)):
                sample = dataset[i]
                # Convert binary tensors to per-tick records
                ticks, neurons = sample["u"].shape

                for tick in range(ticks):
                    record = {
                        "image_index": i,
                        "label": int(sample["label"]),
                        "tick": tick,
                        "layers": [],  # We'll populate this with neuron data
                    }

                    # Create layer data (single layer for simplicity)
                    layer_data = []
                    for neuron_idx in range(neurons):
                        neuron_id = sample["neuron_ids"][neuron_idx]
                        layer_data.append(
                            {
                                "neuron_id": neuron_id,
                                "S": float(sample["u"][tick, neuron_idx]),
                                "t_ref": float(sample["t_ref"][tick, neuron_idx]),
                                "F_avg": float(sample["fr"][tick, neuron_idx]),
                                "fired": 1
                                if (
                                    (sample["spikes"][:, 0] == tick)
                                    & (sample["spikes"][:, 1] == neuron_idx)
                                ).any()
                                else 0,
                            }
                        )

                    record["layers"] = [{"layer_index": 0, "neurons": layer_data}]
                    yield record

        # Approximate total count based on first sample
        first_sample = dataset[0] if len(dataset) > 0 else None
        ticks_per_sample = first_sample["u"].shape[0] if first_sample else 1
        return binary_records_iterator(), len(
            dataset
        ) * ticks_per_sample  # Approximate total count

    # Legacy JSON format - use original logic
    print(f"Loading JSON dataset from: {path}")
    if not use_streaming:
        # Eager load entire file into memory
        with open(path, "r") as f:
            payload = json.load(f)

        records_list = []
        if isinstance(payload, dict) and "records" in payload:
            records_list = payload["records"]
        elif isinstance(payload, list):
            records_list = payload
        else:
            raise ValueError(
                "Unsupported dataset JSON format: Expected a list of records or a dict with a 'records' key."
            )

        # Return iterator and total count
        return iter(records_list), len(records_list)

    # Use ijson streaming
    try:
        import ijson  # type: ignore
    except ImportError:
        print("Warning: ijson not available, falling back to eager loading")
        # Fallback to eager load if ijson is unavailable
        with open(path, "r") as f:
            payload = json.load(f)

        records_list = []
        if isinstance(payload, dict) and "records" in payload:
            records_list = payload["records"]
        elif isinstance(payload, list):
            records_list = payload
        else:
            raise ValueError(
                "Unsupported dataset JSON format: Expected a list of records or a dict with a 'records' key."
            )

        # Return iterator and total count for fallback
        return iter(records_list), len(records_list)

    # Detect top-level container by inspecting the first non-whitespace byte
    with open(path, "rb") as fh_probe:
        head = fh_probe.read(2048)
        first_non_ws = None
        for b in head:
            if chr(b) not in [" ", "\n", "\r", "\t"]:
                first_non_ws = chr(b)
                break
    if first_non_ws is None:
        return iter(()), 0  # empty file

    def stream_records():
        if first_non_ws == "{":
            with open(path, "rb") as fh:
                for rec in ijson.items(fh, "records.item"):
                    yield rec
        elif first_non_ws == "[":
            with open(path, "rb") as fh:
                for rec in ijson.items(fh, "item"):
                    yield rec
        else:
            raise ValueError(
                "Unsupported JSON format: expected object or array at top level"
            )

    # Return streaming iterator with no total count
    return stream_records(), None


def group_by_image(
    records: Iterable[Dict[str, Any]],
    total_records: Optional[int] = None,
    max_ticks: Optional[int] = None,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Groups records by (label, image_index) to collect per-tick data for each image presentation.

    Args:
        records: Iterable of record dictionaries
        total_records: Total number of records (if known). Used for progress bar when not streaming.
        max_ticks: Maximum number of ticks to keep per image. If specified, only first N ticks are kept.
    """
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

    # Configure progress bar based on whether we know the total
    if total_records is not None:
        # Non-streaming mode: show progress with total count
        progress_bar = tqdm(
            records, desc="Grouping records", total=total_records, unit="records"
        )
    else:
        # Streaming mode: show progress without total count
        progress_bar = tqdm(records, desc="Grouping records", unit="records")

    for rec in progress_bar:
        label = rec.get("label", -1)
        img_idx = rec.get("image_index", -1)
        tick = rec.get("tick", 0)

        if label == -1 or img_idx == -1:
            continue

        # Skip records beyond max_ticks limit to save memory
        if max_ticks is not None and tick >= max_ticks:
            continue

        key = (int(label), int(img_idx))
        buckets.setdefault(key, []).append(rec)

    # Sort records within each bucket by tick to ensure correct temporal order
    # and apply max_ticks limit after sorting to ensure we get the first N ticks
    for key in buckets:
        buckets[key].sort(key=lambda r: r.get("tick", 0))
        if max_ticks is not None:
            buckets[key] = buckets[key][:max_ticks]

    return buckets


def extract_firings_time_series(image_records: List[Dict[str, Any]]) -> torch.Tensor:
    """Extracts a time series of firings from records of a single image presentation."""
    if not image_records:
        return torch.empty(0)

    # Pre-allocate numpy array for better performance
    num_ticks = len(image_records)
    # Get neuron count from first record
    first_layer_data = extract_fired_from_layer(image_records[0].get("layers", [{}])[0])
    num_neurons = len(first_layer_data)

    time_series = np.zeros((num_ticks, num_neurons), dtype=np.float32)

    for tick_idx, record in enumerate(image_records):
        neuron_idx = 0
        for layer in record.get("layers", []):
            layer_data = extract_fired_from_layer(layer)
            for fired_value in layer_data:
                time_series[tick_idx, neuron_idx] = fired_value
                neuron_idx += 1

    return torch.from_numpy(time_series)


def extract_avg_S_time_series(image_records: List[Dict[str, Any]]) -> torch.Tensor:
    """Extracts a time series of average membrane potentials (S) from records."""
    if not image_records:
        return torch.empty(0)

    # Pre-allocate numpy array for better performance
    num_ticks = len(image_records)
    # Get neuron count from first record
    first_layer_data = extract_S_from_layer(image_records[0].get("layers", [{}])[0])
    num_neurons = len(first_layer_data)

    time_series = np.zeros((num_ticks, num_neurons), dtype=np.float32)

    for tick_idx, record in enumerate(image_records):
        neuron_idx = 0
        for layer in record.get("layers", []):
            layer_data = extract_S_from_layer(layer)
            for s_value in layer_data:
                time_series[tick_idx, neuron_idx] = s_value
                neuron_idx += 1

    return torch.from_numpy(time_series)


def extract_avg_t_ref_time_series(image_records: List[Dict[str, Any]]) -> torch.Tensor:
    """Extracts a time series of average refractory window (t_ref) values per neuron."""
    if not image_records:
        return torch.empty(0)

    # Pre-allocate numpy array for better performance
    num_ticks = len(image_records)
    # Get neuron count from first record
    first_layer_data = extract_t_ref_from_layer(image_records[0].get("layers", [{}])[0])
    num_neurons = len(first_layer_data)

    time_series = np.zeros((num_ticks, num_neurons), dtype=np.float32)

    for tick_idx, record in enumerate(image_records):
        neuron_idx = 0
        for layer in record.get("layers", []):
            layer_data = extract_t_ref_from_layer(layer)
            for t_ref_value in layer_data:
                time_series[tick_idx, neuron_idx] = t_ref_value
                neuron_idx += 1

    return torch.from_numpy(time_series)


def extract_multi_feature_time_series(
    image_records: List[Dict[str, Any]], feature_types: List[str]
) -> torch.Tensor:
    """Extracts a time series combining multiple features from records of a single image presentation."""
    if not image_records:
        return torch.empty(0)

    # Pre-allocate numpy array for better performance
    num_ticks = len(image_records)
    # Get neuron count from first record
    first_record = image_records[0]
    total_features = 0
    for layer in first_record.get("layers", []):
        layer_neurons = len(
            extract_S_from_layer(layer)
        )  # All features have same neuron count
        total_features += layer_neurons * len(feature_types)

    time_series = np.zeros((num_ticks, total_features), dtype=np.float32)

    for tick_idx, record in enumerate(image_records):
        feature_idx = 0
        for layer in record.get("layers", []):
            # Extract all requested features for this layer
            layer_features = {}
            if "firings" in feature_types:
                layer_features["firings"] = extract_fired_from_layer(layer)
            if "avg_S" in feature_types:
                layer_features["avg_S"] = extract_S_from_layer(layer)
            if "avg_t_ref" in feature_types:
                layer_features["avg_t_ref"] = extract_t_ref_from_layer(layer)

            # Interleave features in requested order
            for neuron_idx in range(len(layer_features[feature_types[0]])):
                for ft in feature_types:
                    time_series[tick_idx, feature_idx] = layer_features[ft][neuron_idx]
                    feature_idx += 1

    return torch.from_numpy(time_series)


class FeatureScaler:
    """Fits per-dimension scaling statistics over variable-length time series.

    Scaling is computed across ALL timesteps of the training samples for each
    feature dimension independently to avoid temporal information leakage.
    """

    def __init__(self, method: str, eps: float = 1e-8) -> None:
        self.method = method
        self.eps = float(eps)
        self._fitted = False
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._min: Optional[torch.Tensor] = None
        self._max: Optional[torch.Tensor] = None
        self._max_abs: Optional[torch.Tensor] = None

    def fit(self, samples: List[torch.Tensor]) -> None:
        if self.method == "none":
            self._fitted = True
            return

        if not samples:
            raise ValueError("Cannot fit scaler on empty sample list")

        # Determine feature dimension D
        first = samples[0]
        if first.ndim != 2:
            raise ValueError("Each sample must be a 2D tensor [T, D]")
        feature_dim = int(first.shape[1])

        if self.method == "standard":
            running_sum = torch.zeros(feature_dim, dtype=torch.float32)
            running_sumsq = torch.zeros(feature_dim, dtype=torch.float32)
            total_count = 0
            for ts in samples:
                if ts.numel() == 0:
                    continue
                # Sum over time dimension
                running_sum += ts.sum(dim=0)
                running_sumsq += (ts * ts).sum(dim=0)
                total_count += int(ts.shape[0])
            if total_count == 0:
                raise ValueError("All training samples are empty; cannot fit scaler")
            mean = running_sum / float(total_count)
            var = torch.clamp(running_sumsq / float(total_count) - mean * mean, min=0.0)
            std = torch.sqrt(var)
            self._mean = mean
            self._std = std
            self._fitted = True
            return

        if self.method == "minmax":
            cur_min = torch.full((feature_dim,), float("inf"), dtype=torch.float32)
            cur_max = torch.full((feature_dim,), float("-inf"), dtype=torch.float32)
            for ts in samples:
                if ts.numel() == 0:
                    continue
                # Per-dimension min/max over time
                tmin = ts.min(dim=0).values
                tmax = ts.max(dim=0).values
                cur_min = torch.minimum(cur_min, tmin)
                cur_max = torch.maximum(cur_max, tmax)
            self._min = cur_min
            self._max = cur_max
            self._fitted = True
            return

        if self.method == "maxabs":
            cur_max_abs = torch.zeros(feature_dim, dtype=torch.float32)
            for ts in samples:
                if ts.numel() == 0:
                    continue
                tmax_abs = ts.abs().max(dim=0).values
                cur_max_abs = torch.maximum(cur_max_abs, tmax_abs)
            self._max_abs = cur_max_abs
            self._fitted = True
            return

        raise ValueError(f"Unknown scaler method: {self.method}")

    def transform(self, sample: torch.Tensor) -> torch.Tensor:
        if not self._fitted or self.method == "none" or sample.numel() == 0:
            return sample
        if sample.ndim != 2:
            raise ValueError("Each sample must be a 2D tensor [T, D]")

        if self.method == "standard":
            assert self._mean is not None and self._std is not None
            denom = self._std + self.eps
            return (sample - self._mean) / denom

        if self.method == "minmax":
            assert self._min is not None and self._max is not None
            scale = self._max - self._min
            scale = torch.where(scale == 0, torch.full_like(scale, 1.0), scale)
            return (sample - self._min) / (scale + self.eps)

        if self.method == "maxabs":
            assert self._max_abs is not None
            denom = self._max_abs + self.eps
            return sample / denom

        return sample

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"method": self.method, "eps": self.eps}
        if self._mean is not None:
            state["mean"] = self._mean.detach().cpu()
        if self._std is not None:
            state["std"] = self._std.detach().cpu()
        if self._min is not None:
            state["min"] = self._min.detach().cpu()
        if self._max is not None:
            state["max"] = self._max.detach().cpu()
        if self._max_abs is not None:
            state["max_abs"] = self._max_abs.detach().cpu()
        return state


def main():
    parser = argparse.ArgumentParser(
        description="Prepare network activity data for SNN training."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input dataset directory (binary) or JSON file (with --legacy-json).",
    )
    parser.add_argument(
        "--legacy-json",
        action="store_true",
        help="Force loading as legacy JSON format instead of binary",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prepared_data",
        help="Directory to save the processed datasets.",
    )
    parser.add_argument(
        "--feature-types",
        type=str,
        nargs="+",
        default=["firings"],
        choices=["firings", "avg_S", "avg_t_ref"],
        help=(
            "The temporal characteristics to extract. Can specify multiple features like: "
            "--feature-types firings avg_S avg_t_ref"
        ),
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="The fraction of data to use for the training set.",
    )
    parser.add_argument(
        "--use-streaming",
        action="store_true",
        help="Enable streaming mode for loading large JSON files. Uses ijson for memory-efficient processing. Off by default.",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="none",
        choices=["none", "standard", "minmax", "maxabs"],
        help=(
            "Feature scaling method applied after the train/test split. "
            "'standard' uses z-score; 'minmax' scales to [0,1]; 'maxabs' scales by max absolute value. "
            "Default: none."
        ),
    )
    parser.add_argument(
        "--scale-eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon to avoid division by zero during scaling.",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Maximum number of ticks to include per image presentation. If specified, only the first N ticks will be used.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of image samples to process. If specified, only the first N samples will be used.",
    )
    args = parser.parse_args()

    # Create a structured output directory based on the input file name, feature types
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    feature_suffix = "_".join(args.feature_types)
    structured_output_dir = os.path.join(
        args.output_dir, f"{input_basename}_{feature_suffix}"
    )
    os.makedirs(structured_output_dir, exist_ok=True)

    print("Starting data preparation")
    print(f"Feature types: {args.feature_types}")
    print(f"Streaming mode: {'enabled' if args.use_streaming else 'disabled'}")
    print(f"Scaler: {args.scaler}")
    if args.max_ticks is not None:
        print(f"Max ticks per image: {args.max_ticks}")
    if args.max_samples is not None:
        print(f"Max samples to process: {args.max_samples}")
    print(f"Output will be saved in: {structured_output_dir}")

    # 1. Load dataset directly (skip record conversion for HDF5)
    if os.path.isdir(args.input_file) and not args.legacy_json:
        # Direct HDF5 processing for maximum performance
        print(f"Loading HDF5 dataset directly: {args.input_file}")
        hdf5_dataset = LazyActivityDataset(args.input_file)
        num_samples = len(hdf5_dataset)
        print(f"Dataset contains {num_samples} samples")

        # Limit samples if specified
        if args.max_samples is not None:
            num_samples = min(num_samples, args.max_samples)
            print(f"Limited to {num_samples} samples")

        # Pre-allocate data structures
        all_data = []
        all_labels = []
    else:
        # Fallback to record-based processing for JSON
        records, total_records = load_dataset(
            args.input_file,
            use_streaming=args.use_streaming,
            legacy_json=args.legacy_json,
        )
        image_buckets = group_by_image(records, total_records, max_ticks=args.max_ticks)

        all_data = []
        all_labels = []

        # Limit number of samples if specified
        image_items = list(image_buckets.items())
        if args.max_samples is not None:
            image_items = image_items[: args.max_samples]
            print(
                f"Limited to {len(image_items)} samples (from {len(image_buckets)} total)"
            )

    # 2. Extract features for each image presentation
    if os.path.isdir(args.input_file) and not args.legacy_json:
        # Direct HDF5 processing
        for i in tqdm(range(num_samples), desc="Extracting features"):
            sample = hdf5_dataset[i]
            label = int(sample["label"])

            # Extract time series directly from tensors
            u_tensor = sample["u"]  # Shape: [ticks, neurons]
            spikes = sample["spikes"]  # Shape: [N, 2] - (tick, neuron_idx) pairs

            # Convert spikes to firing tensor
            ticks, neurons = u_tensor.shape
            firing_tensor = torch.zeros(ticks, neurons, dtype=torch.float32)
            if len(spikes) > 0:
                spike_ticks, spike_neurons = spikes[:, 0], spikes[:, 1]
                firing_tensor[spike_ticks, spike_neurons] = 1.0

            # Build feature tensor based on requested types
            if len(args.feature_types) == 1:
                # Single feature
                ft = args.feature_types[0]
                if ft == "firings":
                    time_series = firing_tensor
                elif ft == "avg_S":
                    time_series = u_tensor.float()
                elif ft == "avg_t_ref":
                    # For t_ref, we need to expand it to time series
                    t_ref_vals = sample["t_ref"][0]  # Assume constant across ticks
                    time_series = t_ref_vals.unsqueeze(0).expand(ticks, -1).float()
                else:
                    raise ValueError(f"Unknown feature type: {ft}")
            else:
                # Multi-feature: concatenate in requested order
                feature_tensors = []
                for ft in args.feature_types:
                    if ft == "firings":
                        feature_tensors.append(firing_tensor)
                    elif ft == "avg_S":
                        feature_tensors.append(u_tensor.float())
                    elif ft == "avg_t_ref":
                        t_ref_vals = sample["t_ref"][0]
                        feature_tensors.append(
                            t_ref_vals.unsqueeze(0).expand(ticks, -1).float()
                        )
                    else:
                        raise ValueError(f"Unknown feature type: {ft}")
                time_series = torch.cat(feature_tensors, dim=1)

            if time_series.numel() > 0:
                all_data.append(time_series)
                all_labels.append(label)
    else:
        # Record-based processing (JSON fallback)
        for (label, _), image_records in tqdm(image_items, desc="Extracting features"):
            if len(args.feature_types) == 1:
                # Single feature extraction (backward compatibility)
                if args.feature_types[0] == "firings":
                    time_series = extract_firings_time_series(image_records)
                elif args.feature_types[0] == "avg_S":
                    time_series = extract_avg_S_time_series(image_records)
                elif args.feature_types[0] == "avg_t_ref":
                    time_series = extract_avg_t_ref_time_series(image_records)
                else:
                    raise ValueError(f"Unknown feature type: {args.feature_types[0]}")
            else:
                # Multi-feature extraction
                # Build multi-feature by concatenating requested features in order
                per_feature = []
                for ft in args.feature_types:
                    if ft == "firings":
                        per_feature.append(extract_firings_time_series(image_records))
                    elif ft == "avg_S":
                        per_feature.append(extract_avg_S_time_series(image_records))
                    elif ft == "avg_t_ref":
                        per_feature.append(extract_avg_t_ref_time_series(image_records))
                    else:
                        raise ValueError(f"Unknown feature type: {ft}")
                time_series = (
                    torch.cat(per_feature, dim=1) if per_feature else torch.empty(0)
                )

            if time_series.numel() > 0:
                all_data.append(time_series)
                all_labels.append(label)

    if not all_data:
        print("No data was extracted. Check the input file and feature type.")
        return

    # 3. Shuffle and split data
    print(f"Extracted {len(all_data)} samples. Shuffling and splitting data...")

    # Set random seed for reproducible train/test splits
    torch.manual_seed(42)
    np.random.seed(42)

    # Convert to tensors for shuffling
    # Note: Samples can have different time lengths, so we keep them in a list for now.
    # They will be handled by a custom collate function in the training script.
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # Create a permutation
    indices = torch.randperm(len(all_data))

    # Apply permutation
    shuffled_data = [all_data[i] for i in indices]
    shuffled_labels = labels_tensor[indices]

    # Split
    split_idx = int(len(shuffled_data) * args.train_split)
    train_data = shuffled_data[:split_idx]
    train_labels = shuffled_labels[:split_idx]
    test_data = shuffled_data[split_idx:]
    test_labels = shuffled_labels[split_idx:]

    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    # 4. Fit and apply scaler (on training data only), then transform both train and test
    scaler_path = None
    if args.scaler != "none":
        print(f"Fitting '{args.scaler}' scaler on training data...")
        scaler = FeatureScaler(args.scaler, eps=args.scale_eps)
        scaler.fit(train_data)
        print("Transforming datasets with fitted scaler...")
        train_data = [scaler.transform(ts) for ts in train_data]
        test_data = [scaler.transform(ts) for ts in test_data]
        scaler_state = scaler.state_dict()
        scaler_path = os.path.join(structured_output_dir, "scaler.pt")
        torch.save(scaler_state, scaler_path)

    # 5. Save datasets
    torch.save(train_data, os.path.join(structured_output_dir, "train_data.pt"))
    torch.save(train_labels, os.path.join(structured_output_dir, "train_labels.pt"))
    torch.save(test_data, os.path.join(structured_output_dir, "test_data.pt"))
    torch.save(test_labels, os.path.join(structured_output_dir, "test_labels.pt"))

    # 6. Save metadata about the feature configuration and target architecture
    # Determine per-feature dimensionality based on a sample (post-scaling if applied)
    sample_dim = int(train_data[0].shape[1]) if len(train_data) > 0 else 0
    metadata = {
        "feature_types": args.feature_types,
        "num_features": len(args.feature_types),
        "total_feature_dim": sample_dim,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "input_file": args.input_file,
        "train_split": args.train_split,
        "use_streaming": args.use_streaming,
        "scaler": args.scaler,
        "scale_eps": args.scale_eps,
    }
    if scaler_path is not None:
        metadata["scaler_state_file"] = scaler_path

    # Save metadata as JSON
    import json

    metadata_path = os.path.join(structured_output_dir, "dataset_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully saved datasets to {structured_output_dir}")
    print(f"Feature configuration: {args.feature_types}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
