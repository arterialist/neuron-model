"""Dataset loading for binary (HDF5) and legacy JSON formats."""

import json
import os
from typing import Any, Iterable, Iterator, Optional

from tqdm import tqdm

from snn_classification_realtime.build_activity_dataset import LazyActivityDataset


def load_dataset(
    path: str,
    use_streaming: bool = False,
    legacy_json: bool = False,
) -> tuple[Iterator[dict[str, Any]], Optional[int]]:
    """Load dataset, preferring binary format unless legacy_json is True.

    Supports both binary tensor format and legacy JSON formats.

    Args:
        path: Path to dataset directory (binary) or JSON file
        use_streaming: If True, use ijson for streaming JSON
        legacy_json: Force loading as legacy JSON format instead of binary

    Returns:
        Tuple of (records_iterator, total_count). total_count is None when streaming.
    """
    if os.path.isdir(path) and not legacy_json:
        print(f"Loading binary dataset from: {path}")
        dataset = LazyActivityDataset(path)

        def binary_records_iterator() -> Iterator[dict[str, Any]]:
            for i in range(len(dataset)):
                sample = dataset[i]
                ticks, neurons = sample["u"].shape
                for tick in range(ticks):
                    record: dict[str, Any] = {
                        "image_index": i,
                        "label": int(sample["label"]),
                        "tick": tick,
                        "layers": [],
                    }
                    layer_data = []
                    for neuron_idx in range(neurons):
                        neuron_id = sample["neuron_ids"][neuron_idx]
                        layer_data.append({
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
                        })
                    record["layers"] = [{"layer_index": 0, "neurons": layer_data}]
                    yield record

        first_sample = dataset[0] if len(dataset) > 0 else None
        ticks_per_sample = first_sample["u"].shape[0] if first_sample else 1
        return binary_records_iterator(), len(dataset) * ticks_per_sample

    print(f"Loading JSON dataset from: {path}")
    if not use_streaming:
        return _load_json_eager(path)

    try:
        import ijson
    except ImportError:
        print("Warning: ijson not available, falling back to eager loading")
        return _load_json_eager(path)

    with open(path, "rb") as fh_probe:
        head = fh_probe.read(2048)
        first_non_ws = None
        for b in head:
            if chr(b) not in (" ", "\n", "\r", "\t"):
                first_non_ws = chr(b)
                break

    if first_non_ws is None:
        return iter(()), 0

    def stream_records() -> Iterator[dict[str, Any]]:
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

    return stream_records(), None


def _load_json_eager(path: str) -> tuple[Iterator[dict[str, Any]], int]:
    """Load entire JSON file into memory and return iterator."""
    with open(path, "r") as f:
        payload = json.load(f)

    records_list: list[dict[str, Any]] = []
    if isinstance(payload, dict) and "records" in payload:
        records_list = payload["records"]
    elif isinstance(payload, list):
        records_list = payload
    else:
        raise ValueError(
            "Unsupported dataset JSON format: Expected a list of records "
            "or a dict with a 'records' key."
        )
    return iter(records_list), len(records_list)


def group_by_image(
    records: Iterable[dict[str, Any]],
    total_records: Optional[int] = None,
    max_ticks: Optional[int] = None,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Group records by (label, image_index) for per-tick data per image.

    Args:
        records: Iterable of record dictionaries
        total_records: Total count (if known), for progress bar
        max_ticks: Maximum ticks to keep per image
    """
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}

    if total_records is not None:
        progress_bar = tqdm(
            records, desc="Grouping records", total=total_records, unit="records"
        )
    else:
        progress_bar = tqdm(records, desc="Grouping records", unit="records")

    for rec in progress_bar:
        label = rec.get("label", -1)
        img_idx = rec.get("image_index", -1)
        tick = rec.get("tick", 0)

        if label == -1 or img_idx == -1:
            continue

        if max_ticks is not None and tick >= max_ticks:
            continue

        key = (int(label), int(img_idx))
        buckets.setdefault(key, []).append(rec)

    for key in buckets:
        buckets[key].sort(key=lambda r: r.get("tick", 0))
        if max_ticks is not None:
            buckets[key] = buckets[key][:max_ticks]

    return buckets
