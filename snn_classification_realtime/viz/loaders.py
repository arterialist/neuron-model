"""Dataset loading utilities for visualization scripts."""

from __future__ import annotations

from typing import Any

from snn_classification_realtime.build_activity_dataset import LazyActivityDataset


def load_activity_dataset(path: str) -> LazyActivityDataset:
    """Load activity dataset from directory (binary/HDF5 format)."""
    if not path or not __import__("os").path.isdir(path):
        raise ValueError(
            f"Dataset path must be a directory (binary format): {path}"
        )
    print(f"Loading activity dataset from: {path}")
    return LazyActivityDataset(path)


def group_images_by_label(
    dataset: LazyActivityDataset,
) -> dict[int, list[int]]:
    """Group sample indices by label."""
    label_to_indices: dict[int, list[int]] = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        label = int(sample["label"])
        label_to_indices.setdefault(label, []).append(i)
    for lbl in label_to_indices:
        label_to_indices[lbl].sort()
    return label_to_indices


def group_by_image(
    dataset: LazyActivityDataset,
    max_ticks: int | None = None,
) -> dict[int, dict[str, Any]]:
    """Group binary dataset samples by image index."""
    from tqdm import tqdm

    buckets: dict[int, dict[str, Any]] = {}
    for img_idx in tqdm(range(len(dataset)), desc="Grouping by image"):
        sample = dataset[img_idx]
        bucket: dict[str, Any] = {
            "label": int(sample["label"]),
            "binary_sample": sample,
            "dataset": dataset,
        }
        if max_ticks is not None:
            bucket["max_ticks"] = max_ticks
        buckets[img_idx] = bucket
    return buckets


def get_sample_as_numpy(dataset: LazyActivityDataset, idx: int) -> dict[str, Any]:
    """Get a sample as numpy arrays (convert torch tensors if present)."""
    sample = dataset[idx]
    result: dict[str, Any] = {}
    for k, v in sample.items():
        if hasattr(v, "numpy"):
            result[k] = v.numpy()
        else:
            result[k] = v
    return result
