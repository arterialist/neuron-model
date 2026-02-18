"""Main orchestration for activity data preparation."""

import json
import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from build_activity_dataset import LazyActivityDataset

from snn_classification_realtime.activity_preparer.config import PrepareConfig
from snn_classification_realtime.activity_preparer.loaders import (
    load_dataset,
    group_by_image,
)
from snn_classification_realtime.activity_preparer.features import (
    extract_firings_time_series,
    extract_avg_S_time_series,
    extract_avg_t_ref_time_series,
)
from snn_classification_realtime.activity_preparer.scaler import FeatureScaler
from snn_classification_realtime.activity_preparer.hdf5_features import (
    extract_all_from_hdf5,
)


def _extract_from_json_buckets(
    image_items: list[tuple[tuple[int, int], list[dict[str, Any]]]],
    feature_types: list[str],
) -> tuple[list[torch.Tensor], list[int]]:
    """Extract features from JSON record buckets."""
    all_data: list[torch.Tensor] = []
    all_labels: list[int] = []

    for (label, _), image_records in tqdm(image_items, desc="Extracting features"):
        if len(feature_types) == 1:
            ft = feature_types[0]
            if ft == "firings":
                time_series = extract_firings_time_series(image_records)
            elif ft == "avg_S":
                time_series = extract_avg_S_time_series(image_records)
            elif ft == "avg_t_ref":
                time_series = extract_avg_t_ref_time_series(image_records)
            else:
                raise ValueError(f"Unknown feature type: {ft}")
        else:
            per_feature = []
            for ft in feature_types:
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

    return all_data, all_labels


def run_prepare(config: PrepareConfig) -> None:
    """Run the full activity data preparation workflow."""
    os.makedirs(config.structured_output_dir, exist_ok=True)

    print("Starting data preparation")
    print(f"Feature types: {config.feature_types}")
    print(f"Streaming mode: {'enabled' if config.use_streaming else 'disabled'}")
    print(f"Scaler: {config.scaler}")
    if config.max_ticks is not None:
        print(f"Max ticks per image: {config.max_ticks}")
    if config.max_samples is not None:
        print(f"Max samples to process: {config.max_samples}")
    print(f"Output will be saved in: {config.structured_output_dir}")

    if os.path.isdir(config.input_file) and not config.legacy_json:
        hdf5_dataset = LazyActivityDataset(config.input_file)
        num_samples = len(hdf5_dataset)
        print(f"Dataset contains {num_samples} samples")
        if config.max_samples is not None:
            num_samples = min(num_samples, config.max_samples)
            print(f"Limited to {num_samples} samples")

        all_data, all_labels = extract_all_from_hdf5(
            hdf5_dataset,
            config.feature_types,
            max_samples=config.max_samples,
        )
    else:
        records, total_records = load_dataset(
            config.input_file,
            use_streaming=config.use_streaming,
            legacy_json=config.legacy_json,
        )
        image_buckets = group_by_image(
            records,
            total_records,
            max_ticks=config.max_ticks,
        )
        image_items = list(image_buckets.items())
        if config.max_samples is not None:
            image_items = image_items[: config.max_samples]
            print(
                f"Limited to {len(image_items)} samples "
                f"(from {len(image_buckets)} total)"
            )

        all_data, all_labels = _extract_from_json_buckets(
            image_items,
            config.feature_types,
        )

    if not all_data:
        print("No data was extracted. Check the input file and feature type.")
        return

    print(f"Extracted {len(all_data)} samples. Shuffling and splitting data...")

    torch.manual_seed(42)
    np.random.seed(42)

    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    indices = torch.randperm(len(all_data))
    shuffled_data = [all_data[i] for i in indices]
    shuffled_labels = labels_tensor[indices]

    split_idx = int(len(shuffled_data) * config.train_split)
    train_data = shuffled_data[:split_idx]
    train_labels = shuffled_labels[:split_idx]
    test_data = shuffled_data[split_idx:]
    test_labels = shuffled_labels[split_idx:]

    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    scaler_path: str | None = None
    if config.scaler != "none":
        print(f"Fitting '{config.scaler}' scaler on training data...")
        scaler = FeatureScaler(config.scaler, eps=config.scale_eps)
        scaler.fit(train_data)
        print("Transforming datasets with fitted scaler...")
        train_data = [scaler.transform(ts) for ts in train_data]
        test_data = [scaler.transform(ts) for ts in test_data]
        scaler_path = os.path.join(config.structured_output_dir, "scaler.pt")
        torch.save(scaler.state_dict(), scaler_path)

    torch.save(train_data, os.path.join(config.structured_output_dir, "train_data.pt"))
    torch.save(train_labels, os.path.join(config.structured_output_dir, "train_labels.pt"))
    torch.save(test_data, os.path.join(config.structured_output_dir, "test_data.pt"))
    torch.save(test_labels, os.path.join(config.structured_output_dir, "test_labels.pt"))

    sample_dim = int(train_data[0].shape[1]) if train_data else 0
    metadata: dict[str, Any] = {
        "feature_types": config.feature_types,
        "num_features": len(config.feature_types),
        "total_feature_dim": sample_dim,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "input_file": config.input_file,
        "train_split": config.train_split,
        "use_streaming": config.use_streaming,
        "scaler": config.scaler,
        "scale_eps": config.scale_eps,
    }
    if scaler_path is not None:
        metadata["scaler_state_file"] = scaler_path

    metadata_path = os.path.join(config.structured_output_dir, "dataset_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully saved datasets to {config.structured_output_dir}")
    print(f"Feature configuration: {config.feature_types}")
    print(f"Metadata saved to: {metadata_path}")
