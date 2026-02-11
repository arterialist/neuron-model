"""
Data preparer step for the pipeline.

Prepares activity data for classifier training by extracting features.
Wraps functionality from snn_classification_realtime/prepare_activity_data.py.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.config import DataPreparationConfig
from pipeline.steps.base import (
    PipelineStep,
    StepContext,
    StepResult,
    StepStatus,
    Artifact,
    StepRegistry,
    StepCancelledException,
)
from pipeline.utils.activity_data import LazyActivityDataset, is_binary_dataset


def extract_S_from_layer(layer: Dict[str, Any]) -> List[float]:
    """Extract S values from layer."""
    return layer.get("S", [])


def extract_fired_from_layer(layer: Dict[str, Any]) -> List[int]:
    """Extract fired values from layer."""
    return layer.get("fired", [])


def extract_t_ref_from_layer(layer: Dict[str, Any]) -> List[float]:
    """Extract t_ref values from layer."""
    return layer.get("t_ref", [])


def load_dataset_json(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """Load activity dataset from JSON file.

    Returns:
        Tuple of (records, total_count)
    """
    with open(path, "r") as f:
        data = json.load(f)

    records = data.get("records", [])
    return records, len(records)


def extract_features_from_h5_sample(
    sample: Dict[str, Any], feature_types: List[str]
) -> torch.Tensor:
    """Extracts combined time series from an HDF5 sample.

    Args:
        sample: Dictionary containing sample data (u, t_ref, spikes, etc.)
        feature_types: List of feature types to extract

    Returns:
        Combined feature tensor of shape [T, N * len(feature_types)]
    """
    # u and t_ref are already [T, N] tensors (or numpy arrays)
    # spikes is [N_spikes, 2] where col 0 is tick, col 1 is neuron_idx

    if torch.is_tensor(sample["u"]):
        T, N = sample["u"].shape
        u = sample["u"]
        t_ref = sample["t_ref"]
        spikes = sample["spikes"]
    else:
        # Numpy fallback
        T, N = sample["u"].shape
        u = torch.from_numpy(sample["u"])
        t_ref = torch.from_numpy(sample["t_ref"])
        spikes = torch.from_numpy(sample["spikes"])

    feature_series = []

    for ft in feature_types:
        if ft == "firings":
            # Create dense from sparse spikes
            dense = torch.zeros((T, N), dtype=torch.float32)
            if len(spikes) > 0:
                # spikes[:, 0] are ticks, spikes[:, 1] are neuron indices
                # Ensure long type for indexing
                tick_indices = spikes[:, 0].long()
                neuron_indices = spikes[:, 1].long()

                # Filter out any out-of-bounds indices (safety check)
                mask = (tick_indices < T) & (neuron_indices < N)
                if not mask.all():
                    tick_indices = tick_indices[mask]
                    neuron_indices = neuron_indices[mask]

                dense[tick_indices, neuron_indices] = 1.0
            feature_series.append(dense)
        elif ft == "avg_S":
            feature_series.append(u.float())
        elif ft == "avg_t_ref":
            feature_series.append(t_ref.float())

    if not feature_series:
        return torch.empty(0)

    return torch.cat(feature_series, dim=1)


def group_by_image(
    records: List[Dict[str, Any]],
    max_ticks: Optional[int] = None,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Groups records by (label, image_index) to collect per-tick data."""
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

    for record in records:
        key = (int(record["label"]), int(record["image_index"]))
        buckets.setdefault(key, []).append(record)

    # Sort by tick and limit
    result: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for key, recs in buckets.items():
        recs.sort(key=lambda r: r.get("tick", 0))
        if max_ticks is not None:
            recs = recs[:max_ticks]
        result[key] = {"label": key[0], "image_index": key[1], "records": recs}

    return result


def extract_firings_time_series(image_records: List[Dict[str, Any]]) -> torch.Tensor:
    """Extracts time series of firings from records."""
    T = len(image_records)
    if T == 0:
        return torch.empty(0)

    sample_layers = image_records[0].get("layers", [])
    num_neurons = sum(len(layer.get("fired", [])) for layer in sample_layers)

    series = torch.zeros(T, num_neurons)
    for t, record in enumerate(image_records):
        offset = 0
        for layer in record.get("layers", []):
            fired = extract_fired_from_layer(layer)
            for i, v in enumerate(fired):
                series[t, offset + i] = float(v)
            offset += len(fired)

    return series


def extract_avg_S_time_series(image_records: List[Dict[str, Any]]) -> torch.Tensor:
    """Extracts time series of average S values from records."""
    T = len(image_records)
    if T == 0:
        return torch.empty(0)

    sample_layers = image_records[0].get("layers", [])
    num_neurons = sum(len(layer.get("S", [])) for layer in sample_layers)

    series = torch.zeros(T, num_neurons)
    for t, record in enumerate(image_records):
        offset = 0
        for layer in record.get("layers", []):
            S_vals = extract_S_from_layer(layer)
            for i, v in enumerate(S_vals):
                series[t, offset + i] = float(v)
            offset += len(S_vals)

    return series


def extract_avg_t_ref_time_series(image_records: List[Dict[str, Any]]) -> torch.Tensor:
    """Extracts time series of average t_ref values from records."""
    T = len(image_records)
    if T == 0:
        return torch.empty(0)

    sample_layers = image_records[0].get("layers", [])
    num_neurons = sum(len(layer.get("t_ref", [])) for layer in sample_layers)

    series = torch.zeros(T, num_neurons)
    for t, record in enumerate(image_records):
        offset = 0
        for layer in record.get("layers", []):
            t_ref_vals = extract_t_ref_from_layer(layer)
            for i, v in enumerate(t_ref_vals):
                series[t, offset + i] = float(v)
            offset += len(t_ref_vals)

    return series


def extract_multi_feature_time_series(
    image_records: List[Dict[str, Any]], feature_types: List[str]
) -> torch.Tensor:
    """Extracts combined time series for multiple features."""
    feature_series = []

    for ft in feature_types:
        if ft == "firings":
            feature_series.append(extract_firings_time_series(image_records))
        elif ft == "avg_S":
            feature_series.append(extract_avg_S_time_series(image_records))
        elif ft == "avg_t_ref":
            feature_series.append(extract_avg_t_ref_time_series(image_records))

    if not feature_series:
        return torch.empty(0)

    return torch.cat(feature_series, dim=1)


class FeatureScaler:
    """Fits per-dimension scaling statistics."""

    def __init__(self, method: str = "minmax", eps: float = 1e-8):
        self.method = method
        self.eps = eps
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._min: Optional[torch.Tensor] = None
        self._max: Optional[torch.Tensor] = None
        self._max_abs: Optional[torch.Tensor] = None

    def fit(self, samples: List[torch.Tensor]):
        """Compute scaling statistics from samples."""
        if not samples:
            return

        # Stack all timesteps from all samples
        all_data = torch.cat([s.reshape(-1, s.shape[-1]) for s in samples], dim=0)

        if self.method == "zscore":
            self._mean = all_data.mean(dim=0)
            self._std = all_data.std(dim=0) + self.eps
        elif self.method == "minmax":
            self._min = all_data.min(dim=0).values
            self._max = all_data.max(dim=0).values + self.eps
        elif self.method == "maxabs":
            self._max_abs = all_data.abs().max(dim=0).values + self.eps

    def transform(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply scaling to a sample."""
        if self.method == "none":
            return sample
        elif self.method == "zscore" and self._mean is not None:
            return (sample - self._mean) / self._std
        elif self.method == "minmax" and self._min is not None:
            return (sample - self._min) / (self._max - self._min + self.eps)
        elif self.method == "maxabs" and self._max_abs is not None:
            return sample / self._max_abs
        return sample

    def state_dict(self) -> Dict[str, Any]:
        """Get scaler state for serialization."""
        return {
            "method": self.method,
            "eps": self.eps,
            "mean": self._mean.tolist() if self._mean is not None else None,
            "std": self._std.tolist() if self._std is not None else None,
            "min": self._min.tolist() if self._min is not None else None,
            "max": self._max.tolist() if self._max is not None else None,
            "max_abs": self._max_abs.tolist() if self._max_abs is not None else None,
        }


@StepRegistry.register
class DataPreparerStep(PipelineStep):
    """Pipeline step for preparing activity data for training."""

    @property
    def name(self) -> str:
        return "data_preparation"

    @property
    def display_name(self) -> str:
        return "Data Preparation"

    def run(self, context: StepContext) -> StepResult:
        start_time = datetime.now()
        logs: list[str] = []

        try:
            config: DataPreparationConfig = context.config
            log = context.logger or logging.getLogger(__name__)

            # Get activity dataset from previous step
            activity_artifacts = context.previous_artifacts.get(
                "activity_recording", []
            )
            if not activity_artifacts:
                raise ValueError("Activity recording artifact not found")

            activity_artifact = activity_artifacts[0]
            activity_path = str(activity_artifact.path)

            # Create output directory
            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)

            # Load activity data
            log.info(f"Loading activity data from {activity_path}")
            logs.append(f"Loading activity data from {activity_path}")

            all_samples: List[torch.Tensor] = []
            all_labels: List[int] = []
            feature_types = config.feature_types
            log.info(f"Extracting features: {feature_types}")
            logs.append(f"Extracting features: {feature_types}")

            if is_binary_dataset(activity_path):
                # HDF5 Loading Path
                start_time_load = datetime.now()
                # If path is a file, use its directory for LazyActivityDataset
                data_dir = os.path.dirname(activity_path)
                try:
                    dataset = LazyActivityDataset(data_dir)
                    log.info(f"Opened HDF5 dataset with {len(dataset)} samples")
                    logs.append(f"Using HDF5 dataset with {len(dataset)} samples")

                    for i in tqdm(range(len(dataset)), desc="Extracting features"):
                        sample = dataset[i]
                        label = int(sample["label"])

                        features = extract_features_from_h5_sample(
                            sample, feature_types
                        )

                        # Apply max_ticks limiting
                        if (
                            config.max_ticks is not None
                            and features.shape[0] > config.max_ticks
                        ):
                            features = features[: config.max_ticks]

                        all_samples.append(features)
                        all_labels.append(label)

                    dataset.close()
                    log.info(
                        f"HDF5 processing complete in {(datetime.now() - start_time_load).total_seconds():.2f}s"
                    )

                except Exception as e:
                    # Provide helpful error if h5py is missing or file is corrupt
                    raise RuntimeError(f"Failed to load HDF5 dataset: {e}") from e

            else:
                # JSON Loading Path (Legacy)
                records, total_count = load_dataset_json(activity_path)
                log.info(f"Loaded {total_count} records")
                logs.append(f"Loaded {total_count} records")

                # Group by image
                image_buckets = group_by_image(records, max_ticks=config.max_ticks)
                log.info(f"Grouped into {len(image_buckets)} image presentations")
                logs.append(f"Grouped into {len(image_buckets)} image presentations")

                for key, bucket in tqdm(
                    image_buckets.items(), desc="Extracting features"
                ):
                    label = bucket["label"]
                    recs = bucket["records"]

                    if not recs:
                        continue

                    features = extract_multi_feature_time_series(recs, feature_types)
                    all_samples.append(features)
                    all_labels.append(label)

            if not all_samples:
                raise ValueError("No valid samples extracted")

            # Train/test split
            n_samples = len(all_samples)
            n_train = int(n_samples * config.train_split)

            indices = list(range(n_samples))
            np.random.shuffle(indices)

            train_indices = indices[:n_train]
            test_indices = indices[n_train:]

            train_samples = [all_samples[i] for i in train_indices]
            train_labels = [all_labels[i] for i in train_indices]
            test_samples = [all_samples[i] for i in test_indices]
            test_labels = [all_labels[i] for i in test_indices]

            log.info(f"Split: {len(train_samples)} train, {len(test_samples)} test")
            logs.append(f"Split: {len(train_samples)} train, {len(test_samples)} test")

            # Fit scaler on training data
            scaler = FeatureScaler(method=config.scaling_method)
            scaler.fit(train_samples)

            # Apply scaling
            train_samples_scaled = [scaler.transform(s) for s in train_samples]
            test_samples_scaled = [scaler.transform(s) for s in test_samples]

            # Save datasets
            torch.save(train_samples_scaled, step_dir / "train_data.pt")
            torch.save(torch.tensor(train_labels), step_dir / "train_labels.pt")
            torch.save(test_samples_scaled, step_dir / "test_data.pt")
            torch.save(torch.tensor(test_labels), step_dir / "test_labels.pt")

            # Save metadata
            metadata = {
                "feature_types": feature_types,
                "scaling_method": config.scaling_method,
                "num_train": len(train_samples),
                "num_test": len(test_samples),
                "num_features": train_samples_scaled[0].shape[-1]
                if train_samples_scaled
                else 0,
                "scaler_state": scaler.state_dict(),
            }

            with open(step_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logs.append(f"Saved prepared data to {step_dir}")

            # Create artifacts
            artifacts = [
                Artifact(
                    name="train_data.pt",
                    path=step_dir / "train_data.pt",
                    artifact_type="dataset",
                    size_bytes=(step_dir / "train_data.pt").stat().st_size,
                ),
                Artifact(
                    name="train_labels.pt",
                    path=step_dir / "train_labels.pt",
                    artifact_type="dataset",
                    size_bytes=(step_dir / "train_labels.pt").stat().st_size,
                ),
                Artifact(
                    name="test_data.pt",
                    path=step_dir / "test_data.pt",
                    artifact_type="dataset",
                    size_bytes=(step_dir / "test_data.pt").stat().st_size,
                ),
                Artifact(
                    name="test_labels.pt",
                    path=step_dir / "test_labels.pt",
                    artifact_type="dataset",
                    size_bytes=(step_dir / "test_labels.pt").stat().st_size,
                ),
                Artifact(
                    name="metadata.json",
                    path=step_dir / "metadata.json",
                    artifact_type="metadata",
                    size_bytes=(step_dir / "metadata.json").stat().st_size,
                    metadata=metadata,
                ),
            ]

            return StepResult(
                status=StepStatus.COMPLETED,
                artifacts=artifacts,
                metrics={
                    "num_train": len(train_samples),
                    "num_test": len(test_samples),
                    "num_features": metadata["num_features"],
                },
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs,
            )

        except StepCancelledException:
            raise
        except Exception as e:
            import traceback

            return StepResult(
                status=StepStatus.FAILED,
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs + [f"ERROR: {e}", traceback.format_exc()],
            )
