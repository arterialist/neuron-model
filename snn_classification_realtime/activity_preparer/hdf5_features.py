"""Extract features directly from HDF5 (LazyActivityDataset) samples."""

from typing import Any

import torch

from snn_classification_realtime.build_activity_dataset import LazyActivityDataset


def extract_features_from_hdf5_sample(
    sample: dict[str, Any],
    feature_types: list[str],
) -> torch.Tensor:
    """Extract combined time series from an HDF5 sample.

    Args:
        sample: Dict with u, t_ref, fr, spikes, label, neuron_ids
        feature_types: List of feature types (firings, avg_S, avg_t_ref)

    Returns:
        Tensor of shape [T, N] or [T, N * len(feature_types)]
    """
    u_tensor = sample["u"]
    spikes = sample["spikes"]
    ticks, neurons = u_tensor.shape

    firing_tensor = torch.zeros(ticks, neurons, dtype=torch.float32)
    if len(spikes) > 0:
        spike_ticks, spike_neurons = spikes[:, 0], spikes[:, 1]
        firing_tensor[spike_ticks, spike_neurons] = 1.0

    if len(feature_types) == 1:
        ft = feature_types[0]
        if ft == "firings":
            return firing_tensor
        if ft == "avg_S":
            return u_tensor.float()
        if ft == "avg_t_ref":
            t_ref_vals = sample["t_ref"][0]
            return t_ref_vals.unsqueeze(0).expand(ticks, -1).float()
        raise ValueError(f"Unknown feature type: {ft}")

    feature_tensors = []
    for ft in feature_types:
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
    return torch.cat(feature_tensors, dim=1)


def extract_all_from_hdf5(
    dataset: LazyActivityDataset,
    feature_types: list[str],
    max_samples: int | None = None,
) -> tuple[list[torch.Tensor], list[int]]:
    """Extract features from all HDF5 samples.

    Returns:
        Tuple of (list of time series tensors, list of labels)
    """
    all_data: list[torch.Tensor] = []
    all_labels: list[int] = []
    num_samples = len(dataset)
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    for i in range(num_samples):
        sample = dataset[i]
        label = int(sample["label"])
        time_series = extract_features_from_hdf5_sample(sample, feature_types)
        if time_series.numel() > 0:
            all_data.append(time_series)
            all_labels.append(label)

    return all_data, all_labels
