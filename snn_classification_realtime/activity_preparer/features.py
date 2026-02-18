"""Extract time series features from image records."""

from typing import Any

import numpy as np
import torch

from snn_classification_realtime.activity_preparer.extraction import (
    extract_S_from_layer,
    extract_fired_from_layer,
    extract_t_ref_from_layer,
)


def extract_firings_time_series(
    image_records: list[dict[str, Any]],
) -> torch.Tensor:
    """Extract firing time series from records of a single image presentation."""
    if not image_records:
        return torch.empty(0)

    num_ticks = len(image_records)
    first_layer = image_records[0].get("layers", [{}])[0]
    num_neurons = len(extract_fired_from_layer(first_layer))

    time_series = np.zeros((num_ticks, num_neurons), dtype=np.float32)
    for tick_idx, record in enumerate(image_records):
        neuron_idx = 0
        for layer in record.get("layers", []):
            for fired_value in extract_fired_from_layer(layer):
                time_series[tick_idx, neuron_idx] = fired_value
                neuron_idx += 1

    return torch.from_numpy(time_series)


def extract_avg_S_time_series(
    image_records: list[dict[str, Any]],
) -> torch.Tensor:
    """Extract average membrane potential (S) time series from records."""
    if not image_records:
        return torch.empty(0)

    num_ticks = len(image_records)
    first_layer = image_records[0].get("layers", [{}])[0]
    num_neurons = len(extract_S_from_layer(first_layer))

    time_series = np.zeros((num_ticks, num_neurons), dtype=np.float32)
    for tick_idx, record in enumerate(image_records):
        neuron_idx = 0
        for layer in record.get("layers", []):
            for s_value in extract_S_from_layer(layer):
                time_series[tick_idx, neuron_idx] = s_value
                neuron_idx += 1

    return torch.from_numpy(time_series)


def extract_avg_t_ref_time_series(
    image_records: list[dict[str, Any]],
) -> torch.Tensor:
    """Extract refractory window (t_ref) time series per neuron."""
    if not image_records:
        return torch.empty(0)

    num_ticks = len(image_records)
    first_layer = image_records[0].get("layers", [{}])[0]
    num_neurons = len(extract_t_ref_from_layer(first_layer))

    time_series = np.zeros((num_ticks, num_neurons), dtype=np.float32)
    for tick_idx, record in enumerate(image_records):
        neuron_idx = 0
        for layer in record.get("layers", []):
            for t_ref_value in extract_t_ref_from_layer(layer):
                time_series[tick_idx, neuron_idx] = t_ref_value
                neuron_idx += 1

    return torch.from_numpy(time_series)


def extract_multi_feature_time_series(
    image_records: list[dict[str, Any]],
    feature_types: list[str],
) -> torch.Tensor:
    """Extract combined time series from multiple features."""
    if not image_records:
        return torch.empty(0)

    num_ticks = len(image_records)
    first_record = image_records[0]
    total_features = 0
    for layer in first_record.get("layers", []):
        layer_neurons = len(extract_S_from_layer(layer))
        total_features += layer_neurons * len(feature_types)

    time_series = np.zeros((num_ticks, total_features), dtype=np.float32)

    for tick_idx, record in enumerate(image_records):
        feature_idx = 0
        for layer in record.get("layers", []):
            layer_features: dict[str, list[float | int]] = {}
            if "firings" in feature_types:
                layer_features["firings"] = extract_fired_from_layer(layer)
            if "avg_S" in feature_types:
                layer_features["avg_S"] = extract_S_from_layer(layer)
            if "avg_t_ref" in feature_types:
                layer_features["avg_t_ref"] = extract_t_ref_from_layer(layer)

            num_neurons = len(layer_features[feature_types[0]])
            for neuron_idx in range(num_neurons):
                for ft in feature_types:
                    time_series[tick_idx, feature_idx] = layer_features[ft][neuron_idx]
                    feature_idx += 1

    return torch.from_numpy(time_series)
