"""Feature collection from network state (consistent with prepare_activity_data)."""

from typing import Any

import torch

from neuron.network import NeuronNetwork


def collect_activity_snapshot(
    network_sim: NeuronNetwork,
    layers: list[list[int]],
    feature_type: str,
) -> list[float]:
    """Collect a snapshot of the specified feature from the network."""
    snapshot = []
    net = network_sim.network
    for layer_ids in layers:
        if feature_type == "firings":
            snapshot.extend([1 if net.neurons[nid].O > 0 else 0 for nid in layer_ids])
        elif feature_type == "avg_S":
            snapshot.extend([float(net.neurons[nid].S) for nid in layer_ids])
        elif feature_type == "avg_t_ref":
            snapshot.extend([float(net.neurons[nid].t_ref) for nid in layer_ids])
    return snapshot


def collect_multi_feature_snapshot(
    network_sim: NeuronNetwork,
    layers: list[list[int]],
    feature_types: list[str],
) -> list[float]:
    """Collect a snapshot combining multiple features from the network."""
    combined_snapshot = []
    for feature_type in feature_types:
        feature_snapshot = collect_activity_snapshot(
            network_sim, layers, feature_type
        )
        combined_snapshot.extend(feature_snapshot)
    return combined_snapshot


def _extract_firings_time_series(
    image_records: list[dict[str, Any]],
) -> torch.Tensor:
    """Extract firings time series from records (layer format: fired, S, t_ref)."""
    if not image_records:
        return torch.empty(0)
    time_series = []
    for record in image_records:
        tick_firings = []
        for layer in record.get("layers", []):
            tick_firings.extend(layer.get("fired", []))
        time_series.append(tick_firings)
    return torch.tensor(time_series, dtype=torch.float32)


def _extract_avg_S_time_series(
    image_records: list[dict[str, Any]],
) -> torch.Tensor:
    """Extract avg_S time series from records."""
    if not image_records:
        return torch.empty(0)
    time_series = []
    for record in image_records:
        tick_s_values = []
        for layer in record.get("layers", []):
            tick_s_values.extend(layer.get("S", []))
        time_series.append(tick_s_values)
    return torch.tensor(time_series, dtype=torch.float32)


def _extract_avg_t_ref_time_series(
    image_records: list[dict[str, Any]],
) -> torch.Tensor:
    """Extract avg_t_ref time series from records."""
    if not image_records:
        return torch.empty(0)
    time_series = []
    for record in image_records:
        tick_tref_values = []
        for layer in record.get("layers", []):
            tick_tref_values.extend(layer.get("t_ref", []))
        time_series.append(tick_tref_values)
    return torch.tensor(time_series, dtype=torch.float32)


def _extract_multi_feature_time_series(
    image_records: list[dict[str, Any]],
    feature_types: list[str],
) -> torch.Tensor:
    """Extract combined time series from records."""
    if not image_records:
        return torch.empty(0)
    feature_series = {}
    for feature_type in feature_types:
        if feature_type == "firings":
            feature_series[feature_type] = _extract_firings_time_series(
                image_records
            )
        elif feature_type == "avg_S":
            feature_series[feature_type] = _extract_avg_S_time_series(
                image_records
            )
        elif feature_type == "avg_t_ref":
            feature_series[feature_type] = _extract_avg_t_ref_time_series(
                image_records
            )
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    return torch.cat(list(feature_series.values()), dim=1)


def collect_features_consistently(
    network_sim: NeuronNetwork,
    layers: list[list[int]],
    feature_types: list[str],
) -> list[float]:
    """Collect features using the same method as prepare_activity_data.

    Ensures consistency between training and inference data collection.
    """
    mock_record: dict[str, Any] = {"layers": []}

    for layer_ids in layers:
        layer_data: dict[str, list] = {"fired": [], "S": [], "t_ref": []}
        for nid in layer_ids:
            neuron = network_sim.network.neurons[nid]
            layer_data["fired"].append(1 if neuron.O > 0 else 0)
            layer_data["S"].append(float(neuron.S))
            layer_data["t_ref"].append(float(neuron.t_ref))
        mock_record["layers"].append(layer_data)

    if len(feature_types) == 1:
        ft = feature_types[0]
        if ft == "firings":
            time_series = _extract_firings_time_series([mock_record])
        elif ft == "avg_S":
            time_series = _extract_avg_S_time_series([mock_record])
        elif ft == "avg_t_ref":
            time_series = _extract_avg_t_ref_time_series([mock_record])
        else:
            raise ValueError(f"Unknown feature type: {ft}")
    else:
        time_series = _extract_multi_feature_time_series(
            [mock_record], feature_types
        )

    if time_series.numel() > 0:
        return time_series[0].tolist()
    return []
