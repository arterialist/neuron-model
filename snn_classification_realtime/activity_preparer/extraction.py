"""Extract neuron values from layer records (neurons dict format)."""

from typing import Any


def extract_S_from_layer(layer: dict[str, Any]) -> list[float]:
    """Extract S values from layer (optimized for neurons dict format)."""
    neurons = layer.get("neurons", [])
    return [neuron.get("S", 0.0) for neuron in neurons]


def extract_fired_from_layer(layer: dict[str, Any]) -> list[int]:
    """Extract fired values from layer (optimized for neurons dict format)."""
    neurons = layer.get("neurons", [])
    return [neuron.get("fired", 0) for neuron in neurons]


def extract_t_ref_from_layer(layer: dict[str, Any]) -> list[float]:
    """Extract t_ref values from layer (optimized for neurons dict format)."""
    neurons = layer.get("neurons", [])
    return [neuron.get("t_ref", 0.0) for neuron in neurons]
