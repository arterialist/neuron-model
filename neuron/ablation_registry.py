"""
Ablation registry: maps ablation names to neuron module classes.
Used by build_activity_dataset.py and realtime_classification.py to load
the appropriate neuron model for ablation studies.
"""

import importlib
from typing import Type

# Map ablation name -> (module_path, class_name)
ABLATION_REGISTRY = {
    "tref_frozen": ("neuron.neuron_tref_frozen", "Neuron"),
    "retrograde_disabled": ("neuron.neuron_retrograde_disabled", "Neuron"),
    "weight_update_disabled": ("neuron.neuron_weight_update_disabled", "Neuron"),
    "thresholds_frozen": ("neuron.neuron_thresholds_frozen", "Neuron"),
    "directional_error_disabled": (
        "neuron.neuron_directional_error_disabled",
        "Neuron",
    ),
}


def get_neuron_class_for_ablation(ablation_name: str | None) -> Type:
    """
    Return the Neuron class for the given ablation name.

    Args:
        ablation_name: One of tref_frozen, retrograde_disabled, weight_update_disabled,
            thresholds_frozen, directional_error_disabled, or None for full model.

    Returns:
        The Neuron class to use for network instantiation.
    """
    if not ablation_name or ablation_name.lower() == "none":
        from neuron.neuron import Neuron

        return Neuron

    ablation_name = ablation_name.strip().lower()
    if ablation_name not in ABLATION_REGISTRY:
        raise ValueError(
            f"Unknown ablation '{ablation_name}'. "
            f"Valid options: {list(ABLATION_REGISTRY.keys())}, none"
        )

    module_path, class_name = ABLATION_REGISTRY[ablation_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def list_ablations() -> list[str]:
    """Return list of valid ablation names."""
    return list(ABLATION_REGISTRY.keys())
