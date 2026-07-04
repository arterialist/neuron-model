"""Input-drive calibration: estimate per-neuron drive vs firing threshold.

The neuron integrates dS = (dt/lambda) * (-S + I_t), so at steady state
S ~= I_t. Each external signal of strength `x` arriving at synapse `s`
contributes x * (u_i.info + u_i.plast) * delta_decay**distance(s) to I_t
per tick (signals are re-sent every tick by the build loop). This module
computes that contribution exactly from the network weights and the actual
image->signal mapping, and compares it with the firing thresholds.

Interpretation of the drive/threshold ratio (with default c=10, lambda=20):
  ratio < 1.0   -> neuron never fires from external drive alone (silent)
  1.0 - ~2.5    -> firing rate depends on the image (the useful regime)
  ratio > ~2.5  -> S crosses threshold within the refractory period, so the
                   neuron fires at the refractory-limited max rate for every
                   image (saturated: activity carries no class information)
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from neuron.network import NeuronNetwork

from snn_classification_realtime.core.config import DatasetConfig
from snn_classification_realtime.core.input_mapping import image_to_signals


@dataclass
class DriveReport:
    """Per-input-layer drive statistics estimated from probe images."""

    num_probe_images: int
    num_input_neurons: int
    drive_mean: float
    drive_p05: float
    drive_p95: float
    threshold_mean: float
    threshold_min: float
    threshold_max: float
    ratio_mean: float
    ratio_p05: float
    ratio_p95: float
    frac_silent: float
    frac_image_driven: float
    frac_saturated: float
    saturation_ratio_limit: float
    suggested_gain: float
    target_ratio: float
    verdict: str


def _synapse_weight(neuron: Any, synapse_id: int) -> float:
    """Per-tick current contribution of a unit-strength signal at a synapse."""
    synapse = neuron.postsynaptic_points.get(synapse_id)
    if synapse is None:
        return 0.0
    u_i = synapse.u_i
    weight = float(u_i.info) + float(u_i.plast)
    distance = float(neuron.distances.get(synapse_id, 0))
    decay = float(getattr(neuron.params, "delta_decay", 1.0))
    return weight * (decay**distance)


def _saturation_ratio_limit(neuron: Any) -> float:
    """Ratio I/r above which firing becomes refractory-limited.

    From S(t) = I * (1 - exp(-t/lambda)): threshold is reached within the
    refractory period c whenever I/r > 1 / (1 - exp(-c/lambda)).
    """
    lam = float(getattr(neuron.params, "lambda_param", 20.0))
    c = float(getattr(neuron.params, "c", 10.0))
    denom = 1.0 - math.exp(-c / max(lam, 1e-9))
    return 1.0 / max(denom, 1e-9)


def probe_image_indices(
    label_to_indices: dict[int, list[int]],
    per_label: int = 2,
) -> list[int]:
    """Deterministic probe selection: first `per_label` images of each label."""
    chosen: list[int] = []
    for label in sorted(label_to_indices.keys()):
        chosen.extend(label_to_indices[label][:per_label])
    return chosen


def estimate_input_drive(
    network_sim: NeuronNetwork,
    input_layer_ids: list[int],
    input_synapses_per_neuron: int,
    dataset_config: DatasetConfig,
    probe_indices: list[int],
    target_ratio: float = 1.5,
) -> DriveReport:
    """Estimate steady-state drive of input-layer neurons for probe images.

    Drive is linear in dataset_config.signal_gain, so suggested_gain rescales
    the *current* gain to bring the mean ratio to target_ratio.
    """
    neurons = network_sim.network.neurons
    id_to_pos = {nid: i for i, nid in enumerate(input_layer_ids)}
    ds = dataset_config.dataset

    per_image_drive = np.zeros(
        (len(probe_indices), len(input_layer_ids)), dtype=np.float64
    )
    for row, img_idx in enumerate(probe_indices):
        image_tensor, _ = ds[img_idx]
        signals = image_to_signals(
            image_tensor,
            input_layer_ids,
            input_synapses_per_neuron,
            network_sim,
            dataset_config,
        )
        for neuron_id, synapse_id, strength in signals:
            pos = id_to_pos.get(neuron_id)
            if pos is None:
                continue
            per_image_drive[row, pos] += strength * _synapse_weight(
                neurons[neuron_id], synapse_id
            )

    # Mean drive per neuron across probe images
    drive = per_image_drive.mean(axis=0)
    thresholds = np.array(
        [
            float(getattr(neurons[nid], "r", getattr(neurons[nid].params, "r_base", 1.0)))
            for nid in input_layer_ids
        ],
        dtype=np.float64,
    )
    thresholds = np.maximum(thresholds, 1e-9)
    ratios = drive / thresholds

    sat_limit = _saturation_ratio_limit(neurons[input_layer_ids[0]])
    frac_silent = float(np.mean(ratios < 1.0))
    frac_saturated = float(np.mean(ratios > sat_limit))
    frac_image_driven = max(0.0, 1.0 - frac_silent - frac_saturated)

    ratio_mean = float(ratios.mean())
    current_gain = float(getattr(dataset_config, "signal_gain", 1.0))
    if ratio_mean > 1e-12:
        suggested_gain = current_gain * target_ratio / ratio_mean
    else:
        suggested_gain = current_gain

    if frac_saturated > 0.5:
        verdict = "SATURATED"
    elif frac_silent > 0.5:
        verdict = "UNDER-DRIVEN"
    else:
        verdict = "OK"

    return DriveReport(
        num_probe_images=len(probe_indices),
        num_input_neurons=len(input_layer_ids),
        drive_mean=float(drive.mean()),
        drive_p05=float(np.percentile(drive, 5)),
        drive_p95=float(np.percentile(drive, 95)),
        threshold_mean=float(thresholds.mean()),
        threshold_min=float(thresholds.min()),
        threshold_max=float(thresholds.max()),
        ratio_mean=ratio_mean,
        ratio_p05=float(np.percentile(ratios, 5)),
        ratio_p95=float(np.percentile(ratios, 95)),
        frac_silent=frac_silent,
        frac_image_driven=frac_image_driven,
        frac_saturated=frac_saturated,
        saturation_ratio_limit=sat_limit,
        suggested_gain=suggested_gain,
        target_ratio=target_ratio,
        verdict=verdict,
    )


def format_drive_report(report: DriveReport, effective_gain: float) -> str:
    """Human-readable calibration summary for the build log."""
    lines = [
        "--- Input drive calibration ---",
        f"probe images: {report.num_probe_images}, "
        f"input neurons: {report.num_input_neurons}, "
        f"signal gain: {effective_gain:.6g}",
        f"steady-state drive per neuron: mean {report.drive_mean:.3f} "
        f"(p05 {report.drive_p05:.3f}, p95 {report.drive_p95:.3f})",
        f"firing threshold r: mean {report.threshold_mean:.3f} "
        f"(range {report.threshold_min:.3f}-{report.threshold_max:.3f})",
        f"drive/threshold ratio: mean {report.ratio_mean:.2f} "
        f"(p05 {report.ratio_p05:.2f}, p95 {report.ratio_p95:.2f})",
        f"input neurons: {report.frac_silent:.0%} silent (<1.0), "
        f"{report.frac_image_driven:.0%} image-driven, "
        f"{report.frac_saturated:.0%} saturated (>{report.saturation_ratio_limit:.2f})",
        f"verdict: {report.verdict}",
    ]
    if report.verdict != "OK":
        lines.append(
            f"suggested gain for mean ratio {report.target_ratio:.2f}: "
            f"--signal-gain {report.suggested_gain:.6g} "
            f"(or use --auto-gain {report.target_ratio:.2f})"
        )
    return "\n".join(lines)
