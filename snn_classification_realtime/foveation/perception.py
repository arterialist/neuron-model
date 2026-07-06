"""Fovea-sized PAULA perception network wrapper.

Wraps a NeuronNetwork built at retina resolution (C, k, k). Every tick it is
fed the current k x k patch (all inputs driven), advanced one tick, and its
population state (S, F_avg, O, t_ref) is read out. The state arrays are what
the free-energy probe and the decoder consume.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.ablation_registry import get_neuron_class_for_ablation
from neuron import setup_neuron_logger

from snn_classification_realtime.core.config import DatasetConfig
from snn_classification_realtime.core.input_mapping import image_to_signals
from snn_classification_realtime.core.network_utils import (
    infer_layers_from_metadata,
    determine_input_mapping,
)
from snn_classification_realtime.network_builder_direct import (
    build_network_config_direct,
)


@dataclass
class PopulationState:
    """Per-tick snapshot of the perception population."""

    S: np.ndarray
    F_avg: np.ndarray
    O: np.ndarray
    t_ref: np.ndarray

    def concat(self) -> np.ndarray:
        """Flat feature vector [S | F_avg | O] for a decoder."""
        return np.concatenate([self.S, self.F_avg, self.O]).astype(np.float32)


def build_fovea_network_json(
    path: str,
    *,
    channels: int = 1,
    size: int = 16,
    layers: list[dict] | None = None,
    seed: int | None = None,
) -> str:
    """Build and save a perception network sized for a (channels, size, size)
    retina. Returns the path. Uses the direct JSON builder (no heavy objects).
    """
    if seed is not None:
        import random

        random.seed(seed)
        np.random.seed(seed)
    if layers is None:
        # Small conv stack; layer-0 conv is the CNN input (one neuron per
        # kernel position), which image_to_signals feeds directly.
        layers = [
            {"type": "conv", "kernel_size": 4, "stride": 2, "filters": 2},
            {"type": "conv", "kernel_size": 3, "stride": 2, "filters": 3},
        ]
    cfg = {
        "dataset": "cifar10_grayscale" if channels == 1 else "cifar10",
        "layers": layers,
    }
    config_out = build_network_config_direct(
        cfg, input_shape=(channels, size, size)
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(config_out, f)
    return path


class PerceptionNetwork:
    """A fovea-sized perception substrate advanced one tick at a time."""

    def __init__(
        self,
        network_path: str,
        dataset_config: DatasetConfig,
        ablation: str = "none",
        log_level: str = "CRITICAL",
    ) -> None:
        self.network_path = network_path
        self.dataset_config = dataset_config
        neuron_cls = get_neuron_class_for_ablation(ablation)
        self.sim = NetworkConfig.load_network_config(
            network_path, neuron_class=neuron_cls
        )
        self.core = NNCore()
        self.core.neural_net = self.sim
        setup_neuron_logger(log_level)
        self.core.set_log_level(log_level)

        self.layers = infer_layers_from_metadata(self.sim)
        self.input_layer_ids, self.input_synapses_per_neuron = (
            determine_input_mapping(self.sim, self.layers)
        )
        # Stable neuron ordering for state readout.
        self._ids = list(self.sim.network.neurons.keys())
        self._n = len(self._ids)
        self._saved_eta: dict[int, tuple[float, float]] = {}
        # Per-readout-position layer index (for per-layer participation / fade).
        self._layer_of_pos = np.array(
            [int(self.sim.network.neurons[nid].metadata.get("layer", 0))
             for nid in self._ids],
            dtype=np.int64,
        )

    @property
    def num_neurons(self) -> int:
        return self._n

    @property
    def layer_of_pos(self) -> np.ndarray:
        """Layer index for each readout position (aligns with read_state arrays)."""
        return self._layer_of_pos

    @property
    def layer_indices(self) -> list[int]:
        return sorted(set(int(x) for x in self._layer_of_pos))

    def set_layer_threshold(self, layer_idx: int, r_base: float) -> None:
        """Override the firing threshold r_base for every neuron in a layer.

        Takes effect on the next reset() (reset re-reads params.r_base into r).
        This is the 'careful per-layer threshold tuning' lever against
        signal-fade-through-depth.
        """
        for nrn in self.sim.network.neurons.values():
            if int(nrn.metadata.get("layer", 0)) == layer_idx:
                nrn.params.r_base = float(r_base)

    def scale_layer_threshold(self, layer_idx: int, factor: float) -> None:
        for nrn in self.sim.network.neurons.values():
            if int(nrn.metadata.get("layer", 0)) == layer_idx:
                nrn.params.r_base = float(nrn.params.r_base * factor)

    def reset(self) -> None:
        self.sim.reset_simulation()
        self.core.state.current_tick = 0
        self.sim.current_tick = 0

    def patch_to_signals(
        self, patch: torch.Tensor
    ) -> list[tuple[int, int, float]]:
        return image_to_signals(
            patch,
            self.input_layer_ids,
            self.input_synapses_per_neuron,
            self.sim,
            self.dataset_config,
        )

    def step(self, signals: list[tuple[int, int, float]]) -> PopulationState:
        """Inject signals for the current patch and advance one tick."""
        self.core.send_batch_signals(signals)
        self.core.do_tick()
        return self.read_state()

    def read_state(self) -> PopulationState:
        neurons = self.sim.network.neurons
        S = np.empty(self._n, dtype=np.float32)
        F = np.empty(self._n, dtype=np.float32)
        O = np.empty(self._n, dtype=np.float32)
        tr = np.empty(self._n, dtype=np.float32)
        for i, nid in enumerate(self._ids):
            nrn = neurons[nid]
            S[i] = nrn.S
            F[i] = nrn.F_avg
            O[i] = 1.0 if nrn.O > 0 else 0.0
            tr[i] = nrn.t_ref
        return PopulationState(S=S, F_avg=F, O=O, t_ref=tr)

    def set_learning(self, enabled: bool) -> None:
        """Freeze/unfreeze local plasticity by zeroing the learning rates.

        reset_simulation() preserves learned synaptic efficacies (u_i.info), so
        freezing eta lets us probe a fixed learned attractor landscape without
        it drifting during the probe scan.
        """
        if enabled:
            for nid, nrn in self.sim.network.neurons.items():
                if nid in self._saved_eta:
                    nrn.params.eta_post, nrn.params.eta_retro = self._saved_eta[nid]
        else:
            for nid, nrn in self.sim.network.neurons.items():
                self._saved_eta[nid] = (nrn.params.eta_post, nrn.params.eta_retro)
                nrn.params.eta_post = 0.0
                nrn.params.eta_retro = 0.0

    def set_plasticity(self, mode: str | None = None, lr_error: float | None = None,
                       weight_decay_tau: float | None = None,
                       weight_baseline: float | None = None,
                       nm_plasticity_kappa: float | None = None) -> None:
        """Select the plastic rule at runtime for every neuron (no JSON changes).

        mode: 'legacy_multiplicative' (default) or 'error_correcting'.
        weight_decay_tau: >0 enables per-tick passive decay toward weight_baseline.
        """
        for nrn in self.sim.network.neurons.values():
            if mode is not None:
                nrn.params.plasticity_mode = mode
            if lr_error is not None:
                nrn.params.lr_error = float(lr_error)
            if weight_decay_tau is not None:
                nrn.params.weight_decay_tau = float(weight_decay_tau)
            if weight_baseline is not None:
                nrn.params.weight_baseline = float(weight_baseline)
            if nm_plasticity_kappa is not None:
                nrn.params.nm_plasticity_kappa = float(nm_plasticity_kappa)

    def mean_efficacy(self) -> float:
        """Mean informational synaptic efficacy across the net (learning proxy)."""
        vals = self.efficacy_vector()
        return float(vals.mean()) if vals.size else 0.0

    def efficacy_index(self) -> dict[tuple[int, int], int]:
        """Map (neuron_id, synapse_id) -> position in efficacy_vector(). Lets an
        experiment mask the plastic state by which synapses a given input drives."""
        idx: dict[tuple[int, int], int] = {}
        pos = 0
        for nid in self._ids:
            nrn = self.sim.network.neurons[nid]
            for sid in sorted(nrn.postsynaptic_points.keys()):
                idx[(nid, sid)] = pos
                pos += 1
        return idx

    def efficacy_vector(self) -> np.ndarray:
        """Flat vector of every postsynaptic informational efficacy (u_i.info),
        in a stable (neuron, synapse) order. This is the slow/plastic state — the
        substrate's persistent trace of what it has learned."""
        vals: list[float] = []
        for nid in self._ids:
            nrn = self.sim.network.neurons[nid]
            for sid in sorted(nrn.postsynaptic_points.keys()):
                vals.append(float(nrn.postsynaptic_points[sid].u_i.info))
        return np.asarray(vals, dtype=np.float32)

    def broadcast_neuromod(self, m0: float, m1: float) -> None:
        """ALERM volume transmission: nudge every neuron's M_vector toward the
        global (m0, m1) via its gamma-leaky integrator (mirrors the C. elegans
        _volume_broadcast). The substrate only ever sees the scalar modulators,
        never any class information.
        """
        if m0 < 1e-8 and m1 < 1e-8:
            return
        for nrn in self.sim.network.neurons.values():
            gamma = nrn.params.gamma
            if m0 > 1e-8:
                nrn.M_vector[0] += (1.0 - gamma[0]) * m0
            if len(nrn.M_vector) > 1 and m1 > 1e-8:
                nrn.M_vector[1] += (1.0 - gamma[1]) * m1
