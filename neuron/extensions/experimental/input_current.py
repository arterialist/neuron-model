"""Opt-in effective synaptic-current relaxation within a spiking PAULA cell.

Each declared input port owns decaying current components. A native queued
arrival adds its pre-learning, attenuated potential once. This never creates
extra receptor inputs, plasticity updates, or retrograde events. Spiking,
membrane integration, modulation and ongoing adaptation remain inherited.

This is an effective current kernel, not an identified receptor, conductance
model, dendritic cable, or voltage-clamp observation model. The exponential
tail starts at the hillock arrival, with instantaneous rise. Its origin must
be tested independently. All time constants are in model time units.
"""
from __future__ import annotations

import heapq

import numpy as np

from ...neuron import Neuron


class InputCurrentNeuron(Neuron):
    """Configure explicit filtered ports before ticking; other ports bypass.

    With d_j = exp(-dt/tau_j), x_pj <- d_j*x_pj + a_j*arrival_p.
    I_p = sum_j x_pj. In peak mode a_j = fraction_j. In area mode
    a_j = fraction_j / sum_k(fraction_k/(1-d_k)), preserving the same shape
    while conserving every native impulse's discrete charge at constant dt.
    Area mode is the conservative default,
    not a claim that biological synaptic charge has been measured in PAULA units.

    Parameters, addresses and constructor defaults of ordinary neurons do not
    change. No cell type, stimulus identity or decoded neural state is used.
    """

    def configure_current_kernel(self, ports, decay_ticks, peak_fractions,
                                 normalization="area"):
        if hasattr(self, "current_state"):
            raise RuntimeError("Current kernel already configured; use a fresh neuron")
        ports = np.asarray(ports)
        tau = np.asarray(decay_ticks, dtype=float)
        fractions = np.asarray(peak_fractions, dtype=float)
        if (ports.ndim != 1 or ports.dtype.kind not in "iu" or not len(ports)
                or len(np.unique(ports)) != len(ports)
                or any(int(p) not in self.postsynaptic_points for p in ports)):
            raise ValueError("Expected distinct existing input ports")
        if (tau.ndim != 1 or not len(tau) or fractions.shape != tau.shape
                or not np.isfinite(tau).all() or np.any(tau <= 0)
                or not np.isfinite(fractions).all() or np.any(fractions < 0)
                or not np.isclose(fractions.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError("Positive finite decay times and normalized nonnegative fractions required")
        if normalization not in {"area", "peak"}:
            raise ValueError("Normalization must be area or peak")
        self.current_ports = ports.astype(np.int64, copy=True)
        self.current_decay_ticks = tau.copy()
        self.current_fractions = fractions.copy()
        self.current_normalization = normalization
        self._current_rows = {int(p): i for i, p in enumerate(ports)}
        self.current_state = np.zeros((len(ports), len(tau)))
        self.arrived_port_impulse = np.zeros(self.params.num_inputs)
        self.last_port_current = np.zeros(self.params.num_inputs)
        self.total_current = 0.0
        self._current_dt = None
        self._current_tick = None

    def reset_additional_state(self):
        self.current_state.fill(0)
        self.arrived_port_impulse.fill(0)
        self.last_port_current.fill(0)
        self.total_current = 0.0
        self._current_dt = self._current_tick = None

    def tick(self, external_inputs, current_tick, dt=1.0):
        # Validate before native tick mutates any state or schedules arrivals.
        if not hasattr(self, "current_state"):
            raise RuntimeError("Input current kernel is not configured")
        if not np.isfinite(dt) or dt <= 0 or dt > self.params.lambda_param:
            raise ValueError("Require 0 < dt/lambda <= 1")
        if self._current_dt is not None and dt != self._current_dt:
            raise ValueError("Current kernel requires constant dt until reset")
        if self._current_tick is not None and current_tick != self._current_tick + 1:
            raise ValueError("Current kernel requires consecutive ticks until reset")
        if self._current_dt is None:
            self._current_dt = dt
            self._current_decay = np.exp(-dt / self.current_decay_ticks)
            self._current_add = self.current_fractions.copy()
            if self.current_normalization == "area":
                self._current_add /= np.sum(
                    self.current_fractions / -np.expm1(-dt / self.current_decay_ticks))
        events = super().tick(external_inputs, current_tick, dt)
        self._current_tick = current_tick
        return events

    def _hillock_current(self, current_tick, dt):
        self.current_state *= self._current_decay
        self.arrived_port_impulse.fill(0)
        self.last_port_current.fill(0)
        while self.propagation_queue and self.propagation_queue[0][0] <= current_tick:
            _, _, initial, port = heapq.heappop(self.propagation_queue)
            arrived = float(initial * self.params.delta_decay**self.distances[port])
            self.arrived_port_impulse[port] += arrived
            row = self._current_rows.get(port)
            if row is None:
                self.last_port_current[port] += arrived
            else:
                self.current_state[row] += self._current_add * arrived
        self.last_port_current[self.current_ports] = self.current_state.sum(axis=1)
        self.total_current = float(self.last_port_current.sum())
        if not np.isfinite(self.current_state).all() or not np.isfinite(self.total_current):
            raise FloatingPointError("Nonfinite input current; no silent reset")
        return self.total_current
