"""Opt-in use-dependent terminal release in a spiking PAULA neuron.

Each declared terminal owns an available fraction R. Between spikes it
recovers toward one with a fixed time constant. A native forward event emits
its existing coefficient times R, then consumes a fraction u of R. The native
weights, electrical state, chemical inputs and return events are unchanged.

This is an effective short-term depression model, not a claim about vesicle
counts or a substitute for long-term PAULA learning. No stimulus labels,
neuronal identities or experiment schedules enter the cellular mechanism.
"""
from __future__ import annotations

import numpy as np

from ...neuron import Neuron


class DepressingReleaseNeuron(Neuron):
    def configure_release_depression(self, terminals, depletion_fraction, recovery_ticks):
        if hasattr(self, "release_available"):
            raise RuntimeError("Release dynamics already configured")
        terminals = np.asarray(terminals)
        if (terminals.ndim != 1 or terminals.dtype.kind not in "iu" or not len(terminals)
                or len(np.unique(terminals)) != len(terminals)
                or any(int(t) not in self.presynaptic_points for t in terminals)):
            raise ValueError("Need distinct existing terminal ids")
        if (not np.isfinite(depletion_fraction) or not 0 <= depletion_fraction <= 1
                or not np.isfinite(recovery_ticks) or recovery_ticks <= 0):
            raise ValueError("Need 0 <= depletion <= 1 and positive recovery time")
        self.release_terminals = terminals.astype(np.int64, copy=True)
        self.release_depletion = float(depletion_fraction)
        self.release_recovery_ticks = float(recovery_ticks)
        self.release_available = np.ones(len(terminals))
        self.release_used_fraction = np.zeros(len(terminals))
        self.release_native_amplitude = np.zeros(len(terminals))
        self.release_effective_amplitude = np.zeros(len(terminals))
        self._release_rows = {int(t): i for i,t in enumerate(terminals)}
        self._release_tick = self._release_dt = None

    def reset_additional_state(self):
        self.release_available.fill(1.)
        self.release_used_fraction.fill(0.)
        self.release_native_amplitude.fill(0.)
        self.release_effective_amplitude.fill(0.)
        self._release_tick = self._release_dt = None

    def tick(self, external_inputs, current_tick, dt=1.):
        if not hasattr(self, "release_available"):
            raise RuntimeError("Release depression is not configured")
        if (not np.isfinite(dt) or dt <= 0 or dt > self.params.lambda_param
                or (self._release_dt is not None and dt != self._release_dt)
                or (self._release_tick is not None and current_tick != self._release_tick+1)):
            raise ValueError("Release dynamics require consecutive ticks and constant stable dt")
        if self._release_dt is None:
            self._release_dt = dt
            self._release_recovery = -np.expm1(-dt/self.release_recovery_ticks)
        self.release_available += (1.-self.release_available)*self._release_recovery
        self.release_used_fraction.fill(0.)
        self.release_native_amplitude.fill(0.)
        self.release_effective_amplitude.fill(0.)
        native = super().tick(external_inputs, current_tick, dt)
        result = []
        for event in native:
            row = self._release_rows.get(event[1]) if isinstance(event, tuple) else None
            if row is None:
                result.append(event)
                continue
            available = float(self.release_available[row])
            self.release_used_fraction[row] = available
            self.release_native_amplitude[row] = event[2]
            amplitude = event[2] if available == 1. else event[2]*available
            self.release_effective_amplitude[row] = amplitude
            self.release_available[row] *= 1.-self.release_depletion
            result.append(event if available == 1. else (event[0], event[1], amplitude))
        if (not np.isfinite(self.release_available).all()
                or not np.isfinite(self.release_effective_amplitude).all()
                or np.any(self.release_available < 0) or np.any(self.release_available > 1)):
            raise FloatingPointError("Invalid release state; no clipping or silent reset")
        self._release_tick = current_tick
        return result
