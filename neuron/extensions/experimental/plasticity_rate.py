"""Experimental local neuromodulatory control of PAULA plasticity rates.

This adds no behaviour, target error, global loss, clock or sample/hold. A
local receptor occupancy scales the existing learning rates. It is a
phenomenological mechanism, not a reconstruction of a particular molecule.

g(M) = 1 + boost * max(M, 0) / (half_saturation + max(M, 0)).

Both adaptation paths use the completed previous tick's M, consistent with
network.py delivering retrograde events before executing neuronal ticks.
The inherited tick updates M normally. Thus a newly delivered transmitter
changes the rate on the next tick, without a special timer.

Metadata: plasticity_rate_boost (default 0, exact base path),
plasticity_rate_index (default 0), plasticity_rate_half_saturation (default .1).
The basal eta_post and eta_retro must both be strictly positive when enabled.
Initially supports legacy_multiplicative only; other rules are not silently
given different semantics. Passive weight decay, if enabled, is not gated.

Motivation: Zenke, Gerstner & Ganguli (2017), doi:10.1016/j.conb.2017.03.015.
That review motivates gating, not this specific equation or its parameters.
"""
from copy import deepcopy
import math

from ...neuron import Neuron


class PlasticityRateNeuron(Neuron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rate_boost = float(self.metadata.get("plasticity_rate_boost", 0.))
        self._rate_index = int(self.metadata.get("plasticity_rate_index", 0))
        self._rate_half = float(self.metadata.get("plasticity_rate_half_saturation", .1))
        self.last_tick_rate_multiplier = 1.
        if not math.isfinite(self._rate_boost) or self._rate_boost < 0:
            raise ValueError("plasticity_rate_boost must be finite and nonnegative")
        if self._rate_boost:
            if not 0 <= self._rate_index < len(self.M_vector):
                raise ValueError("plasticity_rate_index is outside M_vector")
            if not math.isfinite(self._rate_half) or self._rate_half <= 0:
                raise ValueError("plasticity_rate_half_saturation must be finite and positive")
            if self.params.plasticity_mode != "legacy_multiplicative":
                raise ValueError("Rate-gating experiment currently supports legacy_multiplicative only")
            if not all(math.isfinite(x) and x > 0 for x in (self.params.eta_post, self.params.eta_retro)):
                raise ValueError("Enabled rate gate requires strictly positive basal adaptation rates")
            # Base Neuron accepts a shared parameters object. Temporary local
            # rate scaling must never leak to another neuron sharing that object.
            self.params = deepcopy(self.params)

    def rate_multiplier(self):
        if not self._rate_boost:
            return 1.
        concentration = float(self.M_vector[self._rate_index])
        if not math.isfinite(concentration):
            raise FloatingPointError("Non-finite local plasticity modulator")
        concentration = max(0., concentration)
        return 1. + self._rate_boost * concentration / (self._rate_half + concentration)

    def tick(self, external_inputs, current_tick, dt=1.):
        if not self._rate_boost:
            return super().tick(external_inputs, current_tick, dt)
        gain = self.rate_multiplier()
        self.last_tick_rate_multiplier = gain
        basal = self.params.eta_post
        self.params.eta_post = basal * gain
        try:
            return super().tick(external_inputs, current_tick, dt)
        finally:
            self.params.eta_post = basal

    def process_retrograde_signal(self, event):
        if not self._rate_boost:
            return super().process_retrograde_signal(event)
        basal = self.params.eta_retro
        self.params.eta_retro = basal * self.rate_multiplier()
        try:
            return super().process_retrograde_signal(event)
        finally:
            self.params.eta_retro = basal
