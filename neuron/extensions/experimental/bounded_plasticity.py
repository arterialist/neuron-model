"""Experimental signed-magnitude plasticity with local neuromodulatory rates.

Opt in with metadata bounded_plasticity=True. The default calls the existing
PlasticityRateNeuron unchanged. This is a phenomenological hypothesis, not a
faithful receptor model or an implementation of the cited papers' STDP rules.

For q=abs(w), held local error magnitude e and native timing direction d:
  d=+1: dq/ds = q * [e * (1-q/C) - decay]
  d=-1: dq/ds = -q * (e+decay)
Integrate this scalar equation exactly over s=effective_eta_post. This prevents
an Euler overshoot from changing a synapse's sign. It is exact only for held e
within an update, not for the entire coupled neural system. Positive basal
adaptation and the existing previous-tick neural rate receptor remain active.

C defaults to 10, the native positive soft-bound scale; decay defaults to .02.
Neither is fitted to stimulus identity. Zero weights remain zero, as in the
native multiplicative rule. This is not structural growth or synaptogenesis.
Extreme depression can underflow; it is reported, not replaced by a hidden floor.

The base currently has no postsynaptic-update hook. This subclass captures local
pre-update signals, runs the inherited tick, then replaces ONLY active incoming
info weights before returning. Inherited propagation and retrograde events use
pre-update weights/errors, so they retain the native causal order. Base DEBUG
weight messages describe provisional native updates; final replacements are
logged separately. No other cell can observe the provisional weights through
the standard Network.run_tick loop. Passive decay and ablations are rejected
when enabled rather than silently changing their semantics.

Motivation, not derivation: van Rossum, Bi & Turrigiano (2000),
doi:10.1523/JNEUROSCI.20-23-08812.2000; Gutig et al. (2003),
doi:10.1523/JNEUROSCI.23-09-03697.2003. Stable bounds do not prove useful learning.
"""
import math

import numpy as np

from .plasticity_rate import PlasticityRateNeuron


def magnitude_step(q, error, direction, eta, cap=10., decay=.02):
    """Frozen-coefficient exact flow; no hard clipping or minimum weight floor."""
    if not all(math.isfinite(x) for x in (q, error, eta, cap, decay)):
        raise ValueError("Plasticity inputs must be finite")
    if not 0 <= q <= cap or error < 0 or eta < 0 or cap <= 0 or decay < 0 or direction not in (-1, 1):
        raise ValueError("Invalid local magnitude dynamics")
    if q == 0 or eta == 0:
        return q
    if direction < 0:
        return q*math.exp(-eta*(error+decay))
    a, b = error-decay, error/cap
    if abs(a*eta) < 1e-8:
        # expm1 avoids cancellation; use the a=0 limit exactly.
        h = eta if a == 0 else math.expm1(a*eta)/a
        return q*math.exp(a*eta)/(1.+b*q*h)
    if a > 0:
        # This reciprocal form cannot overflow for large potentiation steps.
        z = math.exp(-a*eta)
        return q/(z+(b*q/a)*(-math.expm1(-a*eta)))
    z = math.exp(a*eta)
    return q*z/(1.+(b*q/a)*math.expm1(a*eta))


class BoundedPlasticityNeuron(PlasticityRateNeuron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bounded_enabled = bool(self.metadata.get("bounded_plasticity", False))
        self._magnitude_cap = float(self.metadata.get("plasticity_magnitude_cap", 10.))
        self._magnitude_decay = float(self.metadata.get("plasticity_magnitude_decay", .02))
        self.bounded_updates = 0
        self.bounded_underflows = 0
        if self._bounded_enabled:
            if (self.params.plasticity_mode != "legacy_multiplicative" or
                    self.params.weight_decay_tau != 0 or self._ablation):
                raise ValueError("Bounded rule requires legacy timing, no passive decay and no ablations")
            if not all(math.isfinite(x) and x > 0 for x in (self.params.eta_post, self.params.eta_retro)):
                raise ValueError("Bounded rule requires positive basal adaptation")
            if not math.isfinite(self._magnitude_cap) or self._magnitude_cap <= 0:
                raise ValueError("Magnitude cap must be finite and positive")
            if not math.isfinite(self._magnitude_decay) or self._magnitude_decay < 0:
                raise ValueError("Magnitude decay must be finite and nonnegative")
            if self.params.w_min > -self._magnitude_cap or self.params.w_max < self._magnitude_cap:
                raise ValueError("Native bounds must contain the signed magnitude interval")

    def tick(self, external_inputs, current_tick, dt=1.):
        if not self._bounded_enabled:
            return super().tick(external_inputs, current_tick, dt)
        active = []
        for sid in np.flatnonzero(self.input_buffer[:, 0] > 0):
            syn = self.postsynaptic_points.get(sid)
            if syn is None:
                continue
            before = float(syn.u_i.info)
            if not math.isfinite(before) or abs(before) > self._magnitude_cap:
                raise ValueError("Active weight is outside the declared magnitude interval")
            row = self.input_buffer[sid]
            error = float(np.linalg.norm(np.array([row[0]-syn.u_i.info,
                          row[1]-syn.u_i.plast, *row[2:]])))
            if not math.isfinite(error):
                raise FloatingPointError("Nonfinite local plasticity error")
            active.append((sid, before, error))
        eta = self.params.eta_post*self.rate_multiplier()
        events = super().tick(external_inputs, current_tick, dt)
        direction = 1 if current_tick-self.t_last_fire <= self.t_ref else -1
        for sid, before, error in active:
            magnitude = magnitude_step(abs(before), error, direction, eta,
                                       self._magnitude_cap, self._magnitude_decay)
            after = math.copysign(magnitude, before)
            self.postsynaptic_points[sid].u_i.info = after
            self.bounded_updates += int(before != after)
            self.bounded_underflows += int(before != 0 and after == 0)
            if self._debug_ticks:
                self.logger.debug(f"Bounded info update {sid}: {before!r} -> {after!r}")
        return events
